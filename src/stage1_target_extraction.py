"""
Stage 1: 目标抽取模块

功能:
1. 使用 LLM 从文本中抽取立场目标
2. 控制目标粒度（只保留层级 2 和 3）
3. 判断目标的显性/隐性
4. 语义去重
"""

import json
import re
from typing import List, Dict, Any

# 延迟导入，避免初始化时失败
SENTENCE_TRANSFORMERS_AVAILABLE = False

def _try_import_sentence_transformers():
    global SENTENCE_TRANSFORMERS_AVAILABLE
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            global SentenceTransformer, DBSCAN, cosine_similarity
            from sentence_transformers import SentenceTransformer
            from sklearn.cluster import DBSCAN
            from sklearn.metrics.pairwise import cosine_similarity
            SENTENCE_TRANSFORMERS_AVAILABLE = True
        except (ImportError, ValueError) as e:
            print(f"注意：sentence-transformers 不可用 ({e})，将使用简单去重方法")
            SentenceTransformer = None
            DBSCAN = None
            cosine_similarity = None
    return SENTENCE_TRANSFORMERS_AVAILABLE


# 目标抽取 Prompt
TARGET_EXTRACTION_SYSTEM_PROMPT = """你是一个立场目标抽取专家。

任务：从文本中抽取所有被表达立场的目标（显性和隐性）。

## 目标定义
立场目标是文本中表达态度、观点或立场的对象，包括：
1. **实体型**：人名、组织、产品、事件
2. **议题型**：抽象概念、问题、现象
3. **问题型**：文本中讨论的核心问题

## 关键原则
1. **简洁性**：目标是精炼的名词短语（2-10 字），不是完整句子
2. **完整性**：抽取所有目标，不要遗漏
3. **抽象层级**：
   - 讨论具体实体 → 用实体名（如"杨子"）
   - 讨论抽象问题 → 用抽象表述（如"涉事产品"）
   
粒度控制规则（必须严格遵守）：
- 层级 1（太宏观，不要抽取）：抽象理念如"环保"、"经济发展"、"口碑"
- 层级 2（合适）：具体对象/作品/政策如"《信条》"、"《盗梦空间》"、"碳税政策"
- 层级 3（合适）：具体方面/细节如"电影剧情"、"演员演技"、"碳税税率设定"
- 层级 4（太细，不要抽取）：具体事件/评论如"某次抗议活动"、"某条微博"

显性/隐性判断标准：
- 显性目标：文本直接提及，有明确词汇指向（如"支持"、"反对"、"喜欢"的对象）
- 隐性目标：文本暗示但未直接提及，需要从上下文中推断（如被比较、被因果关联的对象）

请按以下步骤思考（Chain of Thought）：
1. 通读全文，标记所有可能被讨论的对象
2. 对每个对象，判断其粒度层级（1-4）
3. 只保留层级 2 和 3 的目标
4. 对每个目标，判断是显性还是隐性
5. 检查是否有语义重复的目标，进行合并

输出格式（严格 JSON）：
{
  "targets": [
    {
      "id": "T1",
      "text": "原文表述",
      "normalized": "规范化表述（去掉修饰词）",
      "granularity_level": 2,
      "type": "explicit",
      "evidence": "支撑的原文片段",
      "reasoning": "判断理由（简述）"
    }
  ]
}
"""

TARGET_EXTRACTION_USER_PROMPT = """文本：
{text}

请抽取目标（只返回 JSON，不要其他内容）："""


class TargetExtractor:
    """目标抽取器"""

    def __init__(self, llm_client=None, dedup_threshold=0.85):
        """
        Args:
            llm_client: LLM 客户端，需要有 generate 方法
            dedup_threshold: 语义去重阈值
        """
        self.llm_client = llm_client
        self.dedup_threshold = dedup_threshold
        self.embedding_model = None

    def _get_embedding_model(self):
        """懒加载 embedding 模型"""
        if not _try_import_sentence_transformers():
            raise RuntimeError(
                "sentence-transformers 未安装或不可用，请运行：pip install sentence-transformers"
            )
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return self.embedding_model

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中抽取目标

        Args:
            text: 输入文本

        Returns:
            目标列表
        """
        if self.llm_client is None:
            raise ValueError("LLM client is not set")

        # 构建 prompt
        user_prompt = TARGET_EXTRACTION_USER_PROMPT.format(text=text)

        # 调用 LLM
        response = self.llm_client.generate(
            system_prompt=TARGET_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        # 解析 JSON
        targets = self._parse_json_response(response)

        return targets

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """解析 LLM 的 JSON 响应"""
        # 尝试提取 JSON 部分
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, response)

        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                return data.get('targets', [])
            except json.JSONDecodeError:
                print(f"JSON 解析失败：{json_str}")
                return []
        else:
            print(f"未找到 JSON 内容：{response}")
            return []

    def deduplicate(self, targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        语义去重

        Args:
            targets: 目标列表

        Returns:
            去重后的目标列表
        """
        if len(targets) <= 1:
            return targets

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            # 如果没有安装 sentence-transformers，使用简单的字符串匹配去重
            print("警告：使用简单去重（建议安装 sentence-transformers 以获得更好的效果）")
            return self._simple_dedup(targets)

        model = self._get_embedding_model()

        # 编码 normalized 表述
        texts = [t['normalized'] for t in targets]
        embeddings = model.encode(texts)

        # 计算余弦相似度矩阵
        similarity_matrix = cosine_similarity(embeddings)

        # 聚类去重
        clustering = DBSCAN(
            eps=1 - self.dedup_threshold,
            metric='precomputed'
        ).fit(1 - similarity_matrix)

        # 每簇保留一个代表
        deduped = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # 噪声点，单独保留
                for i, label in enumerate(clustering.labels_):
                    if label == -1:
                        deduped.append(targets[i])
            else:
                cluster_targets = [
                    t for t, l in zip(targets, clustering.labels_)
                    if l == cluster_id
                ]
                # 优先级：显性目标 > 隐性目标，层级 2 > 层级 3
                cluster_targets.sort(
                    key=lambda x: (
                        0 if x['type'] == 'explicit' else 1,
                        x['granularity_level']
                    )
                )
                deduped.append(cluster_targets[0])

        # 重新编号
        for i, target in enumerate(deduped):
            target['id'] = f"T{i + 1}"

        return deduped

    def _simple_dedup(self, targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """简单去重（基于字符串相似度）"""
        deduped = []
        seen_texts = set()

        for target in sorted(targets, key=lambda x: (-len(x['normalized']), x['id'])):
            text = target['normalized'].lower()
            # 检查是否与已有目标重复
            is_dup = False
            for seen in seen_texts:
                if text in seen or seen in text:
                    is_dup = True
                    break

            if not is_dup:
                deduped.append(target)
                seen_texts.add(text)

        # 重新编号
        for i, target in enumerate(deduped):
            target['id'] = f"T{i + 1}"

        return deduped

    def process(self, text: str) -> List[Dict[str, Any]]:
        """
        完整的目标抽取流程（抽取 + 去重）

        Args:
            text: 输入文本

        Returns:
            去重后的目标列表
        """
        targets = self.extract(text)
        deduped_targets = self.deduplicate(targets)
        return deduped_targets


if __name__ == "__main__":
    # 测试简单去重
    test_targets = [
        {'id': 'T1', 'text': '《信条》', 'normalized': '《信条》',
         'granularity_level': 2, 'type': 'explicit'},
        {'id': 'T2', 'text': '《盗梦空间》', 'normalized': '《盗梦空间》',
         'granularity_level': 2, 'type': 'explicit'},
        {'id': 'T3', 'text': '信条电影', 'normalized': '信条电影',
         'granularity_level': 2, 'type': 'implicit'},
    ]

    extractor = TargetExtractor()
    deduped = extractor._simple_dedup(test_targets)
    print(f"去重结果：{len(deduped)} 个目标")
    for t in deduped:
        print(f"  - {t['id']}: {t['normalized']}")
