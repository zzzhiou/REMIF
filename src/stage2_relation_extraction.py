"""
Stage 2: 关系抽取和图构建模块

6 种关系类型:
1. contrast: 对立关系（对 A 和 B 立场相反/冲突）
2. consistent: 一致关系（对 A 和 B 立场相同）
3. causal: 因果关系（A 导致/影响 B）
4. hierarchical: 层级关系（A 是 B 的上位/下位概念）
5. parallel: 并列关系（A 和 B 同级，无立场关联）
6. analogy: 类比关系（用 A 类比 B）
"""

import json
import re
from typing import List, Dict, Any, Tuple
import networkx as nx


# 关系抽取 Prompt
RELATION_EXTRACTION_SYSTEM_PROMPT = """你是一个关系抽取专家。

任务：判断给定目标对之间的关系类型。

关系类型（只能从以下 6 种中选择）：
1. contrast: 对立关系（对 A 和 B 立场相反/冲突，如"支持 A 但反对 B"）
2. consistent: 一致关系（对 A 和 B 立场相同，如"都支持"或"都反对"）
3. causal: 因果关系（A 导致/影响 B，如"A 会导致 B"）
4. hierarchical: 层级关系（A 是 B 的上位/下位概念，如"A 包括 B"）
5. parallel: 并列关系（A 和 B 同级并列，无明显立场关联）
6. analogy: 类比关系（用 A 类比 B，如"A 就像 B"）

方向性说明：
- contrast: 双向关系
- consistent: 双向关系
- causal: 有向关系 (A→B 表示 A 导致 B)
- hierarchical: 有向关系 (A→B 表示 A 是 B 的上位概念)
- parallel: 无向关系
- analogy: 有向关系 (A→B 表示用 A 类比 B)

输出格式（严格 JSON）：
{
  "relation": "causal",
  "direction": "A→B",
  "evidence": "原文证据片段",
  "confidence": 0.85,
  "reasoning": "简短理由"
}

如果两个目标之间没有明显关系，输出：
{
  "relation": "none",
  "direction": "none",
  "evidence": "",
  "confidence": 1.0,
  "reasoning": "无明显关系"
}
"""

RELATION_EXTRACTION_USER_PROMPT = """目标 A: {target_a}
目标 B: {target_b}

原文：
{context}

请判断关系（只返回 JSON，不要其他内容）："""


class RelationExtractor:
    """关系抽取器"""

    def __init__(self, llm_client=None, confidence_threshold=0.5):
        """
        Args:
            llm_client: LLM 客户端
            confidence_threshold: 置信度阈值
        """
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

    def extract(self, target_a: Dict[str, Any], target_b: Dict[str, Any],
                context: str) -> Dict[str, Any]:
        """
        抽取两个目标之间的关系

        Args:
            target_a: 目标 A
            target_b: 目标 B
            context: 原文上下文

        Returns:
            关系信息
        """
        if self.llm_client is None:
            raise ValueError("LLM client is not set")

        # 构建 prompt
        user_prompt = RELATION_EXTRACTION_USER_PROMPT.format(
            target_a=target_a['text'],
            target_b=target_b['text'],
            context=context
        )

        # 调用 LLM
        response = self.llm_client.generate(
            system_prompt=RELATION_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        # 解析 JSON
        relation = self._parse_json_response(response)
        relation['target_a'] = target_a['id']
        relation['target_b'] = target_b['id']

        return relation

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 的 JSON 响应"""
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, response)

        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                return {
                    'relation': data.get('relation', 'none'),
                    'direction': data.get('direction', 'none'),
                    'evidence': data.get('evidence', ''),
                    'confidence': float(data.get('confidence', 0.5)),
                    'reasoning': data.get('reasoning', '')
                }
            except json.JSONDecodeError:
                print(f"JSON 解析失败：{json_str}")
                return {'relation': 'none', 'direction': 'none', 'evidence': '',
                        'confidence': 1.0, 'reasoning': '解析失败'}
        else:
            return {'relation': 'none', 'direction': 'none', 'evidence': '',
                    'confidence': 1.0, 'reasoning': '未找到 JSON'}

    def extract_all(self, targets: List[Dict[str, Any]],
                    context: str) -> List[Dict[str, Any]]:
        """
        抽取所有目标对之间的关系

        Args:
            targets: 目标列表
            context: 原文上下文

        Returns:
            关系列表
        """
        relations = []
        n = len(targets)

        for i in range(n):
            for j in range(i + 1, n):
                relation = self.extract(targets[i], targets[j], context)
                # 只保留置信度高于阈值的关系
                if relation['relation'] != 'none' and \
                   relation['confidence'] >= self.confidence_threshold:
                    relations.append(relation)

        return relations


class HeterogeneousGraphBuilder:
    """异质图构建器"""

    # 关系类型映射
    RELATION_TYPES = {
        'contrast': 0,
        'consistent': 1,
        'causal': 2,
        'hierarchical': 3,
        'parallel': 4,
        'analogy': 5,
        'none': -1,
    }

    def __init__(self):
        pass

    def build(self, targets: List[Dict[str, Any]],
              relations: List[Dict[str, Any]]) -> nx.MultiDiGraph:
        """
        构建异质图

        Args:
            targets: 目标列表
            relations: 关系列表

        Returns:
            NetworkX 图
        """
        G = nx.MultiDiGraph()

        # 添加节点
        for target in targets:
            G.add_node(
                target['id'],
                text=target['text'],
                normalized=target['normalized'],
                granularity_level=target['granularity_level'],
                type=target['type'],
                evidence=target.get('evidence', '')
            )

        # 添加边
        for relation in relations:
            if relation['relation'] == 'none':
                continue

            src = relation['target_a']
            tgt = relation['target_b']
            rel_type = relation['relation']

            # 根据方向添加边
            if relation['direction'] == 'A→B':
                G.add_edge(
                    src, tgt,
                    relation_type=rel_type,
                    relation_id=self.RELATION_TYPES.get(rel_type, -1),
                    evidence=relation['evidence'],
                    confidence=relation['confidence']
                )
            elif relation['direction'] == 'B→A':
                G.add_edge(
                    tgt, src,
                    relation_type=rel_type,
                    relation_id=self.RELATION_TYPES.get(rel_type, -1),
                    evidence=relation['evidence'],
                    confidence=relation['confidence']
                )
            else:  # bidirectional
                G.add_edge(
                    src, tgt,
                    relation_type=rel_type,
                    relation_id=self.RELATION_TYPES.get(rel_type, -1),
                    evidence=relation['evidence'],
                    confidence=relation['confidence']
                )
                G.add_edge(
                    tgt, src,
                    relation_type=rel_type,
                    relation_id=self.RELATION_TYPES.get(rel_type, -1),
                    evidence=relation['evidence'],
                    confidence=relation['confidence']
                )

        return G

    def to_pyg_format(self, G: nx.MultiDiGraph,
                      node_embeddings: Any) -> Tuple[Any, Any, Any]:
        """
        转换为 PyTorch Geometric 格式

        Args:
            G: NetworkX 图
            node_embeddings: 节点特征矩阵 [num_nodes, feature_dim]

        Returns:
            edge_index: [2, num_edges]
            edge_type: [num_edges]
            node_ids: 节点 ID 列表
        """
        import torch

        # 节点 ID 映射
        node_ids = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_ids)}

        # 构建边
        edge_list = []
        edge_types = []

        for u, v, data in G.edges(data=True):
            edge_list.append([node_to_idx[u], node_to_idx[v]])
            edge_types.append(data.get('relation_id', 0))

        if len(edge_list) == 0:
            # 空图处理
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_type = torch.tensor(edge_types, dtype=torch.long)

        return edge_index, edge_type, node_ids

    def get_graph_description(self, G: nx.MultiGraph) -> str:
        """
        获取图的文本描述（用于 LLM 输入）

        Args:
            G: NetworkX 图

        Returns:
            图的文本描述
        """
        lines = []

        # 节点信息
        lines.append("【目标节点】")
        for node_id, data in G.nodes(data=True):
            lines.append(f"- {node_id}: {data['normalized']} "
                        f"(层级:{data['granularity_level']}, {data['type']})")

        # 关系信息
        lines.append("\n【目标关系】")
        edges_seen = set()
        for u, v, data in G.edges(data=True):
            edge_key = (min(u, v), max(u, v))
            if edge_key in edges_seen:
                continue
            edges_seen.add(edge_key)
            lines.append(f"- {u} → {v}: {data['relation_type']} "
                        f"(置信度:{data['confidence']:.2f})")

        return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    test_targets = [
        {'id': 'T1', 'text': '《信条》', 'normalized': '《信条》',
         'granularity_level': 2, 'type': 'explicit', 'evidence': '...'},
        {'id': 'T2', 'text': '《盗梦空间》', 'normalized': '《盗梦空间》',
         'granularity_level': 2, 'type': 'explicit', 'evidence': '...'},
    ]

    test_relations = [
        {'target_a': 'T1', 'target_b': 'T2', 'relation': 'contrast',
         'direction': 'bidirectional', 'evidence': '口碑不如',
         'confidence': 0.9, 'reasoning': '...'}
    ]

    builder = HeterogeneousGraphBuilder()
    G = builder.build(test_targets, test_relations)

    print(f"图：{G.number_of_nodes()} 节点，{G.number_of_edges()} 边")
    print(builder.get_graph_description(G))