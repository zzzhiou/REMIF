"""
Stage 3: R-GCN + LLM 立场检测模块

架构:
1. 使用 BERT 编码目标节点
2. 使用 R-GCN 聚合图结构信息
3. 将 R-GCN 输出 + 图结构线性化输入 LLM
4. LLM 推理出每个目标的立场
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import numpy as np

# 延迟导入 torch_geometric
PYG_AVAILABLE = False

def _try_import_pyg():
    global PYG_AVAILABLE
    if not PYG_AVAILABLE:
        try:
            global RGCNConv, from_networkx
            from torch_geometric.nn import RGCNConv
            from torch_geometric.utils import from_networkx
            PYG_AVAILABLE = True
        except (ImportError, ModuleNotFoundError) as e:
            print(f"注意：torch_geometric 不可用 ({e})，R-GCN 将无法使用")
            RGCNConv = None
            from_networkx = None
    return PYG_AVAILABLE


# ============ R-GCN 模型 ============

class TargetRGCN(nn.Module):
    """
    用于目标立场检测的 R-GCN 编码器
    """

    def __init__(self, num_node_features: int, hidden_dim: int,
                 num_relations: int, num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            num_node_features: 节点特征维度（BERT hidden size）
            hidden_dim: 隐藏层维度
            num_relations: 关系类型数量
            num_layers: RGCN 层数
            dropout: Dropout 比率
        """
        super().__init__()

        if not _try_import_pyg():
            raise RuntimeError("torch_geometric 不可用，无法创建 R-GCN 模型")

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # R-GCN 层
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # 第一层
        self.convs.append(RGCNConv(num_node_features, hidden_dim, num_relations))
        self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # 中间层和输出层
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, num_features] 节点特征
            edge_index: [2, num_edges] 边索引
            edge_type: [num_edges] 边类型
        Returns:
            [num_nodes, hidden_dim] 节点表示
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_type)
            x = self.layer_norms[i](x)

            if i < self.num_layers - 1:  # 最后一层不加激活
                x = self.relu(x)
                x = self.dropout(x)

        return x


class StanceClassifier(nn.Module):
    """
    立场分类头（可选，用于端到端训练）
    """

    def __init__(self, input_dim: int, num_classes: int = 3,
                 num_intensity: int = 3):
        """
        Args:
            input_dim: 输入维度
            num_classes: 立场类别数（支持/中立/反对）
            num_intensity: 强度类别数（强/中/弱）
        """
        super().__init__()

        self.polarity_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, num_classes)
        )

        self.intensity_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, num_intensity)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [num_nodes, input_dim]
        Returns:
            polarity: [num_nodes, num_classes]
            intensity: [num_nodes, num_intensity]
        """
        polarity = self.polarity_head(x)
        intensity = self.intensity_head(x)
        return polarity, intensity


class GraphEnhancedStanceDetector(nn.Module):
    """
    图增强的立场检测器（R-GCN + 分类头）
    """

    def __init__(self, bert_hidden_size: int = 768, rgcn_hidden_dim: int = 256,
                 num_relations: int = 6, num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            bert_hidden_size: BERT 隐藏层维度
            rgcn_hidden_dim: R-GCN 隐藏层维度
            num_relations: 关系类型数
            num_layers: R-GCN 层数
            dropout: Dropout 比率
        """
        super().__init__()

        self.rgcn = TargetRGCN(
            num_node_features=bert_hidden_size,
            hidden_dim=rgcn_hidden_dim,
            num_relations=num_relations,
            num_layers=num_layers,
            dropout=dropout
        )

        self.classifier = StanceClassifier(
            input_dim=rgcn_hidden_dim,
            num_classes=3,
            num_intensity=3
        )

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [num_nodes, bert_hidden_size]
            edge_index: [2, num_edges]
            edge_type: [num_edges]
        Returns:
            polarity: [num_nodes, 3]
            intensity: [num_nodes, 3]
        """
        # R-GCN 编码
        graph_repr = self.rgcn(node_features, edge_index, edge_type)

        # 立场分类
        polarity, intensity = self.classifier(graph_repr)

        return polarity, intensity


# ============ 图线性化 Prompt ============

STANCE_DETECTION_PROMPT = """你是立场分析领域的资深专家，擅长判断文本对特定目标的立场态度。

任务：分析文本中对每个目标的立场。

【目标列表】（按重要性排序）
{target_list}

【目标关系】
{relation_list}

【原始文本】
{text}

【R-GCN 计算的目标重要性】
{rgcn_importance}

【立场分类体系】
| 类别 | 定义 | 典型信号词 |
|-----|------|-----------|
| **支持** | 赞同、肯定、推荐、期待 | "支持"、"涨"、"好"、"期待"、"推荐" |
| **反对** | 批评、否定、质疑、担忧 | "反对"、"说谎"、"为啥"、"质疑"、"抵制" |
| **中立** | 客观描述、无明显倾向 | 纯描述性语言、疑问但无倾向 |



【输出要求】
对每个目标，判断：
1. 立场极性：支持 /反对 /中立
2. 证据：原文片段

输出格式（严格 JSON）：
{{
  "stances": [
    {{
      "target_id": "T1",
      "polarity": "support",
      "evidence": "原文证据",
      "reasoning": "推理过程简述"
    }}
  ]
}}
"""


class RGCNLLMStanceDetector:
    """
    R-GCN + LLM 联合立场检测器
    """

    def __init__(self, bert_model, rgcn_model=None, llm_client=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            bert_model: BERT 模型（用于编码目标）
            rgcn_model: R-GCN 模型（可选）
            llm_client: LLM 客户端
            device: 设备
        """
        self.bert_model = bert_model
        self.rgcn_model = rgcn_model
        self.llm_client = llm_client
        self.device = device

        if bert_model is not None:
            self.bert_model.to(device)
            self.bert_model.eval()

    def encode_targets(self, targets: List[Dict[str, Any]],
                       context: str) -> torch.Tensor:
        """
        使用 BERT 编码目标

        Args:
            targets: 目标列表
            context: 原文上下文

        Returns:
            [num_targets, hidden_size] 目标表示
        """
        from transformers import BertTokenizer

        if self.bert_model is None:
            raise ValueError("BERT model is not set")

        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        embeddings = []
        for target in targets:
            # 使用目标文本 + 上下文编码
            text = f"{target['normalized']}。{context[:200]}"

            inputs = tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(**inputs)

            # 取 [CLS] token 表示
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding)

        return torch.cat(embeddings, dim=0)  # [num_targets, hidden_size]

    def compute_target_importance(self, graph, rgcn_output: torch.Tensor) -> Dict[str, float]:
        """
        计算目标重要性（基于 R-GCN 输出和图中心性）

        Args:
            graph: NetworkX 图
            rgcn_output: R-GCN 输出 [num_nodes, hidden_dim]

        Returns:
            目标重要性字典
        """
        import networkx as nx

        importance = {}

        # 1. R-GCN 向量范数
        if rgcn_output is not None:
            norms = torch.norm(rgcn_output, dim=1).cpu().numpy()
            for i, node_id in enumerate(graph.nodes()):
                importance[node_id] = norms[i]
        else:
            # 2. 使用度中心性
            centrality = nx.degree_centrality(graph)
            importance = centrality

        return importance

    def create_stance_prompt(self, graph, targets: List[Dict[str, Any]],
                             relations: List[Dict[str, Any]],
                             text: str,
                             rgcn_output: torch.Tensor = None) -> str:
        """
        创建立场检测的 Prompt

        Args:
            graph: NetworkX 图
            targets: 目标列表
            relations: 关系列表
            text: 原文
            rgcn_output: R-GCN 输出

        Returns:
            Prompt 字符串
        """
        # 计算重要性
        importance = self.compute_target_importance(graph, rgcn_output)

        # 排序
        sorted_targets = sorted(
            targets,
            key=lambda t: importance.get(t['id'], 0),
            reverse=True
        )

        # 构建目标列表
        target_list = []
        for t in sorted_targets:
            imp = importance.get(t['id'], 0)
            target_list.append(
                f"- {t['id']}: {t['normalized']} "
                f"(重要性:{imp:.3f}, 层级:{t['granularity_level']}, {t['type']})"
            )

        # 构建关系列表
        relation_list = []
        seen = set()
        for r in relations:
            key = (min(r['target_a'], r['target_b']),
                   max(r['target_a'], r['target_b']))
            if key in seen:
                continue
            seen.add(key)
            relation_list.append(
                f"- {r['target_a']} → {r['target_b']}: {r['relation']} "
                f"(置信度:{r['confidence']:.2f})"
            )

        # R-GCN 重要性描述
        rgcn_importance = "\n".join([
            f"- {t['id']}: {importance.get(t['id'], 0):.3f}"
            for t in sorted_targets
        ])

        prompt = STANCE_DETECTION_PROMPT.format(
            target_list="\n".join(target_list),
            relation_list="\n".join(relation_list),
            text=text,
            rgcn_importance=rgcn_importance
        )

        return prompt

    def detect(self, graph, targets: List[Dict[str, Any]],
               relations: List[Dict[str, Any]],
               text: str,
               use_rgcn: bool = True) -> List[Dict[str, Any]]:
        """
        检测立场

        Args:
            graph: NetworkX 图
            targets: 目标列表
            relations: 关系列表
            text: 原文
            use_rgcn: 是否使用 R-GCN

        Returns:
            立场列表
        """
        # 编码目标
        rgcn_output = None
        if use_rgcn and self.rgcn_model is not None and self.bert_model is not None:
            node_features = self.encode_targets(targets, text)

            # 转换为 PyG 格式
            from torch_geometric.utils import from_networkx
            pyg_graph = from_networkx(graph)
            edge_index = pyg_graph.edge_index.to(self.device)
            edge_type = pyg_graph.edge_type.to(self.device) if hasattr(pyg_graph, 'edge_type') else None

            if edge_type is None:
                edge_type = torch.zeros_like(edge_index[0])

            # R-GCN 编码
            self.rgcn_model.eval()
            with torch.no_grad():
                rgcn_output = self.rgcn_model(
                    node_features.to(self.device),
                    edge_index,
                    edge_type
                )

        # 创建 Prompt
        prompt = self.create_stance_prompt(
            graph, targets, relations, text, rgcn_output
        )

        # 调用 LLM
        if self.llm_client is None:
            raise ValueError("LLM client is not set")

        response = self.llm_client.generate(
            system_prompt="你是一个立场检测专家。请严格按照 JSON 格式输出。",
            user_prompt=prompt
        )

        # 解析响应
        stances = self._parse_stance_response(response)

        return stances

    def _parse_stance_response(self, response: str) -> List[Dict[str, Any]]:
        """解析立场响应"""
        import re
        import json

        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, response)

        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                return data.get('stances', [])
            except json.JSONDecodeError:
                print(f"JSON 解析失败：{json_str}")
                return []
        else:
            print(f"未找到 JSON 内容：{response}")
            return []


if __name__ == "__main__":
    # 测试模型
    print("测试 R-GCN 模型...")

    # 创建模型
    model = GraphEnhancedStanceDetector(
        bert_hidden_size=768,
        rgcn_hidden_dim=256,
        num_relations=6,
        num_layers=2
    )

    # 随机输入
    num_nodes = 5
    num_edges = 8

    node_features = torch.randn(num_nodes, 768)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, 6, (num_edges,))

    # 前向传播
    polarity, intensity = model(node_features, edge_index, edge_type)

    print(f"Polarity 输出形状：{polarity.shape}")
    print(f"Intensity 输出形状：{intensity.shape}")
    print("模型测试通过!")
