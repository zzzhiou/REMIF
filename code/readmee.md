# 多目标立场检测系统 - 完整实现方案

## 项目概述

基于 **LLM + R-GCN** 的多目标立场检测系统，用于识别社交媒体文本中多个目标的立场，并建模目标之间的复杂关系。

---

## 6 种关系类型（已确定）

| 关系 ID | 关系名 | 定义 | 方向性 | 示例 |
|--------|-------|------|--------|------|
| 0 | **contrast** | 对立关系（对 A 和 B 立场相反/冲突） | 双向 | "支持 A 但反对 B" |
| 1 | **consistent** | 一致关系（对 A 和 B 立场相同） | 双向 | "都支持"或"都反对" |
| 2 | **causal** | 因果关系（A 导致/影响 B） | 有向 (A→B) | "碳税会减轻企业负担" |
| 3 | **hierarchical** | 层级关系（A 是 B 的上位/下位概念） | 有向 | "环保政策包括碳税" |
| 4 | **parallel** | 并列关系（A 和 B 同级并列） | 无向 | "碳税和补贴政策" |
| 5 | **analogy** | 类比关系（用 A 类比 B） | 有向 (A→B) | "碳税就像罚款" |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入：社交媒体文本                        │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: 目标抽取 (LLM + CoT + 粒度控制)                         │
│                                                                  │
│  - 使用 LLM 抽取显性和隐性目标                                     │
│  - 粒度控制：只保留层级 2-3（具体对象/措施）                       │
│  - CoT 推理：逐步判断目标类型和层级                               │
│  - 语义去重：DBSCAN 聚类（或简单字符串匹配）                       │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: 关系抽取 + 异质图构建                                    │
│                                                                  │
│  - LLM 判断目标对之间的 6 种关系                                    │
│  - 构建 MultiDiGraph（ NetworkX）                                 │
│  - 节点属性：id, text, normalized, level, type, evidence        │
│  - 边属性：relation_type, direction, confidence, evidence       │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: R-GCN + LLM 立场检测                                     │
│                                                                  │
│  - BERT 编码目标节点                                              │
│  - R-GCN 聚合图结构信息（2 层）                                    │
│  - 计算目标重要性（向量范数 + 中心性）                            │
│  - 图线性化 + Prompt 输入 LLM                                     │
│  - LLM 推理输出立场（polarity + intensity）                       │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      输出：每个目标的立场                         │
│  - polarity: support/neutral/oppose                             │
│  - intensity: strong/moderate/weak                              │
│  - evidence: 原文证据                                            │
│  - reasoning: 推理过程                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
stance_detection_project/
├── configs/
│   ├── __init__.py
│   └── config.py              # 配置文件的配置
├── src/
│   ├── __init__.py
│   ├── stage1_target_extraction.py   # Stage 1: 目标抽取
│   ├── stage2_relation_extraction.py # Stage 2: 关系抽取 + 图构建
│   └── stage3_stance_detection.py    # Stage 3: R-GCN+LLM 立场检测
├── utils/
│   ├── __init__.py
│   ├── data_processor.py    # 数据处理
│   └── llm_client.py        # LLM 客户端封装
├── models/                   # 模型保存目录
├── results/                  # 结果输出目录
├── data/                     # 数据目录
├── main.py                   # 主运行脚本
├── test_run.py               # 测试脚本
├── requirements.txt          # 依赖
└── README.md                 # 使用说明
```

---

## 核心代码说明

### 1. Stage 1: 目标抽取 (`src/stage1_target_extraction.py`)

**关键技术**：
- **CoT (Chain of Thought)**: 逐步推理目标层级和类型
- **约束提示**: 明确定义 4 个粒度层级
- **语义去重**: Sentence-BERT + DBSCAN 聚类

**Prompt 设计**：
```
粒度控制规则：
- 层级 1（太宏观）：抽象理念如"环保"、"经济发展"
- 层级 2（合适）：具体对象/作品/政策
- 层级 3（合适）：具体方面/细节
- 层级 4（太细）：具体事件/评论

输出：JSON 格式，包含 id, text, normalized, granularity_level, type
```

---

### 2. Stage 2: 关系抽取 (`src/stage2_relation_extraction.py`)

**关系抽取 Prompt**：
```
关系类型（6 种）：
1. contrast: 对立关系
2. consistent: 一致关系
3. causal: 因果关系
4. hierarchical: 层级关系
5. parallel: 并列关系
6. analogy: 类比关系

输出：JSON 格式，包含 relation, direction, evidence, confidence
```

**图构建**：
- 使用 `nx.MultiDiGraph` 允许重边
- 节点 ID 映射：T1, T2, T3...
- 边类型映射：relation_id (0-5)

---

### 3. Stage 3: 立场检测 (`src/stage3_stance_detection.py`)

**R-GCN 模型**：
```python
class TargetRGCN(nn.Module):
    # 输入：节点特征 [num_nodes, 768] (BERT)
    # 输出：图增强表示 [num_nodes, 256]

    def forward(x, edge_index, edge_type):
        # 2 层 R-GCN + LayerNorm + ReLU
        # 聚合邻居信息（按关系类型加权）
```

**立场推理 Prompt**：
```
【目标列表】（按重要性排序）
- T1: 目标 1 (重要性:0.85, 层级:2, explicit)
- T2: 目标 2 (重要性:0.63, 层级:3, implicit)

【目标关系】
- T1 → T2: causal (置信度:0.9)
- T1 ↔ T3: contrast (置信度:0.8)

【R-GCN 计算的目标重要性】
- T1: 0.92
- T2: 0.45
- T3: 0.31

请判断每个目标的立场...
```

---

## 使用方法

### 安装依赖

```bash
# 基础依赖
pip install pandas openpyxl networkx numpy

# LLM 依赖（可选）
pip install dashscope

# 完整依赖（包含 R-GCN）
pip install torch torch-geometric transformers sentence-transformers scikit-learn
```

### 运行流程

```bash
# 1. 测试运行（Mock LLM）
python test_run.py

# 2. 完整运行（需要 API Key）
export DASHSCOPE_API_KEY=your_key
python main.py --data_path ../测试-data-最终版.xlsx --output results/predictions.json

# 3. 纯 LLM 方案（不使用 R-GCN）
python main.py --no_rgcn --limit 10
```

### 输出格式

```json
{
  "timestamp": "2024-xx-xxTxx:xx:xx",
  "results": [
    {
      "sample_id": 0,
      "text": "...",
      "extracted_targets": [
        {
          "id": "T1",
          "text": "《信条》",
          "normalized": "《信条》",
          "granularity_level": 2,
          "type": "explicit",
          "evidence": "为什么信条口碑好像不如..."
        }
      ],
      "relations": [
        {
          "target_a": "T1",
          "target_b": "T2",
          "relation": "contrast",
          "direction": "bidirectional",
          "confidence": 0.9
        }
      ],
      "graph_stats": {"num_nodes": 2, "num_edges": 2},
      "stances": [
        {
          "target_id": "T1",
          "polarity": "support",
          "intensity": "moderate",
          "evidence": "炫技式结尾我还挺受用的",
          "reasoning": "正面评价"
        }
      ]
    }
  ]
}
```

---

## 技术方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| **LLM-only** | 零训练、可解释、部署简单 | 效果上限较低 | 工程落地、数据少 |
| **R-GCN + LLM** | 利用图结构、可端到端训练、适合论文 | 需要训练、误差累积 | 学术研究、追求 SOTA |
| **纯 R-GCN** | 推理快、成本低 | 需要大量标注数据 | 大规模部署 |

**推荐方案**：R-GCN + LLM 混合架构
- R-GCN 编码图结构信息
- LLM 进行语义推理
- 兼顾效果和可解释性

---

## 评估指标（待实现）

### Stage 1: 目标抽取
- Precision: 抽取正确的目标数 / 抽取总数
- Recall: 抽取正确的目标数 / 金标总数
- F1: 调和平均

### Stage 2: 关系抽取
- Accuracy: 正确关系数 / 总关系数
- Macro-F1: 各类关系的 F1 平均

### Stage 3: 立场检测
- Weighted-F1: 考虑类别不平衡
- End-to-End F1: 全流程评估

---

## 注意事项

1. **API Key**: 需要通义千问 DashScope API Key（或替换为其他 LLM）
2. **显存**: R-GCN 需要 GPU（可选，CPU 也可运行但较慢）
3. **Python 版本**: 建议 3.9+（Python 3.8 可能有兼容性问题）
4. **数据路径**: 确保 `测试-data-最终版.xlsx` 路径正确

---

## 扩展方向

1. **多语言支持**: 替换 BERT 为多语言模型
2. **在线学习**: 增量更新 R-GCN 参数
3. **可视化**: 使用 PyVis 可视化目标关系图
4. **主动学习**: 选择困难样本进行标注

---

## 参考资料

1. Schlichtkrull et al. "Modeling Relational Data with Graph Convolutional Networks" (R-GCN 论文)
2. DashScope API 文档：https://help.aliyun.com/zh/dashscope/
3. PyTorch Geometric 文档：https://pytorch-geometric.readthedocs.io/
