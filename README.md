# 多目标立场检测

基于 LLM + R-GCN 的多目标立场检测。

## 架构设计

```
输入：社交媒体文本
  ↓
Stage 1: 目标抽取 (LLM + CoT + 粒度控制)
  ↓
Stage 2: 关系抽取 + 异质图构建 (6 种关系类型)
  ↓
Stage 3: R-GCN + LLM 立场检测
  ↓
输出：每个目标的立场（极性 + 强度）
```

## 6 种关系类型

| 关系 | 说明 | 方向性 |
|-----|------|-------|
| contrast | 对立关系（立场冲突） | 双向 |
| consistent | 一致关系（立场相同） | 双向 |
| causal | 因果关系 | 有向 (A→B) |
| hierarchical | 层级关系 | 有向 |
| parallel | 并列关系 | 无向 |
| analogy | 类比关系 | 有向 |

## 安装

```bash
# 创建虚拟环境（可选）
conda create -n stance python=3.9
conda activate stance

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
# 设置 API Key
export DASHSCOPE_API_KEY=your_api_key

# 运行完整流程
python main.py --data_path ../测试-data-最终版.xlsx --output results/predictions.json
```

### 参数说明

```bash
python main.py --help

# --data_path: 数据文件路径
# --api_key: DashScope API Key（或设置环境变量）
# --use_rgcn: 使用 R-GCN（默认）
# --no_rgcn: 不使用 R-GCN（纯 LLM 方案）
# --limit: 限制处理样本数（测试用）
# --output: 输出文件路径
```

### 示例

```bash
# 测试前 5 个样本
python main.py --limit 5 --output results/test.json

# 纯 LLM 方案（不使用 R-GCN）
python main.py --no_rgcn --limit 10

# 完整运行
python main.py --output results/full_results.json
```

## 项目结构

```
stance_detection_project/
├── configs/
│   ├── __init__.py
│   └── config.py          # 配置文件
├── src/
│   ├── __init__.py
│   ├── stage1_target_extraction.py   # 目标抽取
│   ├── stage2_relation_extraction.py # 关系抽取 + 图构建
│   └── stage3_stance_detection.py    # R-GCN + LLM 立场检测
├── utils/
│   ├── __init__.py
│   ├── data_processor.py   # 数据处理
│   └── llm_client.py       # LLM 客户端
├── models/                  # 模型保存目录
├── results/                 # 结果输出目录
├── data/                    # 数据目录
├── main.py                  # 主运行脚本
└── requirements.txt         # 依赖
```

## 输出格式

```json
{
  "timestamp": "2024-xx-xxTxx:xx:xx",
  "config": {
    "use_rgcn": true,
    "relation_types": {...}
  },
  "results": [
    {
      "sample_id": 0,
      "text": "...",
      "gold_targets": [...],
      "gold_stances": [...],
      "extracted_targets": [
        {
          "id": "T1",
          "text": "原文表述",
          "normalized": "规范化表述",
          "granularity_level": 2,
          "type": "explicit",
          "evidence": "...",
          "reasoning": "..."
        }
      ],
      "relations": [...],
      "graph_stats": {"num_nodes": 5, "num_edges": 3},
      "stances": [
        {
          "target_id": "T1",
          "polarity": "support",
          "intensity": "moderate",
          "evidence": "...",
          "reasoning": "..."
        }
      ]
    }
  ]
}
```

## 评估指标（待实现）

- 目标抽取：Precision / Recall / F1
- 关系抽取：Accuracy / Macro-F1
- 立场检测：Weighted-F1 / End-to-End F1

## 注意事项

1. **API Key**: 需要通义千问 DashScope API Key
2. **显存**: 使用 R-GCN 需要 GPU（可选）
3. **数据路径**: 确保数据文件路径正确

## 技术栈

- **LLM**: 通义千问 (Qwen)
- **Embedding**: Sentence-BERT
- **图神经网络**: PyTorch Geometric (R-GCN)
- **语言模型**: BERT (中文)
