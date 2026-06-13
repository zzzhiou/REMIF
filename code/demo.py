#!/usr/bin/env python3
"""
演示脚本 - 展示使用真实 LLM 的完整流程（使用示例响应）

由于 API Key 无效，这里使用预设的示例响应来演示流程
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_processor import DataProcessor


# ============ 模拟真实 LLM 的响应（基于实际样本） ============

MOCK_RESPONSES = {
    "target_extraction": {
        # 样本 1：《信条》影评
        0: {
            "targets": [
                {
                    "id": "T1",
                    "text": "《信条》口碑",
                    "normalized": "《信条》口碑",
                    "granularity_level": 2,
                    "type": "explicit",
                    "evidence": "为什么信条口碑好像不如盗梦空间呢",
                    "reasoning": "直接提及，比较对象"
                },
                {
                    "id": "T2",
                    "text": "《盗梦空间》口碑",
                    "normalized": "《盗梦空间》口碑",
                    "granularity_level": 2,
                    "type": "explicit",
                    "evidence": "信条口碑好像不如盗梦空间",
                    "reasoning": "比较对象"
                },
                {
                    "id": "T3",
                    "text": "炫技式结尾",
                    "normalized": "《信条》的炫技式结尾",
                    "granularity_level": 3,
                    "type": "explicit",
                    "evidence": "炫技式结尾我还挺受用的",
                    "reasoning": "直接评价"
                }
            ]
        },
        # 样本 2：妈妈朋友的儿子
        1: {
            "targets": [
                {
                    "id": "T1",
                    "text": "妈妈朋友的儿子",
                    "normalized": "妈妈朋友的儿子（剧集）",
                    "granularity_level": 2,
                    "type": "explicit",
                    "evidence": "妈妈朋友的儿子 丁海寅 郑素敏",
                    "reasoning": "剧集名称，讨论主题"
                },
                {
                    "id": "T2",
                    "text": "丁海寅",
                    "normalized": "丁海寅",
                    "granularity_level": 2,
                    "type": "explicit",
                    "evidence": "丁海寅 郑素敏",
                    "reasoning": "演员，被提及"
                },
                {
                    "id": "T3",
                    "text": "郑素敏",
                    "normalized": "郑素敏",
                    "granularity_level": 2,
                    "type": "explicit",
                    "evidence": "丁海寅 郑素敏",
                    "reasoning": "演员，被提及"
                },
                {
                    "id": "T4",
                    "text": "确定关系后开了倍速",
                    "normalized": "剧情节奏（确定关系后）",
                    "granularity_level": 3,
                    "type": "explicit",
                    "evidence": "确定关系后开了倍速，爽啊",
                    "reasoning": "评价剧情节奏"
                }
            ]
        }
    },
    "relations": {
        # 样本 1 的关系
        0: [
            {
                "target_a": "T1",
                "target_b": "T2",
                "relation": "contrast",
                "direction": "bidirectional",
                "evidence": "信条口碑好像不如盗梦空间",
                "confidence": 0.95,
                "reasoning": "两部电影的口碑进行对比"
            },
            {
                "target_a": "T3",
                "target_b": "T1",
                "relation": "causal",
                "direction": "T3→T1",
                "evidence": "炫技式结尾我还挺受用的",
                "confidence": 0.7,
                "reasoning": "结尾质量影响口碑"
            }
        ],
        # 样本 2 的关系
        1: [
            {
                "target_a": "T2",
                "target_b": "T3",
                "relation": "parallel",
                "direction": "none",
                "evidence": "丁海寅 郑素敏",
                "confidence": 0.9,
                "reasoning": "两位演员并列提及"
            },
            {
                "target_a": "T4",
                "target_b": "T1",
                "relation": "causal",
                "direction": "T4→T1",
                "evidence": "确定关系后开了倍速，爽啊",
                "confidence": 0.85,
                "reasoning": "剧情节奏影响观看体验"
            }
        ]
    },
    "stances": {
        # 样本 1 的立场
        0: [
            {
                "target_id": "T1",
                "polarity": "neutral",
                "intensity": "moderate",
                "evidence": "为什么信条口碑好像不如盗梦空间呢",
                "reasoning": "提出疑问，没有明确立场"
            },
            {
                "target_id": "T2",
                "polarity": "neutral",
                "intensity": "weak",
                "evidence": "信条口碑好像不如盗梦空间",
                "reasoning": "作为比较基准，无明显立场"
            },
            {
                "target_id": "T3",
                "polarity": "support",
                "intensity": "moderate",
                "evidence": "炫技式结尾我还挺受用的🤧",
                "reasoning": "明确表达喜欢，emoji 加强语气"
            }
        ],
        # 样本 2 的立场
        1: [
            {
                "target_id": "T1",
                "polarity": "support",
                "intensity": "strong",
                "evidence": "爽啊！",
                "reasoning": "感叹号表达强烈情感"
            },
            {
                "target_id": "T2",
                "polarity": "support",
                "intensity": "moderate",
                "evidence": "丁海寅 郑素敏",
                "reasoning": "提及演员，无明显评价但语境正面"
            },
            {
                "target_id": "T3",
                "polarity": "support",
                "intensity": "moderate",
                "evidence": "丁海寅 郑素敏",
                "reasoning": "提及演员，无明显评价但语境正面"
            },
            {
                "target_id": "T4",
                "polarity": "support",
                "intensity": "strong",
                "evidence": "确定关系后开了倍速，爽啊！",
                "reasoning": "倍速+ 爽啊表达正面评价"
            }
        ]
    }
}


def demo_single_sample(text: str, sample_id: int, mock_data: dict):
    """演示单个样本的处理流程"""
    idx = sample_id - 1

    print(f"\n{'='*70}")
    print(f"样本 {sample_id}")
    print(f"{'='*70}")
    print(f"文本：{text[:120]}...")

    # Stage 1: 目标抽取
    print(f"\n【Stage 1: 目标抽取】")
    targets = mock_data["target_extraction"][idx]["targets"]
    print(f"  抽取到 {len(targets)} 个目标:")
    for t in targets:
        type_str = "显性" if t["type"] == "explicit" else "隐性"
        print(f"    [{t['id']}] {t['normalized']}")
        print(f"         类型：{type_str} | 层级：{t['granularity_level']}")
        print(f"         证据：{t['evidence'][:50]}...")
        print(f"         理由：{t['reasoning']}")

    # Stage 2: 关系抽取
    print(f"\n【Stage 2: 关系抽取】")
    relations = mock_data["relations"][idx]
    print(f"  抽取到 {len(relations)} 条关系:")
    for r in relations:
        dir_str = r['direction'] if r['direction'] != 'none' else '无向'
        print(f"    [{r['target_a']}] ↔ [{r['target_b']}]")
        print(f"         关系类型：{r['relation']} | 方向：{dir_str}")
        print(f"         置信度：{r['confidence']}")
        print(f"         证据：{r['evidence']}")
        print(f"         理由：{r['reasoning']}")

    # 构建图
    from src.stage2_relation_extraction import HeterogeneousGraphBuilder
    builder = HeterogeneousGraphBuilder()
    graph = builder.build(targets, relations)

    print(f"\n【图构建完成】")
    print(f"  节点数：{graph.number_of_nodes()}")
    print(f"  边数：{graph.number_of_edges()}")

    if graph.number_of_edges() > 0:
        print(f"\n  图结构可视化:")
        print(builder.get_graph_description(graph))

    # Stage 3: 立场检测
    print(f"\n【Stage 3: 立场检测】")
    stances = mock_data["stances"][idx]
    print(f"  检测到 {len(stances)} 个立场:")

    polarity_map = {
        "support": "✓ 支持",
        "oppose": "✗ 反对",
        "neutral": "○ 中立"
    }

    intensity_map = {
        "strong": "★★★ 强烈",
        "moderate": "★★☆ 中等",
        "weak": "★☆☆ 微弱"
    }

    for s in stances:
        polarity_cn = polarity_map.get(s['polarity'], s['polarity'])
        intensity_cn = intensity_map.get(s['intensity'], s['intensity'])
        print(f"    [{s['target_id']}] {polarity_cn} | {intensity_cn}")
        print(f"         证据：{s['evidence']}")
        print(f"         理由：{s['reasoning']}")

    return {
        'sample_id': sample_id,
        'text': text,
        'targets': targets,
        'relations': relations,
        'stances': stances,
        'graph_stats': {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges()
        }
    }


def main():
    """主函数"""
    print("="*70)
    print("多目标立场检测系统 - 演示（模拟真实 LLM 响应）")
    print("="*70)

    # 加载数据
    data_path = '../测试-data-最终版.xlsx'
    print(f"\n加载数据：{data_path}")

    processor = DataProcessor(data_path)
    processor.load()

    stats = processor.get_statistics()
    print(f"  总样本数：{stats['total_samples']}")
    print(f"  平均目标数：{stats['avg_targets']:.2f}")

    # 演示前 2 个样本
    samples = processor.get_samples(limit=2)

    results = []
    for i, sample in enumerate(samples):
        result = demo_single_sample(sample['text'], i + 1, MOCK_RESPONSES)
        results.append(result)

    # 保存结果
    output_file = 'results/demo_results.json'
    os.makedirs('results', exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'demo_info': '演示结果（模拟真实 LLM 响应）',
            'results': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("演示完成!")
    print(f"结果已保存到：{output_file}")
    print(f"{'='*70}")

    # 打印总结
    print("\n【总结】")
    for r in results:
        print(f"\n样本 {r['sample_id']}:")
        print(f"  - 目标数：{len(r['targets'])}")
        print(f"  - 关系数：{len(r['relations'])}")
        print(f"  - 立场数：{len(r['stances'])}")

        # 立场分布
        polarity_count = {}
        for s in r['stances']:
            p = s['polarity']
            polarity_count[p] = polarity_count.get(p, 0) + 1
        print(f"  - 立场分布：{polarity_count}")


if __name__ == "__main__":
    main()
