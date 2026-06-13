#!/usr/bin/env python3
"""
测试脚本 - 测试完整流程（不使用 R-GCN）

使用方法:
    python test_run.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_processor import DataProcessor
from utils.llm_client import DashScopeClient
from src.stage1_target_extraction import TargetExtractor
from src.stage2_relation_extraction import RelationExtractor, HeterogeneousGraphBuilder


def test_single_sample(text: str, sample_id: int):
    """测试单个样本"""
    print(f"\n{'='*60}")
    print(f"样本 {sample_id}")
    print(f"{'='*60}")
    print(f"文本：{text[:100]}...")

    # 创建 Mock LLM 客户端（用于测试）
    class MockLLMClient:
        def generate(self, system_prompt, user_prompt, **kwargs):
            # 根据 prompt 类型返回不同的 mock 响应
            if '目标抽取' in system_prompt or '抽取专家' in system_prompt:
                # 从文本中提取一些假目标
                return json.dumps({
                    "targets": [
                        {
                            "id": "T1",
                            "text": "测试目标 1",
                            "normalized": "测试目标 1",
                            "granularity_level": 2,
                            "type": "explicit",
                            "evidence": "文本片段 1",
                            "reasoning": "直接提及"
                        },
                        {
                            "id": "T2",
                            "text": "测试目标 2",
                            "normalized": "测试目标 2",
                            "granularity_level": 2,
                            "type": "implicit",
                            "evidence": "文本片段 2",
                            "reasoning": "暗示提及"
                        }
                    ]
                }, ensure_ascii=False)
            elif '关系抽取' in system_prompt or '关系专家' in system_prompt:
                return json.dumps({
                    "relation": "parallel",
                    "direction": "none",
                    "evidence": "文本片段",
                    "confidence": 0.8,
                    "reasoning": "两个目标并列提及"
                }, ensure_ascii=False)
            elif '立场检测' in system_prompt or '立场检测专家' in system_prompt:
                return json.dumps({
                    "stances": [
                        {
                            "target_id": "T1",
                            "polarity": "support",
                            "intensity": "moderate",
                            "evidence": "证据 1",
                            "reasoning": "正面评价"
                        },
                        {
                            "target_id": "T2",
                            "polarity": "neutral",
                            "intensity": "weak",
                            "evidence": "证据 2",
                            "reasoning": "中立态度"
                        }
                    ]
                }, ensure_ascii=False)
            else:
                return '{"result": "mock"}'

    llm_client = MockLLMClient()

    # Stage 1: 目标抽取
    print("\nStage 1: 目标抽取...")
    extractor = TargetExtractor(llm_client=llm_client, dedup_threshold=0.85)
    targets = extractor.extract(text)
    print(f"  抽取到 {len(targets)} 个目标:")
    for t in targets:
        print(f"    - {t['id']}: {t['normalized']} ({t['type']}, 层级{t['granularity_level']})")

    if len(targets) == 0:
        print("  无目标，停止测试")
        return None

    # Stage 2: 关系抽取
    print("\nStage 2: 关系抽取...")
    relation_extractor = RelationExtractor(llm_client=llm_client, confidence_threshold=0.5)
    relations = relation_extractor.extract_all(targets, text)
    print(f"  抽取到 {len(relations)} 条关系:")
    for r in relations:
        print(f"    - {r['target_a']} → {r['target_b']}: {r['relation']} (置信度：{r['confidence']})")

    # 构建图
    graph_builder = HeterogeneousGraphBuilder()
    graph = graph_builder.build(targets, relations)
    print(f"\n  图构建完成：{graph.number_of_nodes()} 节点，{graph.number_of_edges()} 边")

    # 输出图描述
    print("\n  图结构:")
    print(graph_builder.get_graph_description(graph))

    return {
        'sample_id': sample_id,
        'text': text,
        'targets': targets,
        'relations': relations,
        'graph_stats': {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges()
        }
    }


def main():
    """主测试函数"""
    print("="*60)
    print("多目标立场检测系统 - 流程测试")
    print("="*60)

    # 加载数据
    data_path = '../测试-data-最终版.xlsx'
    print(f"\n加载数据：{data_path}")

    processor = DataProcessor(data_path)
    processor.load()

    stats = processor.get_statistics()
    print(f"  总样本数：{stats['total_samples']}")
    print(f"  平均目标数：{stats['avg_targets']:.2f}")

    # 测试前 3 个样本
    samples = processor.get_samples(limit=3)

    results = []
    for i, sample in enumerate(samples):
        result = test_single_sample(sample['text'], i + 1)
        if result:
            results.append(result)

    # 保存测试结果
    output_file = 'results/test_results.json'
    os.makedirs('results', exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_info': 'Mock LLM 测试（未调用真实 API）',
            'results': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"测试完成!")
    print(f"结果已保存到：{output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
