#!/usr/bin/env python3
"""
真实 LLM 测试脚本 - 使用通义千问 API 运行完整流程

使用方法:
    python run_real_llm.py --api_key YOUR_KEY
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_processor import DataProcessor
from utils.llm_client import DashScopeClient
from src.stage1_target_extraction import TargetExtractor
from src.stage2_relation_extraction import RelationExtractor, HeterogeneousGraphBuilder


def test_single_sample(text: str, sample_id: int, llm_client):
    """测试单个样本"""
    print(f"\n{'='*60}")
    print(f"样本 {sample_id}")
    print(f"{'='*60}")
    print(f"文本：{text[:150]}...")

    # Stage 1: 目标抽取
    print("\nStage 1: 目标抽取...")
    extractor = TargetExtractor(llm_client=llm_client, dedup_threshold=0.85)
    targets = extractor.extract(text)

    if not targets:
        print("  未抽取到目标，跳过")
        return None

    print(f"  抽取到 {len(targets)} 个目标:")
    for t in targets:
        print(f"    - {t['id']}: {t['normalized']} ({t['type']}, 层级{t['granularity_level']})")

    # 去重
    targets = extractor.deduplicate(targets)
    print(f"  去重后：{len(targets)} 个目标")

    if len(targets) < 2:
        print("  目标少于 2 个，无法抽取关系")
        # 单目标也可以继续
        relations = []
    else:
        # Stage 2: 关系抽取
        print("\nStage 2: 关系抽取...")
        relation_extractor = RelationExtractor(llm_client=llm_client, confidence_threshold=0.5)
        relations = relation_extractor.extract_all(targets, text)
        print(f"  抽取到 {len(relations)} 条关系:")
        for r in relations:
            print(f"    - {r['target_a']} → {r['target_b']}: {r['relation']} (置信度：{r['confidence']:.2f})")

    # 构建图
    graph_builder = HeterogeneousGraphBuilder()
    graph = graph_builder.build(targets, relations)
    print(f"\n  图构建完成：{graph.number_of_nodes()} 节点，{graph.number_of_edges()} 边")

    # 输出图描述
    if graph.number_of_edges() > 0:
        print("\n  图结构:")
        print(graph_builder.get_graph_description(graph))

    # 简化的立场检测（纯 LLM）
    print("\nStage 3: 立场检测 (LLM)...")
    stances = detect_stance_llm_only(llm_client, targets, relations, text)
    print(f"  检测到 {len(stances)} 个立场:")
    for s in stances:
        print(f"    - {s['target_id']}: {s['polarity']} ({s['intensity']})")

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


def detect_stance_llm_only(llm_client, targets, relations, text):
    """使用纯 LLM 进行立场检测"""

    # 构建目标列表
    target_list = "\n".join([
        f"- {t['id']}: {t['normalized']} ({t['type']})"
        for t in targets
    ])

    # 构建关系列表
    if relations:
        relation_list = "\n".join([
            f"- {r['target_a']} → {r['target_b']}: {r['relation']}"
            for r in relations
        ])
    else:
        relation_list = "无明显关系"

    prompt = f"""你是一个立场检测专家。

【目标列表】
{target_list}

【目标关系】
{relation_list}

【原始文本】
{text}

请判断每个目标的立场：
1. 立场极性：support（支持）/ neutral（中立）/ oppose（反对）
2. 立场强度：strong（强烈）/ moderate（中等）/ weak（微弱）
3. 证据：原文片段

输出格式（JSON）：
{{
  "stances": [
    {{
      "target_id": "T1",
      "polarity": "support",
      "intensity": "moderate",
      "evidence": "原文证据",
      "reasoning": "推理过程"
    }}
  ]
}}
"""

    response = llm_client.generate(
        system_prompt="你是一个立场检测专家。请严格按照 JSON 格式输出。",
        user_prompt=prompt
    )

    # 解析响应
    import re
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, response)

    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            return data.get('stances', [])
        except json.JSONDecodeError:
            print(f"JSON 解析失败：{json_str[:100]}")
            return []
    return []


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='真实 LLM 测试脚本')
    parser.add_argument('--api_key', type=str, default=None,
                       help='DashScope API Key')
    parser.add_argument('--limit', type=int, default=3,
                       help='测试样本数量')
    parser.add_argument('--output', type=str, default='results/real_llm_results.json',
                       help='输出文件路径')

    args = parser.parse_args()

    # 检查 API Key
    api_key = args.api_key or os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        print("错误：请提供 API Key")
        print("使用方法：python run_real_llm.py --api_key YOUR_KEY")
        sys.exit(1)

    print("="*60)
    print("多目标立场检测系统 - 真实 LLM 测试")
    print("="*60)

    # 初始化 LLM
    print("\n初始化 LLM 客户端...")
    llm_client = DashScopeClient(api_key=api_key, model="qwen-plus")

    # 加载数据
    data_path = '../测试-data-最终版.xlsx'
    print(f"\n加载数据：{data_path}")

    processor = DataProcessor(data_path)
    processor.load()

    stats = processor.get_statistics()
    print(f"  总样本数：{stats['total_samples']}")
    print(f"  平均目标数：{stats['avg_targets']:.2f}")

    # 测试样本
    samples = processor.get_samples(limit=args.limit)
    print(f"\n测试前 {args.limit} 个样本")

    results = []
    for i, sample in enumerate(samples):
        try:
            result = test_single_sample(sample['text'], i + 1, llm_client)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  样本 {i+1} 处理失败：{e}")
            continue

    # 保存结果
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'test_info': f'真实 LLM 测试（{args.limit}个样本）',
            'api_key_used': api_key[:10] + '...' if len(api_key) > 10 else api_key,
            'results': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"测试完成!")
    print(f"结果已保存到：{args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
