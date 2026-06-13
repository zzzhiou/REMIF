#!/usr/bin/env python3
"""
查看和分析批量处理结果
"""

import json
import pandas as pd
from collections import Counter


def analyze_results(results_path: str):
    """分析结果文件"""

    # 加载结果
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data['results']

    print("="*80)
    print("多目标立场检测结果分析")
    print("="*80)

    # 1. 基本统计
    print("\n【基本统计】")
    print(f"  总样本数：{len(results)}")

    # 2. 目标统计
    target_counts = [len(r['extracted_targets']) for r in results]
    print(f"\n【目标统计】")
    print(f"  平均目标数：{sum(target_counts)/len(target_counts):.2f}")
    print(f"  中位数：{sorted(target_counts)[len(target_counts)//2]}")
    print(f"  最多：{max(target_counts)}")
    print(f"  最少：{min(target_counts)}")

    # 目标数量分布
    target_dist = Counter(target_counts)
    print(f"\n  目标数量分布:")
    for count in sorted(target_dist.keys())[:10]:
        print(f"    {count}个目标：{target_dist[count]}样本 ({target_dist[count]/len(results)*100:.1f}%)")

    # 3. 关系统计
    relation_counts = [len(r['relations']) for r in results]
    relation_types = []
    for r in results:
        for rel in r['relations']:
            relation_types.append(rel['relation'])

    print(f"\n【关系统计】")
    print(f"  总关系数：{sum(relation_counts)}")
    print(f"  平均每样本：{sum(relation_counts)/len(results):.2f}")

    # 关系类型分布
    relation_dist = Counter(relation_types)
    print(f"\n  关系类型分布:")
    for rel_type, count in relation_dist.most_common():
        print(f"    {rel_type}: {count} ({count/len(relation_types)*100:.1f}%)")

    # 4. 立场统计
    polarities = []
    intensities = []
    for r in results:
        for s in r['stances']:
            polarities.append(s['polarity'])
            intensities.append(s['intensity'])

    print(f"\n【立场统计】")
    polarity_dist = Counter(polarities)
    total_stances = len(polarities)
    print(f"  总立场数：{total_stances}")
    print(f"\n  立场极性分布:")
    polarity_cn = {"support": "支持", "oppose": "反对", "neutral": "中立"}
    for pol, count in polarity_dist.most_common():
        cn = polarity_cn.get(pol, pol)
        print(f"    {cn} ({pol}): {count} ({count/total_stances*100:.1f}%)")

    print(f"\n  立场强度分布:")
    intensity_cn = {"strong": "强烈", "moderate": "中等", "weak": "微弱"}
    intensity_dist = Counter(intensities)
    for inte, count in intensity_dist.most_common():
        cn = intensity_cn.get(inte, inte)
        print(f"    {cn} ({inte}): {count} ({count/total_stances*100:.1f}%)")

    # 5. 图统计
    node_counts = [r['graph_stats']['num_nodes'] for r in results]
    edge_counts = [r['graph_stats']['num_edges'] for r in results]

    print(f"\n【图结构统计】")
    print(f"  平均节点数：{sum(node_counts)/len(node_counts):.2f}")
    print(f"  平均边数：{sum(edge_counts)/len(edge_counts):.2f}")
    print(f"  平均密度：{sum(e/(n*(n-1)) if n>1 else 0 for n, e in zip(node_counts, edge_counts))/len(results):.3f}")

    # 6. 保存分析结果
    analysis_df = pd.DataFrame({
        'sample_id': range(len(results)),
        'target_count': target_counts,
        'relation_count': relation_counts,
        'node_count': node_counts,
        'edge_count': edge_counts,
        'support_count': [sum(1 for s in r['stances'] if s['polarity']=='support') for r in results],
        'oppose_count': [sum(1 for s in r['stances'] if s['polarity']=='oppose') for r in results],
        'neutral_count': [sum(1 for s in r['stances'] if s['polarity']=='neutral') for r in results],
    })

    # 保存统计结果
    output_excel = 'results/analysis_summary.xlsx'
    analysis_df.to_excel(output_excel, index=False)
    print(f"\n分析摘要已保存到：{output_excel}")

    return analysis_df


def view_sample(results_path: str, sample_id: int):
    """查看指定样本的详细信息"""

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = data['results'][sample_id]

    print("\n" + "="*80)
    print(f"样本 {sample_id} 详细信息")
    print("="*80)

    print(f"\n【原文】")
    print(f"{result['text'][:200]}...")

    print(f"\n【金标数据】")
    print(f"  目标：{result['gold_targets']}")
    print(f"  立场：{result['gold_stances']}")

    print(f"\n【Stage 1: 抽取的目标】")
    for t in result['extracted_targets']:
        print(f"  [{t['id']}] {t['normalized']}")
        print(f"       层级:{t['granularity_level']}, 类型:{t['type']}")
        print(f"       证据:{t['evidence'][:50]}")

    print(f"\n【Stage 2: 关系】")
    for r in result['relations'][:10]:
        print(f"  {r['target_a']} ↔ {r['target_b']}: {r['relation']}")
    if len(result['relations']) > 10:
        print(f"  ... 还有 {len(result['relations'])-10} 条")

    print(f"\n【图统计】节点:{result['graph_stats']['num_nodes']}, 边:{result['graph_stats']['num_edges']}")

    print(f"\n【Stage 3: 立场】")
    polarity_cn = {"support": "✓支持", "oppose": "✗反对", "neutral": "○中立"}
    for s in result['stances']:
        p_cn = polarity_cn.get(s['polarity'], s['polarity'])
        print(f"  [{s['target_id']}] {p_cn} | {s['intensity']}")
        print(f"       证据:{s['evidence'][:50]}...")


if __name__ == "__main__":
    import sys

    results_path = 'results/batch_results.json'

    if len(sys.argv) > 1:
        # 查看指定样本
        sample_id = int(sys.argv[1])
        view_sample(results_path, sample_id)
    else:
        # 分析所有结果
        analyze_results(results_path)
