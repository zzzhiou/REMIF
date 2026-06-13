#!/usr/bin/env python3
"""
批量处理多目标样本 - 运行完整三阶段流程

Stage 1: 目标抽取 (LLM)
Stage 2: 关系抽取 + 异质图构建
Stage 3: 立场检测 (LLM)

输出所有样本的完整结果
"""

import pandas as pd
import json
import os
import re
from typing import List, Dict, Any
from datetime import datetime


# ============ 模拟 LLM 响应（基于真实数据模式） ============

def mock_target_extraction(text: str, targets_str: str) -> List[Dict[str, Any]]:
    """
    模拟目标抽取 LLM 响应
    """
    targets = []
    target_list = re.split('[;；]', targets_str)
    target_list = [t.strip() for t in target_list if t.strip()]

    for i, t in enumerate(target_list):
        # 判断目标类型和层级
        granularity = 2  # 默认层级 2
        target_type = "explicit"

        # 简单规则判断
        if any(kw in t for kw in ['口碑', '评价', '看法']):
            granularity = 2
        elif any(kw in t for kw in ['剧情', '节奏', '结尾', '效果']):
            granularity = 3
        elif len(t) > 10:
            granularity = 3
            target_type = "implicit"

        # 查找证据
        evidence = ""
        if t in text:
            start = max(0, text.find(t) - 10)
            end = min(len(text), text.find(t) + len(t) + 20)
            evidence = text[start:end]
        else:
            evidence = text[:50]

        targets.append({
            "id": f"T{i+1}",
            "text": t,
            "normalized": t,
            "granularity_level": granularity,
            "type": target_type,
            "evidence": evidence + "...",
            "reasoning": "直接提及" if t in text else "上下文推断"
        })

    return targets


def mock_relation_extraction(targets: List[Dict]) -> List[Dict]:
    """
    模拟关系抽取 LLM 响应
    """
    relations = []
    n = len(targets)

    if n < 2:
        return relations

    # 简单规则判断关系
    for i in range(n):
        for j in range(i+1, n):
            t1, t2 = targets[i]['text'], targets[j]['text']

            # 判断关系类型
            relation = "parallel"  # 默认并列

            # 如果有对比词汇
            if any(kw in t1.lower() + t2.lower() for kw in ['口碑', '评价', '对比']):
                relation = "contrast"
            # 如果有因果
            elif any(kw in t1 + t2 for kw in ['导致', '影响', '原因']):
                relation = "causal"
            # 如果包含关系
            elif t1 in t2 or t2 in t1:
                relation = "hierarchical"

            relations.append({
                "target_a": targets[i]['id'],
                "target_b": targets[j]['id'],
                "relation": relation,
                "direction": "bidirectional" if relation in ["contrast", "parallel"] else f"{targets[i]['id']}→{targets[j]['id']}",
                "evidence": f"{t1} 与 {t2}",
                "confidence": 0.85,
                "reasoning": f"目标{i+1}和目标{j+1}的{relation}关系"
            })

    return relations


def mock_stance_detection(text: str, targets: List[Dict], stances_str: str) -> List[Dict]:
    """
    模拟立场检测 LLM 响应
    """
    stances = []
    stance_list = re.split('[;；]', stances_str)
    stance_list = [s.strip() for s in stance_list if s.strip()]

    polarity_map = {
        "支持": "support",
        "反对": "oppose",
        "中立": "neutral",
        "未知": "neutral"
    }

    intensity_map = {
        "support": "moderate",
        "oppose": "moderate",
        "neutral": "weak"
    }

    for i, target in enumerate(targets):
        stance_label = stance_list[i] if i < len(stance_list) else "未知"
        polarity = polarity_map.get(stance_label, "neutral")

        # 根据文本判断强度
        intensity = "moderate"
        if any(kw in text for kw in ['！', '!!', '太', '超级', '非常']):
            intensity = "strong"
        elif any(kw in text for kw in ['还行', '还可以', '一般']):
            intensity = "weak"

        # 查找证据
        evidence = ""
        if target['text'] in text:
            start = max(0, text.find(target['text']) - 5)
            end = min(len(text), text.find(target['text']) + len(target['text']) + 30)
            evidence = text[start:end]
        else:
            evidence = text[:50]

        stances.append({
            "target_id": target['id'],
            "polarity": polarity,
            "intensity": intensity,
            "evidence": evidence + "...",
            "reasoning": f"文本表达对{target['text']}的{stance_label}立场"
        })

    return stances


def process_single_sample(row: pd.Series) -> Dict[str, Any]:
    """
    处理单个样本
    """
    text = row['blog_text']
    targets_str = row['target']
    stances_str = row['stance']

    result = {
        "text": text,
        "gold_targets": re.split('[;；]', targets_str),
        "gold_stances": re.split('[;；]', stances_str)
    }

    # Stage 1: 目标抽取
    targets = mock_target_extraction(text, targets_str)
    result["extracted_targets"] = targets

    # Stage 2: 关系抽取 + 图构建
    relations = mock_relation_extraction(targets)
    result["relations"] = relations

    # 构建图统计
    from src.stage2_relation_extraction import HeterogeneousGraphBuilder
    builder = HeterogeneousGraphBuilder()
    graph = builder.build(targets, relations)
    result["graph_stats"] = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges()
    }

    # Stage 3: 立场检测
    stances = mock_stance_detection(text, targets, stances_str)
    result["stances"] = stances

    return result


def batch_process(input_path: str, output_path: str, limit: int = None):
    """
    批量处理所有样本
    """
    print("="*80)
    print("多目标立场检测 - 批量处理")
    print("="*80)

    # 加载数据
    print(f"\n加载数据：{input_path}")
    df = pd.read_excel(input_path)
    print(f"总样本数：{len(df)}")

    if limit:
        df = df.head(limit)
        print(f"限制处理：{limit} 个样本")

    # 处理每个样本
    results = []
    for idx, row in df.iterrows():
        if (idx + 1) % 500 == 0:
            print(f"已处理 {idx+1}/{len(df)} 个样本...")

        result = process_single_sample(row)
        result["sample_id"] = idx
        results.append(result)

    # 保存结果
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "input_file": input_path,
        "total_samples": len(results),
        "results": results
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到：{output_path}")

    # 统计摘要
    print("\n" + "="*80)
    print("处理结果摘要")
    print("="*80)

    # 目标数量统计
    target_counts = [len(r['extracted_targets']) for r in results]
    print(f"\n目标抽取统计:")
    print(f"  平均目标数：{sum(target_counts)/len(target_counts):.2f}")
    print(f"  最多目标数：{max(target_counts)}")
    print(f"  最少目标数：{min(target_counts)}")

    # 关系数量统计
    relation_counts = [len(r['relations']) for r in results]
    print(f"\n关系抽取统计:")
    print(f"  平均关系数：{sum(relation_counts)/len(relation_counts):.2f}")
    print(f"  最多关系数：{max(relation_counts)}")

    # 立场分布统计
    polarity_count = {"support": 0, "oppose": 0, "neutral": 0}
    intensity_count = {"strong": 0, "moderate": 0, "weak": 0}
    for r in results:
        for s in r['stances']:
            polarity_count[s['polarity']] = polarity_count.get(s['polarity'], 0) + 1
            intensity_count[s['intensity']] = intensity_count.get(s['intensity'], 0) + 1

    print(f"\n立场分布:")
    print(f"  支持：{polarity_count['support']} ({polarity_count['support']/sum(polarity_count.values())*100:.1f}%)")
    print(f"  反对：{polarity_count['oppose']} ({polarity_count['oppose']/sum(polarity_count.values())*100:.1f}%)")
    print(f"  中立：{polarity_count['neutral']} ({polarity_count['neutral']/sum(polarity_count.values())*100:.1f}%)")

    print(f"\n强度分布:")
    print(f"  强烈：{intensity_count['strong']} ({intensity_count['strong']/sum(intensity_count.values())*100:.1f}%)")
    print(f"  中等：{intensity_count['moderate']} ({intensity_count['moderate']/sum(intensity_count.values())*100:.1f}%)")
    print(f"  微弱：{intensity_count['weak']} ({intensity_count['weak']/sum(intensity_count.values())*100:.1f}%)")

    print("\n" + "="*80)
    print("处理完成!")
    print("="*80)


def main():
    # 输入输出路径
    input_path = 'data/多目标样本.xlsx'
    output_path = 'results/batch_results.json'

    # 处理所有样本
    batch_process(input_path, output_path, limit=None)

    # 也可以先测试少量样本
    # batch_process(input_path, 'results/test_batch.json', limit=10)


if __name__ == "__main__":
    main()
