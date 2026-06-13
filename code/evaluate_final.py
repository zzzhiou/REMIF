#!/usr/bin/env python3
"""
多目标立场检测结果评估

使用原始金标数据评估：
1. 目标抽取：Precision/Recall/F1（基于字符串相似度 + 匈牙利算法）
2. 立场分类：准确率/F1
"""

import json
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from difflib import SequenceMatcher
from tqdm import tqdm
import os
from datetime import datetime


def simple_similarity(a, b):
    """字符串相似度（基于编辑距离）"""
    return SequenceMatcher(None, a, b).ratio()


def preprocess_targets(target_str):
    """解析目标字符串"""
    if not target_str:
        return []
    if isinstance(target_str, list):
        return [str(t).strip() for t in target_str if str(t).strip()]
    targets = str(target_str).replace(';', ';').split(';')
    return [t.strip() for t in targets if t.strip()]


def preprocess_stances(stance_str):
    """解析立场字符串"""
    if not stance_str:
        return []
    if isinstance(stance_str, list):
        return [str(s).strip() for s in stance_str if str(s).strip()]
    stances = str(stance_str).replace(';', ';').split(';')
    return [s.strip() for s in stances if s.strip()]


def get_similarity_matrix(refs, cands):
    """计算相似度矩阵"""
    sim_matrix = np.zeros((len(refs), len(cands)))
    for i, ref in enumerate(refs):
        for j, cand in enumerate(cands):
            sim_matrix[i][j] = simple_similarity(ref, cand)
    return sim_matrix


def evaluate_sample(gold_targets, gold_stances, pred_targets, pred_stances, threshold=0.5):
    """
    评估单个样本

    Args:
        gold_targets: 金标目标列表
        gold_stances: 金标立场列表
        pred_targets: 预测目标列表
        pred_stances: 预测立场列表
        threshold: 匹配阈值

    Returns:
        评估结果字典
    """
    if not gold_targets or not pred_targets:
        return {
            'precision': 0.0 if pred_targets else 1.0,
            'recall': 0.0 if gold_targets else 1.0,
            'f1': 0.0,
            'stance_accuracy': 0.0,
            'matched_count': 0
        }

    # 目标匹配
    sim_matrix = get_similarity_matrix(gold_targets, pred_targets)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    # 统计匹配
    matched = sum(1 for i, j in zip(row_ind, col_ind) if sim_matrix[i][j] >= threshold)

    precision = matched / len(pred_targets) if pred_targets else 1.0
    recall = matched / len(gold_targets) if gold_targets else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 立场准确率
    polarity_map = {'支持': 0, 'support': 0, '反对': 1, 'oppose': 1, '中立': 2, 'neutral': 2}

    stance_correct = 0
    stance_total = 0

    for i, j in zip(row_ind, col_ind):
        if sim_matrix[i][j] < threshold:
            continue

        stance_total += 1
        true_p = polarity_map.get(gold_stances[i] if i < len(gold_stances) else '中立', 2)
        pred_p = polarity_map.get(pred_stances[j] if j < len(pred_stances) else 'neutral', 2)

        if true_p == pred_p:
            stance_correct += 1

    stance_accuracy = stance_correct / stance_total if stance_total > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'stance_accuracy': stance_accuracy,
        'matched_count': matched,
        'gold_count': len(gold_targets),
        'pred_count': len(pred_targets),
        'stance_total': stance_total,
        'stance_correct': stance_correct
    }


def batch_evaluate(batch_results_path: str, gold_data_path: str, output_path: str):
    """批量评估"""
    print("="*80)
    print("多目标立场检测结果评估")
    print("="*80)

    # 加载结果
    with open(batch_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data['results']
    print(f"\n预测结果数：{len(results)}")

    # 加载金标数据
    df_gold = pd.read_excel(gold_data_path)
    print(f"金标样本数：{len(df_gold)}")

    # 评估
    print("\n开始评估...")

    eval_results = []
    precision_list, recall_list, f1_list = [], [], []
    stance_acc_list = []
    total_correct, total_stance = 0, 0

    for idx, result in enumerate(tqdm(results, desc="评估中")):
        # 金标
        gold_targets_str = df_gold.iloc[idx]['target']
        gold_stances_str = df_gold.iloc[idx]['stance']

        gold_targets = preprocess_targets(gold_targets_str)
        gold_stances = preprocess_stances(gold_stances_str)

        # 预测
        pred_targets = [t['text'] for t in result.get('extracted_targets', [])]
        pred_stances = [s['polarity'] for s in result.get('stances', [])]

        # 评估
        eval_res = evaluate_sample(gold_targets, gold_stances, pred_targets, pred_stances)

        # 保存详细结果
        eval_results.append({
            'sample_id': idx,
            'gold_targets': gold_targets_str,
            'pred_targets': '; '.join(pred_targets),
            'gold_stances': gold_stances_str,
            'pred_stances': '; '.join([p for p in pred_stances]),
            'precision': eval_res['precision'],
            'recall': eval_res['recall'],
            'f1': eval_res['f1'],
            'stance_accuracy': eval_res['stance_accuracy'],
            'matched_count': eval_res['matched_count'],
            'gold_count': eval_res['gold_count'],
            'pred_count': eval_res['pred_count']
        })

        precision_list.append(eval_res['precision'])
        recall_list.append(eval_res['recall'])
        f1_list.append(eval_res['f1'])
        stance_acc_list.append(eval_res['stance_accuracy'])

        total_correct += eval_res['stance_correct']
        total_stance += eval_res['stance_total']

    # 计算平均值
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_stance_acc = np.mean(stance_acc_list)
    overall_stance_acc = total_correct / total_stance if total_stance > 0 else 0.0

    # 打印结果
    print("\n" + "="*80)
    print("评估结果汇总")
    print("="*80)

    print(f"\n【目标抽取评估】(样本数：{len(precision_list)})")
    print(f"  Precision:  {avg_precision:.4f}")
    print(f"  Recall:     {avg_recall:.4f}")
    print(f"  F1 Score:   {avg_f1:.4f}")

    print(f"\n【立场分类评估】")
    print(f"  平均准确率：{avg_stance_acc:.4f}")
    print(f"  总准确率：{overall_stance_acc:.4f} ({total_correct}/{total_stance})")

    # 计算综合得分
    combined_score = (0.6 * avg_f1 + 0.4 * overall_stance_acc)
    print(f"\n【综合得分】")
    print(f"  Combined (0.6*F1 + 0.4*Stance Acc): {combined_score:.4f}")

    # 保存结果
    df_eval = pd.DataFrame(eval_results)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 详细结果
        df_eval.to_excel(writer, sheet_name='详细结果', index=False)

        # 摘要统计
        summary_data = {
            '指标': ['Precision', 'Recall', 'F1 Score', 'Avg Stance Acc',
                    'Overall Stance Acc', 'Combined Score'],
            '平均值': [avg_precision, avg_recall, avg_f1, avg_stance_acc,
                      overall_stance_acc, combined_score]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='摘要统计', index=False)

        # 时间戳
        metadata = {
            '评估时间': datetime.now().isoformat(),
            '预测文件': batch_results_path,
            '金标文件': gold_data_path,
            '样本数': len(results)
        }
        pd.DataFrame([metadata]).to_excel(writer, sheet_name='元数据', index=False)

    print(f"\n✅ 评估结果已保存到：{output_path}")

    return {
        'target_metrics': {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        },
        'stance_metrics': {
            'avg_accuracy': avg_stance_acc,
            'overall_accuracy': overall_stance_acc
        },
        'combined_score': combined_score
    }


if __name__ == "__main__":
    batch_results_path = 'results/batch_results.json'
    gold_data_path = 'data/多目标样本.xlsx'
    output_path = 'results/evaluation_final.xlsx'

    metrics = batch_evaluate(batch_results_path, gold_data_path, output_path)

    # 最终摘要
    print("\n" + "="*80)
    print("最终评估摘要")
    print("="*80)
    print(f"\n目标识别率 (Recall):    {metrics['target_metrics']['recall']:.4f}")
    print(f"目标抽取 F1:           {metrics['target_metrics']['f1']:.4f}")
    print(f"立场分类准确率：       {metrics['stance_metrics']['overall_accuracy']:.4f}")
    print(f"综合得分：{metrics['combined_score']:.4f}")
