#!/usr/bin/env python3
"""
多目标立场检测结果评估 - 简化版

使用编辑距离和简单相似度评估目标抽取质量
计算立场分类准确率
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
import json
from tqdm import tqdm
import os
from difflib import SequenceMatcher


def simple_similarity(a, b):
    """简单字符串相似度（基于编辑距离）"""
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


def evaluate_targets(true_targets, pred_targets, threshold=0.6):
    """
    评估目标抽取质量

    使用匈牙利算法进行最优匹配，计算 Precision/Recall/F1
    """
    true = preprocess_targets(true_targets)
    pred = preprocess_targets(pred_targets)

    if not true and not pred:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'matched_count': 0,
            'true_count': 0,
            'pred_count': 0
        }

    if not true or not pred:
        return {
            'precision': 0.0 if pred else 1.0,
            'recall': 0.0 if true else 1.0,
            'f1': 0.0,
            'matched_count': 0,
            'true_count': len(true),
            'pred_count': len(pred)
        }

    # 计算相似度矩阵
    sim_matrix = get_similarity_matrix(true, pred)

    # 匈牙利算法最优匹配
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    # 统计匹配数量（相似度超过阈值）
    matched_count = 0
    matched_scores = []
    for i, j in zip(row_ind, col_ind):
        if sim_matrix[i][j] >= threshold:
            matched_count += 1
            matched_scores.append(sim_matrix[i][j])

    # 计算 Precision/Recall/F1
    precision = matched_count / len(pred) if pred else 1.0
    recall = matched_count / len(true) if true else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 计算平均匹配相似度
    avg_sim = np.mean(matched_scores) if matched_scores else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched_count': matched_count,
        'true_count': len(true),
        'pred_count': len(pred),
        'avg_similarity': avg_sim,
        'matched_pairs': [(true[i], pred[j], sim_matrix[i][j]) for i, j in zip(row_ind, col_ind)]
    }


def evaluate_stances(true_stances, pred_stances, true_targets, pred_targets, threshold=0.6):
    """
    评估立场分类准确率

    首先匹配目标，然后判断立场是否一致
    """
    true = preprocess_stances(true_stances)
    pred = preprocess_stances(pred_stances)
    true_t = preprocess_targets(true_targets)
    pred_t = preprocess_targets(pred_targets)

    if not true_t or not pred_t:
        return {
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }

    # 匹配目标
    sim_matrix = get_similarity_matrix(true_t, pred_t)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    # 立场极性映射
    polarity_map = {
        '支持': 0, 'support': 0,
        '反对': 1, 'oppose': 1,
        '中立': 2, 'neutral': 2
    }

    # 计算立场准确率
    correct = 0
    tp, fp, fn = 0, 0, 0
    matched_stances = []

    for i, j in zip(row_ind, col_ind):
        if sim_matrix[i][j] < threshold:
            continue

        true_polarity = polarity_map.get(true[i] if i < len(true) else '中立', 2)
        pred_polarity = polarity_map.get(pred[j] if j < len(pred) else 'neutral', 2)

        match = (true_polarity == pred_polarity)
        if match:
            correct += 1
            tp += 1
        else:
            fp += 1
            fn += 1

        matched_stances.append({
            'true_target': true_t[i],
            'pred_target': pred_t[j],
            'true_stance': true[i] if i < len(true) else '',
            'pred_stance': pred[j] if j < len(pred) else '',
            'similarity': sim_matrix[i][j],
            'match': match
        })

    total_matched = len([m for m in matched_stances if m['similarity'] >= threshold])
    accuracy = correct / total_matched if total_matched > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'matched_stances': matched_stances
    }


def evaluate_batch_results(batch_results_path: str, output_path: str):
    """批量评估结果"""
    print("="*80)
    print("多目标立场检测结果评估")
    print("="*80)

    # 加载结果
    with open(batch_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data['results']
    print(f"\n总样本数：{len(results)}")

    # 评估结果列表
    precision_list, recall_list, f1_list = [], [], []
    sim_scores = []
    stance_acc_list, stance_f1_list = [], []

    eval_results = []

    print("\n开始评估...")
    for idx, result in enumerate(tqdm(results, desc="评估中")):
        # 金标数据
        gold_targets = result.get('gold_targets', [])
        gold_stances = result.get('gold_stances', [])

        # 预测数据
        pred_targets = [t['text'] for t in result.get('extracted_targets', [])]
        pred_stances = [s['polarity'] for s in result.get('stances', [])]

        # 将 polarity 转换为中文
        polarity_cn_map = {'support': '支持', 'oppose': '反对', 'neutral': '中立'}
        pred_stances_cn = [polarity_cn_map.get(p, p) for p in pred_stances]

        # 评估目标
        target_res = evaluate_targets(gold_targets, pred_targets)

        # 评估立场
        stance_res = evaluate_stances(
            gold_stances, pred_stances_cn,
            gold_targets, pred_targets
        )

        # 记录指标
        precision_list.append(target_res['precision'])
        recall_list.append(target_res['recall'])
        f1_list.append(target_res['f1'])
        sim_scores.append(target_res.get('avg_similarity', 0.0))

        # 立场指标
        stance_acc_list.append(stance_res['accuracy'])
        stance_f1_list.append(stance_res['f1'])

        # 计算综合得分
        combined = (target_res['f1'] * target_res['recall'])

        # 保存评估结果
        eval_results.append({
            'sample_id': idx,
            'gold_targets': gold_targets,
            'pred_targets': pred_targets,
            'gold_stances': gold_stances,
            'pred_stances': pred_stances_cn,
            'precision': target_res['precision'],
            'recall': target_res['recall'],
            'f1': target_res['f1'],
            'similarity': target_res.get('avg_similarity', 0.0),
            'combined': combined,
            'stance_accuracy': stance_res['accuracy'],
            'stance_f1': stance_res['f1'],
            'matched_count': target_res['matched_count'],
            'true_count': target_res['true_count'],
            'pred_count': target_res['pred_count']
        })

    # 计算平均值
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_sim = np.mean(sim_scores)
    avg_combined = np.mean([e['combined'] for e in eval_results])

    avg_stance_acc = np.mean(stance_acc_list)
    avg_stance_f1 = np.mean(stance_f1_list)

    # 打印结果
    print("\n" + "="*80)
    print("评估结果汇总")
    print("="*80)

    print("\n【目标抽取评估】")
    print(f"  Precision:  {avg_precision:.4f}")
    print(f"  Recall:     {avg_recall:.4f}")
    print(f"  F1 Score:   {avg_f1:.4f}")
    print(f"  平均相似度：{avg_sim:.4f}")

    print("\n【立场分类评估】")
    print(f"  准确率：    {avg_stance_acc:.4f}")
    print(f"  F1 分数：   {avg_stance_f1:.4f}")

    # 保存结果到 Excel
    df_summary = pd.DataFrame(eval_results)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 所有详细结果
        df_summary.to_excel(writer, sheet_name='详细结果', index=False)

        # 摘要统计
        summary_data = {
            '指标': ['Precision', 'Recall', 'F1 Score', 'Avg Similarity',
                    'Combined', 'Stance Accuracy', 'Stance F1'],
            '平均值': [avg_precision, avg_recall, avg_f1, avg_sim,
                      avg_combined, avg_stance_acc, avg_stance_f1]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='摘要统计', index=False)

    print(f"\n✅ 评估结果已保存到：{output_path}")

    return {
        'target_metrics': {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'similarity': avg_sim
        },
        'stance_metrics': {
            'accuracy': avg_stance_acc,
            'f1': avg_stance_f1
        }
    }


if __name__ == "__main__":
    batch_results_path = 'results/batch_results.json'
    output_path = 'results/evaluation_results.xlsx'

    metrics = evaluate_batch_results(batch_results_path, output_path)

    # 打印最终摘要
    print("\n" + "="*80)
    print("最终评估摘要")
    print("="*80)
    print(f"\n目标识别率 (Recall): {metrics['target_metrics']['recall']:.4f}")
    print(f"目标抽取 F1:         {metrics['target_metrics']['f1']:.4f}")
    print(f"立场分类准确率：    {metrics['stance_metrics']['accuracy']:.4f}")
    print(f"立场分类 F1:        {metrics['stance_metrics']['f1']:.4f}")
