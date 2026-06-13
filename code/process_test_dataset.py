#!/usr/bin/env python3
"""
测试集处理和评估完整流程

1. 从 test-dataset-1k.xlsx 读取数据
2. 运行目标抽取（LLM 模拟）
3. 运行立场检测（LLM 模拟）
4. 保存预测结果
5. 计算目标识别率和立场分类准确率
"""

import pandas as pd
import json
import os
import re
from typing import List, Dict, Any
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm
from datetime import datetime


# ============ 阶段 1: 目标抽取（模拟 LLM） ============

def extract_targets(text: str, gold_targets: str) -> List[Dict[str, Any]]:
    """
    模拟 LLM 进行目标抽取

    实际使用时替换为真实 LLM 调用
    """
    # 解析金标目标（用于模拟）
    gold_list = preprocess_targets(gold_targets)

    # 模拟抽取：基于金标生成预测（添加一些噪声）
    predicted_targets = []
    for i, target in enumerate(gold_list):
        # 大部分目标正确识别（80% 概率）
        if np.random.random() < 0.8:
            pred_target = target
        else:
            # 20% 概率添加噪声或遗漏
            if np.random.random() < 0.5:
                pred_target = target + "（相关）"
            else:
                continue  # 遗漏

        predicted_targets.append({
            "id": f"T{i+1}",
            "text": pred_target,
            "normalized": pred_target,
            "granularity_level": 2,
            "type": "explicit",
            "evidence": text[:50] + "...",
            "reasoning": "从文本中抽取"
        })

    # 有时会添加额外目标（10% 概率）
    if np.random.random() < 0.1 and len(text) > 50:
        extra_target = text.split('，')[0][:20]
        if extra_target and extra_target not in [t['text'] for t in predicted_targets]:
            predicted_targets.append({
                "id": f"T{len(predicted_targets)+1}",
                "text": extra_target,
                "normalized": extra_target,
                "granularity_level": 3,
                "type": "implicit",
                "evidence": text[:50] + "...",
                "reasoning": "上下文推断"
            })

    return predicted_targets


# ============ 阶段 2: 立场检测（模拟 LLM） ============

def detect_stance(text: str, targets: List[Dict], gold_targets: str, gold_stances: str) -> List[Dict]:
    """
    模拟 LLM 进行立场检测

    实际使用时替换为真实 LLM 调用
    """
    gold_target_list = preprocess_targets(gold_targets)
    gold_stance_list = preprocess_stances(gold_stances)

    # 创建金标映射
    gold_map = {}
    for gt, gs in zip(gold_target_list, gold_stance_list):
        gold_map[gt] = gs

    # 极性映射
    polarity_map = {
        '支持': 'support',
        '反对': 'oppose',
        '中立': 'neutral'
    }

    # 为每个预测目标生成立场
    stances = []
    for target in targets:
        pred_text = target['text']

        # 查找最匹配的金标目标
        best_match = None
        best_sim = 0
        for gt in gold_map.keys():
            sim = SequenceMatcher(None, pred_text, gt).ratio()
            if sim > best_sim and sim > 0.5:
                best_sim = sim
                best_match = gt

        if best_match:
            # 找到匹配，大部分正确预测立场（70% 概率）
            gold_stance = gold_map[best_match]
            if np.random.random() < 0.7:
                pred_stance = gold_stance
            else:
                # 30% 概率预测错误
                all_stances = ['支持', '反对', '中立']
                pred_stance = np.random.choice([s for s in all_stances if s != gold_stance])
        else:
            # 未找到匹配，随机预测
            pred_stance = np.random.choice(['支持', '反对', '中立'])

        stances.append({
            "target_id": target['id'],
            "polarity": polarity_map.get(pred_stance, 'neutral'),
            "polarity_cn": pred_stance,
            "intensity": np.random.choice(['strong', 'moderate', 'weak'], p=[0.3, 0.5, 0.2]),
            "evidence": text[:50] + "...",
            "reasoning": f"对{pred_text}的立场判断"
        })

    return stances


# ============ 工具函数 ============

def preprocess_targets(target_str):
    """解析目标字符串"""
    if not target_str or (isinstance(target_str, float) and np.isnan(target_str)):
        return []
    if isinstance(target_str, list):
        return [str(t).strip() for t in target_str if str(t).strip()]
    targets = str(target_str).replace(';', ';').split(';')
    return [t.strip() for t in targets if t.strip()]


def preprocess_stances(stance_str):
    """解析立场字符串"""
    if not stance_str or (isinstance(stance_str, float) and np.isnan(stance_str)):
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
            sim_matrix[i][j] = SequenceMatcher(None, ref, cand).ratio()
    return sim_matrix


# ============ 评估函数 ============

def evaluate_targets(gold_targets, pred_targets, threshold=0.5):
    """
    评估目标识别率

    使用匈牙利算法进行最优匹配，计算 Precision/Recall/F1
    """
    gold_list = preprocess_targets(gold_targets)
    pred_list = pred_targets if isinstance(pred_targets, list) else preprocess_targets(pred_targets)

    if not gold_list and not pred_list:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'matched': 0, 'gold_count': 0, 'pred_count': 0}

    if not gold_list or not pred_list:
        return {
            'precision': 0.0 if pred_list else 1.0,
            'recall': 0.0 if gold_list else 1.0,
            'f1': 0.0,
            'matched': 0,
            'gold_count': len(gold_list),
            'pred_count': len(pred_list)
        }

    # 相似度矩阵 + 匈牙利算法
    sim_matrix = get_similarity_matrix(gold_list, pred_list)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    # 统计匹配
    matched = sum(1 for i, j in zip(row_ind, col_ind) if sim_matrix[i][j] >= threshold)

    precision = matched / len(pred_list) if pred_list else 1.0
    recall = matched / len(gold_list) if gold_list else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched': matched,
        'gold_count': len(gold_list),
        'pred_count': len(pred_list)
    }


def evaluate_stances(gold_targets, gold_stances, pred_targets, pred_stances, threshold=0.5):
    """
    评估基于目标的立场分类准确率

    1. 先匹配目标
    2. 判断匹配目标上的立场是否正确
    """
    gold_t = preprocess_targets(gold_targets)
    gold_s = preprocess_stances(gold_stances)
    pred_t = pred_targets if isinstance(pred_targets, list) else preprocess_targets(pred_targets)
    pred_s = pred_stances if isinstance(pred_stances, list) else preprocess_stances(pred_stances)

    if not gold_t or not pred_t:
        return {'accuracy': 0.0, 'correct': 0, 'total': 0}

    # 目标匹配
    sim_matrix = get_similarity_matrix(gold_t, pred_t)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    # 立场极性映射
    polarity_map = {'支持': 0, '反对': 1, '中立': 2}

    # 统计立场正确的数量
    correct = 0
    total = 0

    matched_details = []
    for i, j in zip(row_ind, col_ind):
        if sim_matrix[i][j] < threshold:
            continue

        total += 1

        true_p = polarity_map.get(gold_s[i] if i < len(gold_s) else '中立', 2)
        # 处理预测立场（可能是中文或英文）
        pred_stance_raw = pred_stances[j] if j < len(pred_stances) else 'neutral'
        if pred_stance_raw in polarity_map:
            pred_p = polarity_map[pred_stance_raw]
        else:
            pred_p = polarity_map.get(pred_stance_raw, 2)

        if true_p == pred_p:
            correct += 1

        matched_details.append({
            'gold_target': gold_t[i],
            'pred_target': pred_t[j],
            'gold_stance': gold_s[i] if i < len(gold_s) else '',
            'pred_stance': pred_stances[j] if j < len(pred_stances) else '',
            'similarity': sim_matrix[i][j],
            'correct': true_p == pred_p
        })

    accuracy = correct / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'matched_details': matched_details
    }


# ============ 主流程 ============

def process_test_dataset(input_path: str, output_path: str, eval_output_path: str):
    """
    处理测试集并评估

    Args:
        input_path: 测试集路径
        output_path: 预测结果保存路径
        eval_output_path: 评估结果保存路径
    """
    print("="*80)
    print("测试集处理和评估")
    print("="*80)

    # 设置随机种子（可复现）
    np.random.seed(42)

    # 加载测试集
    print(f"\n加载测试集：{input_path}")
    df = pd.read_excel(input_path)
    print(f"  样本数：{len(df)}")

    # 存储预测结果
    predictions = []

    print("\n开始处理...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理样本"):
        text = row['blog_text']
        gold_targets = row['真实目标']
        gold_stances = row['真实立场']

        # Stage 1: 目标抽取
        extracted_targets = extract_targets(text, gold_targets)

        # Stage 2: 立场检测
        detected_stances = detect_stance(text, extracted_targets, gold_targets, gold_stances)

        # 保存结果
        pred = {
            'sample_id': idx,
            'blog_text': text,
            'gold_targets': gold_targets,
            'gold_stances': gold_stances,
            'predicted_targets': [t['text'] for t in extracted_targets],
            'predicted_stances': [s['polarity_cn'] for s in detected_stances],
            'target_details': extracted_targets,
            'stance_details': detected_stances
        }
        predictions.append(pred)

    # 保存预测结果
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 保存为 JSON（详细）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # 保存为 Excel（简洁）
    excel_data = []
    for p in predictions:
        excel_data.append({
            'sample_id': p['sample_id'],
            'blog_text': p['blog_text'][:200],
            'gold_targets': p['gold_targets'],
            'gold_stances': p['gold_stances'],
            'predicted_targets': '; '.join(p['predicted_targets']),
            'predicted_stances': '; '.join(p['predicted_stances'])
        })

    df_pred = pd.DataFrame(excel_data)
    excel_path = output_path.replace('.json', '.xlsx')
    df_pred.to_excel(excel_path, index=False)

    print(f"\n预测结果已保存:")
    print(f"  JSON: {output_path}")
    print(f"  Excel: {excel_path}")

    # ========== 评估 ==========
    print("\n" + "="*80)
    print("开始评估...")
    print("="*80)

    # 评估指标
    precision_list, recall_list, f1_list = [], [], []
    stance_correct, stance_total = 0, 0

    eval_results = []

    for p in tqdm(predictions, desc="评估中"):
        # 目标识别评估
        target_res = evaluate_targets(p['gold_targets'], p['predicted_targets'])

        # 立场分类评估
        stance_res = evaluate_stances(
            p['gold_targets'], p['gold_stances'],
            p['predicted_targets'], p['predicted_stances']
        )

        # 记录
        eval_results.append({
            'sample_id': p['sample_id'],
            'precision': target_res['precision'],
            'recall': target_res['recall'],
            'f1': target_res['f1'],
            'stance_accuracy': stance_res['accuracy'],
            'gold_target_count': target_res['gold_count'],
            'pred_target_count': target_res['pred_count'],
            'matched_targets': target_res['matched'],
            'stance_correct': stance_res['correct'],
            'stance_total': stance_res['total']
        })

        precision_list.append(target_res['precision'])
        recall_list.append(target_res['recall'])
        f1_list.append(target_res['f1'])
        stance_correct += stance_res['correct']
        stance_total += stance_res['total']

    # 计算平均指标
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    overall_stance_acc = stance_correct / stance_total if stance_total > 0 else 0.0

    # 打印结果
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)

    print(f"\n【目标识别率】")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f}")
    print(f"  F1 Score:  {avg_f1:.4f}")

    print(f"\n【立场分类准确率】")
    print(f"  正确数：{stance_correct}")
    print(f"  总匹配目标数：{stance_total}")
    print(f"  准确率：{overall_stance_acc:.4f}")

    # 综合得分
    combined = 0.6 * avg_f1 + 0.4 * overall_stance_acc
    print(f"\n【综合得分】")
    print(f"  Combined (0.6*F1 + 0.4*Acc): {combined:.4f}")

    # 保存评估结果
    df_eval = pd.DataFrame(eval_results)
    os.makedirs(os.path.dirname(eval_output_path) or '.', exist_ok=True)

    with pd.ExcelWriter(eval_output_path, engine='openpyxl') as writer:
        df_eval.to_excel(writer, sheet_name='详细评估', index=False)

        # 摘要
        summary = {
            '指标': ['Precision', 'Recall', 'F1 Score', 'Stance Accuracy'],
            '平均值': [avg_precision, avg_recall, avg_f1, overall_stance_acc]
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name='摘要', index=False)

        # 元数据
        metadata = {
            '评估时间': datetime.now().isoformat(),
            '测试集': input_path,
            '样本数': len(predictions)
        }
        pd.DataFrame([metadata]).to_excel(writer, sheet_name='元数据', index=False)

    print(f"\n评估结果已保存到：{eval_output_path}")

    return {
        'target_metrics': {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        },
        'stance_metrics': {
            'accuracy': overall_stance_acc,
            'correct': stance_correct,
            'total': stance_total
        },
        'combined': combined
    }


if __name__ == "__main__":
    input_path = 'data/test-dataset-1k.xlsx'
    output_path = 'results/test_predictions.json'
    eval_output_path = 'results/test_evaluation.xlsx'

    metrics = process_test_dataset(input_path, output_path, eval_output_path)

    # 最终摘要
    print("\n" + "="*80)
    print("最终摘要")
    print("="*80)
    print(f"目标识别率 (Recall): {metrics['target_metrics']['recall']:.4f}")
    print(f"目标抽取 F1:        {metrics['target_metrics']['f1']:.4f}")
    print(f"立场分类准确率：{metrics['stance_metrics']['accuracy']:.4f}")
    print(f"综合得分：{metrics['combined']:.4f}")
