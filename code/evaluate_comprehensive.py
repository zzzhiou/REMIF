#!/usr/bin/env python3
"""
将 test_predictions_real.json 转换为评估格式并计算完整指标

输出指标包括：
- Target Identification: BERT-S, BLEU, ROUGE, Recall, C-Score
- Stance Detection: Precision, Recall, F1
"""

import numpy as np
from bert_score import score
from scipy.optimize import linear_sum_assignment
import torch
import jieba
import pandas as pd
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer
import os


class SpaceTokenizer(Tokenizer):
    """用于处理预分词的文本（空格分隔）"""
    def tokenize(self, text):
        return text.split()


class ComprehensiveEvaluator:
    def __init__(self):
        self.bert_model = 'bert-base-chinese'
        self.lang = 'zh'
        self.smoothing = SmoothingFunction().method3

        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rougeL'],
            use_stemmer=False,
            tokenizer=SpaceTokenizer()
        )
        jieba.initialize()

    def _preprocess_targets(self, target_str):
        if not target_str or (isinstance(target_str, float) and np.isnan(target_str)):
            return []
        if isinstance(target_str, list):
            return [str(t).strip() for t in target_str if str(t).strip()]
        targets = str(target_str).replace(';', ';').split(';')
        return [t.strip() for t in targets if t.strip()]

    def _preprocess_stances(self, stance_str):
        if not stance_str or (isinstance(stance_str, float) and np.isnan(stance_str)):
            return []
        if isinstance(stance_str, list):
            return [str(s).strip() for s in stance_str if str(s).strip()]
        stances = str(stance_str).replace(';', ';').split(';')
        return [s.strip() for s in stances if s.strip()]

    def _get_bert_score_matrix(self, refs, cands):
        if not refs or not cands:
            return np.zeros((max(len(refs), 1), max(len(cands), 1)))

        pairs = [(cand, ref) for ref in refs for cand in cands]
        cand_texts = [cand for cand, _ in pairs]
        ref_texts = [ref for _, ref in pairs]

        try:
            _, _, f1s = score(
                cand_texts, ref_texts,
                model_type=self.bert_model,
                lang=self.lang,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=False
            )
            return np.array(f1s).reshape(len(refs), len(cands))
        except Exception as e:
            print(f"BERTScore 错误：{e}")
            return np.zeros((len(refs), len(cands)))

    def _get_bleu_score_matrix(self, refs, cands):
        sim_matrix = np.zeros((len(refs), len(cands)))
        for i, ref in enumerate(refs):
            ref_tokens = list(jieba.cut(ref))
            for j, cand in enumerate(cands):
                cand_tokens = list(jieba.cut(cand))
                sim_matrix[i][j] = sentence_bleu(
                    [ref_tokens], cand_tokens,
                    smoothing_function=self.smoothing
                )
        return sim_matrix

    def _get_rouge_l_matrix(self, refs, cands):
        sim_matrix = np.zeros((len(refs), len(cands)))
        for i, ref in enumerate(refs):
            processed_ref = ' '.join(jieba.cut(ref))
            for j, cand in enumerate(cands):
                processed_cand = ' '.join(jieba.cut(cand))
                score_ = self.rouge_scorer.score(processed_ref, processed_cand)
                sim_matrix[i][j] = score_['rougeL'].fmeasure
        return sim_matrix

    def evaluate_targets(self, true_targets, pred_targets):
        """
        评估目标抽取质量 - 按照论文指标

        返回：
        - bert_f1: BERTScore F1
        - bleu: BLEU
        - rouge_l: ROUGE-L F1
        - recall_ratio: Recall (数量召回率)
        - c_score: Combined Score
        """
        true = self._preprocess_targets(true_targets)
        pred = self._preprocess_targets(pred_targets)

        if not true and not pred:
            return {
                'bert_f1': 1.0,
                'bleu': 1.0,
                'rouge_l': 1.0,
                'recall_ratio': 1.0,
                'precision_ratio': 1.0,
                'c_score': 1.0,
                'matched_pairs': []
            }

        if not true or not pred:
            return {
                'bert_f1': 0.0,
                'bleu': 0.0,
                'rouge_l': 0.0,
                'recall_ratio': 0.0 if not true else 1.0,
                'precision_ratio': 0.0 if not pred else 1.0,
                'c_score': 0.0,
                'matched_pairs': []
            }

        # 计算相似度矩阵
        bert_matrix = self._get_bert_score_matrix(true, pred)
        bleu_matrix = self._get_bleu_score_matrix(true, pred)
        rouge_matrix = self._get_rouge_l_matrix(true, pred)

        # 匈牙利算法最优匹配（基于 BERTScore）
        row_ind, col_ind = linear_sum_assignment(-bert_matrix)

        bert_scores, bleu_scores, rouge_scores = [], []
        matched_pairs = []

        for i, j in zip(row_ind, col_ind):
            bert_scores.append(bert_matrix[i][j])
            bleu_scores.append(bleu_matrix[i][j])
            rouge_scores.append(rouge_matrix[i][j])
            matched_pairs.append({
                'true': true[i],
                'pred': pred[j],
                'bert': bert_matrix[i][j],
                'bleu': bleu_matrix[i][j],
                'rouge': rouge_matrix[i][j]
            })

        # 计算各项指标
        recall_ratio = len(set(row_ind)) / len(true) if true else 0.0
        precision_ratio = len(set(col_ind)) / len(pred) if pred else 0.0

        avg_bert = np.mean(bert_scores) if bert_scores else 0.0
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0

        # C-Score: (0.6*BERT + 0.2*BLEU + 0.2*ROUGE) * Recall
        c_score = (0.6 * avg_bert + 0.2 * avg_bleu + 0.2 * avg_rouge) * recall_ratio

        return {
            'bert_f1': float(avg_bert),
            'bleu': float(avg_bleu),
            'rouge_l': float(avg_rouge),
            'recall_ratio': float(recall_ratio),
            'precision_ratio': float(precision_ratio),
            'c_score': float(c_score),
            'matched_pairs': matched_pairs
        }

    def evaluate_stances(self, true_stances, pred_stances, true_targets, pred_targets):
        """
        评估立场分类 - 按照论文指标

        返回：
        - stance_precision: Precision
        - stance_recall: Recall
        - stance_f1: F1-score
        """
        true = self._preprocess_stances(true_stances)
        pred = self._preprocess_stances(pred_stances)
        true_t = self._preprocess_targets(true_targets)
        pred_t = self._preprocess_targets(pred_targets)

        if not true or not pred or not true_t or not pred_t:
            return {
                'stance_precision': 0.0,
                'stance_recall': 0.0,
                'stance_f1': 0.0,
                'matched_stances': []
            }

        # 通过 BERTScore 匹配目标
        bert_matrix = self._get_bert_score_matrix(true_t, pred_t)
        row_ind, col_ind = linear_sum_assignment(-bert_matrix)

        # 立场极性映射
        polarity_map = {
            '支持': 0, 'support': 0,
            '反对': 1, 'oppose': 1,
            '中立': 2, 'neutral': 2
        }

        # 统计 TP, FP, FN
        tp = 0
        fp = 0
        fn = 0

        matched_stances = []

        for i, j in zip(row_ind, col_ind):
            true_polarity = polarity_map.get(true[i] if i < len(true) else '中立', 2)
            pred_polarity = polarity_map.get(pred[j] if j < len(pred) else 'neutral', 2)

            match = (true_polarity == pred_polarity)

            if match:
                tp += 1
            else:
                fp += 1
                fn += 1

            matched_stances.append({
                'true_target': true_t[i],
                'pred_target': pred_t[j],
                'true_stance': true[i] if i < len(true) else '',
                'pred_stance': pred[j] if j < len(pred) else '',
                'match': match
            })

        # 计算 Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'stance_precision': float(precision),
            'stance_recall': float(recall),
            'stance_f1': float(f1),
            'matched_stances': matched_stances
        }


def convert_and_evaluate(predictions_path: str, output_path: str):
    """转换预测结果格式并进行评估"""
    print("="*80)
    print("多目标立场检测 - 完整评估（论文指标）")
    print("="*80)

    # 加载预测结果
    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    print(f"\n加载预测结果：{len(predictions)} 个样本")

    # 转换为评估格式
    print("\n转换为评估格式...")
    eval_results = []

    for pred in predictions:
        gold_targets_raw = pred['gold_targets']
        gold_stances_raw = pred['gold_stances']
        pred_targets = pred['predicted_targets']
        pred_stances = pred['predicted_stances']

        # 解析金标
        gold_targets = preprocess_targets(gold_targets_raw)
        gold_stances = preprocess_stances(gold_stances_raw)

        eval_results.append({
            'gold_targets': gold_targets,
            'gold_stances': gold_stances,
            'pred_targets': pred_targets,
            'pred_stances': pred_stances
        })

    print(f"转换完成：{len(eval_results)} 个样本")

    # 初始化评估器
    evaluator = ComprehensiveEvaluator()

    # 评估指标列表
    bert_list, bleu_list, rouge_list = [], [], []
    recall_list, precision_list = [], []
    c_score_list = []
    stance_p_list, stance_r_list, stance_f1_list = [], [], []

    detailed_results = []

    print("\n开始评估...")
    for idx, result in enumerate(tqdm(eval_results, desc="评估中")):
        # 评估目标
        target_res = evaluator.evaluate_targets(
            result['gold_targets'],
            result['pred_targets']
        )

        # 评估立场
        stance_res = evaluator.evaluate_stances(
            result['gold_stances'],
            result['pred_stances'],
            result['gold_targets'],
            result['pred_targets']
        )

        # 记录指标
        bert_list.append(target_res['bert_f1'])
        bleu_list.append(target_res['bleu'])
        rouge_list.append(target_res['rouge_l'])
        recall_list.append(target_res['recall_ratio'])
        precision_list.append(target_res['precision_ratio'])
        c_score_list.append(target_res['c_score'])

        stance_p_list.append(stance_res['stance_precision'])
        stance_r_list.append(stance_res['stance_recall'])
        stance_f1_list.append(stance_res['stance_f1'])

        # 保存详细结果
        detailed_results.append({
            'sample_id': idx,
            'gold_targets': result['gold_targets'],
            'pred_targets': result['pred_targets'],
            'gold_stances': result['gold_stances'],
            'pred_stances': result['pred_stances'],
            'bert_f1': target_res['bert_f1'],
            'bleu': target_res['bleu'],
            'rouge_l': target_res['rouge_l'],
            'recall': target_res['recall_ratio'],
            'precision': target_res['precision_ratio'],
            'c_score': target_res['c_score'],
            'stance_p': stance_res['stance_precision'],
            'stance_r': stance_res['stance_recall'],
            'stance_f1': stance_res['stance_f1'],
            'matched_pairs': target_res['matched_pairs'],
            'matched_stances': stance_res['matched_stances']
        })

    # 计算平均指标
    avg_bert = np.mean(bert_list)
    avg_bleu = np.mean(bleu_list)
    avg_rouge = np.mean(rouge_list)
    avg_recall = np.mean(recall_list)
    avg_precision = np.mean(precision_list)
    avg_c_score = np.mean(c_score_list)

    avg_stance_p = np.mean(stance_p_list)
    avg_stance_r = np.mean(stance_r_list)
    avg_stance_f1 = np.mean(stance_f1_list)

    # 打印结果
    print("\n" + "="*80)
    print("评估结果汇总")
    print("="*80)

    print("\n【Target Identification】")
    print(f"  BERT-S:   {avg_bert:.4f}")
    print(f"  BLEU:     {avg_bleu:.4f}")
    print(f"  ROUGE-L:  {avg_rouge:.4f}")
    print(f"  Recall:   {avg_recall:.4f}")
    print(f"  C-Score:  {avg_c_score:.4f}")

    print("\n【Stance Detection】")
    print(f"  Precision:  {avg_stance_p:.4f}")
    print(f"  Recall:     {avg_stance_r:.4f}")
    print(f"  F1-score:   {avg_stance_f1:.4f}")

    # 保存结果
    df_summary = pd.DataFrame(detailed_results)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='详细结果', index=False)

        summary_data = {
            '指标': ['BERT-S', 'BLEU', 'ROUGE-L', 'Recall', 'C-Score',
                    'Stance-P', 'Stance-R', 'Stance-F1'],
            '平均值': [avg_bert, avg_bleu, avg_rouge, avg_recall, avg_c_score,
                      avg_stance_p, avg_stance_r, avg_stance_f1]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='摘要统计', index=False)

    print(f"\n✅ 评估结果已保存到：{output_path}")

    return {
        'target_metrics': {
            'bert_s': avg_bert,
            'bleu': avg_bleu,
            'rouge_l': avg_rouge,
            'recall': avg_recall,
            'c_score': avg_c_score
        },
        'stance_metrics': {
            'precision': avg_stance_p,
            'recall': avg_stance_r,
            'f1': avg_stance_f1
        }
    }


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


if __name__ == "__main__":
    predictions_path = 'results/test_predictions_real.json'
    output_path = 'results/test_evaluation_comprehensive.xlsx'

    metrics = convert_and_evaluate(predictions_path, output_path)

    print("\n" + "="*80)
    print("最终摘要（与论文指标对比）")
    print("="*80)
    print(f"\n目标指标:")
    print(f"  BERT-S:   {metrics['target_metrics']['bert_s']:.4f}  (目标：0.8018)")
    print(f"  BLEU:     {metrics['target_metrics']['bleu']:.4f}    (目标：0.2831)")
    print(f"  ROUGE-L:  {metrics['target_metrics']['rouge_l']:.4f}  (目标：0.5732)")
    print(f"  Recall:   {metrics['target_metrics']['recall']:.4f}   (目标：0.9113)")
    print(f"  C-Score:  {metrics['target_metrics']['c_score']:.4f}  (目标：0.5945)")
    print(f"\n立场指标:")
    print(f"  Precision: {metrics['stance_metrics']['precision']:.4f}  (目标：0.8180)")
    print(f"  Recall:    {metrics['stance_metrics']['recall']:.4f}    (目标：0.8255)")
    print(f"  F1-score:  {metrics['stance_metrics']['f1']:.4f}    (目标：0.8096)")
