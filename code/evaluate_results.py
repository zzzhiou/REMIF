#!/usr/bin/env python3
"""
多目标立场检测结果评估

使用 BERTScore、BLEU、ROUGE-L 评估目标抽取质量
使用匈牙利算法进行最优匹配
计算立场分类准确率
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


class TargetSimilarityEvaluator:
    def __init__(self):
        self.bert_model = 'bert-base-chinese'
        self.lang = 'zh'
        self.smoothing = SmoothingFunction().method3

        # 修正 ROUGE 配置
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
        """评估目标抽取质量"""
        true = self._preprocess_targets(true_targets)
        pred = self._preprocess_targets(pred_targets)

        if not true or not pred:
            return {
                'bert_f1': 0.0,
                'bleu': 0.0,
                'rouge_l': 0.0,
                'matched_pairs': [],
                'true_count': len(true),
                'pred_count': len(pred),
                'recall_ratio': 0.0 if not true else 1.0,
                'precision_ratio': 0.0 if not pred else 1.0
            }

        bert_matrix = self._get_bert_score_matrix(true, pred)
        bleu_matrix = self._get_bleu_score_matrix(true, pred)
        rouge_matrix = self._get_rouge_l_matrix(true, pred)

        # 匈牙利算法最优匹配
        row_ind, col_ind = linear_sum_assignment(-bert_matrix)

        bert_scores, bleu_scores, rouge_scores = [], []
        for i, j in zip(row_ind, col_ind):
            bert_scores.append(bert_matrix[i][j])
            bleu_scores.append(bleu_matrix[i][j])
            rouge_scores.append(rouge_matrix[i][j])

        recall_ratio = len(set(row_ind)) / len(true) if true else 0.0
        precision_ratio = len(set(col_ind)) / len(pred) if pred else 0.0

        avg_bert = np.mean(bert_scores) if bert_scores else 0.0
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0

        return {
            'bert_f1': float(avg_bert),
            'bleu': float(avg_bleu),
            'rouge_l': float(avg_rouge),
            'matched_pairs': list(zip(
                [true[i] for i in row_ind],
                [pred[j] for j in col_ind],
                map(float, bert_scores),
                map(float, bleu_scores),
                map(float, rouge_scores)
            )),
            'true_count': len(true),
            'pred_count': len(pred),
            'recall_ratio': float(recall_ratio),
            'precision_ratio': float(precision_ratio)
        }

    def evaluate_stances(self, true_stances, pred_stances, true_targets, pred_targets):
        """评估立场分类准确率（基于目标匹配后）"""
        true = self._preprocess_stances(true_stances)
        pred = self._preprocess_stances(pred_stances)
        true_t = self._preprocess_targets(true_targets)
        pred_t = self._preprocess_targets(pred_targets)

        if not true or not pred or not true_t or not pred_t:
            return {
                'stance_accuracy': 0.0,
                'stance_f1': 0.0,
                'stance_precision': 0.0,
                'stance_recall': 0.0
            }

        # 首先通过 BERTScore 匹配目标
        bert_matrix = self._get_bert_score_matrix(true_t, pred_t)
        row_ind, col_ind = linear_sum_assignment(-bert_matrix)

        # 基于目标匹配计算立场准确率
        correct = 0
        total = min(len(true), len(pred))

        # 立场极性映射
        polarity_map = {
            '支持': 'support', 'support': 'support',
            '反对': 'oppose', 'oppose': 'oppose',
            '中立': 'neutral', 'neutral': 'neutral'
        }

        matched_stances = []
        for i, j in zip(row_ind, col_ind):
            true_polarity = polarity_map.get(true[i] if i < len(true) else '', 'neutral')
            pred_polarity = polarity_map.get(pred[j] if j < len(pred) else '', 'neutral')

            if true_polarity == pred_polarity:
                correct += 1
            matched_stances.append({
                'true_target': true_t[i],
                'pred_target': pred_t[j],
                'true_stance': true[i] if i < len(true) else '',
                'pred_stance': pred[j] if j < len(pred) else '',
                'match': true_polarity == pred_polarity
            })

        stance_accuracy = correct / total if total > 0 else 0.0

        # 计算立场分类的 Precision/Recall/F1
        tp = sum(1 for m in matched_stances if m['match'])
        fp = total - tp
        fn = len(true) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'stance_accuracy': float(stance_accuracy),
            'stance_f1': float(f1),
            'stance_precision': float(precision),
            'stance_recall': float(recall),
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

    # 初始化评估器
    evaluator = TargetSimilarityEvaluator()

    # 评估结果列表
    bert_list, bleu_list, rouge_l_list = [], []
    precision_list, recall_list = [], []
    qty_f1_list, combined_list = [], []
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
        target_res = evaluator.evaluate_targets(gold_targets, pred_targets)

        # 评估立场
        stance_res = evaluator.evaluate_stances(
            gold_stances, pred_stances_cn,
            gold_targets, pred_targets
        )

        # 记录指标
        bert_list.append(target_res['bert_f1'])
        bleu_list.append(target_res['bleu'])
        rouge_l_list.append(target_res['rouge_l'])
        precision_list.append(target_res['precision_ratio'])
        recall_list.append(target_res['recall_ratio'])

        # 数量 F1
        p = target_res['precision_ratio']
        r = target_res['recall_ratio']
        qty_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        qty_f1_list.append(qty_f1)

        # 综合得分
        current_combined = (0.6 * target_res['bert_f1'] +
                          0.2 * target_res['bleu'] +
                          0.2 * target_res['rouge_l']) * target_res['recall_ratio']
        combined_list.append(current_combined)

        # 立场指标
        stance_acc_list.append(stance_res['stance_accuracy'])
        stance_f1_list.append(stance_res['stance_f1'])

        # 保存评估结果
        eval_results.append({
            'sample_id': idx,
            'gold_targets': gold_targets,
            'pred_targets': pred_targets,
            'gold_stances': gold_stances,
            'pred_stances': pred_stances_cn,
            'bert_f1': target_res['bert_f1'],
            'bleu': target_res['bleu'],
            'rouge_l': target_res['rouge_l'],
            'precision': target_res['precision_ratio'],
            'recall': target_res['recall_ratio'],
            'qty_f1': qty_f1,
            'combined_score': current_combined,
            'stance_accuracy': stance_res['stance_accuracy'],
            'stance_f1': stance_res['stance_f1'],
            'matched_pairs': target_res['matched_pairs'],
            'matched_stances': stance_res['matched_stances']
        })

    # 计算平均值
    avg_bert = np.mean(bert_list)
    avg_bleu = np.mean(bleu_list)
    avg_rouge = np.mean(rouge_l_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_qty_f1 = np.mean(qty_f1_list)
    avg_combined = (0.6 * avg_bert + 0.2 * avg_bleu + 0.2 * avg_rouge) * avg_recall

    avg_stance_acc = np.mean(stance_acc_list)
    avg_stance_f1 = np.mean(stance_f1_list)

    # 打印结果
    print("\n" + "="*80)
    print("评估结果汇总")
    print("="*80)

    print("\n【目标抽取评估】")
    print(f"  BERTScore F1:  {avg_bert:.4f}")
    print(f"  BLEU:          {avg_bleu:.4f}")
    print(f"  ROUGE-L:       {avg_rouge:.4f}")
    print(f"  Precision:     {avg_precision:.4f}")
    print(f"  Recall:        {avg_recall:.4f}")
    print(f"  Quantity F1:   {avg_qty_f1:.4f}")
    print(f"  Combined:      {avg_combined:.4f}")

    print("\n【立场分类评估】")
    print(f"  准确率：       {avg_stance_acc:.4f}")
    print(f"  F1 分数：       {avg_stance_f1:.4f}")

    # 保存结果到 Excel
    df_summary = pd.DataFrame(eval_results)

    # 创建 Excel writer
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 所有详细结果
        df_summary.to_excel(writer, sheet_name='详细结果', index=False)

        # 摘要统计
        summary_data = {
            '指标': ['BERTScore F1', 'BLEU', 'ROUGE-L', 'Precision', 'Recall',
                    'Quantity F1', 'Combined', 'Stance Accuracy', 'Stance F1'],
            '平均值': [avg_bert, avg_bleu, avg_rouge, avg_precision, avg_recall,
                      avg_qty_f1, avg_combined, avg_stance_acc, avg_stance_f1]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='摘要统计', index=False)

    print(f"\n✅ 评估结果已保存到：{output_path}")

    return {
        'target_metrics': {
            'bert_f1': avg_bert,
            'bleu': avg_bleu,
            'rouge_l': avg_rouge,
            'precision': avg_precision,
            'recall': avg_recall,
            'qty_f1': avg_qty_f1,
            'combined': avg_combined
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
