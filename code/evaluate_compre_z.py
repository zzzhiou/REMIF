#!/usr/bin/env python3
"""
针对 z1.py (纯 Prompt 基线) 输出结果的评估脚本
读取 test_predictions_real_z1.json，输出 test_evaluation_z1.xlsx
"""
import os

# ---------- 所有环境变量必须在这儿设置 ----------
os.environ['JIEBA_CACHE_FILE'] = '/data/stance_detection_project/stance_detection_project/.jieba.cache'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
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



class SpaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return text.split()

class BaselineEvaluator:
    def __init__(self):
        self.bert_model = '/data/stance_detection_project/stance_detection_project/bert-base-chinese'
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
        targets = str(target_str).replace('；', ';').split(';')
        return [t.strip() for t in targets if t.strip()]

    def _preprocess_stances(self, stance_str):
        if not stance_str or (isinstance(stance_str, float) and np.isnan(stance_str)):
            return []
        if isinstance(stance_str, list):
            return [str(s).strip() for s in stance_str if str(s).strip()]
        stances = str(stance_str).replace('；', ';').split(';')
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
                model_type=self.bert_model,   # 本地路径（旧版只认这个参数作为模型来源）
                num_layers=12,                # 强制指定层数，避免查询内置字典
                lang=self.lang,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=False,
                
                
            )
            return np.array(f1s).reshape(len(refs), len(cands))
        except Exception as e:           
            import traceback
            print(f"BERTScore 错误：{e}")
            traceback.print_exc()
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
        true = self._preprocess_targets(true_targets)
        pred = self._preprocess_targets(pred_targets)

        if not true and not pred:
            return {'bert_f1': 1.0, 'bleu': 1.0, 'rouge_l': 1.0, 'recall_ratio': 1.0, 'precision_ratio': 1.0, 'c_score': 1.0, 'matched_pairs': []}
        if not true or not pred:
            return {'bert_f1': 0.0, 'bleu': 0.0, 'rouge_l': 0.0, 'recall_ratio': 0.0 if not true else 1.0, 'precision_ratio': 0.0 if not pred else 1.0, 'c_score': 0.0, 'matched_pairs': []}

        bert_matrix = self._get_bert_score_matrix(true, pred)
        bleu_matrix = self._get_bleu_score_matrix(true, pred)
        rouge_matrix = self._get_rouge_l_matrix(true, pred)

        row_ind, col_ind = linear_sum_assignment(-bert_matrix)

        bert_scores, bleu_scores, rouge_scores, matched_pairs = [], [], [], []

        for i, j in zip(row_ind, col_ind):
            bert_scores.append(bert_matrix[i][j])
            bleu_scores.append(bleu_matrix[i][j])
            rouge_scores.append(rouge_matrix[i][j])
            matched_pairs.append({
                'true': true[i], 'pred': pred[j],
                'bert': bert_matrix[i][j], 'bleu': bleu_matrix[i][j], 'rouge': rouge_matrix[i][j]
            })

        recall_ratio = len(set(row_ind)) / len(true) if true else 0.0
        precision_ratio = len(set(col_ind)) / len(pred) if pred else 0.0

        avg_bert = np.mean(bert_scores) if bert_scores else 0.0
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0

        c_score = (0.6 * avg_bert + 0.2 * avg_bleu + 0.2 * avg_rouge) * recall_ratio

        return {
            'bert_f1': float(avg_bert), 'bleu': float(avg_bleu), 'rouge_l': float(avg_rouge),
            'recall_ratio': float(recall_ratio), 'precision_ratio': float(precision_ratio),
            'c_score': float(c_score), 'matched_pairs': matched_pairs
        }

    def evaluate_stances(self, true_stances, pred_stances, true_targets, pred_targets):
        true = self._preprocess_stances(true_stances)
        pred = self._preprocess_stances(pred_stances)
        true_t = self._preprocess_targets(true_targets)
        pred_t = self._preprocess_targets(pred_targets)

        if not true or not pred or not true_t or not pred_t:
            return {'stance_precision': 0.0, 'stance_recall': 0.0, 'stance_f1': 0.0, 'matched_stances': []}

        bert_matrix = self._get_bert_score_matrix(true_t, pred_t)
        row_ind, col_ind = linear_sum_assignment(-bert_matrix)

        # 增强版极性映射，容错大模型的自由输出
        polarity_map = {
            '支持': 0, 'support': 0, 'positive': 0, '赞同': 0,
            '反对': 1, 'oppose': 1, 'negative': 1, '不赞同': 1,
            '中立': 2, 'neutral': 2, 'none': 2, '无': 2
        }

        tp, fp, fn = 0, 0, 0
        matched_stances = []

        for i, j in zip(row_ind, col_ind):
            true_polarity = polarity_map.get(true[i].lower() if i < len(true) else '中立', 2)
            pred_polarity = polarity_map.get(pred[j].lower() if j < len(pred) else 'neutral', 2)

            match = (true_polarity == pred_polarity)

            if match:
                tp += 1
            else:
                fp += 1
                fn += 1

            matched_stances.append({
                'true_target': true_t[i], 'pred_target': pred_t[j],
                'true_stance': true[i] if i < len(true) else '', 'pred_stance': pred[j] if j < len(pred) else '',
                'match': match
            })

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {'stance_precision': float(precision), 'stance_recall': float(recall), 'stance_f1': float(f1), 'matched_stances': matched_stances}

def run_baseline_evaluation(predictions_path: str, output_path: str):
    print("="*80)
    print("纯 Prompt 基线 (zwt1.py) - 评估启动")
    print("="*80)

    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    evaluator = BaselineEvaluator()
    detailed_results = []
    
    bert_list, bleu_list, rouge_list, recall_list, c_score_list = [], [], [], [], []
    stance_p_list, stance_r_list, stance_f1_list = [], [], []

    for idx, result in enumerate(tqdm(predictions, desc="评估进度")):
        target_res = evaluator.evaluate_targets(result['gold_targets'], result['predicted_targets'])
        stance_res = evaluator.evaluate_stances(result['gold_stances'], result['predicted_stances'], result['gold_targets'], result['predicted_targets'])

        bert_list.append(target_res['bert_f1'])
        bleu_list.append(target_res['bleu'])
        rouge_list.append(target_res['rouge_l'])
        recall_list.append(target_res['recall_ratio'])
        c_score_list.append(target_res['c_score'])

        stance_p_list.append(stance_res['stance_precision'])
        stance_r_list.append(stance_res['stance_recall'])
        stance_f1_list.append(stance_res['stance_f1'])

        detailed_results.append({
            'sample_id': result.get('id', idx),
            'gold_targets': result['gold_targets'], 'pred_targets': result['predicted_targets'],
            'gold_stances': result['gold_stances'], 'pred_stances': result['predicted_stances'],
            'bert_f1': target_res['bert_f1'], 'bleu': target_res['bleu'], 'rouge_l': target_res['rouge_l'],
            'recall': target_res['recall_ratio'], 'c_score': target_res['c_score'],
            'stance_p': stance_res['stance_precision'], 'stance_r': stance_res['stance_recall'], 'stance_f1': stance_res['stance_f1']
        })

    avg_bert, avg_bleu, avg_rouge, avg_recall, avg_c_score = np.mean(bert_list), np.mean(bleu_list), np.mean(rouge_list), np.mean(recall_list), np.mean(c_score_list)
    avg_stance_p, avg_stance_r, avg_stance_f1 = np.mean(stance_p_list), np.mean(stance_r_list), np.mean(stance_f1_list)

    print("\n" + "="*80)
    print("Baseline 最终评估指标")
    print("="*80)
    print("\n【Target Identification (目标识别)】")
    print(f"  BERT-S:   {avg_bert:.4f}")
    print(f"  BLEU:     {avg_bleu:.4f}")
    print(f"  ROUGE-L:  {avg_rouge:.4f}")
    print(f"  Recall:   {avg_recall:.4f}")
    print(f"  C-Score:  {avg_c_score:.4f}")

    print("\n【Stance Detection (立场判定)】")
    print(f"  Precision: {avg_stance_p:.4f}")
    print(f"  Recall:    {avg_stance_r:.4f}")
    print(f"  F1-score:  {avg_stance_f1:.4f}")

    df_summary = pd.DataFrame(detailed_results)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='基线详细结果', index=False)
        summary_data = {
            '指标': ['BERT-S', 'BLEU', 'ROUGE-L', 'Recall', 'C-Score', 'Stance-P', 'Stance-R', 'Stance-F1'],
            '平均值': [avg_bert, avg_bleu, avg_rouge, avg_recall, avg_c_score, avg_stance_p, avg_stance_r, avg_stance_f1]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='基线摘要统计', index=False)

if __name__ == "__main__":
    # 配置输入与输出路径
    predictions_path = '/data/stance_detection_project/stance_detection_project/results/test_predictions_real_zwtcot_ds_526_final.json'
    output_path = '/data/stance_detection_project/stance_detection_project/results/test_evaluation_zwtcot_ds_526_final.xlsx'

    run_baseline_evaluation(predictions_path, output_path)