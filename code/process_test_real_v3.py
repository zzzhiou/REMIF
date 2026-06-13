#!/usr/bin/env python3
"""
测试集处理和评估 - 使用真实 LLM API (改进版 v3)

v3 改进：
1. Few-shot prompting - 添加示例帮助 LLM 理解任务
2. 更明确的目标类型定义
3. 添加思考步骤引导
"""

import pandas as pd
import json
import os
import re
import requests
from typing import List, Dict, Any
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm
from datetime import datetime


# ============ LLM API 调用 ============

API_KEY = "sk-4e28c744a36f4ac6a524c119f6317e7"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

def call_llm(system_prompt: str, user_prompt: str, model: str = "qwen-max") -> str:
    """调用通义千问 API"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 3072
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        if "error" in result:
            err_msg = result["error"].get("message", str(result["error"]))
            print(f"API 错误：{err_msg}")
            return ""

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"请求失败：{e}")
        return ""


# ============ 阶段 1: 目标抽取（Few-shot 增强版） ============

TARGET_EXTRACTION_PROMPT = """你是一名立场目标抽取专家。

## 任务
从社交媒体文本中抽取所有被表达立场的目标。

## 目标定义
立场目标是文本中表达态度、观点或立场的对象，包括：
1. **实体型**：人名、组织、产品、事件
2. **议题型**：抽象概念、问题、现象
3. **问题型**：文本中讨论的核心问题

## 关键原则
1. **简洁性**：目标是精炼的名词短语（2-15 字），不是完整句子
2. **完整性**：抽取所有目标，不要遗漏
3. **抽象层级**：
   - 讨论具体实体 → 用实体名（如"杨子"）
   - 讨论抽象问题 → 用抽象表述（如"涉事产品"）

## 示例

输入："杨子故意跟李行亮说谎，这是为啥？避嫌吗？"
输出：{"targets": [{"text": "杨子"}]}

输入："连续调整的阿里巴巴终于开始涨了，港股也基本到底了，A 股即将反弹。"
输出：{"targets": [{"text": "阿里巴巴"}, {"text": "港股"}, {"text": "A 股"}]}

输入："现在有 4 个问题：具体的产品名、添加了哪种禁用原料、有何种危害、百雀羚怎么弥补。"
输出：{"targets": [{"text": "涉事产品名称"}, {"text": "禁用原料种类"}, {"text": "潜在危害"}, {"text": "百雀羚的应对措施"}]}

输入："#lisa 出席 vma#Lisa 身着华丽服装出席 2024 MTV VMAs 活动"
输出：{"targets": [{"text": "Lisa"}, {"text": "2024 MTV VMAs"}]}

## 输出格式（严格 JSON）
{"targets": [{"text": "目标 1"}, {"text": "目标 2"}]}

只返回 JSON，不要其他内容。"""


def extract_targets(text: str) -> List[Dict[str, Any]]:
    """使用 LLM 进行目标抽取（Few-shot 增强版）"""
    user_prompt = f"""请从以下文本中抽取所有立场目标：

{text}

请抽取目标："""

    response = call_llm(TARGET_EXTRACTION_PROMPT, user_prompt)

    if not response:
        return []

    # 解析 JSON
    try:
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, response)
        if match:
            data = json.loads(match.group(0))
            targets = data.get('targets', [])
            return [
                {"id": f"T{i+1}", "text": t.get('text', ''), "type": "explicit"}
                for i, t in enumerate(targets) if t.get('text')
            ]
    except Exception as e:
        print(f"  目标抽取解析失败：{e}")
        pass

    return []


# ============ 阶段 2: 立场检测（增强版） ============

STANCE_DETECTION_PROMPT = """你是一名立场检测专家。

## 任务
判断文本对每个给定目标的立场态度。

## 立场类别
- **支持**：赞同、肯定、推荐、维护、期待等正面态度
- **反对**：批评、否定、抵制、担忧、质疑等负面态度
- **中立**：客观描述、无明显倾向、混合态度

## 判断原则
1. 结合完整上下文判断
2. 注意情感词汇和语气
3. 疑问句通常表达质疑/反对
4. "期待"、"希望"通常是支持/中立

## 示例

文本："杨子故意跟李行亮说谎，这是为啥？避嫌吗？"
目标：杨子
输出：{"stances": [{"target": "杨子", "polarity": "反对"}]}

文本："连续调整的阿里巴巴终于开始涨了"
目标：阿里巴巴
输出：{"stances": [{"target": "阿里巴巴", "polarity": "支持"}]}

文本："具体的产品名、添加了哪种禁用原料、有何种危害"
目标：涉事产品名称
输出：{"stances": [{"target": "涉事产品名称", "polarity": "中立"}]}

## 输出格式（严格 JSON）
{"stances": [{"target": "目标", "polarity": "支持/反对/中立"}]}

只返回 JSON，不要其他内容。"""


def detect_stance(text: str, targets: List[Dict]) -> List[Dict]:
    """使用 LLM 进行立场检测（增强版）"""
    if not targets:
        return []

    target_list = "\n".join([f"- {t['text']}" for t in targets])
    user_prompt = f"""文本：
{text}

需要判断立场的目标：
{target_list}

请判断每个目标的立场："""

    response = call_llm(STANCE_DETECTION_PROMPT, user_prompt)

    if not response:
        return []

    # 解析 JSON
    try:
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, response)
        if match:
            data = json.loads(match.group(0))
            stances = data.get('stances', [])
            return [
                {"target": s.get('target', ''), "polarity": s.get('polarity', '中立'),
                 "polarity_cn": s.get('polarity', '中立')}
                for s in stances if s.get('target')
            ]
    except Exception as e:
        print(f"  立场检测解析失败：{e}")
        pass

    return []


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
    """评估目标识别率"""
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

    sim_matrix = get_similarity_matrix(gold_list, pred_list)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

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
    """评估立场分类准确率"""
    gold_t = preprocess_targets(gold_targets)
    gold_s = preprocess_stances(gold_stances)
    pred_t = pred_targets if isinstance(pred_targets, list) else preprocess_targets(pred_targets)
    pred_s = pred_stances if isinstance(pred_stances, list) else preprocess_stances(pred_stances)

    if not gold_t or not pred_t:
        return {'accuracy': 0.0, 'correct': 0, 'total': 0}

    sim_matrix = get_similarity_matrix(gold_t, pred_t)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    polarity_map = {'支持': 0, '反对': 1, '中立': 2}

    correct = 0
    total = 0

    for i, j in zip(row_ind, col_ind):
        if sim_matrix[i][j] < threshold:
            continue

        total += 1
        true_p = polarity_map.get(gold_s[i] if i < len(gold_s) else '中立', 2)
        pred_stance_raw = pred_stances[j] if j < len(pred_stances) else '中立'
        pred_p = polarity_map.get(pred_stance_raw, 2)

        if true_p == pred_p:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {'accuracy': accuracy, 'correct': correct, 'total': total}


# ============ 主流程 ============

def process_test_dataset(input_path: str, output_path: str, eval_output_path: str):
    """处理测试集并评估"""
    print("="*80)
    print("测试集处理和评估（真实 LLM API - Few-shot v3）")
    print("="*80)

    # 测试 API
    print("\n测试 API 连接...")
    test_response = call_llm("你是一个助手。", "你好")
    if test_response:
        print("  API 连接成功")
    else:
        print("  API 连接失败，请检查 API Key")
        return

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
        extracted_targets = extract_targets(text)

        # Stage 2: 立场检测
        detected_stances = detect_stance(text, extracted_targets)

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

        if (idx + 1) % 100 == 0:
            print(f"  已处理 {idx+1}/{len(df)} 个样本")

    # 保存预测结果
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Excel 格式
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

    precision_list, recall_list, f1_list = [], [], []
    stance_correct, stance_total = 0, 0

    eval_results = []

    for p in tqdm(predictions, desc="评估中"):
        target_res = evaluate_targets(p['gold_targets'], p['predicted_targets'])
        stance_res = evaluate_stances(
            p['gold_targets'], p['gold_stances'],
            p['predicted_targets'], p['predicted_stances']
        )

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

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    overall_stance_acc = stance_correct / stance_total if stance_total > 0 else 0.0

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

    combined = 0.6 * avg_f1 + 0.4 * overall_stance_acc
    print(f"\n【综合得分】")
    print(f"  Combined (0.6*F1 + 0.4*Acc): {combined:.4f}")

    # 保存评估结果
    df_eval = pd.DataFrame(eval_results)
    os.makedirs(os.path.dirname(eval_output_path) or '.', exist_ok=True)

    with pd.ExcelWriter(eval_output_path, engine='openpyxl') as writer:
        df_eval.to_excel(writer, sheet_name='详细评估', index=False)

        summary = {
            '指标': ['Precision', 'Recall', 'F1 Score', 'Stance Accuracy'],
            '平均值': [avg_precision, avg_recall, avg_f1, overall_stance_acc]
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name='摘要', index=False)

        metadata = {
            '评估时间': datetime.now().isoformat(),
            '测试集': input_path,
            '样本数': len(predictions),
            'API': 'qwen-max',
            '版本': 'v3-few-shot'
        }
        pd.DataFrame([metadata]).to_excel(writer, sheet_name='元数据', index=False)

    print(f"\n评估结果已保存到：{eval_output_path}")

    return {
        'target_metrics': {'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1},
        'stance_metrics': {'accuracy': overall_stance_acc, 'correct': stance_correct, 'total': stance_total},
        'combined': combined
    }


if __name__ == "__main__":
    input_path = 'data/test-dataset-1k.xlsx'
    output_path = 'results/test_predictions_real_v3.json'
    eval_output_path = 'results/test_evaluation_real_v3.xlsx'

    metrics = process_test_dataset(input_path, output_path, eval_output_path)

    print("\n" + "="*80)
    print("最终摘要")
    print("="*80)
    print(f"目标识别率 (Recall): {metrics['target_metrics']['recall']:.4f}")
    print(f"目标抽取 F1:        {metrics['target_metrics']['f1']:.4f}")
    print(f"立场分类准确率：{metrics['stance_metrics']['accuracy']:.4f}")
    print(f"综合得分：{metrics['combined']:.4f}")
