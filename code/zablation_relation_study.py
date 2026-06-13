#!/usr/bin/env python3
"""
细粒度关系消融实验 Pipeline - 基于 V6 架构与本地 Qwen2.5-7B-Instruct

核心功能：
1. 提取所有目标和全量关系（单样本仅执行 1 次，节约算力）
2. 在组装图结构时，逐一过滤特定关系（如剔除“因果”，保留其他）
3. 执行多轮 Stage 3 立场检测，对比各关系对立场传递的增益
4. 采用 V6 鲁棒评估函数，支持脏数据解析
"""

import os
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime
from tqdm import tqdm
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment
import numpy as np

# 尝试导入 torch_geometric
try:
    from torch_geometric.nn import RGCNConv
    from torch_geometric.utils import from_networkx
    PYG_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"警告：torch_geometric 不可用 ({e})，将使用简化版 R-GCN")
    PYG_AVAILABLE = False

try:
    from transformers import BertModel, BertTokenizer, AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"警告：transformers 不可用 ({e})")
    TRANSFORMERS_AVAILABLE = False


# ============ 1. 本地 LLM 加载与调用 ============

LOCAL_MODEL_PATH = "/data/models/Qwen2.5-7B-Instruct"

local_tokenizer = None
local_llm_model = None

def load_local_llm_model(model_path: str = None):
    """加载本地 Qwen 模型"""
    global local_tokenizer, local_llm_model

    if local_llm_model is not None:
        return local_llm_model, local_tokenizer

    path = model_path or LOCAL_MODEL_PATH
    print(f"  加载本地模型：{path}")

    local_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side='left')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    local_llm_model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto' if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )

    if local_tokenizer.pad_token is None:
        local_tokenizer.pad_token = local_tokenizer.eos_token

    print(f"  本地模型加载完成")
    return local_llm_model, local_tokenizer

def call_llm_local(system_prompt: str, user_prompt: str, temperature: float = 0.1, max_tokens: int = 1024, model_path: str = None) -> str:
    """调用本地模型生成回复"""
    global local_tokenizer, local_llm_model

    if local_llm_model is None or local_tokenizer is None:
        load_local_llm_model(model_path or LOCAL_MODEL_PATH)

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = local_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = local_tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=4096).to(local_llm_model.device)

        with torch.no_grad():
            generated_ids = local_llm_model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=local_tokenizer.pad_token_id,
                eos_token_id=local_tokenizer.eos_token_id
            )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return local_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"  本地模型调用失败：{e}")
        return ""

# ============ 2. 各阶段 Prompts 及 抽取模块 ============

TARGET_EXTRACTION_PROMPT = """# 角色设定
你是立场分析领域的资深专家，具有 10 年以上社交媒体文本分析经验。你的任务是精确识别文本中被表达立场的**目标对象**。

# 核心定义
**立场目标** = 文本作者对其表达态度、观点或立场的**具体对象**

# 目标识别的三大原则

## 原则一：目标必须是"对象"，不是"陈述"
✅ 正确示例（对象）：
- "杨子"（人名）
- "阿里巴巴"（公司名）
- "涉事产品"（抽象对象）
- "禁用原料种类"（问题对象）

❌ 错误示例（陈述/句子）：
- "杨子故意跟李行亮说谎"（这是事件描述，不是目标）
- "阿里巴巴股价上涨"（这是陈述，不是目标）

## 原则二：抽象层级必须与文本语境匹配
| 文本类型 | 目标形式 | 示例 |
|---------|---------|------|
| 讨论具体实体 | 使用实体名 | "Lisa"、"华为" |
| 讨论抽象议题 | 使用议题名 | "环保政策"、"技术竞争" |
| 讨论待解决问题 | 使用问题描述 | "涉事产品名称"、"潜在危害" |

## 原则三：宁缺毋滥 - 只抽取明确表达立场的目标
如果文本只是提及某个对象但没有表达任何态度（支持/反对/中立），则不应抽取。

# 高质量示例（Few-Shot）

## 示例 1：质疑/反对态度
输入："杨子故意跟李行亮说谎，这是为啥？避嫌吗？"
输出：{"targets": ["杨子"]}
分析：文本质疑杨子的行为，"杨子"是被表达负面立场的目标。

## 示例 2：多个并列目标
输入："连续调整的阿里巴巴终于开始涨了，港股也基本到底了，A 股即将反弹。"
输出：{"targets": ["阿里巴巴", "港股", "A 股"]}
分析：对三个对象都表达了积极期待。

## 示例 3：问题型目标（关键！）
输入："现在有 4 个问题：具体的产品名、添加了哪种禁用原料、有何种危害、百雀羚怎么弥补。"
输出：{"targets": ["涉事产品名称", "禁用原料种类", "潜在危害", "百雀羚的应对措施"]}
分析：这是消费者关心的问题，不是具体实体，需要用抽象表述。

## 示例 4：事件 + 人物
输入："#lisa 出席 vma#Lisa 身着华丽服装出席 2024 MTV VMAs 活动"
输出：{"targets": ["Lisa", "2024 MT VMAs"]}
分析：文本描述 Lisa 出席活动，两者都是被讨论的对象。

## 示例 5：边界情况 - 不要过度抽取
输入："今天天气真好，心情不错。"
输出：{"targets": []}
分析：这是日常描述，没有对任何公共议题/实体表达立场。

# 输出规范
1. 严格使用 JSON 格式：{"targets": ["目标 1", "目标 2", ...]}
2. 每个目标是 2-15 字的名词短语
3. 不要包含解释、分析或其他多余内容
4. 如果没有符合条件的目标，返回空数组：{"targets": []}

现在，请分析以下文本："""

def extract_targets(text: str) -> List[Dict[str, Any]]:
    """使用 LLM 进行目标抽取"""
    user_prompt = f"""文本：{text}

请抽取所有立场目标："""
    response = call_llm_local(TARGET_EXTRACTION_PROMPT, user_prompt, max_tokens=512)
    try:
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            data = json.loads(match.group(0))
            return [{"id": f"T{i+1}", "text": t.strip(), "normalized": t.strip(), "type": "explicit", "granularity_level": 2} for i, t in enumerate(data.get('targets', [])) if t]
    except: pass
    return []

RELATION_EXTRACTION_PROMPT = """# 角色设定
你是关系抽取专家，擅长判断文本中两个目标对象之间的关系类型。

# 关系类型（只能从以下 6 种中选择）
1. **contrast**: 对立关系（对 A 和 B 立场相反/冲突，如"支持 A 但反对 B"）
2. **consistent**: 一致关系（对 A 和 B 立场相同，如"都支持"或"都反对"）
3. **causal**: 因果关系（A 导致/影响 B，如"A 会导致 B"）
4. **hierarchical**: 层级关系（A 是 B 的上位/下位概念，如"A 包括 B"）
5. **parallel**: 并列关系（A 和 B 同级并列，无明显立场关联）
6. **analogy**: 类比关系（用 A 类比 B，如"A 就像 B"）

# 方向性说明
- contrast: 双向关系
- consistent: 双向关系
- causal: 有向关系 (A→B 表示 A 导致 B)
- hierarchical: 有向关系 (A→B 表示 A 是 B 的上位概念)
- parallel: 无向关系
- analogy: 有向关系 (A→B 表示用 A 类比 B)

# 输出格式（严格 JSON）
{
  "relation": "causal",
  "direction": "A→B",
  "evidence": "原文证据片段",
  "confidence": 0.85,
  "reasoning": "简短理由"
}

如果两个目标之间没有明显关系，输出：
{
  "relation": "none",
  "direction": "none",
  "evidence": "",
  "confidence": 1.0,
  "reasoning": "无明显关系"
}"""

def extract_all_relations(targets: List[Dict[str, Any]], context: str) -> List[Dict[str, Any]]:
    relations = []
    n = len(targets)
    for i in range(n):
        for j in range(i + 1, n):
            user_prompt = f"目标 A: {targets[i]['text']}\n目标 B: {targets[j]['text']}\n原文：\n{context}\n\n请判断关系（JSON）："
            response = call_llm_local(RELATION_EXTRACTION_PROMPT, user_prompt, max_tokens=512)
            try:
                match = re.search(r'\{[\s\S]*\}', response)
                if match:
                    data = json.loads(match.group(0))
                    rel = data.get("relation", "none")
                    if rel != "none" and float(data.get("confidence", 0.5)) >= 0.5:
                        relations.append({
                            "target_a": targets[i]['id'],
                            "target_b": targets[j]['id'],
                            "relation": rel,
                            "direction": data.get("direction", "none"),
                            "confidence": float(data.get("confidence", 0.85))
                        })
            except: pass
    return relations

class HeterogeneousGraphBuilder:
    def build(self, targets: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        for t in targets: G.add_node(t['id'], **t)
        for r in relations:
            src, tgt, rel_type = r['target_a'], r['target_b'], r['relation']
            G.add_edge(src, tgt, relation_type=rel_type, confidence=r['confidence'])
            if r['direction'] not in ['A→B', 'B→A']: # 双向或无向都加反向边辅助计算
                G.add_edge(tgt, src, relation_type=rel_type, confidence=r['confidence'])
        return G

STANCE_DETECTION_PROMPT = """# 角色设定
你是立场分析领域的资深专家，擅长判断文本对特定目标的立场态度。

# 立场分类体系
| 类别 | 定义 | 典型信号词 |
|-----|------|-----------|
| **支持** | 赞同、肯定、推荐、期待 | "支持"、"涨"、"好"、"期待"、"推荐" |
| **反对** | 批评、否定、质疑、担忧 | "反对"、"说谎"、"为啥"、"质疑"、"抵制" |
| **中立** | 客观描述、无明显倾向 | 纯描述性语言、疑问但无倾向 |

# 判断原则
1. 结合完整上下文判断，不要仅看关键词
2. 注意情感词汇和语气
3. 疑问句通常表达质疑/反对态度
4. "期待"、"希望"通常是支持或中立态度

# 高质量示例

## 示例 1
文本："杨子故意跟李行亮说谎，这是为啥？避嫌吗？"
目标：杨子
输出：{{"stances": [{{"target": "杨子", "polarity": "反对"}}]}}

## 示例 2
文本："连续调整的阿里巴巴终于开始涨了"
目标：阿里巴巴
输出：{{"stances": [{{"target": "阿里巴巴", "polarity": "支持"}}]}}

## 示例 3
文本："具体的产品名、添加了哪种禁用原料、有何种危害"
目标：涉事产品名称
输出：{{"stances": [{{"target": "涉事产品名称", "polarity": "中立"}}]}}
分析：这是消费者关心的问题列表，没有表达明确态度。

# 输出规范
1. 严格使用 JSON 格式：{{"stances": [{{"target_id": "目标 ID", "polarity": "支持/反对/中立"}}]}}
2. 只返回 JSON，不要解释
3. polarity 必须是"支持"、"反对"或"中立"之一

现在，请判断以下文本对给定目标的立场：

【目标列表】
{target_list}

【目标关系】
{relation_list}

【原始文本】
{text}"""


class GraphLLMStanceDetector:
    def detect(self, graph: nx.MultiDiGraph, targets: List[Dict[str, Any]], relations: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        if not targets: return []
        
        target_list = "\n".join([f"- {t['id']}: {t['normalized']}" for t in targets])
        relation_list = "无明显关系"
        if relations:
            relation_list = "\n".join([f"- {r['target_a']} ↔ {r['target_b']}: {r['relation']}" for r in relations])

        user_prompt = STANCE_DETECTION_PROMPT.format(target_list=target_list, relation_list=relation_list, text=text)
        response = call_llm_local(STANCE_DETECTION_PROMPT, user_prompt, max_tokens=1024)
        
        try:
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                data = json.loads(match.group(0))
                return [{"target_id": s.get('target_id', ''), "polarity": s.get('polarity', '中立')} for s in data.get('stances', [])]
        except: pass
        return []

# ============ 3. 评估指标 (复用 V6 核心鲁棒代码) ============

def preprocess_list(raw_str):
    if not raw_str or pd.isna(raw_str): return []
    if isinstance(raw_str, list): return [str(x).strip() for x in raw_str if str(x).strip()]
    # 将常见的中文分号、中英文逗号全部统一替换为英文分号，然后再切分
    cleaned_str = str(raw_str).replace('；', ';').replace('，', ';').replace(',', ';')
    return [x.strip() for x in cleaned_str.split(';') if x.strip()]


def evaluate_stances(gold_targets, gold_stances, pred_targets, pred_stances, threshold=0.5):
    gold_t, gold_s = preprocess_list(gold_targets), preprocess_list(gold_stances)
    if not gold_t or not pred_targets: return {'correct': 0, 'total': 0}

    sim_matrix = np.zeros((len(gold_t), len(pred_targets)))
    for i, ref in enumerate(gold_t):
        for j, cand in enumerate(pred_targets):
            sim_matrix[i][j] = SequenceMatcher(None, ref, cand).ratio()
            
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    correct, total = 0, 0
    polarity_map = {'支持': 0, '反对': 1, '中立': 2}

    for i, j in zip(row_ind, col_ind):
        if sim_matrix[i][j] >= threshold:
            total += 1
            true_p = polarity_map.get(gold_s[i] if i < len(gold_s) else '中立', 2)
            pred_p = polarity_map.get(pred_stances[j] if j < len(pred_stances) else '中立', 2)
            if true_p == pred_p:
                correct += 1
    return {'correct': correct, 'total': total}

# ============ 4. 主消融实验流程 ============

def run_ablation_experiment(data_path: str, output_path: str, sample_size: int = None, model_path: str = None):
    print("="*80)
    print("细粒度关系消融实验 Pipeline 开始运行")
    print("="*80)

    # 定义实验组（剔除特定关系）
    ablation_groups = {
        "Full_Graph": [],                                # 完整关系
        "No_Contrast": ["contrast"],                     # 剔除对立关系
        "No_Consistent": ["consistent"],                 # 剔除一致关系
        "No_Causal": ["causal"],                         # 剔除因果关系
        "No_Hierarchical": ["hierarchical"],             # 剔除层级关系
        "No_Parallel": ["parallel"],                     # 剔除并列关系
        "No_Analogy": ["analogy"],                       # 剔除类比关系
        "No_Relations": ["contrast", "consistent", "causal", "hierarchical", "parallel", "analogy"] # 无任何图结构
    }

    df = pd.read_excel(data_path)
    if sample_size: df = df.head(sample_size)

    # 自适应列名寻找
    cols = df.columns.tolist()
    text_col = next((c for c in ['text', 'blog_text', 'content', '原文'] if c in cols), None)
    target_col = next((c for c in ['真实目标', 'gold_targets', 'targets'] if c in cols), None)
    stance_col = next((c for c in ['真实立场', 'gold_stances', 'stances'] if c in cols), None)
    
    if not text_col: raise ValueError("无法在数据集中找到合法的文本列。")

    graph_builder = HeterogeneousGraphBuilder()
    stance_detector = GraphLLMStanceDetector()

    # 初始化统计字典
    results_accumulator = {group: {'correct': 0, 'total': 0} for group in ablation_groups}
    
    # 【新增】用来记录每一条数据死活的明细表
    detailed_logs = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="样本级处理（提取共享特征）"):
        text = row[text_col]
        gold_targets = row[target_col] if target_col else ""
        gold_stances = row[stance_col] if stance_col else ""

        # --- 第一阶段与第二阶段（提取目标和全量关系）：全局仅执行1次！节省算力 ---
        targets = extract_targets(text)
        all_relations = extract_all_relations(targets, text) if len(targets) >= 2 else []
        predicted_target_names = [t['text'] for t in targets]
        
        # 【新增】记录这条数据的死因分析
        log_entry = {
            "Excel行号": idx + 2, # +2 是因为索引从0开始，且Excel有表头
            "原文": text,
            "真实目标(Ground Truth)": gold_targets,
            "模型抽取目标(Pred)": str(predicted_target_names),
            "状态/丢失原因": "正常进入评估"
        }

        if pd.isna(gold_targets) or str(gold_targets).strip() == "":
            log_entry["状态/丢失原因"] = "原表缺失真实目标 (GT为空)"
        elif not targets:
            log_entry["状态/丢失原因"] = "模型未抽出任何目标 (Pred为空)"
            
        detailed_logs.append(log_entry)

        if not targets:
            continue

        # --- 第三阶段：按照消融矩阵遍历子图 ---
        for group_name, exclude_rels in ablation_groups.items():
            # 根据过滤规则剥离边
            filtered_rels = [r for r in all_relations if r['relation'] not in exclude_rels]
            
            # 构建过滤后的图
            graph = graph_builder.build(targets, filtered_rels)
            
            # 推理立场
            detected_stances = stance_detector.detect(graph, targets, filtered_rels, text)
            predicted_stances_list = [s['polarity'] for s in detected_stances]

            # 评估分数
            eval_res = evaluate_stances(gold_targets, gold_stances, predicted_target_names, predicted_stances_list)
            results_accumulator[group_name]['correct'] += eval_res['correct']
            results_accumulator[group_name]['total'] += eval_res['total']

    # 聚合指标计算 Accuracy
    print("\n" + "="*80)
    print("消融实验结果总结")
    print("="*80)
    
    final_metrics = []
    for group, stats in results_accumulator.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"[{group:^15}] Accuracy: {acc:.4f} (Correct: {stats['correct']}/{stats['total']})")
        final_metrics.append({'Ablation_Group': group, 'Accuracy': acc, 'Matched_Targets': stats['total'], 'Correct_Stances': stats['correct']})

    # 保存
    res_df = pd.DataFrame(final_metrics)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    res_df.to_excel(output_path, index=False)
    
    # 保存诊断明细表
    details_path = output_path.replace('.xlsx', '_明细诊断.xlsx')
    pd.DataFrame(detailed_logs).to_excel(details_path, index=False)

    json_path = output_path.replace('.xlsx', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print(f"\n结果已分别保存为：\n - {output_path} (宏观汇总)\n - {details_path} (重点看这个！排查失败原因)\n - {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='立场检测消融实验 - 细粒度关系分析')
    parser.add_argument('--data_file', type=str, default='data/test-dataset-1k.xlsx', help='数据文件路径')
    parser.add_argument('--output', type=str, default='results/ablation_fine_grained_v6_1000.xlsx', help='输出结果路径')
    parser.add_argument('--sample_size', type=int, default=50, help='用于快速跑通的测试样本量')
    parser.add_argument('--model_path', type=str, default='/data/models/Qwen2.5-7B-Instruct', help='本地模型路径')

    args = parser.parse_args()
    
    run_ablation_experiment(
        data_path=args.data_file,
        output_path=args.output,
        sample_size=args.sample_size,
        model_path=args.model_path
    )