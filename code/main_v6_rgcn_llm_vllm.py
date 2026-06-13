#!/usr/bin/env python3
"""
多目标立场检测完整 Pipeline - v6 vLLM 加速版 (R-GCN + vLLM)

v6-vLLM 核心改进:
1. 使用 vLLM 框架加速本地 Qwen2.5-7B-Instruct 模型推理
2. 支持张量并行（多卡并行推理）
3. 支持 Continuous Batching，大幅提升吞吐量
4. 保持 v5 完整的三阶段 R-GCN + LLM 架构

模型路径：/home/立场检测/model/Qwen2.5-7B-Instruct

使用方法:
    # 单进程版（直接运行）
    python main_v6_rgcn_llm_vllm.py --tensor_parallel_size 4

    # 或者先启动 vLLM 服务，再运行（推荐，可复用服务）
    python -m vllm.entrypoints.api_server \\
        --model /home/songxuexian/立场检测/model/Qwen2.5-7B-Instruct \\
        --tensor-parallel-size 4 \\
        --port 8000 \\
        --served-model-name qwen2.5-7b

    python main_v6_rgcn_llm_vllm.py --use_vllm_service
"""

import os
import sys
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
import re
import time
import requests
from typing import List, Dict, Any, Tuple
from datetime import datetime
from tqdm import tqdm
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 尝试导入 torch_geometric
try:
    from torch_geometric.nn import RGCNConv
    from torch_geometric.utils import from_networkx
    PYG_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"警告：torch_geometric 不可用 ({e})，将使用简化版 R-GCN")
    PYG_AVAILABLE = False

# 尝试导入 transformers
try:
    from transformers import BertModel, BertTokenizer
    TRANSFORMERS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"警告：transformers 不可用 ({e})")
    TRANSFORMERS_AVAILABLE = False

# 尝试导入 vLLM
VLLM_AVAILABLE = False
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"提示：vLLM 未安装 ({e})，将使用 API 服务模式")
    print(f"安装命令：pip install vllm")


# ============ vLLM 配置 ============

LOCAL_MODEL_PATH = "/home/立场检测/model/Qwen2.5-7B-Instruct"
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

# 全局变量
vllm_llm = None
vllm_sampling_params = None
use_vllm_service_mode = False


def init_vllm_local(model_path: str, tensor_parallel_size: int = 4):
    """初始化本地 vLLM 模型"""
    global vllm_llm, vllm_sampling_params

    if vllm_llm is not None:
        return

    print(f"  初始化 vLLM 模型：{model_path}")
    print(f"  张量并行数：{tensor_parallel_size}")

    vllm_llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        enforce_eager=False,
        enable_cuda_graph=True,
    )

    vllm_sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=2048,
        top_p=0.9,
    )

    print(f"  vLLM 模型初始化完成")


def call_llm_vllm(system_prompt: str, user_prompt: str, temperature: float = 0.1,
                  max_tokens: int = 2048, use_service: bool = False) -> str:
    """调用 vLLM 模型（本地或服务模式）"""
    global vllm_llm, vllm_sampling_params

    if use_service:
        # API 服务模式
        return call_llm_vllm_service(system_prompt, user_prompt, temperature, max_tokens)
    else:
        # 本地模式
        return call_llm_vllm_local(system_prompt, user_prompt, temperature, max_tokens)


def call_llm_vllm_service(system_prompt: str, user_prompt: str, temperature: float = 0.1,
                          max_tokens: int = 2048) -> str:
    """调用 vLLM API 服务"""
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "qwen2.5-7b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(VLLM_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  vLLM 服务调用失败：{e}")
        return ""


def call_llm_vllm_local(system_prompt: str, user_prompt: str, temperature: float = 0.1,
                        max_tokens: int = 2048) -> str:
    """调用本地 vLLM 模型"""
    global vllm_llm, vllm_sampling_params

    if vllm_llm is None:
        raise RuntimeError("vLLM 模型未初始化")

    try:
        # 构建对话
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 使用 tokenizer 构建 prompt
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 生成
        outputs = vllm_llm.generate([prompt], vllm_sampling_params)

        if outputs and len(outputs) > 0:
            return outputs[0].outputs[0].text
        return ""

    except Exception as e:
        print(f"  vLLM 本地调用失败：{e}")
        return ""


def call_llm_batch_vllm(prompts: List[Tuple[str, str]], temperature: float = 0.1,
                        max_tokens: int = 2048, use_service: bool = False) -> List[str]:
    """批量调用 vLLM（大幅提升吞吐量）"""
    if use_service:
        return call_llm_batch_vllm_service(prompts, temperature, max_tokens)
    else:
        return call_llm_batch_vllm_local(prompts, temperature, max_tokens)


def call_llm_batch_vllm_local(prompts: List[Tuple[str, str]], temperature: float = 0.1,
                              max_tokens: int = 2048) -> List[str]:
    """批量调用本地 vLLM 模型"""
    global vllm_llm, vllm_sampling_params

    if vllm_llm is None:
        raise RuntimeError("vLLM 模型未初始化")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)

        # 构建所有 prompts
        all_prompts = []
        for system_prompt, user_prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompts.append(prompt)

        # 批量生成（vLLM 的 Continuous Batching 会自动优化）
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
        )

        outputs = vllm_llm.generate(all_prompts, sampling_params)

        results = [output.outputs[0].text for output in outputs]
        return results

    except Exception as e:
        print(f"  vLLM 批量调用失败：{e}")
        return [""] * len(prompts)


def call_llm_batch_vllm_service(prompts: List[Tuple[str, str]], temperature: float = 0.1,
                                max_tokens: int = 2048) -> List[str]:
    """批量调用 vLLM API 服务（并发请求）"""
    results = []

    def call_single(prompt_pair):
        system_prompt, user_prompt = prompt_pair
        return call_llm_vllm_service(system_prompt, user_prompt, temperature, max_tokens)

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(call_single, p) for p in prompts]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append("")

    return results


# ============ Stage 1: 目标抽取（v4 权威版） ============

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
输出：{"targets": ["Lisa", "2024 MTV VMAs"]}
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


def extract_targets_batch(texts: List[str], use_vllm_service: bool = False) -> List[List[Dict[str, Any]]]:
    """批量抽取目标"""
    prompts = []
    for text in texts:
        user_prompt = f"""文本：{text}

请抽取所有立场目标："""
        prompts.append((TARGET_EXTRACTION_PROMPT, user_prompt))

    # 批量调用 vLLM
    responses = call_llm_batch_vllm(prompts, temperature=0.1, max_tokens=1024, use_service=use_vllm_service)

    all_targets = []
    for response in responses:
        targets = parse_targets_json(response)
        all_targets.append(targets)

    return all_targets


def parse_targets_json(response: str) -> List[Dict[str, Any]]:
    """解析目标抽取的 JSON 响应"""
    if not response:
        return []

    try:
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, response)
        if match:
            data = json.loads(match.group(0))
            targets = data.get('targets', [])
            return [
                {"id": f"T{i+1}", "text": t.strip(), "normalized": t.strip(),
                 "type": "explicit", "granularity_level": 2}
                for i, t in enumerate(targets) if t and isinstance(t, str)
            ]
    except Exception as e:
        pass

    return []


def extract_targets(text: str, use_vllm_service: bool = False) -> List[Dict[str, Any]]:
    """使用 vLLM 进行目标抽取（v4 权威版）"""
    result = extract_targets_batch([text], use_vllm_service)
    return result[0] if result else []


# ============ Stage 2: 关系抽取 ============

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


def extract_relation_batch(pairs: List[Tuple[str, str, str]], use_vllm_service: bool = False) -> List[Dict[str, Any]]:
    """批量抽取关系"""
    prompts = []
    for target_a, target_b, context in pairs:
        user_prompt = f"""目标 A: {target_a}
目标 B: {target_b}

原文：
{context}

请判断关系（只返回 JSON，不要其他内容）："""
        prompts.append((RELATION_EXTRACTION_PROMPT, user_prompt))

    responses = call_llm_batch_vllm(prompts, temperature=0.1, max_tokens=512, use_service=use_vllm_service)

    all_relations = []
    for response in responses:
        relation = parse_relation_json(response)
        all_relations.append(relation)

    return all_relations


def parse_relation_json(response: str) -> Dict[str, Any]:
    """解析关系抽取的 JSON 响应"""
    if not response:
        return {"relation": "none", "direction": "none", "evidence": "", "confidence": 1.0, "reasoning": "响应为空"}

    try:
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, response)
        if match:
            data = json.loads(match.group(0))
            return {
                "relation": data.get("relation", "none"),
                "direction": data.get("direction", "none"),
                "evidence": data.get("evidence", ""),
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": data.get("reasoning", "")
            }
    except Exception as e:
        pass

    return {"relation": "none", "direction": "none", "evidence": "", "confidence": 1.0, "reasoning": "解析失败"}


def extract_relation(target_a: str, target_b: str, context: str, use_vllm_service: bool = False) -> Dict[str, Any]:
    """抽取两个目标之间的关系"""
    result = extract_relation_batch([(target_a, target_b, context)], use_vllm_service)
    return result[0] if result else {"relation": "none", "direction": "none", "evidence": "", "confidence": 1.0, "reasoning": "失败"}


def extract_all_relations(targets: List[Dict[str, Any]], context: str, use_vllm_service: bool = False) -> List[Dict[str, Any]]:
    """抽取所有目标对之间的关系（批量）"""
    relations = []
    n = len(targets)

    # 限制关系抽取数量
    max_pairs = 10

    pairs = []
    pair_info = []

    for i in range(n):
        for j in range(i + 1, n):
            if len(pairs) >= max_pairs:
                break
            pairs.append((targets[i]["text"], targets[j]["text"], context))
            pair_info.append((i, j))

        if len(pairs) >= max_pairs:
            break

    if not pairs:
        return []

    # 批量抽取
    batch_results = extract_relation_batch(pairs, use_vllm_service)

    # 添加目标 ID
    for (i, j), relation in zip(pair_info, batch_results):
        relation["target_a"] = targets[i]["id"]
        relation["target_b"] = targets[j]["id"]

        if relation["relation"] != "none" and relation["confidence"] >= 0.5:
            relations.append(relation)

    return relations


# ============ 图构建 ============

class HeterogeneousGraphBuilder:
    """异质图构建器"""

    RELATION_TYPES = {
        'contrast': 0,
        'consistent': 1,
        'causal': 2,
        'hierarchical': 3,
        'parallel': 4,
        'analogy': 5,
        'none': -1,
    }

    def build(self, targets: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> nx.MultiDiGraph:
        """构建异质图"""
        G = nx.MultiDiGraph()

        # 添加节点
        for target in targets:
            G.add_node(
                target['id'],
                text=target['text'],
                normalized=target['normalized'],
                granularity_level=target.get('granularity_level', 2),
                type=target['type']
            )

        # 添加边
        for relation in relations:
            if relation['relation'] == 'none':
                continue

            src = relation['target_a']
            tgt = relation['target_b']
            rel_type = relation['relation']

            if relation['direction'] == 'A→B':
                G.add_edge(src, tgt, relation_type=rel_type,
                          relation_id=self.RELATION_TYPES.get(rel_type, -1),
                          evidence=relation['evidence'], confidence=relation['confidence'])
            elif relation['direction'] == 'B→A':
                G.add_edge(tgt, src, relation_type=rel_type,
                          relation_id=self.RELATION_TYPES.get(rel_type, -1),
                          evidence=relation['evidence'], confidence=relation['confidence'])
            else:
                G.add_edge(src, tgt, relation_type=rel_type,
                          relation_id=self.RELATION_TYPES.get(rel_type, -1),
                          evidence=relation['evidence'], confidence=relation['confidence'])
                G.add_edge(tgt, src, relation_type=rel_type,
                          relation_id=self.RELATION_TYPES.get(rel_type, -1),
                          evidence=relation['evidence'], confidence=relation['confidence'])

        return G


# ============ R-GCN 模型 ============

class TargetRGCN(nn.Module):
    """用于目标立场检测的 R-GCN 编码器"""

    def __init__(self, num_node_features: int, hidden_dim: int,
                 num_relations: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        self.convs.append(RGCNConv(num_node_features, hidden_dim, num_relations))
        self.layer_norms.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_type)
            x = self.layer_norms[i](x)

            if i < self.num_layers - 1:
                x = self.relu(x)
                x = self.dropout(x)

        return x


# ============ Stage 3: 立场检测（R-GCN + LLM） ============

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
输出：{"stances": [{"target": "杨子", "polarity": "反对"}]}

## 示例 2
文本："连续调整的阿里巴巴终于开始涨了"
目标：阿里巴巴
输出：{"stances": [{"target": "阿里巴巴", "polarity": "支持"}]}

## 示例 3
文本："具体的产品名、添加了哪种禁用原料、有何种危害"
目标：涉事产品名称
输出：{"stances": [{"target": "涉事产品名称", "polarity": "中立"}]}
分析：这是消费者关心的问题列表，没有表达明确态度。

# 输出规范
1. 严格使用 JSON 格式：{"stances": [{"target_id": "目标 ID", "polarity": "支持/反对/中立"}]}
2. 只返回 JSON，不要解释
3. polarity 必须是"支持"、"反对"或"中立"之一

现在，请判断以下文本对给定目标的立场：

【目标列表】
{target_list}

【目标关系】
{relation_list}

【R-GCN 计算的目标重要性】
{rgcn_importance}

【原始文本】
{text}"""


class RGCNLLMStanceDetector:
    """R-GCN + LLM 联合立场检测器（vLLM 加速版）"""

    def __init__(self, bert_model=None, rgcn_model=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.bert_model = bert_model
        self.rgcn_model = rgcn_model
        self.device = device
        self.tokenizer = None

        if bert_model is not None and TRANSFORMERS_AVAILABLE:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.bert_model.to(device)
            self.bert_model.eval()

        if rgcn_model is not None and PYG_AVAILABLE:
            self.rgcn_model.to(device)
            self.rgcn_model.eval()

    def encode_targets(self, targets: List[Dict[str, Any]], context: str) -> torch.Tensor:
        """使用 BERT 编码目标"""
        if self.bert_model is None or self.tokenizer is None:
            return torch.randn(len(targets), 768)

        embeddings = []
        for target in targets:
            text = f"{target['normalized']}。{context[:200]}"

            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = self.bert_model(**inputs)

            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding)

        return torch.cat(embeddings, dim=0)

    def run_rgcn(self, graph: nx.MultiDiGraph, node_features: torch.Tensor) -> torch.Tensor:
        """运行 R-GCN 编码"""
        if self.rgcn_model is None or not PYG_AVAILABLE:
            return node_features

        try:
            pyg_graph = from_networkx(graph)
            edge_index = pyg_graph.edge_index.to(self.device)

            if hasattr(pyg_graph, 'edge_type'):
                edge_type = pyg_graph.edge_type.to(self.device)
            else:
                edge_type = torch.zeros_like(edge_index[0])

            node_features = node_features.to(self.device)

            with torch.no_grad():
                rgcn_output = self.rgcn_model(node_features, edge_index, edge_type)

            return rgcn_output

        except Exception as e:
            print(f"    R-GCN 运行失败：{e}，使用原始特征")
            return node_features

    def compute_target_importance(self, graph: nx.MultiDiGraph, rgcn_output: torch.Tensor) -> Dict[str, float]:
        """计算目标重要性"""
        importance = {}
        node_ids = list(graph.nodes())

        if rgcn_output is not None:
            norms = torch.norm(rgcn_output, dim=1).cpu().numpy()
            for i, node_id in enumerate(node_ids):
                importance[node_id] = float(norms[i]) if i < len(norms) else 0.0
        else:
            centrality = nx.degree_centrality(graph)
            importance = centrality

        return importance

    def detect(self, graph: nx.MultiDiGraph, targets: List[Dict[str, Any]],
               relations: List[Dict[str, Any]], text: str,
               use_rgcn: bool = True, use_vllm_service: bool = False) -> List[Dict[str, Any]]:
        """检测立场"""
        if not targets:
            return []

        node_features = self.encode_targets(targets, text)

        if use_rgcn and self.rgcn_model is not None and graph.number_of_edges() > 0:
            rgcn_output = self.run_rgcn(graph, node_features)
        else:
            rgcn_output = node_features

        importance = self.compute_target_importance(graph, rgcn_output)

        sorted_targets = sorted(targets, key=lambda t: importance.get(t['id'], 0), reverse=True)

        target_list = "\n".join([
            f"- {t['id']}: {t['normalized']} "
            f"(重要性:{importance.get(t['id'], 0):.3f}, 层级:{t.get('granularity_level', 2)}, {t['type']})"
            for t in sorted_targets
        ])

        if relations:
            seen = set()
            relation_lines = []
            for r in relations:
                key = (min(r['target_a'], r['target_b']), max(r['target_a'], r['target_b']))
                if key in seen:
                    continue
                seen.add(key)
                relation_lines.append(
                    f"- {r['target_a']} ↔ {r['target_b']}: {r['relation']} "
                    f"(置信度:{r['confidence']:.2f})"
                )
            relation_list = "\n".join(relation_lines)
        else:
            relation_list = "无明显关系"

        rgcn_importance = "\n".join([
            f"- {t['id']}: {importance.get(t['id'], 0):.3f}"
            for t in sorted_targets
        ])

        user_prompt = STANCE_DETECTION_PROMPT.format(
            target_list=target_list,
            relation_list=relation_list,
            rgcn_importance=rgcn_importance,
            text=text
        )

        response = call_llm_vllm(STANCE_DETECTION_PROMPT, user_prompt, temperature=0.1,
                                 max_tokens=1024, use_service=use_vllm_service)

        if not response:
            return []

        try:
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, response)
            if match:
                data = json.loads(match.group(0))
                stances = data.get('stances', [])
                return [
                    {"target_id": s.get('target_id', ''), "polarity": s.get('polarity', '中立')}
                    for s in stances if s.get('target_id')
                ]
        except Exception as e:
            print(f"  立场检测解析失败：{e}")
            pass

        return []


# ============ 评估函数 ============

def preprocess_targets(target_str):
    if not target_str or (isinstance(target_str, float) and np.isnan(target_str)):
        return []
    if isinstance(target_str, list):
        return [str(t).strip() for t in target_str if str(t).strip()]
    targets = str(target_str).replace(';', ';').split(';')
    return [t.strip() for t in targets if t.strip()]


def preprocess_stances(stance_str):
    if not stance_str or (isinstance(stance_str, float) and np.isnan(stance_str)):
        return []
    if isinstance(stance_str, list):
        return [str(s).strip() for s in stance_str if str(s).strip()]
    stances = str(stance_str).replace(';', ';').split(';')
    return [s.strip() for s in stances if s.strip()]


def get_similarity_matrix(refs, cands):
    sim_matrix = np.zeros((len(refs), len(cands)))
    for i, ref in enumerate(refs):
        for j, cand in enumerate(cands):
            sim_matrix[i][j] = SequenceMatcher(None, ref, cand).ratio()
    return sim_matrix


def evaluate_targets(gold_targets, pred_targets, threshold=0.5):
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

def process_test_dataset(input_path: str, output_path: str, eval_output_path: str,
                         use_rgcn: bool = True, use_vllm_service: bool = False,
                         tensor_parallel_size: int = 4, batch_size: int = 1):
    """处理测试集并评估（vLLM 加速版）"""
    start_time = time.time()
    print("="*80)
    print("测试集处理和评估（vLLM 加速版 v6）")
    print(f"模型路径：{LOCAL_MODEL_PATH}")
    print(f"张量并行数：{tensor_parallel_size}")
    print(f"是否使用 vLLM 服务：{use_vllm_service}")
    print("="*80)

    # 初始化 vLLM
    if not use_vllm_service:
        print("\n初始化 vLLM 模型（本地模式）...")
        try:
            init_vllm_local(LOCAL_MODEL_PATH, tensor_parallel_size)
            print("  vLLM 初始化成功")
        except Exception as e:
            print(f"  vLLM 初始化失败：{e}")
            print("  请确认已安装 vLLM: pip install vllm")
            return
    else:
        print("\n使用 vLLM API 服务模式")
        try:
            test_response = call_llm_vllm("你是一个助手。", "你好", use_service=True)
            if test_response:
                print("  vLLM 服务连接成功")
            else:
                print("  vLLM 服务响应为空，请确认服务已启动")
                return
        except Exception as e:
            print(f"  vLLM 服务连接失败：{e}")
            print("  启动服务命令：python -m vllm.entrypoints.api_server --model " + LOCAL_MODEL_PATH +
                  " --tensor-parallel-size " + str(tensor_parallel_size) + " --port 8000")
            return

    # 初始化 BERT 和 R-GCN
    print("\n初始化 BERT 和 R-GCN 模型...")
    bert_model = None
    rgcn_model = None

    if use_rgcn and TRANSFORMERS_AVAILABLE and PYG_AVAILABLE:
        bert_model = BertModel.from_pretrained('bert-base-chinese')
        rgcn_model = TargetRGCN(
            num_node_features=768,
            hidden_dim=256,
            num_relations=6,
            num_layers=2,
            dropout=0.3
        )
        print("  模型加载完成")
    else:
        if not TRANSFORMERS_AVAILABLE:
            print("  ⚠️ transformers 不可用")
        if not PYG_AVAILABLE:
            print("  ⚠️ torch_geometric 不可用")
        use_rgcn = False

    stance_detector = RGCNLLMStanceDetector(
        bert_model=bert_model,
        rgcn_model=rgcn_model
    )

    # 加载测试集
    print(f"\n加载测试集：{input_path}")
    df = pd.read_excel(input_path)
    print(f"  样本数：{len(df)}")

    predictions = []
    total_graph_nodes = 0
    total_graph_edges = 0

    print(f"\n开始处理（vLLM 加速，batch_size={batch_size}）...")

    # 批量处理
    sample_ids = list(df.index)

    for batch_start in tqdm(range(0, len(sample_ids), batch_size), desc="处理批次"):
        batch_end = min(batch_start + batch_size, len(sample_ids))
        batch_ids = sample_ids[batch_start:batch_end]

        # Stage 1: 批量目标抽取
        batch_texts = [df.loc[idx, 'blog_text'] for idx in batch_ids]
        batch_targets_list = extract_targets_batch(batch_texts, use_vllm_service)

        for i, idx in enumerate(batch_ids):
            text = df.loc[idx, 'blog_text']
            gold_targets = df.loc[idx, '真实目标']
            gold_stances = df.loc[idx, '真实立场']

            extracted_targets = batch_targets_list[i]

            # Stage 2: 关系抽取 + 图构建
            if len(extracted_targets) >= 2:
                relations = extract_all_relations(extracted_targets, text, use_vllm_service)
            else:
                relations = []

            graph_builder = HeterogeneousGraphBuilder()
            graph = graph_builder.build(extracted_targets, relations)
            total_graph_nodes += graph.number_of_nodes()
            total_graph_edges += graph.number_of_edges()

            # Stage 3: 立场检测
            detected_stances = stance_detector.detect(
                graph=graph,
                targets=extracted_targets,
                relations=relations,
                text=text,
                use_rgcn=use_rgcn,
                use_vllm_service=use_vllm_service
            )

            pred = {
                'sample_id': idx,
                'blog_text': text,
                'gold_targets': gold_targets,
                'gold_stances': gold_stances,
                'predicted_targets': [t['text'] for t in extracted_targets],
                'predicted_stances': [s['polarity'] for s in detected_stances],
                'target_details': extracted_targets,
                'relations': relations,
                'stance_details': detected_stances,
                'graph_stats': {
                    'num_nodes': graph.number_of_nodes(),
                    'num_edges': graph.number_of_edges()
                }
            }
            predictions.append(pred)

        if len(predictions) % 100 == 0:
            elapsed = time.time() - start_time
            rate = len(predictions) / elapsed if elapsed > 0 else 0
            print(f"  已处理 {len(predictions)}/{len(df)} 个样本，速度：{rate:.2f} 样本/秒")

    # 保存结果
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    excel_data = []
    for p in predictions:
        excel_data.append({
            'sample_id': p['sample_id'],
            'blog_text': p['blog_text'][:200],
            'gold_targets': p['gold_targets'],
            'gold_stances': p['gold_stances'],
            'predicted_targets': '; '.join(p['predicted_targets']),
            'predicted_stances': '; '.join(p['predicted_stances']),
            'num_targets': len(p['predicted_targets']),
            'num_relations': len(p['relations'])
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

    total_time = time.time() - start_time

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

    print(f"\n【性能统计】")
    print(f"  总耗时：{total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"  处理速度：{len(predictions)/total_time:.2f} 样本/秒")
    print(f"  平均节点数：{total_graph_nodes/len(predictions):.2f}")
    print(f"  平均边数：{total_graph_edges/len(predictions):.2f}")

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
            '本地模型': LOCAL_MODEL_PATH,
            '版本': 'v6-RGCN-vLLM',
            '张量并行数': tensor_parallel_size,
            '总耗时 (秒)': total_time,
            '处理速度': len(predictions)/total_time,
            '使用 R-GCN': use_rgcn,
            '使用 vLLM 服务': use_vllm_service,
        }
        pd.DataFrame([metadata]).to_excel(writer, sheet_name='元数据', index=False)

    print(f"\n评估结果已保存到：{eval_output_path}")

    return {
        'target_metrics': {'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1},
        'stance_metrics': {'accuracy': overall_stance_acc, 'correct': stance_correct, 'total': stance_total},
        'combined': combined,
        'total_time': total_time,
        'samples_per_second': len(predictions)/total_time
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多目标立场检测 Pipeline - v6 vLLM 加速版')
    parser.add_argument('--data_path', type=str, default='data/test-dataset-1k.xlsx',
                        help='数据文件路径')
    parser.add_argument('--output_path', type=str, default='results/test_predictions_real_v6_vllm.json',
                        help='输出 JSON 路径')
    parser.add_argument('--eval_output_path', type=str, default='results/test_evaluation_real_v6_vllm.xlsx',
                        help='评估输出路径')
    parser.add_argument('--no_rgcn', action='store_true',
                        help='不使用 R-GCN（纯 LLM 方案）')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                        help='张量并行数（使用的 GPU 数量）')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批量处理大小（vLLM 会自动优化）')
    parser.add_argument('--use_vllm_service', action='store_true',
                        help='使用 vLLM API 服务模式（需先启动服务）')

    args = parser.parse_args()

    metrics = process_test_dataset(
        input_path=args.data_path,
        output_path=args.output_path,
        eval_output_path=args.eval_output_path,
        use_rgcn=not args.no_rgcn,
        use_vllm_service=args.use_vllm_service,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size
    )

    print("\n" + "="*80)
    print("最终摘要")
    print("="*80)
    if metrics:
        print(f"目标识别率 (Recall): {metrics['target_metrics']['recall']:.4f}")
        print(f"目标抽取 F1:        {metrics['target_metrics']['f1']:.4f}")
        print(f"立场分类准确率：{metrics['stance_metrics']['accuracy']:.4f}")
        print(f"综合得分：{metrics['combined']:.4f}")
        print(f"总耗时：{metrics['total_time']:.1f} 秒")
        print(f"处理速度：{metrics['samples_per_second']:.2f} 样本/秒")
