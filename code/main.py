#!/usr/bin/env python3
"""
多目标立场检测完整 Pipeline

使用方法:
    python main.py [--data_path PATH] [--use_rgcn] [--api_key KEY]

流程:
    Stage 1: 目标抽取 (LLM + CoT + 粒度控制)
    Stage 2: 关系抽取 + 异质图构建
    Stage 3: R-GCN + LLM 立场检测
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    DATA_CONFIG, RELATION_TYPES, RELATION_TYPE_DEFINITIONS,
    TARGET_EXTRACTION_CONFIG, RGCN_CONFIG, STANCE_LABELS
)
from utils.data_processor import DataProcessor
from utils.llm_client import create_llm_client, DashScopeClient
from src.stage1_target_extraction import TargetExtractor
from src.stage2_relation_extraction import RelationExtractor, HeterogeneousGraphBuilder
from src.stage3_stance_detection import (
    RGCNLLMStanceDetector, GraphEnhancedStanceDetector,
    TargetRGCN
)


class MultiTargetStancePipeline:
    """
    多目标立场检测完整流程
    """

    def __init__(self, api_key: str = None, use_rgcn: bool = True,
                 data_path: str = None):
        """
        Args:
            api_key: DashScope API Key
            use_rgcn: 是否使用 R-GCN
            data_path: 数据文件路径
        """
        self.api_key = api_key
        self.use_rgcn = use_rgcn
        self.data_path = data_path or DATA_CONFIG['data_path']

        # 初始化组件
        print("初始化组件...")
        self.llm_client = self._init_llm()
        self.target_extractor = TargetExtractor(
            llm_client=self.llm_client,
            dedup_threshold=TARGET_EXTRACTION_CONFIG['dedup_threshold']
        )
        self.relation_extractor = RelationExtractor(
            llm_client=self.llm_client,
            confidence_threshold=0.5
        )
        self.graph_builder = HeterogeneousGraphBuilder()

        # 初始化 BERT 和 R-GCN
        self.bert_model = None
        self.rgcn_model = None
        self.stance_detector = None

        if use_rgcn:
            self._init_rgcn_models()

        # 结果存储
        self.results = []

    def _init_llm(self):
        """初始化 LLM 客户端"""
        return DashScopeClient(api_key=self.api_key, model="qwen-plus")

    def _init_rgcn_models(self):
        """初始化 R-GCN 模型"""
        try:
            import torch
            from transformers import BertModel, BertTokenizer

            print("加载 BERT 模型...")
            self.bert_model = BertModel.from_pretrained(
                RGCN_CONFIG['bert_model']
            )

            print("初始化 R-GCN 模型...")
            self.rgcn_model = TargetRGCN(
                num_node_features=768,
                hidden_dim=RGCN_CONFIG['hidden_dim'],
                num_relations=len(RELATION_TYPES),
                num_layers=RGCN_CONFIG['num_layers'],
                dropout=RGCN_CONFIG['dropout']
            )

            # 创建立场检测器
            self.stance_detector = RGCNLLMStanceDetector(
                bert_model=self.bert_model,
                rgcn_model=self.rgcn_model,
                llm_client=self.llm_client
            )

            print("R-GCN 模型初始化完成")

        except Exception as e:
            print(f"R-GCN 模型初始化失败：{e}")
            print("将使用纯 LLM 方案")
            self.use_rgcn = False

    def process_sample(self, text: str, gold_targets: List[str] = None) -> Dict[str, Any]:
        """
        处理单个样本

        Args:
            text: 输入文本
            gold_targets: 金标目标（用于评估，可选）

        Returns:
            处理结果
        """
        result = {
            'text': text[:200] + '...' if len(text) > 200 else text,
            'gold_targets': gold_targets or [],
        }

        # Stage 1: 目标抽取
        print("  Stage 1: 目标抽取...")
        try:
            targets = self.target_extractor.process(text)
            result['extracted_targets'] = targets
            print(f"    抽取到 {len(targets)} 个目标")
        except Exception as e:
            print(f"    目标抽取失败：{e}")
            result['extracted_targets'] = []
            result['stage1_error'] = str(e)
            return result

        if len(targets) == 0:
            print("    无目标，跳过后续处理")
            return result

        # Stage 2: 关系抽取 + 图构建
        print("  Stage 2: 关系抽取...")
        try:
            relations = self.relation_extractor.extract_all(targets, text)
            result['relations'] = relations
            print(f"    抽取到 {len(relations)} 条关系")

            # 构建图
            graph = self.graph_builder.build(targets, relations)
            result['graph_stats'] = {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges()
            }
            print(f"    图：{graph.number_of_nodes()} 节点，{graph.number_of_edges()} 边")

        except Exception as e:
            print(f"    关系抽取失败：{e}")
            result['relations'] = []
            result['stage2_error'] = str(e)
            graph = None

        # Stage 3: 立场检测
        print("  Stage 3: 立场检测...")
        try:
            if self.use_rgcn and graph is not None and self.stance_detector is not None:
                stances = self.stance_detector.detect(
                    graph=graph,
                    targets=targets,
                    relations=relations,
                    text=text,
                    use_rgcn=True
                )
            else:
                # 纯 LLM 方案
                stances = self._llm_only_stance_detection(targets, relations, text)

            result['stances'] = stances
            print(f"    检测到 {len(stances)} 个立场")

        except Exception as e:
            print(f"    立场检测失败：{e}")
            result['stances'] = []
            result['stage3_error'] = str(e)

        return result

    def _llm_only_stance_detection(self, targets: List[Dict],
                                    relations: List[Dict],
                                    text: str) -> List[Dict]:
        """纯 LLM 立场检测（不使用 R-GCN）"""
        # 构建简化的 Prompt
        target_list = "\n".join([
            f"- {t['id']}: {t['normalized']}" for t in targets
        ])

        relation_list = "\n".join([
            f"- {r['target_a']} → {r['target_b']}: {r['relation']}"
            for r in relations[:10]  # 限制关系数量
        ])

        prompt = f"""你是一个立场检测专家。

【目标列表】
{target_list}

【目标关系】
{relation_list if relation_list else "无明显关系"}

【原始文本】
{text}

请判断每个目标的立场：
1. 立场极性：support（支持）/ neutral（中立）/ oppose（反对）
2. 立场强度：strong（强烈）/ moderate（中等）/ weak（微弱）
3. 证据：原文片段

输出格式（JSON）：
{{
  "stances": [
    {{
      "target_id": "T1",
      "polarity": "support",
      "intensity": "moderate",
      "evidence": "原文证据",
      "reasoning": "推理过程"
    }}
  ]
}}
"""

        response = self.llm_client.generate(
            system_prompt="你是一个立场检测专家。请严格按照 JSON 格式输出。",
            user_prompt=prompt
        )

        # 解析响应
        return self._parse_stance_response(response)

    def _parse_stance_response(self, response: str) -> List[Dict]:
        """解析立场响应"""
        import re

        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, response)

        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                return data.get('stances', [])
            except json.JSONDecodeError:
                print(f"JSON 解析失败：{json_str}")
                return []
        return []

    def run(self, limit: int = None, output_file: str = None) -> List[Dict]:
        """
        运行完整流程

        Args:
            limit: 限制处理样本数
            output_file: 输出文件路径

        Returns:
            结果列表
        """
        print(f"\n加载数据：{self.data_path}")
        processor = DataProcessor(self.data_path)
        processor.load()

        samples = processor.get_samples(limit=limit)
        print(f"共 {len(samples)} 个样本")

        if limit:
            print(f"限制处理前 {limit} 个样本")

        # 处理每个样本
        for i, sample in enumerate(samples):
            print(f"\n{'='*50}")
            print(f"处理样本 {i+1}/{len(samples)}")
            print(f"{'='*50}")

            result = self.process_sample(
                text=sample['text'],
                gold_targets=sample['targets']
            )

            # 添加金标立场
            result['gold_stances'] = sample['stances']
            result['sample_id'] = sample['id']

            self.results.append(result)

            # 保存中间结果
            if output_file and (i + 1) % 10 == 0:
                self.save_results(output_file)

        # 保存最终结果
        if output_file:
            self.save_results(output_file)

        return self.results

    def save_results(self, output_file: str):
        """保存结果"""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'use_rgcn': self.use_rgcn,
                'relation_types': RELATION_TYPES,
            },
            'results': self.results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到：{output_file}")

    def evaluate(self) -> Dict[str, float]:
        """
        评估结果（如果有金标）

        Returns:
            评估指标
        """
        # TODO: 实现评估逻辑
        print("评估功能待实现")
        return {}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='多目标立场检测 Pipeline'
    )
    parser.add_argument(
        '--data_path', type=str, default='../测试-data-最终版.xlsx',
        help='数据文件路径'
    )
    parser.add_argument(
        '--api_key', type=str, default=None,
        help='DashScope API Key'
    )
    parser.add_argument(
        '--use_rgcn', action='store_true', default=True,
        help='是否使用 R-GCN'
    )
    parser.add_argument(
        '--no_rgcn', action='store_true',
        help='不使用 R-GCN（纯 LLM 方案）'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='限制处理样本数'
    )
    parser.add_argument(
        '--output', type=str, default='results/predictions.json',
        help='输出文件路径'
    )

    args = parser.parse_args()

    # 检查 API Key
    api_key = args.api_key or os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        print("警告：未设置 DashScope API Key")
        print("请设置环境变量 DASHSCOPE_API_KEY 或使用 --api_key 参数")
        print("将使用 Mock 响应进行测试\n")

    # 创建 Pipeline
    pipeline = MultiTargetStancePipeline(
        api_key=api_key,
        use_rgcn=not args.no_rgcn,
        data_path=args.data_path
    )

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # 运行
    pipeline.run(limit=args.limit, output_file=args.output)

    print("\n" + "="*50)
    print("处理完成!")
    print(f"输出文件：{args.output}")


if __name__ == "__main__":
    main()