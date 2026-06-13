"""
多目标立场检测配置文件

6 种关系类型:
1. contrast: 对立关系（对 A 和 B 立场相反/冲突）
2. consistent: 一致关系（对 A 和 B 立场相同）
3. causal: 因果关系（A 导致/影响 B）
4. hierarchical: 层级关系（A 是 B 的上位/下位概念）
5. parallel: 并列关系（A 和 B 同级，无立场关联）
6. analogy: 类比关系（用 A 类比 B）
"""

# ============ 数据配置 ============
DATA_CONFIG = {
    'data_path': '../测试-data-最终版.xlsx',
    'text_column': 'blog_text',
    'target_column': 'target',
    'stance_column': 'stance',
    'test_size': 0.2,
    'random_seed': 42,
}

# ============ 关系类型配置 ============
RELATION_TYPES = {
    'contrast': 0,      # 对立关系
    'consistent': 1,    # 一致关系
    'causal': 2,        # 因果关系
    'hierarchical': 3,  # 层级关系
    'parallel': 4,      # 并列关系
    'analogy': 5,       # 类比关系
}

RELATION_TYPE_NAMES = {v: k for k, v in RELATION_TYPES.items()}

RELATION_TYPE_DEFINITIONS = {
    'contrast': '对立关系（对 A 和 B 立场相反/冲突，如"支持 A 但反对 B"）',
    'consistent': '一致关系（对 A 和 B 立场相同，如"都支持"或"都反对"）',
    'causal': '因果关系（A 导致/影响 B，如"A 会导致 B"）',
    'hierarchical': '层级关系（A 是 B 的上位/下位概念，如"A 包括 B"）',
    'parallel': '并列关系（A 和 B 同级并列，无明显立场关联）',
    'analogy': '类比关系（用 A 类比 B，如"A 就像 B"）',
}

# ============ Stage1 目标抽取配置 ============
TARGET_EXTRACTION_CONFIG = {
    # 粒度层级定义
    'granularity_levels': {
        1: '宏观政策/理念（太宏观，不抽取）',
        2: '具体政策/措施（合适）',
        3: '执行方式/细节（合适）',
        4: '具体案例/事件（太细，不抽取）',
    },
    # 只保留的层级
    'valid_levels': [2, 3],
    # 语义去重阈值
    'dedup_threshold': 0.85,
    # LLM 模型配置
    'llm_model': 'qwen-plus',
    'max_tokens': 2048,
    'temperature': 0.1,
}

# ============ Stage2 关系抽取配置 ============
RELATION_EXTRACTION_CONFIG = {
    # 关系抽取方式：'llm' 或 'rule'
    'method': 'llm',
    # 置信度阈值
    'confidence_threshold': 0.5,
    # LLM 模型配置
    'llm_model': 'qwen-plus',
    'max_tokens': 1024,
    'temperature': 0.1,
}

# ============ Stage3 R-GCN 配置 ============
RGCN_CONFIG = {
    # BERT 模型
    'bert_model': 'bert-base-chinese',
    # R-GCN 参数
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.3,
    # 训练参数
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'batch_size': 32,
    # 是否使用 R-GCN（False 则仅使用 LLM）
    'use_rgcn': True,
}

# ============ 立场标签配置 ============
STANCE_LABELS = {
    '支持': 0,
    '中立': 1,
    '反对': 2,
}

STANCE_LABEL_NAMES = {v: k for k, v in STANCE_LABELS.items()}

INTENSITY_LABELS = {
    'strong': 2,
    'moderate': 1,
    'weak': 0,
}

# ============ 输出配置 ============
OUTPUT_CONFIG = {
    'results_dir': 'results',
    'save_predictions': True,
    'save_graph': True,
    'verbose': True,
}
