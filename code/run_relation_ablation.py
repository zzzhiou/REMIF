import json
import copy
import os
import subprocess
from datetime import datetime

# 假设你在阶段二定义的关系类型集合（根据论文表4-2调整中英文）
RELATION_TYPES = [
    "对立", "同向", "因果", "层级", "并列", "类比"
    # 如果你的 JSON 中是英文，请替换为: "contrast", "consistent", "causal", "hierarchical", "parallel", "analogy"
]

INPUT_STAGE2_DATA = "data/stage2_output.json"  # 阶段二生成的包含全量关系的数据路径
TEMP_FILTERED_DATA = "data/temp_ablation_input.json" # 传递给阶段三的临时文件
EVAL_RESULTS_FILE = "results/ablation_metrics_summary.json" # 最终指标汇总

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_data(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def filter_relations(data, exclude_relation_type):
    """
    核心逻辑：遍历所有样本，剔除掉指定的的关系类型，保留其他关系。
    如果 exclude_relation_type 为 None，则不剔除（作为 Full Model 基线）。
    """
    filtered_data = []
    for item in data:
        new_item = copy.deepcopy(item)
        if "relations" in new_item:
            # 仅保留类型不是 exclude_relation_type 的关系
            new_item["relations"] = [
                r for r in new_item["relations"] 
                if r.get("type", r.get("关系")) != exclude_relation_type
            ]
        filtered_data.append(new_item)
    return filtered_data

def run_stage3_and_evaluate(input_file, output_prefix):
    """
    调用你已有的 stage3 和 evaluate 脚本。
    这里使用 subprocess 调用以保证与你现有环境解耦。
    假设你的 stage3 脚本接受输入文件并输出预测结果。
    """
    pred_output_file = f"results/{output_prefix}_pred.json"
    
    # 1. 运行阶段三立场检测 (请根据 stage3_stance_detection.py 的实际传参修改)
    print(f"  -> 正在调用本地 Qwen2.5-7B-Instruct 进行推理...")
    cmd_stage3 = [
        "python", "stage3_stance_detection.py",
        "--input", input_file,
        "--output", pred_output_file,
        "--model", "qwen2.5-7b-instruct" # 指定本地模型
    ]
    subprocess.run(cmd_stage3, check=True)
    
    # 2. 运行评估脚本 (请根据 evaluate_results.py 的实际传参修改)
    print(f"  -> 正在评估指标...")
    eval_output_file = f"results/{output_prefix}_metrics.json"
    cmd_eval = [
        "python", "evaluate_results.py",
        "--pred", pred_output_file,
        "--output", eval_output_file
    ]
    subprocess.run(cmd_eval, check=True)
    
    # 返回评估结果
    return load_json_data(eval_output_file)

def main():
    print("="*50)
    print("开始细粒度关系消融实验 (Fine-grained Relation Ablation)")
    print("="*50)
    
    # 1. 加载全量关系数据
    if not os.path.exists(INPUT_STAGE2_DATA):
        print(f"错误: 找不到输入数据 {INPUT_STAGE2_DATA}。请先运行 stage2_relation_extraction.py")
        return
        
    full_data = load_json_data(INPUT_STAGE2_DATA)
    summary_results = {}

    # 2. 运行 Full Model (基线)
    print("\n[实验 0] 运行 Full Model (包含所有关系)")
    save_json_data(full_data, TEMP_FILTERED_DATA)
    metrics = run_stage3_and_evaluate(TEMP_FILTERED_DATA, "ablation_full_model")
    summary_results["Full_Model"] = metrics
    print(f"  -> Full Model F1 Score: {metrics.get('F1', 'N/A')}")

    # 3. 逐一剔除每种关系进行实验
    for rel_type in RELATION_TYPES:
        print(f"\n[实验] 剔除关系: w/o {rel_type}")
        
        # 过滤数据
        filtered_data = filter_relations(full_data, exclude_relation_type=rel_type)
        save_json_data(filtered_data, TEMP_FILTERED_DATA)
        
        # 推理与评估
        exp_name = f"wo_relation_{rel_type}"
        metrics = run_stage3_and_evaluate(TEMP_FILTERED_DATA, exp_name)
        
        summary_results[f"w/o_{rel_type}"] = metrics
        print(f"  -> w/o {rel_type} F1 Score: {metrics.get('F1', 'N/A')}")

    # 4. 汇总与保存最终结果
    print("\n" + "="*50)
    print("实验完成！各项F1指标汇总如下：")
    for exp, mets in summary_results.items():
        print(f" - {exp}: F1 = {mets.get('F1', 'N/A')}")
        
    save_json_data(summary_results, EVAL_RESULTS_FILE)
    print(f"\n详细指标已保存至: {EVAL_RESULTS_FILE}")

if __name__ == "__main__":
    main()