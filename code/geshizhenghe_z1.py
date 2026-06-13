import pandas as pd
import json
import os
import re

def extract_all_targets_and_stances(preds_node):
    """
    专门处理多目标/多立场的提取函数
    将形如 [{"target":"A", "stance":"支持"}, {"target":"B", "stance":"反对"}]
    转换为 -> targets: "A;B", stances: "支持;反对"
    """
    # 1. 如果大模型输出的是带 Markdown 的字符串，先把它脱壳成 JSON 对象
    if isinstance(preds_node, str):
        try:
            clean_str = re.sub(r'```json\s*', '', preds_node, flags=re.IGNORECASE)
            clean_str = re.sub(r'```\s*', '', clean_str)
            preds_node = json.loads(clean_str)
        except:
            pass

    targets_list = []
    stances_list = []

    # 2. 如果是列表（包含多个目标和立场）—— 这是你截图里的真实情况！
    if isinstance(preds_node, list):
        for item in preds_node:
            if isinstance(item, dict):
                # 兼容多种可能的键名
                t = item.get('target', item.get('predicted_targets', item.get('评价对象', '')))
                s = item.get('stance', item.get('predicted_stances', item.get('立场极性', '')))
                
                # 只要目标或立场有一个不为空，就加进去，保持一一对应
                if t or s:
                    targets_list.append(str(t).strip())
                    stances_list.append(str(s).strip())

    # 3. 如果只是一个单独的字典（单目标情况备用）
    elif isinstance(preds_node, dict):
        t = preds_node.get('target', preds_node.get('predicted_targets', ''))
        s = preds_node.get('stance', preds_node.get('predicted_stances', ''))
        if t or s:
            targets_list.append(str(t).strip())
            stances_list.append(str(s).strip())

    # 4. 用分号拼接成评估脚本需要的格式
    final_targets = ";".join(targets_list)
    final_stances = ";".join(stances_list)
    
    return final_targets, final_stances


def format_predictions_multi(excel_path, pred_json_path, output_path):
    print("="*60)
    print("启动【多目标分号拼接版】数据整合...")
    print("="*60)

    try:
        df = pd.read_excel(excel_path)
        df.fillna('', inplace=True) 
        excel_records = df.to_dict('records')
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    try:
        with open(pred_json_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    except Exception as e:
        print(f"读取 JSON 失败: {e}")
        return

    formatted_results = []
    min_len = min(len(excel_records), len(predictions))
    
    for i in range(min_len):
        row = excel_records[i]
        json_item = predictions[i]
        
        # 1. 抓取 Excel 金标
        text_val = row.get('blog_text', '')
        gold_t = row.get('真实目标', '')
        gold_s = row.get('真实立场', '')
        
        # 2. 获取大模型的原始输出内容
        raw_preds_node = json_item.get('predictions', '')
        
        # 3. 提取并拼接所有的目标和立场
        pred_t, pred_s = extract_all_targets_and_stances(raw_preds_node)
        
        formatted_item = {
            "id": str(json_item.get('id', i + 1)),
            "text": str(text_val),
            "gold_targets": str(pred_t) if not gold_t and pred_t else str(gold_t), # 防止excel空白
            "gold_stances": str(pred_s) if not gold_s and pred_s else str(gold_s),
            "predicted_targets": str(pred_t),
            "predicted_stances": str(pred_s)
        }
        
        formatted_results.append(formatted_item)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=2)

    print("\n✅ 转换完成！抽查第 1 条结果（多目标测试）：")
    print(f"- Gold Targets: {formatted_results[0]['gold_targets']}")
    print(f"- Pred Targets: {formatted_results[0]['predicted_targets']}")
    print(f"- Pred Stances: {formatted_results[0]['predicted_stances']}")
    print("="*60)

if __name__ == "__main__":
    # 路径请根据你的实际情况核对
    excel_file = '/data/stance_detection_project/stance_detection_project/data/test-dataset-1k.xlsx'
    json_file = '/data/stance_detection_project/stance_detection_project/results/test_predictions_real_zcot_ds_526.json'
    output_file = '/data/stance_detection_project/stance_detection_project/results/test_predictions_real_zcot_ds_526_final.json'

    format_predictions_multi(excel_file, json_file, output_file)
