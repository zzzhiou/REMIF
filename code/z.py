# ================ 强制修复编码：彻底解决ASCII报错 ================
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ================ 导入依赖 ================
import re
import json
import time
import pandas as pd
from openai import OpenAI

# ===================== 核心配置（仅需修改这里！） =====================
API_KEY = "sk-"  # 填入你的真实API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 华北2（北京）
MODEL_NAME = "qwen2.5-7b-instruct"
INPUT_EXCEL_PATH = "/data/stance_detection_project/stance_detection_project/data/test-dataset-1k.xlsx"
OUTPUT_JSON_PATH = "/data/stance_detection_project/stance_detection_project/results/test_predictions_real_z1.json"

# 初始化OpenAI客户端（你测试成功的方式！）
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 模型参数
MODEL_PARAMS = {
    "temperature": 0.1,
    "max_tokens": 512,
    "top_p": 0.7,
}

# 任务提示词（保持不变）
SYSTEM_PROMPT = """
你是一名目标识别与立场检测专家。根据给定的微博评论，识别评论中讨论的主要目标（如人物、事件或行为），并判断对每个目标的立场（支持/反对/中立）。如果评论包含多个目标，请分别评估每个目标的立场。输出格式: (目标: [目标1], 立场: [立场1]) (目标: [目标2], 立场: [立场2])
实例： input"input": "樊振东的那一拍真是太精彩了——他的反手太有力量了！队员们全都站起来为他鼓掌欢呼！"实例 - output"output": "(目标: 樊振东, 立场: 支持)"
"""

# ===================== 解析函数（完全保留，兼容你的评估脚本） =====================
def parse_model_output(text: str) -> list:
    pattern = r"\(目标: (.*?), 立场: (支持|反对|中立)\)"
    matches = re.findall(pattern, text.strip())
    return [{"target": t.strip(), "stance": s.strip()} for t, s in matches]

# ===================== API调用（重写为你测试成功的版本） =====================
def call_qwen_api(text: str, retry_times: int = 3) -> list:
    for attempt in range(retry_times):
        try:
            # 调用模型（和测试脚本完全一致！）
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                **MODEL_PARAMS
            )
            # 获取输出
            output_text = response.choices[0].message.content.strip()
            return parse_model_output(output_text)

        except Exception as e:
            print(f"Retry {attempt+1}, Error: {str(e)[:50]}")
            time.sleep(2)
    
    print(f"Failed: {text[:30]}...")
    return []

# ===================== 主程序（完全保留原有逻辑） =====================
if __name__ == "__main__":
    # 读取Excel
    df = pd.read_excel(INPUT_EXCEL_PATH)
    print(f"Loaded dataset: {len(df)} items")

    all_predictions = []
    for idx, row in df.iterrows():
        # 读取文本（列名blog_text保持不变）
        blog_text = str(row["blog_text"])
        print(f"Processing {idx+1}/{len(df)}")

        # 模型预测
        pred = call_qwen_api(blog_text)
        
        # 组装结果（和原格式完全一致）
        all_predictions.append({
            "id": idx,
            "input_text": blog_text,
            "predictions": pred
        })

        # 限流
        time.sleep(0.5)

    # 保存JSON文件
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Output saved to: {OUTPUT_JSON_PATH}")
    print("Ready for evaluate_comprehensive.py!")