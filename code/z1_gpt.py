# ================ 强制修复编码：彻底解决ASCII报错 ================
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ================ 导入依赖 ================
import re
import json
import time
import logging
import pandas as pd
from openai import OpenAI

# ===================== 日志配置（后台运行核心：所有过程写入日志文件） =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("/data/stance_detection_project/stance_detection_project/rungpt.log", encoding='utf-8')  # 运行日志自动保存到 run.log
    ]
)
logger = logging.getLogger(__name__)

# ===================== 核心配置（仅修改API_KEY） =====================
API_KEY = "sk-Q9dwGdDCaV1BxtzH11A4967440E24f8"  # 填入你的真实API Key
BASE_URL = "https://api.vveai.com/v1"  
MODEL_NAME = "gpt-4o-2024-11-20"
INPUT_EXCEL_PATH = "/data/stance_detection_project/stance_detection_project/data/test-dataset-1k.xlsx"
OUTPUT_JSON_PATH = "/data/stance_detection_project/stance_detection_project/results/test_predictions_real_z1_gpt.json"

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 模型参数
MODEL_PARAMS = {
    "temperature": 0.1,
    "max_tokens": 512,
    "top_p": 0.7,
}

# 任务提示词
SYSTEM_PROMPT = """
你是一名目标识别与立场检测专家。根据给定的微博评论，识别评论中讨论的主要目标（如人物、事件或行为），并判断对每个目标的立场（支持/反对/中立）。如果评论包含多个目标，请分别评估每个目标的立场。输出格式: (目标: [目标1], 立场: [立场1]) (目标: [目标2], 立场: [立场2])
实例： input"input": "樊振东的那一拍真是太精彩了——他的反手太有力量了！队员们全都站起来为他鼓掌欢呼！"实例 - output"output": "(目标: 樊振东, 立场: 支持)"
"""

# ===================== 格式解析函数 =====================
def parse_model_output(text: str) -> list:
    pattern = r"\(目标: (.*?), 立场: (支持|反对|中立)\)"
    matches = re.findall(pattern, text.strip())
    return [{"target": t.strip(), "stance": s.strip()} for t, s in matches]

# ===================== 模型调用（带重试+日志记录） =====================
def call_qwen_api(text: str, retry_times: int = 3) -> list:
    for attempt in range(retry_times):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                **MODEL_PARAMS
            )
            output_text = response.choices[0].message.content.strip()
            return parse_model_output(output_text)
        except Exception as e:
            logger.warning(f"第{attempt+1}次重试失败：{str(e)[:50]}")
            time.sleep(2)
    logger.error(f"文本处理最终失败：{text[:30]}...")
    return []

# ===================== 主程序 =====================
if __name__ == "__main__":
    try:
        # 读取数据集
        df = pd.read_excel(INPUT_EXCEL_PATH)
        logger.info(f"✅ 成功读取数据集，共 {len(df)} 条文本")

        all_predictions = []
        total = len(df)

        for idx, row in df.iterrows():
            blog_text = str(row["blog_text"])
            logger.info(f"正在处理：第 {idx+1}/{total} 条")

            # 调用模型
            pred = call_qwen_api(blog_text)

            # 组装结果
            all_predictions.append({
                "id": idx,
                "input_text": blog_text,
                "predictions": pred
            })

            # 限流保护
            time.sleep(0.5)

        # 保存最终结果
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"🎉 全部处理完成！结果已保存至：{OUTPUT_JSON_PATH}")
        logger.info("✅ 可直接运行 evaluate_comprehensive.py 计算指标")

    except Exception as e:
        logger.critical(f"💥 程序异常终止：{str(e)}", exc_info=True)