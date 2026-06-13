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
        logging.FileHandler("/data/stance_detection_project/stance_detection_project/runcot_qw.log", encoding='utf-8')  # 运行日志自动保存到 run.log
    ]
)
logger = logging.getLogger(__name__)

# ===================== 核心配置（仅修改API_KEY） =====================
API_KEY = "sk-da3862f95da94483b8e"  # 填入你的真实API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 华北2（北京）
MODEL_NAME = "qwen2.5-7b-instruct"
INPUT_EXCEL_PATH = "/data/stance_detection_project/stance_detection_project/data/test.xlsx"
OUTPUT_JSON_PATH = "/data/stance_detection_project/stance_detection_project/results/test_predictions_real_zcot_qw_test.json"

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
你是一名目标识别与立场检测专家，根据给定的微博评论提取讨论的主要目标（单个 / 多个）目标并判断立场，请进行一步一步思考严格按照以下推理格式
步骤 1:[核心目标提取]: 可能的目标有 [目标内容 1]、[目标内容 2]
步骤 2:[精确立场检测] 逐目标分析:
因为...[情感证据]..., 目标 [目标内容 1] 的立场是 (支持 / 反对 / 中立)
因为...[情感证据]..., 目标 [目标内容 2] 的立场是 (支持 / 反对 / 中立)(依次分析每个目标)
实例： input"input": "今天菲律宾海警小船又在冲撞中国海警大船，越来越像当年的越南了。1979 年我已经 10 岁，对越南挑衅中国的做法记忆很深。现在的菲律宾在美国撑腰下，同样有些自我催眠，南海上不断挑衅、反复折腾，若不吃点教训，恐怕很难消停。"
      output"output": "
      步骤 1:[核心目标提取]: 可能的目标有 [菲律宾]、[越南]、[中国]、[美国]
      步骤 2:[精确立场检测] 逐目标分析:
      因为文本中将菲律宾描述为 “冲撞中国海警大船”“挑衅”“反复折腾”，并说 “若不吃点教训，恐怕很难消停”，带有明显否定和谴责色彩，目标 [菲律宾] 的立场是反对。
      因为文本中将当年的越南与今天的菲律宾类比，并提到 “越南挑衅中国的做法”，同样带有负面评价，目标 [越南] 的立场是反对。因为文本整体叙述站在中国立场上，将中国塑造为被挑衅的一方，并认为中国有权对挑衅行为进行回应，体现出认可和维护态度，目标 [中国] 的立场是支持。
      因为文本中说菲律宾是在 “美国的‘加持’下” 变得更加自负，并将这种外部支持视为负面诱因，带有批评意味，目标 [美国] 的立场是反对。
      步骤 3:[最终结果]:
     (目标: 菲律宾, 立场: 反对)
     (目标: 越南, 立场: 反对)
     (目标: 中国, 立场: 支持)
     (目标: 美国, 立场: 反对)"
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