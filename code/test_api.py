# 强制修复编码，解决ASCII报错
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from openai import OpenAI

# ===================== 你的配置 =====================
API_KEY = "sk-"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen2.5-7b-instruct"

# 初始化客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 纯英文测试，无任何中文
test_text = "hi"

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": test_text}],
        temperature=0.1
    )
    # 纯英文输出
    print("API CALL SUCCESS")
    print("RESPONSE:", response.choices[0].message.content)

except Exception as e:
    print("API CALL FAILED:", str(e))