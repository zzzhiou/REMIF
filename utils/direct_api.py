#!/usr/bin/env python3
"""
直接调用 DashScope API（不依赖 SDK）
"""

import requests
import json


class DirectDashScopeClient:
    """直接调用 DashScope API 的客户端"""

    def __init__(self, api_key, model="qwen-plus"):
        self.api_key = api_key
        self.model = model
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    def generate(self, system_prompt, user_prompt, temperature=0.1, max_tokens=2048):
        """生成响应"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "result_format": "message"
            }
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            if result.get("status_code") == 200:
                return result["output"]["choices"][0]["message"]["content"]
            else:
                print(f"API 错误：{result}")
                return '{"error": "' + str(result) + '"}'

        except requests.exceptions.RequestException as e:
            print(f"请求失败：{e}")
            return '{"error": "' + str(e) + '"}'
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"解析失败：{e}")
            return '{"error": "' + str(e) + '"}'


if __name__ == "__main__":
    # 测试
    api_key = "sk-XTHOzGbBP8XwsKWUyouQiqoGgEuGYaHRpZ9Xf9XaTnlOESbn"
    client = DirectDashScopeClient(api_key, model="qwen-plus")

    print("测试 LLM API...")
    response = client.generate(
        system_prompt="你是一个助手。",
        user_prompt="你好，请用一句话介绍你自己。"
    )
    print(f"响应：{response}")
