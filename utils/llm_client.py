"""
LLM 客户端封装 - 修复版

使用 DashScope 正确的 API 端点
"""

import json
import requests
from typing import Optional, Dict, Any


class LLMClient:
    """LLM 客户端基类"""

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        raise NotImplementedError


class DashScopeClient(LLMClient):
    """
    通义千问客户端 (DashScope)
    使用 HTTP 直接调用 API
    """

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "qwen-plus"):
        self.api_key = api_key
        self.model = model
        # 尝试不同的 API 端点
        self.endpoints = [
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        ]

        if api_key is None:
            import os
            self.api_key = os.environ.get('DASHSCOPE_API_KEY')

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.1,
                 max_tokens: int = 2048,
                 **kwargs) -> str:
        if not self.api_key:
            print("错误：未设置 API Key")
            return self._mock_response(system_prompt, user_prompt)

        for endpoint in self.endpoints:
            result = self._try_request(endpoint, system_prompt, user_prompt, temperature, max_tokens)
            if result and '"error"' not in result:
                return result

        return self._mock_response(system_prompt, user_prompt)

    def _try_request(self, endpoint: str, system_prompt: str, user_prompt: str,
                     temperature: float, max_tokens: int) -> str:

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 尝试第一种格式
        payload1 = {
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

        # 尝试第二种格式（OpenAI 兼容）
        payload2 = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        for payload in [payload1, payload2]:
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=60)

                # 如果是 401，直接返回错误
                if response.status_code == 401:
                    try:
                        err = response.json()
                        err_msg = err.get("error", {}).get("message", str(err))
                    except:
                        err_msg = str(response.text[:200])
                    print(f"API Key 错误 (401): {err_msg}")
                    return f'{{"error": "Invalid API Key: {err_msg}"}}'

                response.raise_for_status()
                result = response.json()

                # 检查错误
                if "error" in result:
                    err_msg = result["error"].get("message", str(result["error"]))
                    print(f"API 错误：{err_msg}")
                    return f'{{"error": "{err_msg}"}}'

                # 解析响应（两种格式）
                if "output" in result:  # DashScope 格式
                    return result["output"]["choices"][0]["message"]["content"]
                elif "choices" in result:  # OpenAI 格式
                    return result["choices"][0]["message"]["content"]
                else:
                    return str(result)

            except requests.exceptions.RequestException as e:
                continue
            except (KeyError, IndexError, json.JSONDecodeError):
                continue

        return '{"error": "All endpoints failed"}'

    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        """Mock 响应"""
        if '目标抽取' in system_prompt or '抽取专家' in system_prompt:
            return '{"targets": []}'
        elif '关系抽取' in system_prompt or '关系专家' in system_prompt:
            return '{"relation": "none", "direction": "none", "evidence": "", "confidence": 1.0, "reasoning": "Mock"}'
        elif '立场检测' in system_prompt or '立场检测专家' in system_prompt:
            return '{"stances": []}'
        else:
            return '{"result": "mock"}'


class OpenAICompatibleClient(LLMClient):
    """OpenAI 兼容接口客户端"""

    def __init__(self, api_key: str, base_url: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.1, max_tokens: int = 2048, **kwargs) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers, json=payload, timeout=120
            )
            response.raise_for_status()
            result = response.json()

            return result["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"生成出错：{e}")
            return '{"error": "' + str(e) + '"}'


def create_llm_client(provider: str = "dashscope", api_key: Optional[str] = None, **kwargs) -> LLMClient:
    if provider == "dashscope":
        return DashScopeClient(api_key=api_key, **kwargs)
    elif provider == "openai":
        return OpenAICompatibleClient(api_key=api_key, **kwargs)
    else:
        return DashScopeClient(api_key=api_key, **kwargs)
