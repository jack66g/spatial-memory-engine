"""DeepSeek LLM 客户端（大实验专用）。

API Key 只从环境变量读取（SME_LLM_API_KEY / DEEPSEEK_API_KEY），
绝不写进任何文件。统一走最便宜档 deepseek-v4-flash。
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"


class LLMClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL,
                 api_key: str | None = None,
                 timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get(
            "SME_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
        self.timeout = timeout
        self.calls = 0
        self.total_usage: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.latencies_ms: list[float] = []

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def chat(self, messages: list[dict[str, str]],
             temperature: float = 0.7,
             max_tokens: int = 512) -> str:
        if not self.configured:
            raise RuntimeError(
                "未配置 API Key：请设置环境变量 SME_LLM_API_KEY")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",   # 最便宜档必须关闭推理
        }
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/chat/completions",
                               headers=self._headers(), json=body)
            resp.raise_for_status()
            payload = resp.json()
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        self.calls += 1
        usage = payload.get("usage") or {}
        for key in self.total_usage:
            if key in usage:
                self.total_usage[key] += usage.get(key, 0)
        try:
            return (payload["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"响应格式异常: {payload}") from exc

    def cost_estimate_yuan(self) -> float:
        """deepseek-v4-flash 近似单价（¥/百万 token，输入输出混合估）。"""
        total = self.total_usage["total_tokens"]
        return round(total / 1_000_000 * 3.0, 4)

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "total_tokens": self.total_usage["total_tokens"],
            "prompt_tokens": self.total_usage["prompt_tokens"],
            "completion_tokens": self.total_usage["completion_tokens"],
            "est_cost_yuan": self.cost_estimate_yuan(),
            "avg_latency_ms": round(sum(self.latencies_ms) / len(self.latencies_ms), 1)
            if self.latencies_ms else 0.0,
        }
