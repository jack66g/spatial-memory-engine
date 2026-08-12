"""LLM client: OpenAI-compatible chat completions.

No model is hardcoded - the client talks to any OpenAI-compatible
``/chat/completions`` endpoint:

    base_url + api_key + model

switch between OpenAI, DeepSeek, Qwen, GLM, Claude-compatible gateways,
Gemini-compatible gateways, LM Studio, vLLM, Ollama, OpenRouter,
SiliconFlow, and any other compatible service.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from sme.config import LLMConfig

CHAT_PATH = "/chat/completions"


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.base_url = (config.base_url or "").rstrip("/")
        self.last_usage: dict[str, int] = {}
        self.total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.calls = 0

    @property
    def configured(self) -> bool:
        """True when a real endpoint is reachable.

        The default OpenAI URL without an API key is treated as
        unconfigured so offline demos never accidentally hit the network.
        """
        if not self.base_url:
            return False
        if self.base_url == "https://api.openai.com/v1" and not self.config.api_key:
            return False
        return True

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat completion request and return the assistant text."""
        if not self.configured:
            raise RuntimeError(
                "LLM is not configured: set llm.base_url (and api_key) in config"
            )
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
        }
        if self.config.reasoning_effort:
            body["reasoning_effort"] = self.config.reasoning_effort
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        elif self.config.max_tokens:
            body["max_tokens"] = self.config.max_tokens

        with httpx.Client(timeout=self.config.timeout) as client:
            resp = client.post(
                f"{self.base_url}{CHAT_PATH}",
                headers=self._headers(),
                json=body,
            )
            resp.raise_for_status()
            payload = resp.json()
        self.calls += 1
        usage = payload.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in usage:
                self.last_usage[key] = usage.get(key, 0)
                self.total_usage[key] = self.total_usage.get(key, 0) + usage.get(key, 0)
        try:
            content = payload["choices"][0]["message"]["content"]
            return (content or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected response shape from LLM endpoint: {payload}"
            ) from exc

    def chat_json(self, messages: list[dict[str, str]], **kwargs) -> Any:
        """Chat completion parsed as JSON."""
        text = self.chat(messages, **kwargs)
        return json.loads(text)

    # ------------------------------------------------------------------ #
    def summarize_memories(self, memories: list) -> str:
        """Summarize a group of memories. Template fallback when offline."""
        if self.configured:
            try:
                items = "\n".join(f"- {m.text}" for m in memories)
                prompt = (
                    "You are a memory consolidation engine. Write ONE concise "
                    "summary sentence that captures the common theme of these "
                    "user memories. Keep it factual, in the user's language:\n\n"
                    f"{items}"
                )
                return self.chat(
                    [{"role": "user", "content": prompt}], temperature=0.2
                )
            except Exception:  # noqa: BLE001 - fall back to template
                pass
        return self._template_summary(memories)

    @staticmethod
    def _template_summary(memories: list) -> str:
        if not memories:
            return ""
        if len(memories) == 1:
            return memories[0].text
        preview = "；".join(m.text for m in memories[:4])
        more = f" 等{len(memories)}条记录" if len(memories) > 4 else ""
        return f"相关主题记忆（{len(memories)}条）：{preview}{more}"

    def to_dict(self) -> dict:
        return {
            "base_url": self.base_url,
            "model": self.config.model,
            "configured": self.configured,
            "has_api_key": bool(self.config.api_key),
        }
