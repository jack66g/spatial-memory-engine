"""lab 公共设施：.env 密钥加载、DeepSeek LLM 调用、本地 Qwen3 embedding 单例。

铁律：密钥只从 lab/.env 读，绝不写进代码、日志或结果文件。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent

# --------------------------------------------------------------------- #
# .env（LAB_LLM_BASE_URL / LAB_LLM_API_KEY / LAB_LLM_MODEL）
# --------------------------------------------------------------------- #
_ENV_CACHE: Optional[dict] = None


def load_env() -> dict:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        env: dict = {}
        path = LAB_ROOT / ".env"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
        for key in ("LAB_LLM_BASE_URL", "LAB_LLM_API_KEY", "LAB_LLM_MODEL"):
            if not env.get(key):
                raise ValueError(f"lab/.env missing key: {key}")
        # DeepSeek 是国内服务，走系统代理反而会被掐 SSL（EOF in violation of
        # protocol，2026-09-29 实测代理挂、直连 0.5s 通）。这里对 LLM 域名
        # 强制直连：对 httpx 与 openai 客户端（mem0）同时生效。
        from urllib.parse import urlparse
        _host = urlparse(env["LAB_LLM_BASE_URL"]).hostname or "api.deepseek.com"
        no_proxy = {h.strip() for h in os.environ.get("NO_PROXY", "").split(",") if h.strip()}
        no_proxy.add(_host)
        os.environ["NO_PROXY"] = ",".join(sorted(no_proxy))
        os.environ["no_proxy"] = os.environ["NO_PROXY"]
        _ENV_CACHE = env
    return _ENV_CACHE


# --------------------------------------------------------------------- #
# DeepSeek chat（OpenAI 兼容 /chat/completions，httpx 走 env 代理）
# --------------------------------------------------------------------- #
def llm_chat(
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout: float = 120.0,
    retries: int = 2,
    model: Optional[str] = None,
) -> str:
    """同步调用判分/生成 LLM。reasoning_effort=none 关掉推理，省 token 且内容直出。"""
    env = load_env()
    body = {
        "model": model or env["LAB_LLM_MODEL"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": "none",
    }
    headers = {"Authorization": f"Bearer {env['LAB_LLM_API_KEY']}"}
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(
                    f"{env['LAB_LLM_BASE_URL'].rstrip('/')}/chat/completions",
                    headers=headers, json=body,
                )
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            content = r.json()["choices"][0]["message"]["content"]
            if content and content.strip():
                return content
            raise RuntimeError("empty content")
        except Exception as e:  # noqa: BLE001 - 统一重试
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"llm_chat failed after {retries} tries: {last_err}")


_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def llm_json(
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout: float = 180.0,
):
    """要求 JSON 输出并容错解析（截取第一个平衡的 {...}/[...] 块）。"""
    raw = llm_chat(messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(raw)
        if m:
            return json.loads(m.group(0))
        raise


# --------------------------------------------------------------------- #
# 本地 Qwen3-Embedding-0.6B（1024 维，CPU）进程级单例
# 统一协议：rag / graphiti（及任何需要裸向量的选手）共享这一个实例。
# --------------------------------------------------------------------- #
_EMBEDDER = None
_EMBEDDER_T0 = 0.0


def get_embedder():
    global _EMBEDDER, _EMBEDDER_T0
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER_T0 = time.time()
        _EMBEDDER = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    return _EMBEDDER


def embed_texts(texts: list[str]) -> "list[list[float]]":
    """统一入口：归一化向量，供余弦相似度直接点积。"""
    m = get_embedder()
    vecs = m.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
    return [v.tolist() for v in vecs]


EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM = 1024


def u8(s: str) -> str:
    """Windows 控制台打印兜底。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    return s
