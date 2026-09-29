"""SME 记忆插件的 MCP 服务器（stdio）。

把 Spatial Memory Engine 暴露为 Model Context Protocol 工具，供任意 MCP 宿主
（ZCode / Claude Code 等智能体）在会话中自动读写长期记忆：

    remember(text, tags, importance)   写入一条记忆（决策/偏好/发现）
    recall(query, top_k)               语义检索记忆（返回 id 便于强化）
    reinforce(memory_id)               标记"用上了"（Ebbinghaus 强化，越用越牢）
    memory_stats()                     引擎状态（记忆数/Region 数）

架构：本进程是薄代理，唯一数据落盘方是常驻的 REST 服务
（``python -m sme.api --port 8760``，单写者，自带热重建锁/WAL）。服务未运行时
自动拉起（Windows 分离进程）并等待就绪。这样主会话与多个子智能体各自 spawn 的
MCP 进程都指向同一个记忆库，不会出现多进程并发写同一快照文件。

用法（宿主配置示例）::

    {"mcpServers": {"sme-memory": {
        "command": "python",
        "args": ["C:/path/to/sme/mcp_server.py"],
        "env": {"SME_MCP_PORT": "8760"}   # 可选，默认 8760
    }}}
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time

import httpx
from mcp.server.mcpserver import MCPServer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PORT = int(os.environ.get("SME_MCP_PORT", "8760"))
BASE_URL = f"http://127.0.0.1:{_PORT}"
# 回环直连：httpx 不读环境代理（trust_env=False），但不修改环境变量本身——
# 拉起的子服务要继承完整环境。模型已本地缓存，服务侧另设 HF_HUB_OFFLINE=1
# 跳过 HF 线上校验（无代理/断网时校验会超时重试拖死启动）。
_NO_ENV_PROXY = {"trust_env": False}
mcp = MCPServer(
    name="sme-memory",
    title="SME 长期记忆",
    description="Spatial Memory Engine 长期记忆：跨会话记住用户偏好、项目决策与工作流",
    instructions=(
        "会话开始时先 recall 恢复上下文（用户偏好/项目状态/待办）；"
        "出现新决策、新偏好、重要发现或用户纠正时立即 remember；"
        "用上了某条记忆后 reinforce 它。记忆永不删除，引擎自动融合近重复项。"
    ),
)


def _service_up(timeout: float = 3.0) -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=timeout, **_NO_ENV_PROXY)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


# 拉起服务的单飞锁：同一 MCP 进程内并发的工具调用（宿主多线程派发）
# 只允许一个线程进入 spawn 窗口，避免 N 路同时加载本地 embedding 模型
# 造成内存尖峰；等待就绪的轮询在锁外进行，不互相阻塞。
_SPAWN_LOCK = threading.Lock()


def _ensure_service(wait_seconds: float = 90.0) -> None:
    """服务不在就后台拉起（分离进程，不随 MCP 退出），并等它就绪。

    就绪等待给足模型加载时间（本地 Qwen3 embedding 首载约 10-30s）；
    服务日志写到系统临时目录 sme_service.log 便于排查拉起失败。
    超时不静默返回：裸 ConnectError 栈对用户毫无信息量，显式抛
    RuntimeError 指向日志路径。
    """
    if _service_up():
        return
    log_path = os.path.join(tempfile.gettempdir(), "sme_service.log")
    with _SPAWN_LOCK:
        if _service_up():
            # 双检：排队等锁期间，前面的线程已经把服务拉起来了
            return
        flags = 0
        popen_kwargs: dict = {}
        child_env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        if os.name == "nt":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # POSIX 侧脱离会话（setsid）：MCP 进程退出/终端关闭不带走服务
            popen_kwargs["start_new_session"] = True
        log_fh = open(log_path, "ab")
        try:
            subprocess.Popen(
                [sys.executable, "-m", "sme.api", "--port", str(_PORT)],
                cwd=REPO_ROOT,
                creationflags=flags,
                env=child_env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                close_fds=True,
                **popen_kwargs,
            )
        finally:
            # 句柄已被子进程继承，父端可安全关闭
            log_fh.close()
    # 等待段在锁外：并发调用各自轮询就绪，不占着 spawn 锁
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _service_up():
            return
        time.sleep(1.5)
    raise RuntimeError(
        f"记忆服务 {wait_seconds:.0f} 秒内未就绪，请查看 {log_path}"
    )


def _api(method: str, path: str, payload: dict | None = None) -> dict:
    _ensure_service()
    r = httpx.request(method, f"{BASE_URL}{path}", json=payload, timeout=300, **_NO_ENV_PROXY)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def remember(
    text: str,
    tags: list[str] | None = None,
    importance: float = 0.5,
) -> str:
    """写入一条长期记忆。用于：用户偏好/要求、项目决策、重要发现、工作流约定。

    Args:
        text: 记忆内容（一句话自包含，将来要能脱离上下文看懂）
        tags: 分类标签，如 ["user_pref"] / ["project","decision"] / ["todo"]
        importance: 0.3-0.9，用户明确要求/核心决策用高值
    """
    d = _api("POST", "/memories", {
        "text": text, "tags": tags or [], "importance": importance, "source": "session",
    })
    mid = d.get("id") or d.get("memory", {}).get("id", "?")
    n = _api("GET", "/stats").get("memories", {}).get("total", "?")
    return f"已记住 [{mid}]（库内共 {n} 条）：{text[:80]}"


@mcp.tool()
def recall(query: str, top_k: int = 3) -> str:
    """语义检索长期记忆（口语化查询即可）。会话开始恢复上下文、工作中查"之前怎么定的"都用它。

    Args:
        query: 查询（如"用户对提交代码的要求" / "项目现在最重要的策略"）
        top_k: 返回条数（默认 3）
    """
    d = _api("POST", "/memories/search", {"text": query, "top_k": top_k})
    results = d.get("results", [])
    if not results:
        return "（记忆库中无相关记忆）"
    lines = []
    for h in results:
        tags = ",".join(h.get("tags") or [])
        lines.append(
            f"[{h['id']}] score={h.get('score', 0):.2f} #{tags} {h['text']}"
        )
    return "\n".join(lines) + "\n（用上了某条就调 reinforce(memory_id) 强化它）"


@mcp.tool()
def reinforce(memory_id: str) -> str:
    """标记一条记忆"本次用上了"：Ebbinghaus 强化，越常用的记忆越牢固、排序越靠前。"""
    _api("POST", f"/memories/{memory_id}/hit")
    return f"已强化 {memory_id}"


@mcp.tool()
def memory_stats() -> str:
    """记忆库状态：记忆总数、Region 数、graph 边数、演化计数。"""
    d = _api("GET", "/stats")
    m = d.get("memories", {})
    r = d.get("regions", {})
    return (
        f"记忆 {m.get('total', '?')} 条（活跃 {m.get('active', '?')} / 归档 "
        f"{m.get('archived', '?')}），Region {r.get('count', '?')} 个，"
        f"graph 边 {d.get('graph_edges', 0)}，splits={d.get('splits', 0)} "
        f"merges={d.get('merges', 0)}"
    )


if __name__ == "__main__":
    _ensure_service()
    mcp.run(transport="stdio")
