"""擂台赛选手统一接口。

所有适配器实现同一个 Contestant 协议；run_battle.py 只认这个接口：

    name    : 展示名（结果表主键）
    add(text, ns=None)          写入一轮用户消息（选手内部自行决定抽取/原文存储）
    search(q, top_k=5) -> list[(text, score)]   检索，按相关性降序
    reset()                     清空状态，回到初始（用于多 seed 重放）
    count() -> int              当前真实存储条数（记忆条目数）
    close()                     释放重型资源（模型/DB 连接），之后不可再用
    reinit_cost_s               构造耗时（init 到可用），用于报表

约定：
- search 返回的 score 允许各选手用自己的量纲（judge 只看文本不看分数），
  但顺序必须按"选手认为的相关性"降序。
- add/search 抛出的异常由 run_battle 捕获记账，适配器自身不做重试。
"""

from __future__ import annotations

from typing import Optional


class Contestant:
    name: str = "contestant"
    reinit_cost_s: float = 0.0

    def add(self, text: str, ns: Optional[str] = None) -> None:
        raise NotImplementedError

    def search(self, q: str, top_k: int = 5) -> list[tuple[str, float]]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def count(self) -> int:
        return -1  # 不支持时返回 -1（报表显示 n/a）

    def close(self) -> None:
        pass
