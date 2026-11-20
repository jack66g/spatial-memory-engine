# -*- coding: utf-8 -*-
"""跨进程文件锁(Windows msvcrt / POSIX fcntl)+ 存储写入者谱系。

SME 的快照是"整库单文件"语义,历史上多进程写同一路径存在两种静默失败:
1. 两个 save 交错 → 撕裂文件/孤儿快照(manifest 引用的 sidecar 被对方清掉);
2. last-writer-wins → 先保存进程的写入被后保存者整库覆盖,零提示。

xlock 覆盖"读快照→改→落盘"整段,交叉写要么串行成功,要么如实报错。
谱系(lineage)记录在锁文件偏移 16 处:每次成功 save 推进 (pid, n)。
引擎在 load/save 时记住当时谱系,save 前发现谱系已被别人推进 = 丢更新,
如实拒绝。不能用文件 mtime 做判据:sqlite 后端的 WAL 写在同一个库文件里,
每次 mutation 都会动 mtime/size,指纹法会把自家 WAL 写误判成外部修改。
谱系必须通过持锁句柄读写——锁住首字节后,另开句柄的读写会被 OS 拒绝。

注意:锁的是快照文件级操作;WAL 追加(每条 add)不在锁内,跨进程 WAL 交错
仍属未定义——多进程写请各自独立 storage.path,或由上层编排(如插件线)。
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Iterator


class CrossProcessLockTimeout(RuntimeError):
    """在 timeout 内未能拿到跨进程锁:另一实例正在写同一存储。"""


# 线程级重入表:同线程内嵌套加锁(如 load→WAL恢复→save)直接放行,
# 不再对同字节二次 msvcrt/fcntl(同进程不同句柄会自锁 30s)。
# _held 同时登记持锁句柄访问器,重入分支 yield 同一个访问器。
_reentrant: dict[tuple[int, str], int] = {}
_held: dict[tuple[int, str], "_LockedFile"] = {}
_reentrant_guard = threading.Lock()

#: 谱系 JSON 存放的文件偏移(首字节 0 是锁位,不与数据混用)
_LINEAGE_OFFSET = 16


class _LockedFile:
    """持锁文件的谱系访问器:读写都走已持有区域锁的句柄。"""

    def __init__(self, fh) -> None:
        self._fh = fh

    def read_lineage(self) -> tuple[int, int]:
        try:
            self._fh.seek(_LINEAGE_OFFSET)
            raw = self._fh.read().decode("utf-8", errors="replace").strip()
            if not raw:
                return (0, 0)
            data = json.loads(raw)
            return (int(data.get("pid", 0)), int(data.get("n", 0)))
        except (OSError, ValueError, TypeError):
            return (0, 0)

    def write_lineage(self, pid: int, n: int) -> None:
        """推进谱系(仅在持有 xlock 时调用)。写失败不阻断保存——
        谱系缺失最多让防护降级为"不拦",不会产生误拦或数据风险。"""
        try:
            self._fh.seek(_LINEAGE_OFFSET)
            self._fh.write(json.dumps({"pid": pid, "n": n}).encode("utf-8"))
            self._fh.truncate()   # 截掉旧内容残留
            self._fh.flush()
        except OSError:
            pass


def _lock_path(path: str | os.PathLike) -> str:
    return str(path) + ".lock"


@contextlib.contextmanager
def xlock(path: str | os.PathLike, timeout: float = 30.0, poll: float = 0.05) -> Iterator["_LockedFile"]:
    """对 ``path`` 对应的锁文件(``path + '.lock'``)取独占锁,超时如实抛错。

    yield 一个谱系访问器(read_lineage/write_lineage)。Windows 用
    msvcrt.locking(锁文件首字节),POSIX 用 flock;进程崩溃/退出后由 OS
    自动释放,无孤儿锁。本线程对同一路径重入直接放行并复用访问器;
    跨线程/跨进程仍互斥。
    """
    lock_path = _lock_path(path)
    key = (threading.get_ident(), os.path.normcase(os.path.abspath(lock_path)))
    # 判定+登记必须在 guard 内完成,yield 必须在 guard 外(生成器挂起时不能持锁)
    with _reentrant_guard:
        reentered = _reentrant.get(key, 0) > 0
        if reentered:
            _reentrant[key] += 1
    if reentered:
        try:
            yield _held[key]
        finally:
            with _reentrant_guard:
                _reentrant[key] -= 1
                if _reentrant[key] <= 0:
                    _reentrant.pop(key, None)
        return
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    # 不能用 "a+b":追加模式下 seek+write 无效(写永远落文件尾,谱系读写错位)
    _fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    fh = os.fdopen(_fd, "r+b")
    accessor = _LockedFile(fh)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise CrossProcessLockTimeout(
                        f"另一进程正在写入 {path},已等待 {timeout:g}s 未获得锁;请重试或检查是否有实例未退出"
                    )
                time.sleep(poll)
        with _reentrant_guard:
            _reentrant[key] = 1
            _held[key] = accessor
        try:
            yield accessor
        finally:
            try:
                if os.name == "nt":
                    import msvcrt
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass  # 进程即将退出时释放失败由 OS 兜底
            with _reentrant_guard:
                _reentrant.pop(key, None)
                _held.pop(key, None)
    finally:
        fh.close()
