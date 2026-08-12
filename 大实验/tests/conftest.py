"""Shared fixtures for the SME regression suite."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # SME 插件根

import pytest

from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine


def _make_engine(path: str) -> SpatialMemoryEngine:
    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = path
    return SpatialMemoryEngine(cfg)


@pytest.fixture
def fresh_engine(tmp_path):
    """One isolated engine (autosave off, storage in the tmp dir)."""
    return _make_engine(str(tmp_path / "engine.json"))


@pytest.fixture
def new_engine(tmp_path):
    """Factory for additional isolated engines (unique paths per call)."""
    counter = {"n": 0}

    def _make():
        counter["n"] += 1
        return _make_engine(str(tmp_path / f"engine{counter['n']}.json"))

    return _make


@pytest.fixture
def zh():
    """Chinese test strings as code points (console-encoding safe)."""

    def u(*cps):
        return "".join(chr(c) for c in cps)

    return {
        "likes_coffee": u(0x7528, 0x6237, 0x559C, 0x6B22, 0x559D, 0x5496, 0x5561),       # 用户喜欢喝咖啡
        "lives_beijing": u(0x7528, 0x6237, 0x4F4F, 0x5728, 0x5317, 0x4EAC),              # 用户住在北京
        "corr_no_coffee": u(0x5176, 0x5B9E, 0x7528, 0x6237, 0x4E0D, 0x559C, 0x6B22,     # 其实用户不喜欢喝咖啡了
                           0x559D, 0x5496, 0x5561, 0x4E86),
        "haha": u(0x54C8, 0x54C8),                                                       # 哈哈
        "q_name": u(0x4F60, 0x53EB, 0x4EC0, 0x4E48, 0x540D, 0x5B57),                     # 你叫什么名字
        "a_name": u(0x6211, 0x53EB, 0x5C0F, 0x660E),                                     # 我叫小明
        "works_company": u(0x7528, 0x6237, 0x5728, 0x516C, 0x53F8, 0x4E0A, 0x73ED),      # 用户在公司上班
        "colleague_zhang": u(0x7528, 0x6237, 0x7684, 0x540C, 0x4E8B, 0x662F, 0x5F20, 0x4E09),  # 用户的同事是张三
    }
