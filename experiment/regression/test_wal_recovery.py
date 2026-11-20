"""WAL-only recovery: a crash before the first snapshot checkpoint must not
lose fsync-confirmed writes (the 200-before-persisted contract).

Regression guard for the ``_restore_snapshot`` gap: the snapshot file does
not exist yet (fresh deployment, first ``checkpoint_every`` window) while
the WAL holds pending ops - the service must replay the WAL instead of
silently starting empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sme.api.server import _restore_snapshot
from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine


def _wal_engine(tmp_path, autosave=False):
    cfg = SMEConfig()
    cfg.storage.path = str(tmp_path / "engine.json")
    cfg.storage.autosave = autosave
    cfg.persistence.enabled = True
    cfg.persistence.sync_mode = "fsync"
    return SpatialMemoryEngine(cfg)


def test_wal_only_recovery_without_snapshot(tmp_path):
    eng = _wal_engine(tmp_path)
    for i in range(5):  # < checkpoint_every(10): no snapshot written yet
        eng.add(text=f"confirmed write {i}", ns="u1")
    assert not Path(str(tmp_path / "engine.json")).exists()  # snapshot absent
    assert eng.wal.has_pending()

    eng.wal.close()  # simulate hard stop
    eng2 = _wal_engine(tmp_path)
    assert _restore_snapshot(eng2) is True
    texts = [m.text for m in eng2.memories.values()]
    assert len(texts) == 5
    assert all(f"confirmed write {i}" in texts for i in range(5))


def test_recovery_with_stale_snapshot_plus_wal(tmp_path):
    eng = _wal_engine(tmp_path)
    eng.add(text="before checkpoint", ns="u1")
    eng.save()  # snapshot exists at this point
    eng.add(text="after checkpoint", ns="u1")  # WAL-only tail

    eng.wal.close()
    eng2 = _wal_engine(tmp_path)
    assert _restore_snapshot(eng2) is True
    texts = {m.text for m in eng2.memories.values()}
    assert texts == {"before checkpoint", "after checkpoint"}


def test_empty_state_recovery_returns_false(tmp_path):
    eng = _wal_engine(tmp_path)
    eng.wal.close()
    eng2 = _wal_engine(tmp_path)
    assert _restore_snapshot(eng2) is False
    assert len(eng2.memories) == 0
