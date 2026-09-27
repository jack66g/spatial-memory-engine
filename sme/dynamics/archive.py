"""Memory Archive: hot / cold storage.

Hot storage  - memories active in the spatial space, fully retrievable.
Cold storage - archived memories, kept in a compact JSON store (gzipped
               when the path ends with .gz); they are excluded from the
               space and retrieval until restored.

Archive keeps a full copy (with embedding), so restore is lossless. Writes
are atomic (tmp + rename) so a crash can never corrupt the cold store.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any, Optional

from sme.models import Memory


class ArchiveManager:
    def __init__(self, cold_path: Optional[str] = None) -> None:
        self.cold_path = cold_path
        self.cold: dict[str, dict[str, Any]] = {}
        if cold_path and os.path.exists(cold_path):
            with self._open(cold_path) as fh:
                self.cold = json.load(fh)

    @staticmethod
    def _open(path: str):
        """Open the cold store for reading (gz or plain, both supported)."""
        if path.lower().endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8")
        return open(path, "r", encoding="utf-8")

    # ------------------------------------------------------------------ #
    def archive(self, memory: Memory) -> bool:
        """Move a memory to cold storage."""
        if memory.id in self.cold:
            return False
        self.cold[memory.id] = memory.to_dict()
        memory.archived = True
        self._flush()
        return True

    def restore(self, memory: Memory) -> bool:
        """Restore a memory from cold storage back to hot storage."""
        if memory.id not in self.cold:
            return False
        del self.cold[memory.id]
        memory.archived = False
        self._flush()
        return True

    def cold_count(self) -> int:
        return len(self.cold)

    def _flush(self) -> None:
        if not self.cold_path:
            return
        directory = os.path.dirname(os.path.abspath(self.cold_path))
        os.makedirs(directory, exist_ok=True)
        payload = json.dumps(self.cold, ensure_ascii=False)
        tmp_path = os.path.join(
            directory, f".tmp_{os.getpid()}_{os.path.basename(self.cold_path)}"
        )
        try:
            if self.cold_path.lower().endswith(".gz"):
                with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
                    fh.write(payload)
            else:
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    fh.write(payload)
            os.replace(tmp_path, self.cold_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------ #
    def load_state(self, state: dict[str, Any]) -> None:
        self.cold = dict(state)
