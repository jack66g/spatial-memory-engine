"""Persistence layer: engine snapshots, storage backends and write-ahead log."""

from sme.storage.snapshot import (
    EngineSnapshot,
    embedding_path,
    load_snapshot,
    save_snapshot,
)
from sme.storage.backends import (
    LocalJsonBackend,
    SqliteBackend,
    StorageBackend,
    build_storage_backend,
)
from sme.storage.wal import WriteAheadLog, default_wal_path

__all__ = [
    "EngineSnapshot",
    "embedding_path",
    "save_snapshot",
    "load_snapshot",
    "StorageBackend",
    "LocalJsonBackend",
    "SqliteBackend",
    "build_storage_backend",
    "WriteAheadLog",
    "default_wal_path",
]
