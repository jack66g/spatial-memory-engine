"""Five alternating pairs at equal acknowledgement boundaries (isolated data).

The frozen baseline cannot persist reinforcement through WAL. Its supported
alternative is save(), either after each operation or after a declared batch.
Both the unsafe old path and this durable alternative are reported separately.
No source tree is modified. psutil is used only by this laboratory script.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--worker", choices=["baseline", "current"])
parser.add_argument("--repeat", type=int, default=0)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

if args.worker is None:
    for repeat in range(5):
        order = ["baseline", "current"] if repeat % 2 == 0 else ["current", "baseline"]
        for label in order:
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                            "--out", str(args.out), "--baseline", str(args.baseline),
                            "--source", str(args.source), "--worker", label,
                            "--repeat", str(repeat)], check=True, timeout=240,
                           env={**os.environ, "PYTHONHASHSEED": "1147", "OMP_NUM_THREADS": "4",
                                "OPENBLAS_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1",
                                "TRANSFORMERS_OFFLINE": "1", "NO_PROXY": "127.0.0.1,localhost"})
    raise SystemExit(0)

sys.path.insert(0, str(args.baseline if args.worker == "baseline" else args.source))
import numpy as np
import psutil
from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine
import sme.models as models
import sme.space.region_manager as region_manager

counter = itertools.count()
def deterministic_id(prefix="m"):
    return f"{prefix}_{next(counter):012x}"
models.new_id = region_manager.new_id = deterministic_id
proc = psutil.Process()
rows = []
with tempfile.TemporaryDirectory(prefix="sme-durable-perf-", dir=args.out) as tmp:
    for size in [100, 1000, 10000]:
        for mode in ["fsync", "grouped"]:
            path = Path(tmp) / f"{size}-{mode}.json.gz"
            c = SMEConfig()
            c.storage.path, c.storage.autosave = str(path), False
            c.persistence.enabled, c.persistence.sync_mode = True, mode
            c.region.auto_evolve, c.region.min_join_cosine = False, 0
            c.embedding.dim = 64
            c.policy.decay_enabled = False
            e = SpatialMemoryEngine(c)
            vectors = np.random.default_rng(1147).normal(size=(size, 64))
            vectors /= np.linalg.norm(vectors, axis=1)[:, None]
            e.space.set_bulk(True)
            for i, v in enumerate(vectors):
                e.memory_manager.add_memory(f"project {i} configuration storage", embedding=v)
            e.space.set_bulk(False)
            e.save()
            ids = list(e.memories)[:20]
            rss_before = proc.memory_info().rss
            peak = [rss_before]
            stop = threading.Event()
            def sample():
                while not stop.wait(.005):
                    peak[0] = max(peak[0], proc.memory_info().rss)
            monitor = threading.Thread(target=sample, daemon=True)
            monitor.start()
            t0 = time.perf_counter()
            for mid in ids:
                e.reinforce(mid)
                if mode == "fsync" and args.worker == "baseline":
                    e.save()  # old public way to acknowledge each mutation durably
            if mode == "grouped":
                if args.worker == "baseline":
                    e.save()  # acknowledge this entire declared batch
                else:
                    e.wal.flush()
            elif args.worker == "current":
                e.wal.flush()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            stop.set(); monitor.join()
            logical = {mid: e.get(mid).to_dict() for mid in ids}
            wal_bytes = Path(e.wal.path).stat().st_size if Path(e.wal.path).exists() else 0
            disk_bytes = sum(p.stat().st_size for p in path.parent.glob(path.stem + "*"))
            e.wal.close()
            r = SpatialMemoryEngine(c)
            assert r.load()
            for mid, before in logical.items():
                after = r.get(mid).to_dict()
                assert all(after[k] == value for k, value in before.items() if k != "region_id")
            r.wal.close()
            rows.append({"size": size, "mode": mode, "operations": 20,
                         "boundary": "each operation" if mode == "fsync" else "batch of 20",
                         "durable_ms": elapsed_ms, "rss_before_bytes": rss_before,
                         "sampled_peak_rss_bytes": peak[0], "wal_bytes": wal_bytes,
                         "manifest_plus_wal_bytes_before_recovery": disk_bytes,
                         "disk_scope": "manifest and WAL only; generation vector sidecar excluded",
                         "recovery_exact": True})
            print("durable", args.worker, args.repeat, size, mode, round(elapsed_ms, 2), flush=True)
            del r, e
            gc.collect()
(args.out / f"durable_{args.worker}_{args.repeat}.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
