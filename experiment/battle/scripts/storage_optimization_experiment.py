"""Alternating paired storage runs, without model loading or production data."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline")
    p.add_argument("--workdir", required=True)
    p.add_argument("--out")
    p.add_argument("--source")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--n", type=int, default=2000)
    args = p.parse_args()
    work = Path(args.workdir).resolve()
    if args.source:
        sys.path.insert(0, str(Path(args.source).resolve()))
        from sme.config import SMEConfig
        from sme.engine import SpatialMemoryEngine

        logging.getLogger("sme").setLevel(logging.ERROR)
        runs = {}
        for mode in ("fsync", "grouped"):
            with tempfile.TemporaryDirectory(prefix="paired-storage-", dir=work) as temporary:
                cfg = SMEConfig()
                cfg.storage.path = str(Path(temporary)/"state.json")
                cfg.storage.autosave = False
                cfg.persistence.enabled = True
                cfg.persistence.sync_mode = mode
                e = SpatialMemoryEngine(cfg)
                rng = random.Random(args.seed)
                topics = [f"topic_{i}" for i in range(32)]
                texts = [f"memory {i} seed {args.seed} about {rng.choice(topics)} server operation" for i in range(args.n)]
                start = time.perf_counter()
                for text in texts:
                    e.add(text)
                e.wal.close()  # include the last grouped flush in the timing
                write_s = time.perf_counter()-start
                wal_bytes = Path(e.wal.path).stat().st_size
                start = time.perf_counter()
                e.save()
                checkpoint_ms = (time.perf_counter()-start)*1000
                e.wal.close()
                runs[mode] = {"n": args.n, "writes_per_s": args.n/write_s,
                              "checkpoint_ms": checkpoint_ms, "wal_bytes": wal_bytes}
        print(json.dumps(runs))
        return
    if not args.baseline or not args.out:
        p.error("--baseline and --out required")
    result = {"protocol": f"3 seeds, {args.n} writes per seed/arm/mode, alternating arm order, includes grouped final flush",
              "baseline": [], "optimized": []}
    for i, seed in enumerate((11, 29, 47)):
        arms = [("baseline", args.baseline), ("optimized", str(Path(__file__).resolve().parents[2]))]
        if i % 2:
            arms.reverse()
        for name, source in arms:
            child = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                    "--source", source, "--workdir", str(work), "--seed", str(seed),
                                    "--n", str(args.n)], check=True, capture_output=True, text=True,
                                   encoding="utf-8")
            values = json.loads(child.stdout)
            result[name].append({"seed": seed, **values})
            print(name, seed, json.dumps(values), flush=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
