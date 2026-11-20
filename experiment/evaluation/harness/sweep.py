# -*- coding: utf-8 -*-
"""
扫参编排器:任务队列 → 每任务独立子进程(引擎崩溃不连坐)→ 汇总 CSV/JSON。
用法:
  python sweep.py run jobs.json          # jobs: [{"name": "...", "overrides": {...}}, ...]
  python sweep.py agg results_dir        # 聚合所有 *.json → sweep_summary.csv
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
PY = r"E:\Spatial Memory Engine（SME）\.venv\Scripts\python.exe"


def run_one(job, dataset, workroot, top_k=10, timeout=1800):
    wd = Path(workroot).resolve() / job["name"]
    wd.mkdir(parents=True, exist_ok=True)
    cfg_file = wd / "overrides.json"
    cfg_file.write_text(json.dumps(job.get("overrides", {}), ensure_ascii=False), encoding="utf-8")
    out_file = wd / "result.json"
    cmd = [PY, "-I", str(HERE / "runner.py"),
           "--dataset", dataset, "--config", str(cfg_file), "--out", str(out_file),
           "--workdir", str(wd), "--top-k", str(top_k)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(wd))
    dt = time.time() - t0
    rec = {"name": job["name"], "overrides": job.get("overrides", {}), "wall_s": round(dt, 1),
           "returncode": proc.returncode}
    if out_file.exists():
        try:
            res = json.loads(out_file.read_text(encoding="utf-8"))
            rec.update(res.get("metrics", {}))
            rec["by_type"] = res.get("by_type", {})
            rec["status"] = "ok"
        except json.JSONDecodeError:
            rec["status"] = "bad_result_json"
    else:
        rec["status"] = f"crash(rc={proc.returncode})"
        rec["stderr_tail"] = (proc.stderr or "")[-800:]
    (wd / "meta.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def run_jobs(jobs, dataset, results_dir, workers=2, top_k=10, timeout=1800):
    results = []
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, j, dataset, Path(results_dir) / "work", top_k, timeout): j for j in jobs}
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            done += 1
            m = rec.get("composite", "-")
            print(f"[{done}/{len(jobs)}] {rec['name']:<40} composite={m} ({rec['wall_s']}s, {rec['status']})", flush=True)
    out = Path(results_dir).resolve() / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved ->", out)
    return results


def agg(results_dir, csv_name="sweep_summary.csv"):
    import csv
    results = json.loads((Path(results_dir) / "results.json").read_text(encoding="utf-8"))
    rows = sorted(results, key=lambda r: -r.get("composite", -1))
    keys = ["name", "composite", "hit@1", "hit@3", "hit@5", "hit@10", "mrr", "ndcg@10",
            "recall@5", "lat_p50_ms", "lat_p95_ms", "wall_s", "status"]
    with open(Path(results_dir) / csv_name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"top5 by composite ({Path(results_dir) / csv_name}):")
    for r in rows[:5]:
        print(" ", r["name"], r.get("composite"), r.get("mrr"), r.get("lat_p50_ms"))
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "run":
        jobs = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        dataset = sys.argv[3] if len(sys.argv) > 3 else "small"
        rdir = sys.argv[4] if len(sys.argv) > 4 else str(LAB / "results" / f"sweep_{dataset}_{int(time.time())}")
        workers = int(sys.argv[5]) if len(sys.argv) > 5 else 2
        run_jobs(jobs, dataset, rdir, workers=workers)
    elif sys.argv[1] == "agg":
        agg(sys.argv[2])
