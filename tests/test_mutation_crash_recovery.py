"""Hard process exit after durable mutations and during checkpoint cleanup."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine


PROGRAM = r'''
import json, os, sys
from pathlib import Path
from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine
c=SMEConfig()
c.storage.path=sys.argv[1]
c.storage.backend=sys.argv[2]
c.persistence.enabled=True
c.persistence.sync_mode=sys.argv[3]
e=SpatialMemoryEngine(c)
assert e.load()
ids=list(e.memories)
op=sys.argv[4]
if op=='reinforce': e.reinforce(ids[0])
elif op=='delete': e.delete(ids[0])
elif op=='link': e.link(ids[0],ids[-1],kind='cause',weight=.37,note='durable link')
elif op=='summary': assert e.consolidate()
e.wal.flush()
state={'memories':{mid:m.to_dict() for mid,m in e.memories.items()},
       'graph':e.graph.to_dict(),'modules':e._module_state(),
       'consolidations':e.consolidation.consolidation_count}
Path(sys.argv[5]).write_text(json.dumps(state),encoding='utf-8')
if sys.argv[6]=='before_truncate':
    e.wal.reset=lambda:os._exit(73)
    e.save()
os._exit(73)
'''


@pytest.mark.parametrize("backend", ["json", "sqlite"])
@pytest.mark.parametrize("mode", ["fsync", "grouped"])
@pytest.mark.parametrize("operation", ["reinforce", "delete", "link", "summary"])
@pytest.mark.parametrize("stage", ["after_flush", "before_truncate"])
def test_hard_exit_does_not_lose_or_repeat_committed_mutation(tmp_path, backend, mode, operation, stage):
    c = SMEConfig()
    c.storage.path = str(tmp_path / ("state.db" if backend == "sqlite" else "state.json.gz"))
    c.storage.backend = backend
    c.storage.autosave = False
    c.persistence.enabled = True
    c.persistence.sync_mode = mode
    c.qapair.enabled = True
    c.extraction.store_assistant = True
    c.region.auto_evolve = False
    c.region.min_join_cosine = 0
    c.consolidation.min_group_size = 2
    c.consolidation.similarity_threshold = 0
    e = SpatialMemoryEngine(c)
    e.add("What is the project access code?", ns="alice")
    e.add("SECRET-123", ns="alice", source="assistant")
    e.add("Project Atlas configuration", ns="alice")
    e.save()
    e.wal.close()
    expected_file = tmp_path / "expected.json"
    child = subprocess.run([sys.executable,"-B","-c",PROGRAM,c.storage.path,backend,mode,
                            operation,str(expected_file),stage], cwd=tmp_path,
                           env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1",
                                "PYTHONPATH":str(Path(__file__).resolve().parents[1])},
                           capture_output=True, timeout=30)
    assert child.returncode == 73, child.stderr.decode(errors="replace")
    expected = json.loads(expected_file.read_text(encoding="utf-8"))
    for _ in range(2):
        r = SpatialMemoryEngine(c)
        try:
            assert r.load()
            assert set(r.memories) == set(expected["memories"])
            for mid, before in expected["memories"].items():
                after = r.get(mid).to_dict()
                for key in before.keys() - {"region_id"}:
                    assert before[key] == after[key], (mid,key)
            assert r.graph.to_dict() == expected["graph"]
            assert r._module_state() == expected["modules"]
            assert r.consolidation.consolidation_count == expected["consolidations"]
        finally:
            r.wal.close()
