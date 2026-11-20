# -*- coding: utf-8 -*-
"""P2-3 深度复现:热重建改 mrl_dim → 新旧维度混存 → 写入/检索/演化路径崩溃链"""
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fastapi.testclient import TestClient  # noqa: E402
from sme.api.server import create_app  # noqa: E402
from sme import SpatialMemoryEngine, SMEConfig  # noqa: E402

td = tempfile.mkdtemp(dir=str(HERE.parent / "work"))
cfg = SMEConfig()
cfg.storage.autosave = False
cfg.storage.path = str(Path(td) / "state.json")
e = SpatialMemoryEngine(cfg)
client = TestClient(create_app(e, config_file=str(Path(td) / "c.json")))

for i in range(60):
    client.post("/memories", json={"text": f"记忆{i}关于主题词{i%30}的内容"})

r = client.put("/config", json={"values": {"embedding.mrl_dim": 32}})
print("PUT mrl_dim=32:", r.status_code, r.text[:150])

# 热重建后继续写 → 新记忆 embedding 应为 32 维,与旧 64 维共存
r2 = client.post("/memories", json={"text": "热重建之后的新记忆"})
print("新增记忆:", r2.status_code, r2.text[:150])
for k in range(5):
    client.post("/memories", json={"text": f"热重建后记忆{k}填充内容"})

# 检查实际维度分布
dims = {}
for m in e.memories.values():
    if m.embedding is not None:
        d = len(m.embedding)
        dims[d] = dims.get(d, 0) + 1
print("库内向量维度分布:", dims)

rs = client.post("/memories/search", json={"text": "主题", "top_k": 10})
print("检索:", rs.status_code, (rs.text[:250] if rs.status_code != 200 else "OK"))

re_ = client.post("/regions/evolve")
print("演化:", re_.status_code, (re_.text[:250] if re_.status_code != 200 else "OK"))

# 再触发一次空间插入(走 update_geometry)
try:
    e.add("直接SDK再写一条触发几何更新")
    print("SDK再写: OK")
except Exception as ex:
    print(f"SDK再写: CRASH {type(ex).__name__}: {str(ex)[:200]}")
