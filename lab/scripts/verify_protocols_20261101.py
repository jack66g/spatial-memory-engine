"""Real HTTP + MCP stdio validation; subprocesses and stores are isolated."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time

import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

p = argparse.ArgumentParser(__doc__)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
args = p.parse_args()
source = args.source.resolve()
args.out.mkdir(parents=True, exist_ok=True)
results = []
token = secrets.token_urlsafe(24)
SERVER = '''
import sys, uvicorn
from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine
from sme.api.server import create_app
cfg=SMEConfig()
cfg.storage.path=sys.argv[1]
cfg.storage.autosave=False
cfg.persistence.enabled=True
cfg.qapair.enabled=True
cfg.api.auth_token=sys.argv[3]
e=SpatialMemoryEngine(cfg)
uvicorn.run(create_app(e),host="127.0.0.1",port=int(sys.argv[2]),log_level="warning")
'''


async def session(port, directory, label, supplied, authenticated):
    env = {**os.environ, "SME_MCP_PORT": str(port), "PYTHONPATH": str(source),
           "PYTHONDONTWRITEBYTECODE": "1", "NO_PROXY": "127.0.0.1,localhost"}
    env.pop("SME_API_AUTH_TOKEN", None); env.pop("SME_MCP_AUTH_TOKEN", None)
    if supplied is not None:
        env["SME_API_AUTH_TOKEN" if label == "legacy" else "SME_MCP_AUTH_TOKEN"] = supplied
    params = StdioServerParameters(command=sys.executable, args=["-B", str(source/"sme/mcp_server.py")],
                                   cwd=str(directory), env=env)
    log = directory/(label+".log")
    with log.open("w", encoding="utf-8") as err:
        async with stdio_client(params, errlog=err) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=30) as client:
                await client.initialize()
                stats = await client.call_tool("memory_stats", {})
                if authenticated and supplied != token:
                    assert stats.is_error
                    content = str(stats)
                    assert token not in content
                    assert "SME_MCP_AUTH_TOKEN" in content
                    denied = await client.call_tool("remember", {
                        "text": "This rejected write must never enter storage", "ns": label})
                    assert denied.is_error and token not in str(denied)
                    results.append({"case": label, "auth_rejected": True,
                                    "write_rejected": True, "passed": True})
                else:
                    assert not stats.is_error, str(stats)
                    remember = await client.call_tool("remember", {"text": "Isolated ORION-731 code", "ns": label})
                    assert not remember.is_error
                    recalled = await client.call_tool("recall", {"query": "ORION-731", "ns": label})
                    import re
                    ids = re.findall(r"m_[0-9a-f]+", str(recalled))
                    assert ids and "ORION-731" in str(recalled)
                    reinforced = await client.call_tool("reinforce", {"memory_id": ids[0]})
                    assert not reinforced.is_error
                    other = await client.call_tool("recall", {"query": "ORION-731", "ns": label+"_other"})
                    assert "ORION-731" not in str(other)
                    results.append({"case": label, "remember_recall_reinforce_stats": True,
                                    "namespace_isolated": True, "passed": True})
    assert token not in log.read_text(encoding="utf-8")


with tempfile.TemporaryDirectory(prefix="sme-protocol-", dir=args.out) as tmp:
    directory = Path(tmp)
    for authenticated in [False, True]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        env = {**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"}
        with (directory/"server.log").open("w",encoding="utf-8") as log:
            # CREATE_NO_WINDOW keeps the isolated helper invisible on Windows.
            proc = subprocess.Popen([sys.executable,"-B","-c",SERVER,
                                     str(directory/(f"state-{authenticated}.json.gz")), str(port),
                                     token if authenticated else ""], cwd=directory, env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            try:
                url = f"http://127.0.0.1:{port}"
                with httpx.Client(trust_env=False, timeout=10) as http:
                    deadline = time.monotonic()+30
                    while time.monotonic() < deadline:
                        try:
                            if http.get(url+"/health").status_code == 200: break
                        except httpx.HTTPError: pass
                        time.sleep(.1)
                    else: raise RuntimeError("isolated server not ready")
                    headers = {"Authorization": "Bearer "+token} if authenticated else {}
                    def count():
                        return http.get(url+"/stats",headers=headers).json()["memories"]["total"]
                    before = count()
                    bad = http.post(url+"/memories/batch", headers=headers, json={"memories":[
                        {"text":"valid"},{"text":"invalid","metadata":[1]}]})
                    assert bad.status_code == 422 and count()==before
                    results.append({"case": "HTTP batch prevalidation "+str(authenticated), "passed":True})
                    if authenticated:
                        for label, supplied in [("correct",token),("legacy",token),("wrong","wrong-test-token"),("missing",None)]:
                            before = count()
                            asyncio.run(session(port,directory,label,supplied,True))
                            if supplied != token: assert count()==before
                    else:
                        asyncio.run(session(port,directory,"no_auth",None,False))
            finally:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=10)
        assert token not in (directory/"server.log").read_text(encoding="utf-8")
(args.out/"protocols.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"passed": len(results), "temporary_servers_stopped": True}))
