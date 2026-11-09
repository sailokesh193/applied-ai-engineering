from fastapi.testclient import TestClient

import claimlens.api as api

from .conftest import SAMPLE


def test_health_query_and_ingest(local_pipeline):
    api._pipeline = local_pipeline
    client = TestClient(api.app)

    health = client.get("/health").json()
    assert health["status"] == "ok" and health["config"]["cloud"] == "local"

    r = client.post("/query", json={"question": "Which outcome code is used when records are missing?"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] and "PEND" in body["answer"]

    with open(SAMPLE / "clm-1002-medical-record.md", "rb") as f:
        r = client.post("/ingest", files=[("files", ("clm-1002-medical-record.md", f, "text/markdown"))])
    assert r.status_code == 200 and r.json()["chunks"] > 0

    bad = client.post("/ingest", files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))])
    assert bad.status_code == 415
