import base64
import config
from app import app

AUTH = {"Authorization": "Basic " + base64.b64encode(b"tester:secret").decode()}

def test_v2_flag_and_persistent_flask_session(seeded_db, monkeypatch):
    monkeypatch.setenv("APP_USER", "tester"); monkeypatch.setenv("APP_PASS", "secret")
    monkeypatch.setattr(config, "CHAT_ENGINE_V2_ENABLED", True)
    client = app.test_client()
    first = client.post("/ask", json={"query": "الباحث الاجتماعي الثاني"}, headers=AUTH)
    assert first.status_code == 200
    second = client.post("/ask", json={"query": "ومديره؟"}, headers=AUTH)
    assert second.status_code == 200
    assert "Manager A" in second.get_json()["answer"]


def test_current_deployment_guarantees_single_worker_state():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert "--workers 1" in (root / "Procfile").read_text(encoding="utf-8")
    assert "--workers 1" in (root / "railway.toml").read_text(encoding="utf-8")
