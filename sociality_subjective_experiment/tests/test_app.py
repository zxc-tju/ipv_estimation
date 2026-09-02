from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def build_client(tmp_path: Path, admin_key: str = "test-admin-key"):
    root = Path(__file__).resolve().parents[1]
    os.environ["DATABASE_PATH"] = str(tmp_path / "test.sqlite3")
    os.environ["MEDIA_DIR"] = str(tmp_path / "uploads")
    os.environ["STUDY_CONFIG_PATH"] = str(root / "config" / "study_seed.json")
    os.environ["ADMIN_KEY"] = admin_key
    os.environ["ALLOW_PLACEHOLDER_TRIALS"] = "1"
    sys.path.insert(0, str(root))
    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    return TestClient(module.app)


def start_payload():
    return {
        "consent": True,
        "age_band": "25–34",
        "gender_optional": "",
        "valid_license": True,
        "years_licensed": "4–7年",
        "driving_frequency": "每周数次",
        "annual_mileage_band": "5,001–10,000 km",
        "urban_driving_frequency": "经常",
        "professional_driver": False,
        "user_agent": "pytest",
        "device_type": "desktop",
    }


def test_full_session_and_exports(tmp_path):
    client = build_client(tmp_path)
    assert client.get("/").status_code == 200
    assert client.get("/work").status_code == 200
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/public/config").json()["pairwise_first"] is True

    started = client.post("/api/session/start", json=start_payload())
    assert started.status_code == 200, started.text
    token = started.json()["token"]
    assert started.json()["total_trials"] == 30

    pair_n = single_n = 0
    while True:
        state = client.get(f"/api/session/{token}/state")
        assert state.status_code == 200, state.text
        payload = state.json()
        if payload["status"] == "post":
            break
        trial = payload["trial"]
        # Participant payload must not expose hidden source/verdict fields.
        assert "actor_source" not in trial
        assert "verdict_class" not in trial
        assert "deviation_side" not in trial
        if trial["type"] == "pairwise":
            pair_n += 1
            r = client.post(f"/api/session/{token}/pairwise", json={
                "trial_id": trial["id"],
                "preference_raw": "NO_PREFERENCE",
                "choice_confidence": 3,
                "response_time_ms": 1200,
                "playback_a_complete": True,
                "playback_b_complete": True,
                "replay_count": 0,
            })
        else:
            single_n += 1
            r = client.post(f"/api/session/{token}/single", json={
                "trial_id": trial["id"],
                "acceptability": 4,
                "predictability": 4,
                "comfort": 4,
                "perceived_unsafe": 2,
                "interaction_burden": 3,
                "assertiveness": 3,
                "hesitation": 3,
                "courtesy": None,
                "rating_confidence": 4,
                "free_text_reason": None,
                "response_time_ms": 1400,
                "playback_complete": True,
                "replay_count": 0,
            })
        assert r.status_code == 200, r.text

    assert pair_n == 12
    assert single_n == 18

    post = client.post(f"/api/session/{token}/post", json={
        "task_difficulty": 3,
        "source_guess": "完全无法判断",
        "hypothesis_guess": "驾驶互动评价",
        "open_feedback": "",
        "adas_experience": "ACC",
        "av_ride_experience": "从未",
        "study_familiarity": "完全不了解",
    })
    assert post.status_code == 200, post.text
    assert client.post(f"/api/session/{token}/complete", json={}).status_code == 200
    assert client.get(f"/api/session/{token}/state").json()["status"] == "completed"

    headers = {"X-Admin-Key": "test-admin-key"}
    summary = client.get("/api/admin/summary", headers=headers)
    assert summary.status_code == 200
    assert summary.json()["sessions_completed"] == 1
    assert summary.json()["pairwise_responses"] == 12
    assert summary.json()["single_responses"] == 18
    assert summary.json()["missing_videos"] == 18

    for table in ("participants", "sessions", "session_trials", "pairwise_responses", "single_responses", "stimuli", "pairs"):
        exported = client.get(f"/api/admin/export/{table}.csv", headers=headers)
        assert exported.status_code == 200
        assert len(exported.content) > 20


def test_admin_auth_and_validation(tmp_path):
    client = build_client(tmp_path)
    assert client.get("/api/admin/summary").status_code == 401
    assert client.get("/api/admin/summary", headers={"X-Admin-Key": "wrong"}).status_code == 401
    bad = start_payload()
    bad["valid_license"] = False
    assert client.post("/api/session/start", json=bad).status_code == 400


def test_placeholder_admin_key_is_rejected(tmp_path):
    client = build_client(tmp_path, admin_key="replace-with-a-long-random-admin-key")
    headers = {"X-Admin-Key": "replace-with-a-long-random-admin-key"}
    assert client.get("/api/admin/summary", headers=headers).status_code == 401
