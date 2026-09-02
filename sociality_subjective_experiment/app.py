from __future__ import annotations

import csv
import io
import json
import os
import random
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent


def resolve_path(env_name: str, default: str) -> Path:
    raw = Path(os.getenv(env_name, default))
    return raw if raw.is_absolute() else ROOT / raw


DB_PATH = resolve_path("DATABASE_PATH", "data/experiment.sqlite3")
MEDIA_DIR = resolve_path("MEDIA_DIR", "uploads")
CONFIG_PATH = resolve_path("STUDY_CONFIG_PATH", "config/study_seed.json")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")
UNSAFE_ADMIN_KEYS = {"", "replace-with-a-long-random-admin-key"}
ALLOW_PLACEHOLDER_TRIALS = os.getenv("ALLOW_PLACEHOLDER_TRIALS", "1") == "1"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".m4v"}

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Sociality Subjective Experiment", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def load_study_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Study config not found: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def init_db() -> None:
    schema = """
    CREATE TABLE IF NOT EXISTS participants (
        id TEXT PRIMARY KEY,
        consent INTEGER NOT NULL,
        age_band TEXT NOT NULL,
        gender_optional TEXT,
        valid_license INTEGER NOT NULL,
        years_licensed TEXT NOT NULL,
        driving_frequency TEXT NOT NULL,
        annual_mileage_band TEXT,
        urban_driving_frequency TEXT NOT NULL,
        professional_driver INTEGER,
        adas_experience TEXT,
        av_ride_experience TEXT,
        study_familiarity TEXT,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        token TEXT UNIQUE NOT NULL,
        participant_id TEXT NOT NULL,
        study_id TEXT NOT NULL,
        study_version TEXT NOT NULL,
        status TEXT NOT NULL,
        random_seed INTEGER NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        user_agent TEXT,
        device_type TEXT,
        post_task_difficulty INTEGER,
        post_source_guess TEXT,
        post_hypothesis_guess TEXT,
        post_open_feedback TEXT,
        FOREIGN KEY(participant_id) REFERENCES participants(id)
    );

    CREATE TABLE IF NOT EXISTS stimuli (
        id TEXT PRIMARY KEY,
        scenario_id TEXT NOT NULL,
        trajectory_id TEXT,
        internal_name TEXT NOT NULL,
        actor_source TEXT NOT NULL,
        verdict_class TEXT NOT NULL,
        deviation_side TEXT NOT NULL,
        deviation_magnitude REAL,
        role TEXT,
        priority_state TEXT,
        public_instruction TEXT,
        video_path TEXT,
        active INTEGER NOT NULL DEFAULT 1,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS pair_definitions (
        id TEXT PRIMARY KEY,
        scenario_id TEXT NOT NULL,
        stimulus_1_id TEXT NOT NULL,
        stimulus_2_id TEXT NOT NULL,
        contrast_code TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        FOREIGN KEY(stimulus_1_id) REFERENCES stimuli(id),
        FOREIGN KEY(stimulus_2_id) REFERENCES stimuli(id)
    );

    CREATE TABLE IF NOT EXISTS session_trials (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        trial_type TEXT NOT NULL,
        order_index INTEGER NOT NULL,
        block_index INTEGER NOT NULL,
        pair_definition_id TEXT,
        stimulus_id TEXT,
        stimulus_a_id TEXT,
        stimulus_b_id TEXT,
        display_order TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        started_at TEXT,
        completed_at TEXT,
        FOREIGN KEY(session_id) REFERENCES sessions(id),
        FOREIGN KEY(pair_definition_id) REFERENCES pair_definitions(id),
        FOREIGN KEY(stimulus_id) REFERENCES stimuli(id),
        FOREIGN KEY(stimulus_a_id) REFERENCES stimuli(id),
        FOREIGN KEY(stimulus_b_id) REFERENCES stimuli(id),
        UNIQUE(session_id, order_index)
    );

    CREATE TABLE IF NOT EXISTS pairwise_responses (
        id TEXT PRIMARY KEY,
        session_trial_id TEXT UNIQUE NOT NULL,
        preference_raw TEXT NOT NULL,
        preferred_stimulus_id TEXT,
        no_clear_preference INTEGER NOT NULL,
        choice_confidence INTEGER NOT NULL,
        response_time_ms INTEGER NOT NULL,
        playback_a_complete INTEGER NOT NULL,
        playback_b_complete INTEGER NOT NULL,
        replay_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY(session_trial_id) REFERENCES session_trials(id)
    );

    CREATE TABLE IF NOT EXISTS single_responses (
        id TEXT PRIMARY KEY,
        session_trial_id TEXT UNIQUE NOT NULL,
        acceptability INTEGER NOT NULL,
        predictability INTEGER NOT NULL,
        comfort INTEGER NOT NULL,
        perceived_unsafe INTEGER NOT NULL,
        interaction_burden INTEGER NOT NULL,
        assertiveness INTEGER NOT NULL,
        hesitation INTEGER NOT NULL,
        courtesy INTEGER,
        rating_confidence INTEGER,
        free_text_reason TEXT,
        response_time_ms INTEGER NOT NULL,
        playback_complete INTEGER NOT NULL,
        replay_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY(session_trial_id) REFERENCES session_trials(id)
    );

    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        session_trial_id TEXT,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(session_id) REFERENCES sessions(id),
        FOREIGN KEY(session_trial_id) REFERENCES session_trials(id)
    );

    CREATE INDEX IF NOT EXISTS idx_trials_session_status ON session_trials(session_id, status, order_index);
    CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, created_at);
    """
    cfg = load_study_config()
    with db_conn() as conn:
        conn.executescript(schema)
        if conn.execute("SELECT COUNT(*) AS n FROM stimuli").fetchone()["n"] == 0:
            for s in cfg.get("stimuli", []):
                now = utcnow()
                conn.execute(
                    """
                    INSERT INTO stimuli (
                        id, scenario_id, trajectory_id, internal_name, actor_source,
                        verdict_class, deviation_side, deviation_magnitude, role,
                        priority_state, public_instruction, video_path, active,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        s["id"], s["scenario_id"], s.get("trajectory_id"), s["internal_name"],
                        s["actor_source"], s["verdict_class"], s.get("deviation_side", "none"),
                        s.get("deviation_magnitude"), s.get("role"), s.get("priority_state"),
                        s.get("public_instruction", ""), s.get("video_path", ""),
                        int(bool(s.get("active", True))), json.dumps(s.get("metadata", {}), ensure_ascii=False),
                        now, now,
                    ),
                )
            for p in cfg.get("pairs", []):
                conn.execute(
                    """
                    INSERT INTO pair_definitions (
                        id, scenario_id, stimulus_1_id, stimulus_2_id,
                        contrast_code, active, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (p["id"], p["scenario_id"], p["stimulus_1_id"], p["stimulus_2_id"],
                     p["contrast_code"], int(bool(p.get("active", True))), utcnow()),
                )


class StartSessionRequest(BaseModel):
    consent: bool
    age_band: str = Field(min_length=1, max_length=30)
    gender_optional: Optional[str] = Field(default=None, max_length=30)
    valid_license: bool
    years_licensed: str = Field(min_length=1, max_length=30)
    driving_frequency: str = Field(min_length=1, max_length=50)
    annual_mileage_band: Optional[str] = Field(default=None, max_length=50)
    urban_driving_frequency: str = Field(min_length=1, max_length=30)
    professional_driver: Optional[bool] = None
    user_agent: Optional[str] = Field(default=None, max_length=500)
    device_type: Optional[str] = Field(default=None, max_length=100)


class PairwiseResponseRequest(BaseModel):
    trial_id: str
    preference_raw: Literal["A", "B", "NO_PREFERENCE"]
    choice_confidence: int = Field(ge=1, le=5)
    response_time_ms: int = Field(ge=0, le=3_600_000)
    playback_a_complete: bool
    playback_b_complete: bool
    replay_count: int = Field(ge=0, le=20)


class SingleResponseRequest(BaseModel):
    trial_id: str
    acceptability: int = Field(ge=1, le=7)
    predictability: int = Field(ge=1, le=7)
    comfort: int = Field(ge=1, le=7)
    perceived_unsafe: int = Field(ge=1, le=7)
    interaction_burden: int = Field(ge=1, le=7)
    assertiveness: int = Field(ge=1, le=7)
    hesitation: int = Field(ge=1, le=7)
    courtesy: Optional[int] = Field(default=None, ge=1, le=7)
    rating_confidence: Optional[int] = Field(default=None, ge=1, le=5)
    free_text_reason: Optional[str] = Field(default=None, max_length=2000)
    response_time_ms: int = Field(ge=0, le=3_600_000)
    playback_complete: bool
    replay_count: int = Field(ge=0, le=20)


class PostSessionRequest(BaseModel):
    task_difficulty: int = Field(ge=1, le=7)
    source_guess: str = Field(max_length=200)
    hypothesis_guess: Optional[str] = Field(default=None, max_length=1000)
    open_feedback: Optional[str] = Field(default=None, max_length=4000)
    adas_experience: Optional[str] = Field(default=None, max_length=500)
    av_ride_experience: Optional[str] = Field(default=None, max_length=100)
    study_familiarity: Optional[str] = Field(default=None, max_length=200)


class EventRequest(BaseModel):
    trial_id: Optional[str] = None
    event_type: str = Field(min_length=1, max_length=100)
    payload: dict[str, Any] = Field(default_factory=dict)


class StimulusUpdateRequest(BaseModel):
    scenario_id: str = Field(min_length=1, max_length=100)
    trajectory_id: Optional[str] = Field(default=None, max_length=100)
    internal_name: str = Field(min_length=1, max_length=200)
    actor_source: Literal["human", "av"]
    verdict_class: Literal["inside", "outside"]
    deviation_side: Literal["none", "assertive", "accommodating"]
    deviation_magnitude: Optional[float] = None
    role: Optional[str] = Field(default=None, max_length=100)
    priority_state: Optional[str] = Field(default=None, max_length=100)
    public_instruction: Optional[str] = Field(default=None, max_length=1000)
    active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class PairCreateRequest(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    scenario_id: str = Field(min_length=1, max_length=100)
    stimulus_1_id: str
    stimulus_2_id: str
    contrast_code: str = Field(min_length=1, max_length=200)
    active: bool = True


def session_by_token(conn: sqlite3.Connection, token: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return row


def require_admin(x_admin_key: Optional[str] = Header(default=None)) -> None:
    if ADMIN_KEY in UNSAFE_ADMIN_KEYS or not x_admin_key or not secrets.compare_digest(x_admin_key, ADMIN_KEY):
        raise HTTPException(status_code=401, detail="Invalid admin key")


def safe_media_url(video_path: Optional[str]) -> Optional[str]:
    return f"/media/{Path(video_path).name}" if video_path else None


def balanced_select(rows: list[sqlite3.Row], n: int, rng: random.Random, key_fields: tuple[str, ...]) -> list[sqlite3.Row]:
    groups: dict[tuple[Any, ...], list[sqlite3.Row]] = {}
    for row in rows:
        groups.setdefault(tuple(row[f] for f in key_fields), []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    keys = list(groups)
    rng.shuffle(keys)
    selected: list[sqlite3.Row] = []
    while len(selected) < n and any(groups[k] for k in keys):
        for key in keys:
            if groups[key] and len(selected) < n:
                selected.append(groups[key].pop())
    return selected


def create_session_trials(conn: sqlite3.Connection, session_id: str, seed: int, cfg: dict[str, Any]) -> int:
    rng = random.Random(seed)
    pair_rows = conn.execute(
        """
        SELECT p.* FROM pair_definitions p
        JOIN stimuli s1 ON s1.id=p.stimulus_1_id
        JOIN stimuli s2 ON s2.id=p.stimulus_2_id
        WHERE p.active=1 AND s1.active=1 AND s2.active=1
        """
    ).fetchall()
    pair_selected = balanced_select(list(pair_rows), min(cfg.get("pairwise_trials_per_session", 12), len(pair_rows)), rng, ("contrast_code",))
    stim_rows = conn.execute("SELECT * FROM stimuli WHERE active=1").fetchall()
    single_selected = balanced_select(list(stim_rows), min(cfg.get("single_trials_per_session", 18), len(stim_rows)), rng, ("actor_source", "verdict_class", "deviation_side"))
    blocks = [("pairwise", pair_selected), ("single", single_selected)] if cfg.get("pairwise_first", True) else [("single", single_selected), ("pairwise", pair_selected)]
    order = 0
    for block_index, (trial_type, items) in enumerate(blocks, start=1):
        rng.shuffle(items)
        for item in items:
            order += 1
            trial_id = str(uuid.uuid4())
            if trial_type == "pairwise":
                swap = bool(rng.getrandbits(1))
                a_id = item["stimulus_2_id"] if swap else item["stimulus_1_id"]
                b_id = item["stimulus_1_id"] if swap else item["stimulus_2_id"]
                conn.execute(
                    """
                    INSERT INTO session_trials(id,session_id,trial_type,order_index,block_index,pair_definition_id,stimulus_a_id,stimulus_b_id,display_order,status)
                    VALUES(?,?,'pairwise',?,?,?,?,?,?,'pending')
                    """,
                    (trial_id, session_id, order, block_index, item["id"], a_id, b_id, "swapped" if swap else "original"),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO session_trials(id,session_id,trial_type,order_index,block_index,stimulus_id,status)
                    VALUES(?,?,'single',?,?,?,'pending')
                    """,
                    (trial_id, session_id, order, block_index, item["id"]),
                )
    return order

@app.get("/", response_class=HTMLResponse)
def participant_page() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/work", response_class=HTMLResponse)
def work_page() -> FileResponse:
    return FileResponse(ROOT / "static" / "work.html")


@app.get("/media/{filename}")
def media(filename: str) -> FileResponse:
    path = MEDIA_DIR / Path(filename).name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "database": str(DB_PATH), "time": utcnow()}


@app.get("/api/public/config")
def public_config() -> dict[str, Any]:
    cfg = load_study_config()
    return {
        "study_id": cfg["study_id"], "version": cfg["version"],
        "title": cfg["title"], "subtitle": cfg.get("subtitle", ""),
        "consent_text": cfg.get("consent_text", ""),
        "participant_instruction": cfg.get("participant_instruction", ""),
        "pairwise_first": bool(cfg.get("pairwise_first", True)),
        "max_replays": int(cfg.get("max_replays", 1)),
        "allow_placeholder_trials": ALLOW_PLACEHOLDER_TRIALS,
    }


@app.post("/api/session/start")
def start_session(req: StartSessionRequest) -> dict[str, Any]:
    if not req.consent:
        raise HTTPException(status_code=400, detail="Consent is required")
    if not req.valid_license:
        raise HTTPException(status_code=400, detail="A valid driving licence is required")
    cfg = load_study_config()
    participant_id, session_id = str(uuid.uuid4()), str(uuid.uuid4())
    token, seed = secrets.token_urlsafe(32), secrets.randbelow(2_000_000_000)
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO participants(id,consent,age_band,gender_optional,valid_license,years_licensed,driving_frequency,annual_mileage_band,urban_driving_frequency,professional_driver,created_at)
            VALUES(?,1,?,?,1,?,?,?,?,?,?)
            """,
            (participant_id, req.age_band, req.gender_optional, req.years_licensed,
             req.driving_frequency, req.annual_mileage_band, req.urban_driving_frequency,
             None if req.professional_driver is None else int(req.professional_driver), utcnow()),
        )
        conn.execute(
            """
            INSERT INTO sessions(id,token,participant_id,study_id,study_version,status,random_seed,started_at,user_agent,device_type)
            VALUES(?,?,?,?,?,'active',?,?,?,?)
            """,
            (session_id, token, participant_id, cfg["study_id"], cfg["version"], seed, utcnow(), req.user_agent, req.device_type),
        )
        total = create_session_trials(conn, session_id, seed, cfg)
        conn.execute("INSERT INTO events(id,session_id,event_type,payload_json,created_at) VALUES(?,?,'session_started',?,?)",
                     (str(uuid.uuid4()), session_id, json.dumps({"total_trials": total}), utcnow()))
    return {"session_id": session_id, "token": token, "total_trials": total, "study_version": cfg["version"]}


@app.get("/api/session/{token}/state")
def session_state(token: str) -> dict[str, Any]:
    with db_conn() as conn:
        session = session_by_token(conn, token)
        totals = conn.execute(
            "SELECT COUNT(*) total,SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) completed FROM session_trials WHERE session_id=?",
            (session["id"],),
        ).fetchone()
        completed, total = totals["completed"] or 0, totals["total"] or 0
        if session["status"] == "completed":
            return {"status": "completed", "completed": completed, "total": total}
        trial = conn.execute("SELECT * FROM session_trials WHERE session_id=? AND status!='completed' ORDER BY order_index LIMIT 1", (session["id"],)).fetchone()
        if not trial:
            return {"status": "post", "completed": completed, "total": total}
        if trial["status"] == "pending":
            conn.execute("UPDATE session_trials SET status='started',started_at=? WHERE id=?", (utcnow(), trial["id"]))
        out = {
            "status": "trial", "completed": completed, "total": total,
            "progress_percent": round(100 * completed / max(total, 1), 1),
            "trial": {"id": trial["id"], "type": trial["trial_type"], "order_index": trial["order_index"], "block_index": trial["block_index"]},
        }
        if trial["trial_type"] == "pairwise":
            a = conn.execute("SELECT scenario_id,public_instruction,video_path FROM stimuli WHERE id=?", (trial["stimulus_a_id"],)).fetchone()
            b = conn.execute("SELECT scenario_id,public_instruction,video_path FROM stimuli WHERE id=?", (trial["stimulus_b_id"],)).fetchone()
            out["trial"].update({
                "scenario_label": f"场景 {a['scenario_id']}",
                "instruction": a["public_instruction"] or b["public_instruction"] or "请评价目标车辆。",
                "video_a_url": safe_media_url(a["video_path"]), "video_b_url": safe_media_url(b["video_path"]),
            })
        else:
            s = conn.execute("SELECT scenario_id,public_instruction,video_path FROM stimuli WHERE id=?", (trial["stimulus_id"],)).fetchone()
            out["trial"].update({
                "scenario_label": f"场景 {s['scenario_id']}",
                "instruction": s["public_instruction"] or "请评价目标车辆。",
                "video_url": safe_media_url(s["video_path"]),
            })
        return out


@app.post("/api/session/{token}/pairwise")
def submit_pairwise(token: str, req: PairwiseResponseRequest) -> dict[str, Any]:
    if not req.playback_a_complete or not req.playback_b_complete:
        raise HTTPException(status_code=400, detail="Both videos must be completed")
    with db_conn() as conn:
        session = session_by_token(conn, token)
        trial = conn.execute("SELECT * FROM session_trials WHERE id=? AND session_id=?", (req.trial_id, session["id"])).fetchone()
        if not trial or trial["trial_type"] != "pairwise":
            raise HTTPException(status_code=404, detail="Pairwise trial not found")
        if conn.execute("SELECT 1 FROM pairwise_responses WHERE session_trial_id=?", (trial["id"],)).fetchone():
            raise HTTPException(status_code=409, detail="Trial already submitted")
        preferred = trial["stimulus_a_id"] if req.preference_raw == "A" else trial["stimulus_b_id"] if req.preference_raw == "B" else None
        conn.execute(
            """
            INSERT INTO pairwise_responses(id,session_trial_id,preference_raw,preferred_stimulus_id,no_clear_preference,choice_confidence,response_time_ms,playback_a_complete,playback_b_complete,replay_count,created_at)
            VALUES(?,?,?,?,?,?,?,1,1,?,?)
            """,
            (str(uuid.uuid4()), trial["id"], req.preference_raw, preferred, int(req.preference_raw == "NO_PREFERENCE"),
             req.choice_confidence, req.response_time_ms, req.replay_count, utcnow()),
        )
        conn.execute("UPDATE session_trials SET status='completed',completed_at=? WHERE id=?", (utcnow(), trial["id"]))
    return {"ok": True}


@app.post("/api/session/{token}/single")
def submit_single(token: str, req: SingleResponseRequest) -> dict[str, Any]:
    if not req.playback_complete:
        raise HTTPException(status_code=400, detail="Video must be completed")
    with db_conn() as conn:
        session = session_by_token(conn, token)
        trial = conn.execute("SELECT * FROM session_trials WHERE id=? AND session_id=?", (req.trial_id, session["id"])).fetchone()
        if not trial or trial["trial_type"] != "single":
            raise HTTPException(status_code=404, detail="Single trial not found")
        if conn.execute("SELECT 1 FROM single_responses WHERE session_trial_id=?", (trial["id"],)).fetchone():
            raise HTTPException(status_code=409, detail="Trial already submitted")
        conn.execute(
            """
            INSERT INTO single_responses(id,session_trial_id,acceptability,predictability,comfort,perceived_unsafe,interaction_burden,assertiveness,hesitation,courtesy,rating_confidence,free_text_reason,response_time_ms,playback_complete,replay_count,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)
            """,
            (str(uuid.uuid4()), trial["id"], req.acceptability, req.predictability, req.comfort,
             req.perceived_unsafe, req.interaction_burden, req.assertiveness, req.hesitation,
             req.courtesy, req.rating_confidence, req.free_text_reason, req.response_time_ms, req.replay_count, utcnow()),
        )
        conn.execute("UPDATE session_trials SET status='completed',completed_at=? WHERE id=?", (utcnow(), trial["id"]))
    return {"ok": True}


@app.post("/api/session/{token}/post")
def submit_post(token: str, req: PostSessionRequest) -> dict[str, Any]:
    with db_conn() as conn:
        session = session_by_token(conn, token)
        conn.execute("UPDATE sessions SET post_task_difficulty=?,post_source_guess=?,post_hypothesis_guess=?,post_open_feedback=? WHERE id=?",
                     (req.task_difficulty, req.source_guess, req.hypothesis_guess, req.open_feedback, session["id"]))
        conn.execute("UPDATE participants SET adas_experience=?,av_ride_experience=?,study_familiarity=? WHERE id=?",
                     (req.adas_experience, req.av_ride_experience, req.study_familiarity, session["participant_id"]))
    return {"ok": True}


@app.post("/api/session/{token}/complete")
def complete_session(token: str) -> dict[str, Any]:
    with db_conn() as conn:
        session = session_by_token(conn, token)
        remaining = conn.execute("SELECT COUNT(*) n FROM session_trials WHERE session_id=? AND status!='completed'", (session["id"],)).fetchone()["n"]
        if remaining:
            raise HTTPException(status_code=400, detail=f"{remaining} trials remain")
        conn.execute("UPDATE sessions SET status='completed',completed_at=? WHERE id=?", (utcnow(), session["id"]))
        conn.execute("INSERT INTO events(id,session_id,event_type,payload_json,created_at) VALUES(?,?,'session_completed','{}',?)",
                     (str(uuid.uuid4()), session["id"], utcnow()))
    return {"ok": True}


@app.post("/api/session/{token}/event")
def log_event(token: str, req: EventRequest) -> dict[str, Any]:
    with db_conn() as conn:
        session = session_by_token(conn, token)
        trial_id = None
        if req.trial_id:
            row = conn.execute("SELECT id FROM session_trials WHERE id=? AND session_id=?", (req.trial_id, session["id"])).fetchone()
            trial_id = row["id"] if row else None
        conn.execute("INSERT INTO events(id,session_id,session_trial_id,event_type,payload_json,created_at) VALUES(?,?,?,?,?,?)",
                     (str(uuid.uuid4()), session["id"], trial_id, req.event_type, json.dumps(req.payload, ensure_ascii=False), utcnow()))
    return {"ok": True}

@app.get("/api/admin/summary", dependencies=[Depends(require_admin)])
def admin_summary() -> dict[str, Any]:
    with db_conn() as conn:
        summary = {
            "participants": conn.execute("SELECT COUNT(*) n FROM participants").fetchone()["n"],
            "sessions_started": conn.execute("SELECT COUNT(*) n FROM sessions").fetchone()["n"],
            "sessions_completed": conn.execute("SELECT COUNT(*) n FROM sessions WHERE status='completed'").fetchone()["n"],
            "pairwise_responses": conn.execute("SELECT COUNT(*) n FROM pairwise_responses").fetchone()["n"],
            "single_responses": conn.execute("SELECT COUNT(*) n FROM single_responses").fetchone()["n"],
            "stimuli": conn.execute("SELECT COUNT(*) n FROM stimuli").fetchone()["n"],
            "active_stimuli": conn.execute("SELECT COUNT(*) n FROM stimuli WHERE active=1").fetchone()["n"],
            "missing_videos": conn.execute("SELECT COUNT(*) n FROM stimuli WHERE active=1 AND COALESCE(video_path,'')='' ").fetchone()["n"],
        }
        rows = conn.execute(
            """
            SELECT s.id,s.status,s.started_at,s.completed_at,COUNT(t.id) total_trials,
                   SUM(CASE WHEN t.status='completed' THEN 1 ELSE 0 END) completed_trials
            FROM sessions s LEFT JOIN session_trials t ON t.session_id=s.id
            GROUP BY s.id ORDER BY s.started_at DESC LIMIT 20
            """
        ).fetchall()
        summary["recent_sessions"] = [dict(r) for r in rows]
        return summary


@app.get("/api/admin/stimuli", dependencies=[Depends(require_admin)])
def admin_stimuli() -> list[dict[str, Any]]:
    with db_conn() as conn:
        rows = conn.execute("SELECT * FROM stimuli ORDER BY scenario_id,actor_source,verdict_class,deviation_side").fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["active"] = bool(d["active"])
            d["media_url"] = safe_media_url(d.get("video_path"))
            d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
            out.append(d)
        return out


@app.put("/api/admin/stimuli/{stimulus_id}", dependencies=[Depends(require_admin)])
def update_stimulus(stimulus_id: str, req: StimulusUpdateRequest) -> dict[str, Any]:
    with db_conn() as conn:
        if not conn.execute("SELECT 1 FROM stimuli WHERE id=?", (stimulus_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Stimulus not found")
        conn.execute(
            """
            UPDATE stimuli SET scenario_id=?,trajectory_id=?,internal_name=?,actor_source=?,verdict_class=?,deviation_side=?,deviation_magnitude=?,role=?,priority_state=?,public_instruction=?,active=?,metadata_json=?,updated_at=? WHERE id=?
            """,
            (req.scenario_id, req.trajectory_id, req.internal_name, req.actor_source, req.verdict_class,
             req.deviation_side, req.deviation_magnitude, req.role, req.priority_state,
             req.public_instruction, int(req.active), json.dumps(req.metadata, ensure_ascii=False), utcnow(), stimulus_id),
        )
    return {"ok": True}


@app.post("/api/admin/stimuli/{stimulus_id}/video", dependencies=[Depends(require_admin)])
def upload_video(stimulus_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Allowed types: {', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}")
    with db_conn() as conn:
        row = conn.execute("SELECT video_path FROM stimuli WHERE id=?", (stimulus_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Stimulus not found")
        old = row["video_path"]
    target_name = f"{uuid.uuid4().hex}{suffix}"
    target = MEDIA_DIR / target_name
    size = 0
    try:
        with target.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds 500 MB")
                out.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    with db_conn() as conn:
        conn.execute("UPDATE stimuli SET video_path=?,updated_at=? WHERE id=?", (target_name, utcnow(), stimulus_id))
    if old:
        old_path = MEDIA_DIR / Path(old).name
        if old_path.exists() and old_path != target:
            old_path.unlink(missing_ok=True)
    return {"ok": True, "media_url": safe_media_url(target_name), "bytes": size}


@app.get("/api/admin/pairs", dependencies=[Depends(require_admin)])
def admin_pairs() -> list[dict[str, Any]]:
    with db_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM pair_definitions ORDER BY scenario_id,contrast_code").fetchall()]


@app.post("/api/admin/pairs", dependencies=[Depends(require_admin)])
def create_pair(req: PairCreateRequest) -> dict[str, Any]:
    if req.stimulus_1_id == req.stimulus_2_id:
        raise HTTPException(status_code=400, detail="Pair stimuli must differ")
    with db_conn() as conn:
        for sid in (req.stimulus_1_id, req.stimulus_2_id):
            if not conn.execute("SELECT 1 FROM stimuli WHERE id=?", (sid,)).fetchone():
                raise HTTPException(status_code=400, detail=f"Unknown stimulus: {sid}")
        try:
            conn.execute(
                "INSERT INTO pair_definitions(id,scenario_id,stimulus_1_id,stimulus_2_id,contrast_code,active,created_at) VALUES(?,?,?,?,?,?,?)",
                (req.id, req.scenario_id, req.stimulus_1_id, req.stimulus_2_id, req.contrast_code, int(req.active), utcnow()),
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="Pair ID already exists") from exc
    return {"ok": True}


EXPORT_TABLES = {
    "participants": "participants", "sessions": "sessions", "stimuli": "stimuli",
    "pairs": "pair_definitions", "session_trials": "session_trials",
    "pairwise_responses": "pairwise_responses", "single_responses": "single_responses",
    "events": "events",
}


@app.get("/api/admin/export/{table_name}.csv", dependencies=[Depends(require_admin)])
def export_csv(table_name: str) -> StreamingResponse:
    table = EXPORT_TABLES.get(table_name)
    if not table:
        raise HTTPException(status_code=404, detail="Unknown export")
    with db_conn() as conn:
        cursor = conn.execute(f"SELECT * FROM {table}")
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(columns)
    writer.writerows([[row[col] for col in columns] for row in rows])
    data = output.getvalue().encode("utf-8-sig")
    return StreamingResponse(iter([data]), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="{table_name}.csv"'})


init_db()
