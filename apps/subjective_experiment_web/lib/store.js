'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { DatabaseSync } = require('node:sqlite');

const SCHEMA = `
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS participants (
  participant_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT,
  status TEXT NOT NULL DEFAULT 'started',
  consent_version TEXT,
  consented_at TEXT,
  age_band TEXT,
  gender TEXT,
  valid_licence INTEGER,
  years_licensed TEXT,
  driving_frequency TEXT,
  annual_mileage_band TEXT,
  urban_driving_frequency TEXT,
  professional_driver TEXT,
  device_type TEXT,
  post_survey_json TEXT
);
CREATE TABLE IF NOT EXISTS study_sessions (
  session_id TEXT PRIMARY KEY,
  participant_id TEXT NOT NULL UNIQUE,
  study_id TEXT NOT NULL,
  study_version TEXT NOT NULL,
  csrf_token TEXT NOT NULL,
  current_stage TEXT NOT NULL,
  current_index INTEGER NOT NULL DEFAULT 0,
  pairwise_count INTEGER NOT NULL,
  single_count INTEGER NOT NULL,
  break_taken INTEGER NOT NULL DEFAULT 0,
  assignment_json TEXT NOT NULL,
  assignment_hash TEXT NOT NULL,
  started_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT,
  visibility_loss_count INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY(participant_id) REFERENCES participants(participant_id)
);
CREATE TABLE IF NOT EXISTS trials (
  trial_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  sequence_index INTEGER NOT NULL,
  block TEXT NOT NULL,
  trial_type TEXT NOT NULL,
  comparison_id TEXT,
  matched_set_id TEXT,
  stimulus_a_id TEXT NOT NULL,
  stimulus_b_id TEXT,
  started_at TEXT,
  completed_at TEXT,
  playback_a_complete INTEGER NOT NULL DEFAULT 0,
  playback_b_complete INTEGER NOT NULL DEFAULT 0,
  replay_count INTEGER NOT NULL DEFAULT 0,
  response_time_ms INTEGER,
  UNIQUE(session_id, sequence_index),
  FOREIGN KEY(session_id) REFERENCES study_sessions(session_id)
);
CREATE TABLE IF NOT EXISTS pairwise_responses (
  trial_id TEXT PRIMARY KEY,
  preference_raw TEXT NOT NULL,
  preferred_stimulus_id TEXT,
  no_clear_preference INTEGER NOT NULL,
  confidence INTEGER NOT NULL,
  free_text_reason TEXT,
  submitted_at TEXT NOT NULL,
  FOREIGN KEY(trial_id) REFERENCES trials(trial_id)
);
CREATE TABLE IF NOT EXISTS single_responses (
  trial_id TEXT PRIMARY KEY,
  acceptability INTEGER NOT NULL,
  comfort INTEGER NOT NULL,
  predictability INTEGER NOT NULL,
  interaction_burden INTEGER NOT NULL,
  perceived_unsafe INTEGER NOT NULL,
  too_aggressive INTEGER NOT NULL,
  too_cautious INTEGER NOT NULL,
  confidence INTEGER,
  free_text_reason TEXT,
  submitted_at TEXT NOT NULL,
  FOREIGN KEY(trial_id) REFERENCES trials(trial_id)
);
CREATE TABLE IF NOT EXISTS event_logs (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  participant_id TEXT,
  trial_id TEXT,
  event_type TEXT NOT NULL,
  payload_json TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trials_session ON trials(session_id, sequence_index);
CREATE INDEX IF NOT EXISTS idx_events_session ON event_logs(session_id, created_at);
`;

function utcNow() {
  return new Date().toISOString();
}

function ensureParent(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

class Store {
  constructor(databasePath) {
    ensureParent(databasePath);
    this.databasePath = databasePath;
    this.db = new DatabaseSync(databasePath);
    this.db.exec('PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;');
    this.db.exec(SCHEMA);
  }

  close() {
    this.db.close();
  }

  transaction(callback) {
    this.db.exec('BEGIN IMMEDIATE');
    try {
      const result = callback();
      this.db.exec('COMMIT');
      return result;
    } catch (error) {
      this.db.exec('ROLLBACK');
      throw error;
    }
  }

  createParticipantAndSession({
    participantId,
    sessionId,
    csrfToken,
    studyId,
    studyVersion,
    assignment,
    assignmentHash,
  }) {
    const now = utcNow();
    const pairwiseCount = assignment.filter((item) => item.trial_type === 'pairwise').length;
    const singleCount = assignment.filter((item) => item.trial_type === 'single').length;
    this.transaction(() => {
      this.db.prepare(`
        INSERT INTO participants (participant_id, created_at, updated_at, status)
        VALUES (?, ?, ?, 'started')
      `).run(participantId, now, now);
      this.db.prepare(`
        INSERT INTO study_sessions (
          session_id, participant_id, study_id, study_version, csrf_token,
          current_stage, current_index, pairwise_count, single_count,
          assignment_json, assignment_hash, started_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 'consent', 0, ?, ?, ?, ?, ?, ?)
      `).run(
        sessionId,
        participantId,
        studyId,
        studyVersion,
        csrfToken,
        pairwiseCount,
        singleCount,
        JSON.stringify(assignment),
        assignmentHash,
        now,
        now,
      );
      const insertTrial = this.db.prepare(`
        INSERT INTO trials (
          trial_id, session_id, sequence_index, block, trial_type,
          comparison_id, matched_set_id, stimulus_a_id, stimulus_b_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);
      assignment.forEach((trial) => {
        insertTrial.run(
          trial.trial_id,
          sessionId,
          trial.sequence_index,
          trial.block,
          trial.trial_type,
          trial.comparison_id || null,
          trial.matched_set_id || null,
          trial.stimulus_a_id,
          trial.stimulus_b_id || null,
        );
      });
    });
  }

  getContext(sessionId) {
    return this.db.prepare(`
      SELECT s.*, p.status AS participant_status, p.valid_licence,
             p.completed_at AS participant_completed_at
      FROM study_sessions s
      JOIN participants p ON p.participant_id = s.participant_id
      WHERE s.session_id = ?
    `).get(sessionId);
  }

  getParticipant(participantId) {
    return this.db.prepare('SELECT * FROM participants WHERE participant_id = ?').get(participantId);
  }

  getCurrentTrial(sessionId, index) {
    return this.db.prepare(`
      SELECT * FROM trials WHERE session_id = ? AND sequence_index = ?
    `).get(sessionId, index);
  }

  markTrialStarted(trialId) {
    const now = utcNow();
    this.db.prepare(`
      UPDATE trials SET started_at = COALESCE(started_at, ?) WHERE trial_id = ?
    `).run(now, trialId);
    return this.db.prepare('SELECT * FROM trials WHERE trial_id = ?').get(trialId);
  }

  updateStage(sessionId, stage) {
    this.db.prepare(`
      UPDATE study_sessions SET current_stage = ?, updated_at = ? WHERE session_id = ?
    `).run(stage, utcNow(), sessionId);
  }

  recordConsent(sessionId, consentVersion, accepted) {
    const context = this.getContext(sessionId);
    if (!context) return;
    const now = utcNow();
    this.transaction(() => {
      this.db.prepare(`
        UPDATE participants SET consent_version = ?, consented_at = ?,
          status = ?, updated_at = ? WHERE participant_id = ?
      `).run(
        consentVersion,
        accepted ? now : null,
        accepted ? 'consented' : 'declined',
        now,
        context.participant_id,
      );
      this.db.prepare(`
        UPDATE study_sessions SET current_stage = ?, updated_at = ? WHERE session_id = ?
      `).run(accepted ? 'profile' : 'declined', now, sessionId);
    });
  }

  recordProfile(sessionId, profile) {
    const context = this.getContext(sessionId);
    const now = utcNow();
    this.transaction(() => {
      this.db.prepare(`
        UPDATE participants SET age_band = ?, gender = ?, valid_licence = ?,
          years_licensed = ?, driving_frequency = ?, annual_mileage_band = ?,
          urban_driving_frequency = ?, professional_driver = ?, device_type = ?,
          status = ?, updated_at = ? WHERE participant_id = ?
      `).run(
        profile.age_band,
        profile.gender,
        profile.valid_licence ? 1 : 0,
        profile.years_licensed,
        profile.driving_frequency,
        profile.annual_mileage_band,
        profile.urban_driving_frequency,
        profile.professional_driver,
        profile.device_type,
        profile.valid_licence ? 'eligible' : 'screened_out',
        now,
        context.participant_id,
      );
      this.db.prepare(`
        UPDATE study_sessions SET current_stage = ?, updated_at = ? WHERE session_id = ?
      `).run(profile.valid_licence ? 'instructions' : 'screened_out', now, sessionId);
    });
  }

  completePractice(sessionId) {
    this.updateStage(sessionId, 'pairwise');
  }

  takeBreak(sessionId) {
    this.db.prepare(`
      UPDATE study_sessions SET break_taken = 1, current_stage = 'single', updated_at = ?
      WHERE session_id = ?
    `).run(utcNow(), sessionId);
  }

  submitPairwise({
    sessionId,
    trialId,
    preferenceRaw,
    preferredStimulusId,
    confidence,
    freeTextReason,
    playbackAComplete,
    playbackBComplete,
    replayCount,
    responseTimeMs,
  }) {
    const now = utcNow();
    this.transaction(() => {
      this.db.prepare(`
        INSERT INTO pairwise_responses (
          trial_id, preference_raw, preferred_stimulus_id,
          no_clear_preference, confidence, free_text_reason, submitted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
      `).run(
        trialId,
        preferenceRaw,
        preferredStimulusId,
        preferenceRaw === 'NO_PREFERENCE' ? 1 : 0,
        confidence,
        freeTextReason || '',
        now,
      );
      this.db.prepare(`
        UPDATE trials SET completed_at = ?, playback_a_complete = ?,
          playback_b_complete = ?, replay_count = ?, response_time_ms = ?
        WHERE trial_id = ?
      `).run(
        now,
        playbackAComplete ? 1 : 0,
        playbackBComplete ? 1 : 0,
        replayCount,
        responseTimeMs,
        trialId,
      );
      this.db.prepare(`
        UPDATE study_sessions SET current_index = current_index + 1, updated_at = ?
        WHERE session_id = ?
      `).run(now, sessionId);
    });
  }

  submitSingle({
    sessionId,
    trialId,
    ratings,
    confidence,
    freeTextReason,
    playbackComplete,
    replayCount,
    responseTimeMs,
  }) {
    const now = utcNow();
    this.transaction(() => {
      this.db.prepare(`
        INSERT INTO single_responses (
          trial_id, acceptability, comfort, predictability,
          interaction_burden, perceived_unsafe, too_aggressive,
          too_cautious, confidence, free_text_reason, submitted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        trialId,
        ratings.acceptability,
        ratings.comfort,
        ratings.predictability,
        ratings.interaction_burden,
        ratings.perceived_unsafe,
        ratings.too_aggressive,
        ratings.too_cautious,
        confidence,
        freeTextReason || '',
        now,
      );
      this.db.prepare(`
        UPDATE trials SET completed_at = ?, playback_a_complete = ?,
          playback_b_complete = 1, replay_count = ?, response_time_ms = ?
        WHERE trial_id = ?
      `).run(now, playbackComplete ? 1 : 0, replayCount, responseTimeMs, trialId);
      this.db.prepare(`
        UPDATE study_sessions SET current_index = current_index + 1, updated_at = ?
        WHERE session_id = ?
      `).run(now, sessionId);
    });
  }

  finishSession(sessionId, postSurvey) {
    const context = this.getContext(sessionId);
    const now = utcNow();
    this.transaction(() => {
      this.db.prepare(`
        UPDATE participants SET post_survey_json = ?, status = 'completed',
          completed_at = ?, updated_at = ? WHERE participant_id = ?
      `).run(JSON.stringify(postSurvey), now, now, context.participant_id);
      this.db.prepare(`
        UPDATE study_sessions SET current_stage = 'complete', completed_at = ?, updated_at = ?
        WHERE session_id = ?
      `).run(now, now, sessionId);
    });
  }

  incrementVisibilityLoss(sessionId) {
    this.db.prepare(`
      UPDATE study_sessions SET visibility_loss_count = visibility_loss_count + 1,
        updated_at = ? WHERE session_id = ?
    `).run(utcNow(), sessionId);
  }

  logEvent({ sessionId = null, participantId = null, trialId = null, eventType, payload = {} }) {
    this.db.prepare(`
      INSERT INTO event_logs (
        session_id, participant_id, trial_id, event_type, payload_json, created_at
      ) VALUES (?, ?, ?, ?, ?, ?)
    `).run(
      sessionId,
      participantId,
      trialId,
      eventType,
      JSON.stringify(payload),
      utcNow(),
    );
  }

  dashboardSummary() {
    const participants = this.db.prepare('SELECT COUNT(*) AS n FROM participants').get().n;
    const completed = this.db.prepare("SELECT COUNT(*) AS n FROM participants WHERE status='completed'").get().n;
    const pairwise = this.db.prepare('SELECT COUNT(*) AS n FROM pairwise_responses').get().n;
    const single = this.db.prepare('SELECT COUNT(*) AS n FROM single_responses').get().n;
    const sessions = this.db.prepare(`
      SELECT current_stage, COUNT(*) AS n FROM study_sessions GROUP BY current_stage ORDER BY current_stage
    `).all();
    return { participants, completed, pairwise, single, sessions };
  }

  exportRows(name) {
    const queries = {
      participants: 'SELECT * FROM participants ORDER BY created_at',
      sessions: 'SELECT * FROM study_sessions ORDER BY started_at',
      trials: 'SELECT * FROM trials ORDER BY session_id, sequence_index',
      pairwise: `
        SELECT t.session_id, t.sequence_index, t.comparison_id, t.matched_set_id,
          t.stimulus_a_id, t.stimulus_b_id, t.started_at, t.completed_at,
          t.playback_a_complete, t.playback_b_complete, t.replay_count, t.response_time_ms,
          r.* FROM pairwise_responses r JOIN trials t ON t.trial_id = r.trial_id
        ORDER BY t.session_id, t.sequence_index
      `,
      single: `
        SELECT t.session_id, t.sequence_index, t.stimulus_a_id,
          t.started_at, t.completed_at, t.playback_a_complete,
          t.replay_count, t.response_time_ms, r.*
        FROM single_responses r JOIN trials t ON t.trial_id = r.trial_id
        ORDER BY t.session_id, t.sequence_index
      `,
      events: 'SELECT * FROM event_logs ORDER BY event_id',
    };
    if (!queries[name]) return null;
    return this.db.prepare(queries[name]).all();
  }
}

module.exports = { Store, SCHEMA, utcNow };
