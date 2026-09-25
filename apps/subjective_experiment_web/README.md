# Driving-interaction subjective experiment web system

A self-contained web application for the prospective human evaluation layer of the NMI sociality-monitoring study. The application presents source-blind driving-interaction clips, collects pairwise preference and single-clip ratings, logs playback quality, and exports analysis-ready tables.

## What the system implements

### Participant flow

1. Informed consent.
2. Valid-driving-licence screening and minimal demographics.
3. Task instructions: the participant adopts the grey vehicle and evaluates the blue target vehicle.
4. Practice clip and target-vehicle comprehension check.
5. Pairwise block first:
   - sequential A then B playback;
   - at most one complete replay;
   - A / B / no clear preference;
   - 1–5 confidence;
   - optional reason.
6. Break.
7. Single-clip block:
   - acceptability;
   - comfort;
   - predictability;
   - interaction burden;
   - perceived unsafety;
   - excessive aggressiveness;
   - excessive caution/hesitation;
   - optional confidence and reason.
8. Post-study familiarity, source-guess and open feedback.
9. Completion code.

The production example contains 12 pairwise trials and 18 single-clip trials. The demo uses 4 + 6 so the entire flow can be inspected quickly.

### Hidden experimental structure

The server assigns six conditions without exposing them in participant HTML:

| Source | Inside human range | Assertive-side outside | Accommodating-side outside |
|---|---|---|---|
| Human | `H_IN` | `H_OUT_ASSERTIVE` | `H_OUT_ACCOMMODATING` |
| AV | `AV_IN` | `AV_OUT_ASSERTIVE` | `AV_OUT_ACCOMMODATING` |

Pairwise comparisons are drawn only within the same `matched_set_id`. A/B position is deterministically randomized for each anonymous participant. Real media filenames should be opaque.

### Researcher functions

- SQLite persistence with immutable study and assignment versions.
- Anonymous participant/session/trial IDs.
- Playback-completion checks and replay limits.
- Response time, visibility loss, playback events and errors.
- Resume support through a signed session cookie.
- Password-protected dashboard.
- CSV exports for participants, sessions, trials, pairwise responses, single responses and events.
- Full JSON export, including frozen configuration and hidden stimulus metadata.
- Health endpoint and automated end-to-end tests.

## Quick start

Requires Node.js 22.5 or later. No npm packages are required.

```bash
cd apps/subjective_experiment_web
cp .env.example .env
set -a; source .env; set +a
npm run check
npm run validate
npm test
npm start
```

Open:

- participant site: `http://127.0.0.1:3000/`
- admin login: `http://127.0.0.1:3000/admin/login`
- health check: `http://127.0.0.1:3000/health`

The demo admin token in `.env.example` must be changed.

## Docker

```bash
cp .env.example .env
# Replace all secrets and select production config/manifest.
docker compose up --build
```

The SQLite database is stored in the `experiment-data` volume.

## Preparing production stimuli

1. Copy `config/study.production.example.json` to a frozen production file.
2. Copy `config/stimuli.template.csv` to a frozen production manifest.
3. Place MP4/WebM clips under `public/stimuli/`, using opaque names.
4. Set every formal stimulus to `is_practice=false` and require:
   - `estimability_pass=true`;
   - `human_support_pass=true`;
   - `stimulus_qc_pass=true`.
5. Use the same `matched_set_id` for clips that are allowed to be compared.
6. Do not expose `actor_source`, `verdict_class`, `deviation_side`, deviation magnitude, driver ID or system ID through file names or participant instructions.
7. Run `npm run validate` before opening recruitment.

The manifest can retain research-only columns after the required columns. They are available in the admin full-JSON export and are never included in participant-facing payloads.

## Environment variables

| Variable | Meaning |
|---|---|
| `PORT` | HTTP port, default `3000` |
| `HOST` | Bind host, default `0.0.0.0` |
| `EXPERIMENT_SECRET_KEY` | HMAC key for signed cookies |
| `EXPERIMENT_ADMIN_TOKEN` | Researcher dashboard token |
| `EXPERIMENT_DATABASE` | SQLite path |
| `EXPERIMENT_STUDY_CONFIG` | Study JSON path |
| `EXPERIMENT_STIMULUS_MANIFEST` | Stimulus CSV path |
| `EXPERIMENT_SECURE_COOKIE` | Set `1` behind HTTPS |
| `NODE_ENV` | Set `production` for deployment |

Production mode refuses to start with the example secrets.

## Data model

- `participants`: consent, eligibility, minimal demographics and post-survey.
- `study_sessions`: study version, assignment hash, progress, CSRF token and visibility count.
- `trials`: immutable stimulus assignment plus playback and timing fields.
- `pairwise_responses`: A/B/no-preference, selected stimulus, confidence and reason.
- `single_responses`: seven ratings, confidence and reason.
- `event_logs`: playback, replay, visibility and client-quality events.

The primary analysis joins the exported response tables to the frozen stimulus manifest outside this application. The participant-facing application never receives the hidden condition fields.

## Pre-recruitment freeze checklist

- Ethics-approved consent and debrief text inserted.
- Inclusion and exclusion rules frozen.
- Production trial counts frozen.
- Stimulus inventory and matching audited without viewing subjective outcomes.
- Video appearance, view, duration, playback speed and target marking standardized.
- Production secrets set and HTTPS enabled.
- Persistent database backed up.
- Full pilot completed on the intended devices and network.
- `npm run check`, `npm run validate` and `npm test` pass.
- A copy of the exact JSON, CSV, code commit and media hashes is archived.

## Important scope

The system collects human preference and acceptability evidence for the frozen monitor. It does not calculate IPV, decide whether a clip is inside/outside the reference, or infer harm. Those fields must be generated and frozen upstream.
