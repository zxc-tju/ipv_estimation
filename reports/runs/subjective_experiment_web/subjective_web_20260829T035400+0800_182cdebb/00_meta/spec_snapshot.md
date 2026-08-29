# Subjective experiment web — frozen implementation scope

## Research role

The system collects prospective human judgements for the validation layer:

`conditional social atypicality → human dispreference`

It does not estimate IPV or change the frozen monitor. Hidden conditions are assigned upstream and imported through the stimulus manifest.

## Hidden design

- Actor source: Human / AV
- Monitor verdict: inside / outside
- Deviation side: none / assertive / accommodating
- Hidden from participants: actor source, verdict, side, deviation magnitude, system ID and driver ID

## Participant session

1. Anonymous session creation
2. Consent and valid-driving-licence screening
3. Minimal demographic and driving background
4. Instructions and practice comprehension check
5. Pairwise block first
6. Break
7. Single-clip ratings
8. Post-study familiarity and source-guess survey
9. Completion code

## Pairwise trial

- Sequential Vehicle A and Vehicle B playback
- Both stimuli must share a `matched_set_id`
- A/B order randomized
- At most one complete pair replay
- Primary response: A / B / no clear preference
- Confidence: 1–5

## Single-clip trial

Seven 1–7 ratings:

1. Acceptability
2. Interaction comfort
3. Predictability
4. Interaction burden imposed on the other driver
5. Perceived unsafety
6. Excessive aggressiveness/assertiveness
7. Excessive caution/hesitation

Acceptability, perceived unsafety and interaction burden remain separate fields.

## Operational logging

- playback completion
- replay count
- response time
- visibility changes
- video errors
- page unload
- trial and exclusion reason

## Default sizes

- Demo: 4 pairwise + 6 single-clip trials
- Production example: 12 pairwise + 18 single-clip trials

## Researcher functions

- SQLite persistence
- signed resumable sessions
- CSRF and security headers
- protected administrator dashboard
- CSV exports for participants, sessions, trials, pairwise responses, single-clip responses and events
- full JSON/config export

## Production boundary

Formal recruitment begins only after the six production gates recorded in `run_manifest.json` are closed.
