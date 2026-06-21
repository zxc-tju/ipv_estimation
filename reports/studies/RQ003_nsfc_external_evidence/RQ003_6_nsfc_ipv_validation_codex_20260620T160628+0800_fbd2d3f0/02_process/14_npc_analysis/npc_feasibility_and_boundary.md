# NPC Feasibility and Boundary

Worker: `RQ003_phase6_state_npc_001`
Run ID: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`
Generated: `2026-06-20T21:51:52+08:00`

## Verdict

Pre-onset matching is **not identifiable** from the allowed Phase 6 inputs. No NPC effect analysis was run.

The available logs support descriptive extraction of initial position, speed, heading/course angle, and observed actor IDs. They do not expose the required script version, simulator seed, or a state marker proving that the ego vehicle had not yet influenced the NPC. Actor identity is also only partially available because per-log IDs and names are observed, while cross-session canonical identity stability is not established.

## Field Audit

| Required pre-onset field | Availability | Evidence | Verdict |
| --- | --- | --- | --- |
| initial_position | partial_yes | vehicle_perception_simulation_trajectory.log value fields include latitude/longitude; frame cache includes derived gaps but not raw x/y pose columns | usable only as pre-onset descriptive matching input |
| initial_speed | yes | vehicle_perception_simulation_trajectory.log value field `speed`; frame cache has ego_speed_mps/npc_speed_mps after feature extraction | usable only as pre-onset descriptive matching input |
| initial_pose_heading | partial_yes | vehicle_perception_simulation_trajectory.log value field `courseAngle`; pose frame convention not independently documented in Phase 6 inputs | partial descriptive input |
| script_version | no | top-five manifests/log schema and rg search expose no script/version field | required field missing |
| seed | no | top-five manifests/log schema and rg search expose no seed/random field | required field missing |
| actor_identity | partial_yes | cell-level counterpart_id/counterpart_name and log value `id` exist, but cross-session canonical identity stability is not proven | not enough for strict matching |
| ego_not_yet_influencing_npc_state | no | no open-loop marker, pre-influence flag, or independent NPC baseline trace found in allowed metadata | required state missing |

## Boundary

Because the required pre-onset fields are incomplete, matching on realized NPC trajectory, response timing, post-interaction movement, or any ego-influenced response is forbidden and was not performed. The only permissible future wording, if independent pre-onset matching evidence is later obtained, is **matched opportunity structure**. This run does not instantiate such a match.

## Checked Sources

- `data/onsite_competition/top5_research_subset/tables/top5_session_manifest.csv`: session/log presence, no script version or seed fields.
- `data/onsite_competition/top5_research_subset/tables/top5_team_manifest.csv`: team/session materialization, no script version or seed fields.
- `data/onsite_competition/top5_research_subset/tables/materialized_analysis_files.csv`: materialized file inventory only.
- `vehicle_perception_simulation_trajectory.log` samples: participant roles and per-frame latitude/longitude/speed/courseAngle/id, no required seed/script/pre-influence marker.
- `simulation_trajectory.log` samples: trajectory timestamps/values, no required seed/script/pre-influence marker.
- `rg -i "seed|script|version|controller|behavior|npc|closed|route"` over top-five tables and text/log metadata returned no matches.
