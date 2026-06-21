# IPV Sign Contract Audit

Status: PASS

Contract: theta > 0 means prosocial. Evidence:

- `src/sociality_estimation/core/agent.py:63` defines a symmetric candidate grid from negative to positive IPV.
- `src/sociality_estimation/core/agent.py:748-750` maps positive theta to positive `sin(ipv)` interaction weight.
- `src/sociality_estimation/core/agent.py:1012-1031` combines individual cost with group/social cost as `cos(theta)*interior_cost + sin(theta)*group_cost`; positive theta therefore increases the group-cost term rather than reversing it.
- `src/sociality_estimation/core/ipv_estimation.py:391-399` converts positive IPV to sign `+1`, neutral to `0`, and negative to `-1`.

Canonical unit tests passed 13/13. Formula tests prove `D_comp=max(0,(Q_low-theta_ego)/w)` fires only below `Q_low`, while `D_yield=max(0,(theta_ego-Q_high)/w)` fires only above `Q_high`.

No sign flip was introduced by mirror, role-swap, or time-truncation tests.
