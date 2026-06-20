# Study Execution Reports

This is the execution layer of the research knowledge base. It holds physically
organized report packages by research question.

## Layout

```text
reports/studies/
  RQ001_online_ipv_interval/
    README.md
    executions.csv
    RQ001_1_current_ipv_distribution_20260618/
    RQ001_2_interval_query_20260618/
    RQ001_3_online_interval_lock_20260619/
```

Rules:

- Use `RQxxx_topic/` for a research question.
- Use `RQxxx_n_short_topic_YYYYMMDD/` for an execution report; `n` is the
  execution version under that RQ.
- Keep rendered reports and generated result payloads local/ignored unless a
  task explicitly asks to track them.
- Put reviewer synthesis and final claim decisions in
  `reports/knowledge/<same RQ stem>/`.

`reports/` should not regain dataset-specific first-level roots. Put large
derived data in `data/derived/` and process archives in
`archived/report_process/`.
