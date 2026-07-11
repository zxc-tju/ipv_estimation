from __future__ import annotations

from pipelines.interhub.process_interhub import (
    build_arg_parser,
    load_and_validate_execution_profile,
    production_workflow_log_path,
)


def test_interhub_cli_defaults_match_and_validate_sigma01_profile() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.solver_mode == "exact"
    assert args.history_window == 10
    assert args.min_observation == 4
    assert args.reference_clip_margin_m == 60.0
    assert args.reference_max_points == 40
    assert args.reference_smooth_points == 40
    metadata = load_and_validate_execution_profile(args.execution_profile, args)
    assert metadata["profile"] == "ipv_sigma01_exact"
    assert len(metadata["sha256"]) == 64


def test_production_output_and_log_stay_in_run_root(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--output-root",
            str(run_root / "outputs"),
            "--log-workflow",
        ]
    )
    monkeypatch.setenv("SOCIALITY_PRODUCTION_RUN_ROOT", str(run_root))
    assert production_workflow_log_path(args) == (run_root / "logs" / "workflow.log").resolve()


def test_production_output_outside_run_root_fails(tmp_path, monkeypatch) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--output-root", str(tmp_path / "wrong")])
    monkeypatch.setenv("SOCIALITY_PRODUCTION_RUN_ROOT", str(tmp_path / "run"))
    try:
        production_workflow_log_path(args)
    except ValueError as exc:
        assert "inside run root" in str(exc)
    else:
        raise AssertionError("Production output outside the run root was accepted")
