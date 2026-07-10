from __future__ import annotations

from pipelines.interhub.process_interhub import (
    build_arg_parser,
    load_and_validate_execution_profile,
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
