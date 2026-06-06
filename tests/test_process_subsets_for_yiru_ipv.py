from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import process_subsets_for_yiru_ipv as mod


def _vehicle(
    *,
    positions,
    velocities=None,
    headings=None,
    timestamps=None,
    lane_ids=None,
    frame_lane_ids=None,
):
    positions = np.asarray(positions, dtype=float)
    steps = len(positions)
    if velocities is None:
        velocities = np.ones((steps, 2), dtype=float)
    if headings is None:
        headings = np.zeros(steps, dtype=float)
    if timestamps is None:
        timestamps = list(range(steps))
    return {
        "positions": positions.tolist(),
        "velocities": np.asarray(velocities, dtype=float).tolist(),
        "headings": np.asarray(headings, dtype=float).tolist(),
        "timestamps": list(timestamps),
        "lane_ids": list(lane_ids or []),
        "frame_lane_ids": list(frame_lane_ids or []),
    }


def _event(*, folder="fold", scenario_idx=7, key_agents="a;b", track_ids=None):
    if track_ids is None:
        track_ids = key_agents.split(";")
    return {
        "metadata": {
            "dataset": "demo",
            "folder": folder,
            "scenario_idx": scenario_idx,
            "track_ids": track_ids,
            "key_agents": key_agents,
            "two_multi": "two",
        },
        "vehicles": {
            "a": _vehicle(
                positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                lane_ids=["L1", "L2"],
            ),
            "b": _vehicle(
                positions=[[0, 1, 0], [1, 1, 0], [2, 1, 0]],
                lane_ids=["L3"],
            ),
        },
        "road_info": {
            "all_lane_centerlines": {
                "L1": [[0, 0], [1, 0]],
                "L2": [[1, 0], [2, 0]],
                "L3": [[0, 1], [2, 1]],
                "F1": [[5, 5], [6, 5]],
                "F2": [[6, 5], [7, 5]],
            }
        },
    }


def test_build_event_index_uses_folder_scenario_key_agents_and_track_id(tmp_path):
    event = _event(folder="demo_folder", scenario_idx=42, key_agents="b;a", track_ids=["a", "b"])
    pkl_path = tmp_path / "demo.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump({"segment_1": event}, f)

    index = mod.build_event_index(tmp_path)

    key = ("demo_folder", "42", "b;a", "a;b")
    assert key in index
    assert index[key].segment_id == "segment_1"
    assert index[key].pkl_path == pkl_path


def test_reference_uses_lane_ids_then_frame_lane_ids():
    event = _event()
    lane_ref, lane_source = mod.build_vehicle_reference(event, "a")
    np.testing.assert_allclose(lane_ref, [[0, 0], [1, 0], [2, 0]])
    assert lane_source == "lane_ids"

    event["vehicles"]["a"]["lane_ids"] = []
    event["vehicles"]["a"]["frame_lane_ids"] = ["F1", "F1", "F2"]
    frame_ref, frame_source = mod.build_vehicle_reference(event, "a")
    np.testing.assert_allclose(frame_ref, [[5, 5], [6, 5], [7, 5]])
    assert frame_source == "frame_lane_ids"


def test_reference_falls_back_to_observed_trajectory_when_lanes_missing():
    event = _event()
    event["vehicles"]["a"]["lane_ids"] = ["missing"]
    event["vehicles"]["a"]["frame_lane_ids"] = []

    ref, source = mod.build_vehicle_reference(event, "a")

    np.testing.assert_allclose(ref, [[0, 0], [1, 0], [2, 0]])
    assert source == "observed_trajectory_fallback"


def test_align_key_agent_motion_uses_common_timestamps_in_key_agent_order():
    event = _event(key_agents="b;a")
    event["vehicles"]["b"] = _vehicle(
        positions=[[10, 0, 0], [11, 0, 0], [12, 0, 0]],
        velocities=[[10, 0], [11, 0], [12, 0]],
        headings=[0.1, 0.2, 0.3],
        timestamps=[0, 1, 2],
    )
    event["vehicles"]["a"] = _vehicle(
        positions=[[20, 0, 0], [21, 0, 0]],
        velocities=[[20, 0], [21, 0]],
        headings=[0.4, 0.5],
        timestamps=[1, 2],
    )

    aligned = mod.align_key_agent_motion(event, ["b", "a"], min_steps=2)

    assert aligned.timestamps == [1, 2]
    np.testing.assert_allclose(aligned.primary_motion[:, 0], [11, 12])
    np.testing.assert_allclose(aligned.secondary_motion[:, 0], [20, 21])


def test_dataset_downsampling_converts_nuplan_20hz_to_10hz_only():
    aligned = mod.AlignedMotion(
        primary_motion=np.column_stack(
            (
                np.arange(6),
                np.zeros(6),
                np.ones(6),
                np.zeros(6),
                np.zeros(6),
            )
        ),
        secondary_motion=np.column_stack(
            (
                np.arange(10, 16),
                np.zeros(6),
                np.ones(6),
                np.zeros(6),
                np.zeros(6),
            )
        ),
        timestamps=list(range(6)),
    )

    nuplan = mod.apply_dataset_downsampling(aligned, dataset="nuplan_train", min_steps=3)
    waymo = mod.apply_dataset_downsampling(aligned, dataset="waymo_train", min_steps=3)

    assert nuplan.timestamps == [0, 2, 4]
    np.testing.assert_allclose(nuplan.primary_motion[:, 0], [0, 2, 4])
    assert waymo.timestamps == list(range(6))
    np.testing.assert_allclose(waymo.primary_motion[:, 0], np.arange(6))


def test_compute_valid_mean_ipv_ignores_pre_observation_rows():
    ipv_values = np.array([[99.0, 99.0], [1.0, 3.0], [2.0, 5.0]])
    ipv_errors = np.array([[99.0, 99.0], [0.1, 0.3], [0.2, 0.5]])

    summary = mod.compute_valid_ipv_summary(ipv_values, ipv_errors, min_observation=1)

    assert summary == {
        "ipv_key_agent_1_mean": 1.5,
        "ipv_key_agent_1_error_mean": 0.15,
        "ipv_key_agent_2_mean": 4.0,
        "ipv_key_agent_2_error_mean": 0.4,
    }


def test_build_csv_copy_preserves_key_agents_without_new_id_columns():
    source = pd.DataFrame(
        {
            "folder": ["demo"],
            "scenario_idx": [1],
            "track_id": ["a;b"],
            "key_agents": ["a;b"],
        }
    )
    results = {
        0: {
            "ipv_key_agent_1_mean": 0.25,
            "ipv_key_agent_1_error_mean": 0.1,
            "ipv_key_agent_2_mean": -0.5,
            "ipv_key_agent_2_error_mean": 0.2,
            "ipv_result_status": "ok",
            "ipv_result_case_dir": "case",
            "ipv_result_error": "",
            "ipv_pkl_file": "demo.pkl",
            "ipv_segment_id": "seg",
            "ipv_reference_source_1": "lane_ids",
            "ipv_reference_source_2": "lane_ids",
        }
    }

    output = mod.build_csv_copy(source, results)

    assert "key_agents" in output.columns
    assert "ipv_key_agent_1_id" not in output.columns
    assert "ipv_key_agent_2_id" not in output.columns
    assert output.loc[0, "key_agents"] == "a;b"
    assert output.loc[0, "ipv_key_agent_1_mean"] == 0.25
    assert output.loc[0, "ipv_key_agent_2_mean"] == -0.5


def test_choose_recommended_workers_prefers_lower_worker_when_gain_is_small():
    records = [
        {"workers": 1, "failed": 0, "cases_per_minute": 10.0},
        {"workers": 2, "failed": 0, "cases_per_minute": 19.0},
        {"workers": 4, "failed": 0, "cases_per_minute": 20.0},
        {"workers": 8, "failed": 1, "cases_per_minute": 30.0},
    ]

    assert mod.choose_recommended_workers(records) == 2


def test_case_output_dir_uses_short_hash_instead_of_full_segment_id(tmp_path):
    segment_id = "train_vegas3_4000_1fe61c52847a5cb8_2b28efeda75a5117"
    row = {
        "dataset": "nuplan_train",
        "folder": "train_vegas3",
        "scenario_idx": 4000,
    }

    case_dir = mod.case_output_dir(tmp_path, 0, row, segment_id)

    assert segment_id not in str(case_dir)
    assert "row_00000_" in case_dir.name
    assert len(case_dir.name) < 24


def test_select_shard_rows_uses_modulo_positions_and_preserves_indices():
    source = pd.DataFrame({"value": list(range(10))}, index=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    shard = mod.select_shard_rows(source, shard_index=1, shard_count=3)

    assert shard.index.tolist() == [11, 14, 17]
    assert shard["value"].tolist() == [1, 4, 7]


def test_select_dataset_filter_preserves_only_requested_datasets():
    source = pd.DataFrame({"dataset": ["nuplan_train", "waymo_train", "nuplan_train"], "value": [1, 2, 3]})

    filtered = mod.select_dataset_rows(source, ["nuplan_train"])

    assert filtered["value"].tolist() == [1, 3]


def test_exclude_csv_rows_removes_matching_keys_and_preserves_indices(tmp_path):
    source = pd.DataFrame(
        {
            "folder": ["full_a", "full_b", "full_c"],
            "scenario_idx": [1, 2, 3],
            "track_id": ["ta", "tb", "tc"],
            "key_agents": ["ka1;ka2", "kb1;kb2", "kc1;kc2"],
            "value": [10, 20, 30],
        },
        index=[10, 11, 12],
    )
    exclude_csv = tmp_path / "selected_interactive_segments_equalized.csv"
    pd.DataFrame(
        {
            "folder": ["full_b", "missing"],
            "scenario_idx": [2, 99],
            "track_id": ["tb", "tm"],
            "key_agents": ["kb1;kb2", "km1;km2"],
        }
    ).to_csv(exclude_csv, index=False)

    filtered, summary = mod.exclude_csv_rows(source, exclude_csv)

    assert filtered.index.tolist() == [10, 12]
    assert filtered["value"].tolist() == [10, 30]
    assert summary == {
        "exclude_csv_path": str(exclude_csv),
        "exclude_csv_rows": 2,
        "exclude_csv_unique_keys": 2,
        "excluded_rows": 1,
    }


def test_merge_shard_outputs_combines_csv_columns_by_key(tmp_path):
    source = pd.DataFrame(
        {
            "folder": ["a", "b", "c"],
            "scenario_idx": [1, 2, 3],
            "track_id": ["ta", "tb", "tc"],
            "key_agents": ["ka1;ka2", "kb1;kb2", "kc1;kc2"],
        }
    )
    csv_path = tmp_path / "selected_interactive_segments_equalized.csv"
    source.to_csv(csv_path, index=False)

    shard_0 = source.iloc[[0, 2]].copy()
    shard_1 = source.iloc[[1]].copy()
    for df, prefix in ((shard_0, "s0"), (shard_1, "s1")):
        for column in mod.CSV_OUTPUT_COLUMNS:
            df[column] = ""
        df["ipv_key_agent_1_mean"] = [f"{prefix}-a{i}" for i in range(len(df))]
        df["ipv_key_agent_2_mean"] = [f"{prefix}-b{i}" for i in range(len(df))]
        df["ipv_result_status"] = "ok"

    output_root = tmp_path / "out"
    output_root.mkdir()
    shard_0.to_csv(output_root / "selected_interactive_segments_equalized_with_ipv_shard_00_of_02.csv", index=False)
    shard_1.to_csv(output_root / "selected_interactive_segments_equalized_with_ipv_shard_01_of_02.csv", index=False)

    summary = mod.merge_shard_outputs(csv_path, output_root, shard_count=2)
    merged = pd.read_csv(summary["csv_output"])

    assert summary["merged_rows"] == 3
    assert merged["ipv_result_status"].tolist() == ["ok", "ok", "ok"]
    assert merged["ipv_key_agent_1_mean"].tolist() == ["s0-a0", "s1-a0", "s0-a1"]
    assert "ipv_key_agent_1_id" not in merged.columns


def test_merge_shard_outputs_can_patch_existing_base_csv(tmp_path):
    source = pd.DataFrame(
        {
            "folder": ["a", "b"],
            "scenario_idx": [1, 2],
            "track_id": ["ta", "tb"],
            "key_agents": ["ka1;ka2", "kb1;kb2"],
        }
    )
    csv_path = tmp_path / "selected_interactive_segments_equalized.csv"
    source.to_csv(csv_path, index=False)
    base = mod.build_csv_copy(
        source,
        {
            0: {"ipv_key_agent_1_mean": "old-nuplan", "ipv_result_status": "ok"},
            1: {"ipv_key_agent_1_mean": "keep-waymo", "ipv_result_status": "ok"},
        },
    )
    base_csv = tmp_path / "base.csv"
    base.to_csv(base_csv, index=False)

    patch = source.iloc[[0]].copy()
    for column in mod.CSV_OUTPUT_COLUMNS:
        patch[column] = ""
    patch["ipv_key_agent_1_mean"] = "new-nuplan"
    patch["ipv_result_status"] = "ok"
    output_root = tmp_path / "out"
    output_root.mkdir()
    patch.to_csv(
        output_root / "selected_interactive_segments_equalized_with_ipv_dataset_nuplan_train_shard_00_of_01.csv",
        index=False,
    )

    summary = mod.merge_shard_outputs(
        csv_path,
        output_root,
        shard_count=1,
        dataset_filter=["nuplan_train"],
        base_csv_path=base_csv,
    )
    merged = pd.read_csv(summary["csv_output"])

    assert summary["merged_rows"] == 1
    assert summary["missing_rows"] == 1
    assert merged["ipv_key_agent_1_mean"].tolist() == ["new-nuplan", "keep-waymo"]


def test_merge_shard_outputs_uses_exclude_suffix_and_reports_excluded_rows(tmp_path):
    source = pd.DataFrame(
        {
            "folder": ["a", "b", "c"],
            "scenario_idx": [1, 2, 3],
            "track_id": ["ta", "tb", "tc"],
            "key_agents": ["ka1;ka2", "kb1;kb2", "kc1;kc2"],
        }
    )
    csv_path = tmp_path / "selected_interactive_segments_nuplan_agv_full.csv"
    source.to_csv(csv_path, index=False)
    exclude_csv = tmp_path / "selected_interactive_segments_equalized.csv"
    source.iloc[[1]].to_csv(exclude_csv, index=False)

    patch = source.iloc[[0, 2]].copy()
    for column in mod.CSV_OUTPUT_COLUMNS:
        patch[column] = ""
    patch["ipv_result_status"] = "ok"

    output_root = tmp_path / "out"
    output_root.mkdir()
    suffix = mod.exclude_csv_suffix(exclude_csv)
    patch.to_csv(
        output_root / f"selected_interactive_segments_equalized_with_ipv{suffix}_shard_00_of_01.csv",
        index=False,
    )

    summary = mod.merge_shard_outputs(csv_path, output_root, shard_count=1, exclude_csv_path=exclude_csv)
    merged = pd.read_csv(summary["csv_output"])

    assert summary["merged_rows"] == 2
    assert summary["excluded_rows"] == 1
    assert summary["missing_rows"] == 1
    assert merged["ipv_result_status"].fillna("").tolist() == ["ok", "", "ok"]


def test_run_processing_propagates_save_plots_flag(tmp_path, monkeypatch):
    event = _event(folder="demo_folder", scenario_idx=42, key_agents="a;b", track_ids=["a", "b"])
    pkl_path = tmp_path / "events.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump({"segment_1": event}, f)
    csv_path = tmp_path / "selected_interactive_segments_equalized.csv"
    pd.DataFrame(
        {
            "dataset": ["demo"],
            "folder": ["demo_folder"],
            "scenario_idx": [42],
            "track_id": ["a;b"],
            "key_agents": ["a;b"],
            "two/multi": ["two"],
        }
    ).to_csv(csv_path, index=False)

    seen_flags = []

    def fake_process_case(task):
        seen_flags.append(task.save_plots)
        return {"ipv_result_status": "ok"}

    monkeypatch.setattr(mod, "process_case", fake_process_case)

    mod.run_processing(
        csv_path,
        tmp_path,
        tmp_path / "out",
        workers=1,
        history_window=10,
        min_observation=4,
        save_plots=False,
    )

    assert seen_flags == [False]


def test_inspect_case_completion_requires_metadata_table_and_current_downsample(tmp_path):
    row = {"dataset": "nuplan_train", "folder": "f", "scenario_idx": 1}
    case_dir = mod.case_output_dir(tmp_path, 0, row, "seg")
    data_dir = case_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "ipv_results.xlsx").write_text("fake", encoding="utf-8")
    metadata_path = data_dir / "metadata.json"
    metadata_path.write_text('{"status": "ok", "downsample_factor": 1}', encoding="utf-8")

    stale = mod.inspect_case_completion(tmp_path, 0, row, "seg")
    assert stale["complete"] is False
    assert stale["reason"] == "stale_downsample_factor"

    metadata_path.write_text('{"status": "ok", "downsample_factor": 2}', encoding="utf-8")
    complete = mod.inspect_case_completion(tmp_path, 0, row, "seg")
    assert complete["complete"] is True
    assert complete["reason"] == "complete"


def test_run_processing_only_incomplete_dispatches_missing_rows(tmp_path, monkeypatch):
    events = {
        "seg_0": _event(folder="f0", scenario_idx=0, key_agents="a;b", track_ids=["a", "b"]),
        "seg_1": _event(folder="f1", scenario_idx=1, key_agents="a;b", track_ids=["a", "b"]),
    }
    pkl_path = tmp_path / "events.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(events, f)
    csv_path = tmp_path / "selected_interactive_segments_equalized.csv"
    rows = pd.DataFrame(
        {
            "dataset": ["waymo_train", "waymo_train"],
            "folder": ["f0", "f1"],
            "scenario_idx": [0, 1],
            "track_id": ["a;b", "a;b"],
            "key_agents": ["a;b", "a;b"],
            "two/multi": ["two", "two"],
        }
    )
    rows.to_csv(csv_path, index=False)

    output_root = tmp_path / "out"
    complete_dir = mod.case_output_dir(output_root, 0, rows.iloc[0].to_dict(), "seg_0") / "data"
    complete_dir.mkdir(parents=True)
    (complete_dir / "metadata.json").write_text('{"status": "ok", "downsample_factor": 1}', encoding="utf-8")
    (complete_dir / "ipv_results.xlsx").write_text("fake", encoding="utf-8")

    processed_rows = []

    def fake_process_case(task):
        processed_rows.append(task.row_index)
        return {"ipv_result_status": "ok"}

    monkeypatch.setattr(mod, "process_case", fake_process_case)

    summary = mod.run_processing(
        csv_path,
        tmp_path,
        output_root,
        workers=1,
        history_window=10,
        min_observation=4,
        only_incomplete=True,
    )

    assert processed_rows == [1]
    assert summary["incomplete_total"] == 1
    assert summary["selected_rows"] == 1


def test_run_processing_exclude_csv_skips_existing_subset_rows(tmp_path, monkeypatch):
    events = {
        "seg_0": _event(folder="f0", scenario_idx=0, key_agents="a;b", track_ids=["a", "b"]),
        "seg_1": _event(folder="f1", scenario_idx=1, key_agents="a;b", track_ids=["a", "b"]),
        "seg_2": _event(folder="f2", scenario_idx=2, key_agents="a;b", track_ids=["a", "b"]),
    }
    pkl_path = tmp_path / "events.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(events, f)
    rows = pd.DataFrame(
        {
            "dataset": ["nuplan_train", "nuplan_train", "av2_motion_forecasting"],
            "folder": ["f0", "f1", "f2"],
            "scenario_idx": [0, 1, 2],
            "track_id": ["a;b", "a;b", "a;b"],
            "key_agents": ["a;b", "a;b", "a;b"],
            "two/multi": ["two", "two", "two"],
        }
    )
    csv_path = tmp_path / "selected_interactive_segments_nuplan_agv_full.csv"
    rows.to_csv(csv_path, index=False)
    exclude_csv = tmp_path / "selected_interactive_segments_equalized.csv"
    rows.iloc[[1]].to_csv(exclude_csv, index=False)

    processed_rows = []

    def fake_process_case(task):
        processed_rows.append(task.row_index)
        return {"ipv_result_status": "ok"}

    monkeypatch.setattr(mod, "process_case", fake_process_case)

    summary = mod.run_processing(
        csv_path,
        tmp_path,
        tmp_path / "out",
        workers=1,
        history_window=10,
        min_observation=4,
        exclude_csv_path=exclude_csv,
    )

    assert processed_rows == [0, 2]
    assert summary["excluded_rows"] == 1
    assert summary["selected_rows"] == 2
