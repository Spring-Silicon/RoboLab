"""Fixed-work comparisons must reject changed work or unverified evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("compare_speed", Path(__file__).parents[1] / "scripts" / "compare_speed.py")
_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compare)


def artifact(tmp_path, name, *, step_ms=10.0, kit_args="", joint_delta=0.0, pixels=0):
    directory = tmp_path / f"{name}_artifacts"
    directory.mkdir()
    actions = directory / "actions.npz"
    np.savez(actions, actions=np.zeros((2, 1, 8), dtype=np.float32))
    state = directory / "repeat_0.npz"
    np.savez(state, **{
        "trajectory/articulations/robot/joint_pos": np.full((3, 1, 7), joint_delta, dtype=np.float32),
        "observation_end/image_obs/camera": np.full((1, 2, 2, 3), pixels, dtype=np.uint8),
    })
    def metadata(path):
        return {"path": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {
        "status": "ok", "verification": {"status": "passed"},
        "configuration": {"arguments": {"output": name, "kit_args": kit_args, "steps": 2}},
        "diagnostic_reasons": [], "actions": metadata(actions),
        "provenance": {
            "config": {"recorders": {"dataset_export_dir_path": str(directory)}},
            "isaacsim_version": "5.0.0", "torch_version": "2.7.0", "torch_cuda": "12.8",
            "gpu": {"name": "L40S"}, "packages": [], "constants": {}, "direct_asset_files": {},
            "source_sha256": {}, "upstream_sources": {},
            "nvidia_smi": {"stdout": "uuid, name, driver_version\nGPU-test, NVIDIA L40S, 570.195.03\n"},
        },
        "runs": [{"measured": {"control_step_ms": step_ms}, "verification": metadata(state)}],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(result))
    return path


def test_rejects_faster_changed_render_workload(tmp_path):
    baseline = artifact(tmp_path, "base")
    candidate = artifact(tmp_path, "candidate", step_ms=5, kit_args="--/rtx/shadows/enabled=false")
    result = _compare.compare(baseline, candidate, 0, 0)
    assert result["median_speedup"] == 2
    assert not result["comparable_fixed_work"]
    assert "arguments.kit_args" in result["workload_differences"]
    assert not result["sampled_state_and_images_within_tolerance"]


def test_thread_scheduling_is_distinct_from_workload(tmp_path):
    baseline = artifact(tmp_path, "base")
    candidate = artifact(tmp_path, "candidate", step_ms=5,
                         kit_args="--/plugins/carb.tasking.plugin/threadCount=8")
    result = _compare.compare(baseline, candidate, 0, 0)
    assert result["comparable_fixed_work"]
    assert result["sampled_state_and_images_within_tolerance"]
    assert result["median_wall_time_reduction_percent"] == 50


def test_rejects_faster_divergent_trajectory(tmp_path):
    baseline = artifact(tmp_path, "base")
    candidate = artifact(tmp_path, "candidate", step_ms=1, joint_delta=0.1)
    result = _compare.compare(baseline, candidate, 1e-4, 0)
    assert result["comparable_fixed_work"]
    assert not result["sampled_state_and_images_within_tolerance"]


def test_pixel_subtraction_does_not_wrap_uint8(tmp_path):
    baseline = artifact(tmp_path, "base", pixels=0)
    candidate = artifact(tmp_path, "candidate", pixels=255)
    result = _compare.compare(baseline, candidate, 0, 1)
    image = next(row for row in result["paired_samples"][0]["arrays"] if "/image_obs/" in row["array"])
    assert image["max_abs_delta"] == 255
    assert not image["within_tolerance"]


def test_rejects_modified_evidence(tmp_path):
    baseline = artifact(tmp_path, "base")
    candidate = artifact(tmp_path, "candidate")
    (tmp_path / "candidate_artifacts" / "repeat_0.npz").write_bytes(b"not the recorded artifact")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _compare.compare(baseline, candidate, 0, 0)


def test_rejects_failed_partial_run(tmp_path):
    path = tmp_path / "failed.json"
    path.write_text(json.dumps({"status": "failed", "verification": {"status": "failed"}}))
    with pytest.raises(ValueError, match="not a validated successful benchmark"):
        _compare.load(path)
