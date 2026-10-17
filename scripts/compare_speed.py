"""Compare fixed-work benchmark artifacts without accepting a changed workload as a speedup.

Usage: python scripts/compare_speed.py baseline.json candidate.json --output comparison.json
Both JSON files must retain their sibling <stem>_artifacts directories. Numeric and
pixel tolerances are explicit acceptance criteria, not inferred from faster timings.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import csv
import json
import shlex
import statistics
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    result = json.loads(path.read_text())
    if result.get("status") != "ok" or result.get("verification", {}).get("status") != "passed":
        raise ValueError(f"{path}: not a validated successful benchmark")
    return result


def archive_path(result_path: Path, metadata: dict) -> Path:
    recorded = Path(metadata["path"])
    local = result_path.parent / (result_path.stem + "_artifacts") / recorded.name
    path = local if local.is_file() else recorded
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["file_sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return path


def semantic_config(result: dict) -> dict:
    config = copy.deepcopy(result["provenance"]["config"])
    recorders = config.get("recorders")
    if isinstance(recorders, dict):
        recorders.pop("dataset_export_dir_path", None)
        recorders.pop("dataset_filename", None)
    # PhysX CPU worker count only schedules the same solver work (see THREAD_CONTROLS).
    physx = config.get("sim", {}).get("physx")
    if isinstance(physx, dict):
        physx.pop("worker_thread_count", None)
    return config


THREAD_CONTROLS = {
    "/plugins/carb.tasking.plugin/threadCount",
    "/plugins/omni.tbb.globalcontrol/maxThreadCount",
    "/persistent/physics/numThreads",
    "/app/settings/persistent",
}


def semantic_arguments(result: dict) -> dict:
    args = dict(result["configuration"]["arguments"])
    # limit_cpu_threads only sizes Kit's worker pools (THREAD_CONTROLS below).
    for key in ("output", "actions", "repeats", "limit_cpu_threads"):
        args.pop(key, None)
    # Thread scheduling is the allowed execution-only axis. Any other Kit
    # setting (including render quality/cadence) remains part of the workload.
    args["kit_args"] = [token for token in shlex.split(args.get("kit_args", ""))
                        if token.partition("=")[0].removeprefix("--") not in THREAD_CONTROLS]
    return args


def hardware_identity(result: dict) -> list:
    rows = csv.DictReader(result["provenance"]["nvidia_smi"]["stdout"].splitlines(), skipinitialspace=True)
    return [(row["uuid"], row["name"], row["driver_version"]) for row in rows]


def compare(base_path: Path, new_path: Path, numeric_atol: float, pixel_atol: float) -> dict:
    base, new = load(base_path), load(new_path)
    differences = []
    left_args, right_args = semantic_arguments(base), semantic_arguments(new)
    for key in sorted(left_args.keys() | right_args.keys()):
        if left_args.get(key) != right_args.get(key):
            differences.append(f"arguments.{key}")
    if hardware_identity(base) != hardware_identity(new):
        differences.append("gpu_identity_or_driver")
    if base["diagnostic_reasons"] != new["diagnostic_reasons"]:
        differences.append("diagnostic_reasons")
    if semantic_config(base) != semantic_config(new):
        differences.append("resolved_environment_config")
    for key in ("isaacsim_version", "torch_version", "torch_cuda", "gpu", "packages", "constants", "direct_asset_files"):
        if base["provenance"][key] != new["provenance"][key]:
            differences.append(f"provenance.{key}")
    with np.load(archive_path(base_path, base["actions"]), allow_pickle=False) as left, np.load(archive_path(new_path, new["actions"]), allow_pickle=False) as right:
        if not np.array_equal(left["actions"], right["actions"]):
            differences.append("action_tensor")
    comparisons = []
    pairs = min(len(base["runs"]), len(new["runs"]))
    for index in range(pairs):
        left_meta, right_meta = base["runs"][index]["verification"], new["runs"][index]["verification"]
        with np.load(archive_path(base_path, left_meta), allow_pickle=False) as left, np.load(archive_path(new_path, right_meta), allow_pickle=False) as right:
            if set(left.files) != set(right.files):
                differences.append(f"repeat_{index}.array_schema")
            rows = []
            for key in sorted(set(left.files) & set(right.files)):
                a, b = left[key], right[key]
                if a.shape != b.shape or a.dtype != b.dtype:
                    differences.append(f"repeat_{index}.{key}.shape_or_dtype")
                    continue
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f"Nonfinite saved data: repeat {index}, {key}")
                delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                is_image = "/image_obs/" in key or "/viewport_cam/" in key
                atol = pixel_atol if is_image else numeric_atol
                rows.append({"array": key, "max_abs_delta": float(delta.max()),
                             "mean_abs_delta": float(delta.mean()),
                             "fraction_changed": float(np.count_nonzero(delta) / delta.size),
                             "atol": atol, "within_tolerance": bool((delta <= atol).all())})
            comparisons.append({"repeat": index, "arrays": rows})
    left_times = [r["measured"]["control_step_ms"] for r in base["runs"]]
    right_times = [r["measured"]["control_step_ms"] for r in new["runs"]]
    baseline_ms, candidate_ms = statistics.median(left_times), statistics.median(right_times)
    equivalent = not differences and bool(pairs) and all(row["within_tolerance"] for run in comparisons for row in run["arrays"])
    return {
        "baseline": str(base_path), "candidate": str(new_path),
        "workload_differences": differences, "comparable_fixed_work": not differences,
        "sampled_state_and_images_within_tolerance": equivalent,
        "validation_scope": "All saved trajectory samples; start/end observations only; not policy-success equivalence",
        "baseline_execution_controls": base["provenance"].get("runtime_settings"),
        "candidate_execution_controls": new["provenance"].get("runtime_settings"),
        "changed_source_files": sorted(key for key in base["provenance"]["source_sha256"].keys()
                                       | new["provenance"]["source_sha256"].keys()
                                       if base["provenance"]["source_sha256"].get(key)
                                       != new["provenance"]["source_sha256"].get(key)),
        "changed_upstream_sources": sorted(key for key in base["provenance"]["upstream_sources"].keys()
                                           | new["provenance"]["upstream_sources"].keys()
                                           if base["provenance"]["upstream_sources"].get(key)
                                           != new["provenance"]["upstream_sources"].get(key)),
        "baseline_control_step_ms": left_times, "candidate_control_step_ms": right_times,
        "median_speedup": baseline_ms / candidate_ms,
        "median_wall_time_reduction_percent": 100 * (1 - candidate_ms / baseline_ms),
        "observed_range_conservative_speedup": min(left_times) / max(right_times),
        "numeric_atol": numeric_atol, "pixel_atol": pixel_atol, "paired_samples": comparisons,
        "unpaired_repeat_count": abs(len(base["runs"]) - len(new["runs"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--numeric-atol", type=float, default=0.0)
    parser.add_argument("--pixel-atol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite([args.numeric_atol, args.pixel_atol]).all() or min(args.numeric_atol, args.pixel_atol) < 0:
        parser.error("Tolerances must be finite and nonnegative")
    result = compare(args.baseline, args.candidate, args.numeric_atol, args.pixel_atol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "paired_samples"}, indent=2))
    if not result["sampled_state_and_images_within_tolerance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
