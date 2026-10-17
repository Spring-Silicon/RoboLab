"""Compare two evaluation output folders task by task.

Usage: python scripts/compare_eval_outputs.py BASE_DIR CANDIDATE_DIR --output comparison.json

Both folders come from robolab.eval.runner (for example scripts/scripted_eval.py). For each
task folder in BASE_DIR, the script compares:
  * episode_results.jsonl rows (all fields except timing and duration); a key
    (env_name, episode) that appears twice in one folder is a failure,
  * every run_*.hdf5: the same files on both sides; all non-image datasets and dataset
    attributes must be bit-identical; camera images (uint8 arrays with 3 or more
    dimensions) give mean and max absolute pixel differences, because RTX images are not
    bitwise repeatable between processes,
  * the names of the *.mp4 videos,
  * env_cfg.json (without output paths).
Exit status is 1 if a non-image value differs or a file is missing or extra.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

IGNORED_RESULT_KEYS = {"timing", "duration"}


def load_rows(folder: Path) -> tuple[dict[tuple[str, int], dict], list[list]]:
    rows, duplicates = {}, []
    for line in (folder / "episode_results.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            key = (row["env_name"], row["episode"])
            if key in rows:
                duplicates.append(list(key))
            rows[key] = {k: v for k, v in row.items() if k not in IGNORED_RESULT_KEYS}
    return rows, duplicates


def is_image(dataset: h5py.Dataset) -> bool:
    return dataset.dtype == np.uint8 and dataset.ndim >= 3


def compare_hdf5(base: Path, cand: Path) -> dict:
    mismatches, images = [], {}
    with h5py.File(base) as fb, h5py.File(cand) as fc:
        names_b, names_c = ["/"], ["/"]  # "/" checks the root attributes
        fb.visit(names_b.append)
        fc.visit(names_c.append)
        if sorted(names_b) != sorted(names_c):
            mismatches.append({"layout": sorted(set(names_b) ^ set(names_c))})
        for name in names_b:
            if name not in fc:
                continue
            ob, oc = fb[name], fc[name]
            attrs_b = {k: np.asarray(v).tolist() for k, v in ob.attrs.items() if "time" not in k and "recorded_at" not in k}
            attrs_c = {k: np.asarray(v).tolist() for k, v in oc.attrs.items() if "time" not in k and "recorded_at" not in k}
            if attrs_b != attrs_c:
                mismatches.append({"attrs": name})
            if not isinstance(ob, h5py.Dataset):
                continue
            if is_image(ob):
                if ob.shape != oc.shape:
                    mismatches.append({"shape": name})
                    continue
                total, peak = 0.0, 0
                for i in range(0, ob.shape[0], 16):  # chunks keep 720p video stacks out of RAM
                    diff = np.abs(ob[i:i + 16].astype(np.int16) - oc[i:i + 16].astype(np.int16))
                    total += float(diff.sum(dtype=np.float64))
                    peak = max(peak, int(diff.max()))
                images[name] = {"mean_abs": total / max(ob.size, 1), "max_abs": peak, "frames": int(ob.shape[0])}
                continue
            a, b = ob[()], oc[()]
            if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
                mismatches.append({"dataset": name})
    return {"non_image_mismatches": mismatches, "images": images}


def compare(base_dir: Path, cand_dir: Path) -> dict:
    (rows_b, dup_b), (rows_c, dup_c) = load_rows(base_dir), load_rows(cand_dir)
    row_diffs = [
        {"key": list(key), "base": rows_b.get(key), "candidate": rows_c.get(key)}
        for key in sorted(set(rows_b) | set(rows_c))
        if rows_b.get(key) != rows_c.get(key)
    ]
    tasks = {}
    for task in sorted({key[0] for key in rows_b}):
        tb, tc = base_dir / task, cand_dir / task
        result = {"runs": {}, "missing": []}
        names_b = {p.name for p in tb.glob("run_*.hdf5")}
        names_c = {p.name for p in tc.glob("run_*.hdf5")}
        result["missing"] = sorted(names_b - names_c)
        result["extra"] = sorted(names_c - names_b)
        for name in sorted(names_b & names_c):
            result["runs"][name] = compare_hdf5(tb / name, tc / name)
        result["videos_equal"] = {p.name for p in tb.glob("*.mp4")} == {p.name for p in tc.glob("*.mp4")}
        cfg_b, cfg_c = tb / "env_cfg.json", tc / "env_cfg.json"
        result["env_cfg_equal"] = cfg_c.is_file() and _cfg(cfg_b) == _cfg(cfg_c)
        result["success"] = {str(k[1]): rows_b[k].get("success") for k in rows_b if k[0] == task}
        tasks[task] = result
    passed = not row_diffs and not dup_b and not dup_c and all(
        not t["missing"] and not t["extra"] and t["videos_equal"] and t["env_cfg_equal"]
        and all(not r["non_image_mismatches"] for r in t["runs"].values())
        for t in tasks.values()
    )
    return {"base": str(base_dir), "candidate": str(cand_dir), "passed": passed,
            "duplicate_rows": {"base": dup_b, "candidate": dup_c},
            "result_row_diffs": row_diffs, "tasks": tasks}


def _cfg(path: Path) -> object:
    """env_cfg.json without values that hold the output path."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in ("dataset_export_dir_path", "output_dir")}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    return strip(json.loads(path.read_text()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.base, args.candidate)
    args.output.write_text(json.dumps(report, indent=1))
    summary = {task: {run: {k: round(v["mean_abs"], 3) for k, v in r["images"].items()}
                      for run, r in t["runs"].items()} for task, t in report["tasks"].items()}
    print(json.dumps({"passed": report["passed"], "row_diffs": len(report["result_row_diffs"]),
                      "image_mean_abs": summary}, indent=1))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
