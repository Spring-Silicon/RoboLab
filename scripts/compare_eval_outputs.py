"""Compare two evaluation output folders task by task.

Usage: python scripts/compare_eval_outputs.py BASE_DIR CANDIDATE_DIR --output comparison.json

Both folders come from robolab.eval.runner (for example scripts/scripted_eval.py). For each
task folder in BASE_DIR, the script compares:
  * episode_results.jsonl rows (all fields except timing and duration); a key
    (env_name, episode) that appears twice in one folder is a failure,
  * every run_*.hdf5: the same files on both sides; all non-image datasets and dataset
    attributes must be bit-identical (attributes whose name contains "time" or
    "recorded_at" are skipped); camera images (uint8 arrays with 3 or more dimensions)
    must have the same shape and give mean and max absolute pixel differences, because
    RTX images are not bitwise repeatable between processes,
  * the *.mp4 videos: the same names, frame counts (more than 0) and frame shapes;
    decoded frames give mean and max absolute pixel differences, the count of bit-identical
    frames, and the difference against a one-frame shift (alignment check; needs OpenCV),
  * env_cfg.json (without output paths and Python object addresses).
Exit status is 1 if a non-image value, an image shape, or a video frame count or shape
differs, a video has no frames, or a file is missing or extra.
"""
from __future__ import annotations

import argparse
import json
import re
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
        videos_b = {p.name for p in tb.glob("*.mp4")}
        result["videos_equal"] = videos_b == {p.name for p in tc.glob("*.mp4")}
        result["videos"] = {name: compare_video(tb / name, tc / name) for name in sorted(videos_b) if (tc / name).is_file()}
        cfg_b, cfg_c = tb / "env_cfg.json", tc / "env_cfg.json"
        result["env_cfg_equal"] = cfg_c.is_file() and _cfg(cfg_b) == _cfg(cfg_c)
        result["success"] = {str(k[1]): rows_b[k].get("success") for k in rows_b if k[0] == task}
        tasks[task] = result
    passed = not row_diffs and not dup_b and not dup_c and all(
        not t["missing"] and not t["extra"] and t["videos_equal"] and t["env_cfg_equal"]
        and all(v["frames"][0] == v["frames"][1] and v["frames"][0] > 0 and v["shape_equal"]
                for v in t["videos"].values())
        and all(not r["non_image_mismatches"] for r in t["runs"].values())
        for t in tasks.values()
    )
    return {"base": str(base_dir), "candidate": str(cand_dir), "passed": passed,
            "duplicate_rows": {"base": dup_b, "candidate": dup_c},
            "result_row_diffs": row_diffs, "tasks": tasks}


def compare_video(base: Path, cand: Path) -> dict:
    """Frame counts and shape (must match, non-empty), mean / max absolute pixel difference of the frames,
    the number of bit-identical frames, the worst per-frame mean difference, and the mean difference to the
    neighbouring frame (base frame i against candidate frames i-1 and i+1): aligned frames give a much
    smaller difference than the one-frame shift."""
    import cv2

    cap_b, cap_c = cv2.VideoCapture(str(base)), cv2.VideoCapture(str(cand))
    frames_b = frames_c = 0
    total, pixels, peak = 0.0, 0, 0
    identical, worst_frame = 0, 0.0
    shifted_total, shifted_pixels = 0.0, 0
    prev_b = prev_c = None
    shape_equal = True
    try:
        while True:
            ok_b, img_b = cap_b.read()
            ok_c, img_c = cap_c.read()
            frames_b += ok_b
            frames_c += ok_c
            if not (ok_b and ok_c):
                break
            if img_b.shape != img_c.shape:
                shape_equal = False
                prev_b = prev_c = None
                continue
            diff = cv2.absdiff(img_b, img_c)
            frame_sum = float(diff.sum(dtype=np.float64))
            total += frame_sum
            pixels += diff.size
            peak = max(peak, int(diff.max()))
            identical += frame_sum == 0
            worst_frame = max(worst_frame, frame_sum / diff.size)
            if prev_b is not None and prev_b.shape == img_b.shape:
                for x, y in ((img_b, prev_c), (prev_b, img_c)):
                    shifted_total += float(cv2.absdiff(x, y).sum(dtype=np.float64))
                    shifted_pixels += diff.size
            prev_b, prev_c = img_b, img_c
        while ok_b:
            ok_b, _ = cap_b.read()
            frames_b += ok_b
        while ok_c:
            ok_c, _ = cap_c.read()
            frames_c += ok_c
    finally:
        cap_b.release()
        cap_c.release()
    return {"frames": [frames_b, frames_c], "shape_equal": shape_equal,
            "mean_abs": total / max(pixels, 1), "max_abs": peak,
            "identical_frames": identical, "worst_frame_mean_abs": worst_frame,
            "shift1_mean_abs": shifted_total / max(shifted_pixels, 1)}


def _cfg(path: Path) -> object:
    """env_cfg.json without values that hold the output path or a Python object address."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in ("dataset_export_dir_path", "output_dir")}
        if isinstance(value, list):
            return [strip(v) for v in value]
        if isinstance(value, str):
            return re.sub(r" at 0x[0-9a-f]+", " at 0x", value)
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
    images = {task: {run: {k: round(v["mean_abs"], 3) for k, v in r["images"].items()}
                     for run, r in t["runs"].items()} for task, t in report["tasks"].items()}
    videos = {task: {name: {"frames": v["frames"], "mean_abs": round(v["mean_abs"], 3), "max_abs": v["max_abs"],
                            "identical_frames": v["identical_frames"],
                            "worst_frame_mean_abs": round(v["worst_frame_mean_abs"], 3),
                            "shift1_mean_abs": round(v["shift1_mean_abs"], 3)}
                     for name, v in t["videos"].items()} for task, t in report["tasks"].items()}
    print(json.dumps({"passed": report["passed"], "row_diffs": len(report["result_row_diffs"]),
                      "image_mean_abs": images, "videos": videos}, indent=1))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
