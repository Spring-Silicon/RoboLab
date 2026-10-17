# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU benchmark of the policy query modes with the sleeping mock policy server.

No Isaac Sim. For each env count, server mode and query mode, the script runs
``--steps`` control steps of ``InferenceClient.infer_batch`` with synthetic
production-size observations (two 720x1280 RGB cameras, arm and gripper state)
against ``scripts/mock_policy_server.py``. It times only ``infer_batch`` and
checks that every mode applies the same actions as the serial loop.

Server modes: ``parallel`` (the server computes all connections at the same
time, like several replicas) and ``serialized`` (one request at a time, like a
single-process openpi or GR00T server). A batched request costs one latency.

Example::

    python scripts/benchmark_policy_queries.py --num-envs 1 4 10 --output /tmp/policy_queries.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mock_policy_server as mock  # noqa: E402

SERVER_MODES = {"parallel": 0, "serialized": 1}
QUERY_MODES = ("serial", "concurrent", "batch")


def make_obs(num_envs: int, step: int, base_images: np.ndarray) -> dict:
    arm = np.linspace(-1.0, 1.0, num_envs * 7, dtype=np.float32).reshape(num_envs, 7) + 0.01 * step
    return {
        "image_obs": {
            "over_shoulder_left_camera": base_images + np.uint8(step % 7),
            "wrist_cam": base_images[:, ::-1] + np.uint8(step % 5),
        },
        "proprio_obs": {"arm_joint_pos": arm, "gripper_pos": np.zeros((num_envs, 1), dtype=np.float32)},
    }


def run_case(port: int, num_envs: int, mode: str, args, base_images: np.ndarray) -> dict:
    client = mock.MockPolicyClient(port, open_loop_horizon=args.horizon)
    client.configure_queries(mode=mode, max_workers=args.workers)
    step_ms, query_step_ms, actions = [], [], []
    try:
        for step in range(args.steps):
            obs = make_obs(num_envs, step, base_images)
            queried = any(client._needs_refresh(env_id) for env_id in range(num_envs))
            start = time.perf_counter()
            rets = client.infer_batch(obs, args.instruction, env_ids=list(range(num_envs)))
            elapsed = (time.perf_counter() - start) * 1000.0
            step_ms.append(elapsed)
            if queried:
                query_step_ms.append(elapsed)
            actions.append(np.stack([rets[env_id]["action"] for env_id in sorted(rets)]))
    finally:
        client.close()
    actions = np.stack(actions)
    return {
        "query_mode": mode,
        "effective_mode": client._effective_query_mode(),
        "infer_ms_per_control_step": statistics.fmean(step_ms),
        "infer_ms_per_query_step_mean": statistics.fmean(query_step_ms),
        "infer_ms_per_query_step_median": statistics.median(query_step_ms),
        "query_steps": len(query_step_ms),
        "total_infer_s": sum(step_ms) / 1000.0,
        "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "_actions": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 4, 10])
    parser.add_argument("--server-modes", nargs="+", choices=sorted(SERVER_MODES), default=["parallel", "serialized"])
    parser.add_argument("--steps", type=int, default=40, help="Control steps per case.")
    parser.add_argument("--horizon", type=int, default=10, help="Action chunk length (open-loop horizon).")
    parser.add_argument("--min-ms", type=float, default=20.0)
    parser.add_argument("--max-ms", type=float, default=80.0)
    parser.add_argument("--workers", type=int, default=16, help="--policy-query-workers for 'concurrent'.")
    parser.add_argument("--image-hw", type=int, nargs=2, default=[720, 1280])
    parser.add_argument("--port", type=int, default=6123)
    parser.add_argument("--instruction", default="put the mustard in the right bin")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    results = []
    for server_mode in args.server_modes:
        port = args.port + list(SERVER_MODES).index(server_mode)
        proc = mock.start_server_process(port, args.min_ms, args.max_ms, args.horizon, SERVER_MODES[server_mode])
        try:
            for num_envs in args.num_envs:
                base_images = rng.integers(0, 200, size=(num_envs, *args.image_hw, 3), dtype=np.uint8)
                cases = [run_case(port, num_envs, mode, args, base_images) for mode in QUERY_MODES]
                serial = cases[0]
                reference = serial["_actions"]
                for case in cases:
                    case["actions_equal_serial"] = bool(np.array_equal(case["_actions"], reference))
                for case in cases:
                    case.pop("_actions")
                for case in cases:
                    case.update(server_mode=server_mode, num_envs=num_envs,
                                speedup_vs_serial=serial["infer_ms_per_control_step"] / case["infer_ms_per_control_step"])
                    results.append(case)
                    print(f"{server_mode:10s} envs={num_envs:2d} {case['query_mode']:10s} "
                          f"{case['infer_ms_per_control_step']:8.2f} ms/step "
                          f"{case['infer_ms_per_query_step_mean']:8.2f} ms/query-step "
                          f"x{case['speedup_vs_serial']:.2f} equal={case['actions_equal_serial']}", flush=True)
        finally:
            proc.kill()
            proc.wait()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "benchmark": "policy_queries_mock",
        "host": platform.node(),
        "python": sys.version.split()[0],
        "config": {key: value for key, value in vars(args).items() if key != "output"},
        "results": results,
    }, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
