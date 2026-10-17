# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-episode benchmark of the policy query modes with the sleeping mock policy server.

Runs in Isaac Lab's Python. Starts ``scripts/mock_policy_server.py`` as a
child process, creates one env (``--num-envs`` parallel envs) and runs one
full episode per entry of ``--modes`` through ``robolab.eval.episode.run_episode``
with a ``MockPolicyClient``. The mock chunk is a function of the request
(both camera images, arm and gripper state, prompt), so a mode that sends
other observations or mixes up envs applies other actions.

For each episode the script stores the applied actions and the arm joint
positions of every step (``<mode-index>_<mode>.npz``) and reports the episode
wall time, the ``run_episode`` timing breakdown and whether actions and joint
positions are equal to the first episode. Repeat ``serial`` to see the
episode-to-episode noise of the simulation itself.

The script imports ``robolab`` from ``PYTHONPATH`` (``gpujob`` sets it), so
the same file can measure a tree without query modes: there every mode is
the serial loop. Example::

    python.sh scripts/benchmark_policy_queries_env.py --num-envs 4 --modes serial auto batch serial \\
        --video-mode sensor --output /tmp/pq_env4
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2  # noqa: F401  must be imported before isaaclab
from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="MustardInRightBinTask")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--modes", nargs="+", default=["serial", "auto", "batch"],
                    choices=["auto", "serial", "concurrent", "batch"], help="One episode per entry, in order.")
parser.add_argument("--workers", type=int, default=16, help="--policy-query-workers.")
parser.add_argument("--server-mode", choices=["parallel", "serialized"], default="parallel",
                    help="parallel: the server computes all connections at the same time; serialized: one at a time.")
parser.add_argument("--min-ms", type=float, default=20.0)
parser.add_argument("--max-ms", type=float, default=80.0)
parser.add_argument("--horizon", type=int, default=10, help="Action chunk length (open-loop horizon).")
parser.add_argument("--port", type=int, default=6223)
parser.add_argument("--video-mode", choices=["none", "sensor"], default="sensor")
parser.add_argument("--limit-cpu-threads", "--limit_cpu_threads", type=int, default=4)
parser.add_argument("--output", required=True, help="New output directory.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mock_policy_server as mock  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.eval.episode import run_episode  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402


class RecordingClient(mock.MockPolicyClient):
    """Mock client that keeps the actions it returns (the actions ``run_episode`` applies)."""

    def begin_episode(self, episode_idx):
        super().begin_episode(episode_idx)
        self.actions, self.joints, self.query_steps = [], [], 0

    def infer_batch(self, obs, instruction, *, env_ids):
        self.query_steps += any(self._needs_refresh(env_id) for env_id in env_ids)
        rets = super().infer_batch(obs, instruction, env_ids=env_ids)
        step_actions = np.full((args.num_envs, mock.ARM_DOF + 1), np.nan, dtype=np.float32)
        for env_id, ret in rets.items():
            step_actions[env_id] = ret["action"]
        self.actions.append(step_actions)
        self.joints.append(obs["proprio_obs"]["arm_joint_pos"].detach().cpu().numpy().copy())
        return rets


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
set_output_dir(str(out))
auto_register_droid_envs(task=[args.task])
server = mock.start_server_process(args.port, args.min_ms, args.max_ms, args.horizon,
                                   1 if args.server_mode == "serialized" else 0)
env = None
records = []
try:
    env, env_cfg = create_env(args.task, num_envs=args.num_envs, seed=0, device=args.device)
    has_modes = hasattr(mock.InferenceClient, "configure_queries")
    reference = None
    for index, mode in enumerate(args.modes):
        client = RecordingClient(args.port, open_loop_horizon=args.horizon)
        if has_modes:
            client.configure_queries(mode=mode, max_workers=args.workers)
        env.reset_eval_state()
        start = time.perf_counter()
        env_results, _, timing = run_episode(env, env_cfg, index, client, headless=True,
                                             save_videos=args.video_mode != "none", video_mode=args.video_mode)
        if env.device != "cpu":
            torch.cuda.synchronize()
        episode_s = time.perf_counter() - start
        effective = client._effective_query_mode() if has_modes else "serial"
        client.close()
        actions, joints = np.stack(client.actions), np.stack(client.joints)
        np.savez(out / f"{index}_{mode}.npz", actions=actions, joints=joints)
        if reference is None:
            reference = (actions, joints)
        record = {
            "index": index,
            "mode": mode,
            "effective_mode": effective,
            "episode_s": episode_s,
            "steps": len(actions),
            "query_steps": client.query_steps,
            "timing": timing,
            "results": env_results,
            "actions_sha256": sha(actions),
            "joints_sha256": sha(joints),
            "actions_equal_first": bool(np.array_equal(actions, reference[0], equal_nan=True)),
            "joints_equal_first": bool(np.array_equal(joints, reference[1])),
        }
        records.append(record)
        print(f"EPISODE {index} {mode}->{effective}: {episode_s:.2f} s, {len(actions)} steps, "
              f"actions_equal_first={record['actions_equal_first']} joints_equal_first={record['joints_equal_first']}",
              flush=True)
    payload = {
        "benchmark": "policy_queries_env",
        "config": {key: value for key, value in vars(args).items() if isinstance(value, (int, float, str, list))},
        "query_modes_available": has_modes,
        "torch_num_threads": torch.get_num_threads(),
        "episodes": records,
    }
    (out / "measurement.json").write_text(json.dumps(payload, indent=2, default=str))
    print("PQ_ENV_RESULT", json.dumps({r["mode"] + str(r["index"]): round(r["episode_s"], 3) for r in records}),
          flush=True)
except BaseException:
    import traceback

    (out / "failure.txt").write_text(traceback.format_exc())
    traceback.print_exc()
    raise
finally:
    server.kill()
    server.wait()
    if env is not None:
        env.close()
    app.close()
