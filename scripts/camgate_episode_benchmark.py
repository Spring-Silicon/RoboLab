"""Full-episode timing for render gating (camgate track measurement script).

Same workload as /workspace/perf/evaluation_benchmark.py (original task horizon and checks, scripted joint and
gripper targets, no network), but the client is an action-chunking ``InferenceClient``: it queries its "policy"
every --open-loop-horizon steps and, like real clients, extracts the proprio and image_obs observations as numpy
on every step whose observation groups are rendered. The action sequence does not depend on the horizon or on
gating, so the physics is the same in every configuration.

Writes OUTPUT/measurement.json, run_N.hdf5, the episode videos, and OUTPUT/obs/run_N.npz with the per-step joint
state and, on the query steps, a 1/8 downsampled copy of every image_obs image the client extracted.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2  # noqa: F401  (before Isaac imports, as in the RoboLab scripts)
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="MustardInRightBinTask")
p.add_argument("--num-envs", type=int, default=1)
p.add_argument("--repeats", type=int, default=1)
p.add_argument("--output", required=True)
p.add_argument("--video-mode", choices=["all", "viewport", "sensor", "none"], default="sensor")
p.add_argument("--open-loop-horizon", type=int, default=8)
p.add_argument("--render-policy-steps-only", action="store_true")
p.add_argument("--render-warmup-steps", type=int, default=0)
p.add_argument("--record-image-data", action="store_true")
p.add_argument("--limit-cpu-threads", "--limit_cpu_threads", type=int, default=4)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
a.enable_cameras = True
app = AppLauncher(a).app


def _exit_on_error(*exc_info):
    # Kit keeps the process alive after an uncaught exception; exit so the GPU lock is released.
    import os
    import traceback

    traceback.print_exception(*exc_info)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = _exit_on_error

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.logging.results import init_experiment  # noqa: E402
from robolab.eval.base_client import InferenceClient  # noqa: E402
from robolab.eval.episode import run_episode  # noqa: E402
from robolab.eval.summarize import summarize_run  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402


class Client(InferenceClient):
    """Chunks of the evaluation_benchmark.py scripted actions; extracts observations like a real client."""

    def __init__(self, horizon):
        super().__init__()
        self.open_loop_horizon = horizon
        self.initial = None
        self.next_step = {}
        self.log = None

    def begin_episode(self, episode_idx):
        super().begin_episode(episode_idx)
        self.reset()
        self.log = {"query_steps": [], "image_small": [], "joint_pos": []}

    def reset(self, *, env_id=None):
        super().reset(env_id=env_id)
        if env_id is None:
            self.initial = None
            self.next_step.clear()

    def infer(self, obs, instruction, *, env_id=0):
        if env_id == 0:
            self.log["joint_pos"].append(obs["proprio_obs"]["arm_joint_pos"][0].detach().cpu().numpy().copy())
        if not self._needs_refresh(env_id) and self._has_unrendered_groups(obs):
            return {"action": self._next_action(env_id), "viz": None}
        extracted = self._extract_observation(obs, env_id=env_id)
        if self._needs_refresh(env_id):
            self._set_chunk(env_id, self._chunk(extracted, env_id))
        return {"action": self._next_action(env_id), "viz": None}

    def _chunk(self, extracted, env_id):
        joints = extracted["arm_joint_pos"]
        if self.initial is None:
            self.initial = np.concatenate([joints, np.zeros(1, dtype=joints.dtype)])
        step = self.next_step.get(env_id, 0)
        steps = np.arange(step, step + self.open_loop_horizon)
        chunk = np.repeat(self.initial[None], self.open_loop_horizon, axis=0)
        chunk[:, 0] += 0.04 * np.sin(steps * 0.05)
        chunk[:, 7] = ((steps // 50) % 2).astype(chunk.dtype)
        self.next_step[env_id] = step + self.open_loop_horizon
        if env_id == 0:
            self.log["query_steps"].append(step)
            self.log["image_small"].append(
                np.stack([extracted[k][::8, ::8, :3] for k in sorted(extracted) if k != "arm_joint_pos"])
            )
        return chunk

    def _extract_observation(self, raw_obs, *, env_id=0):
        extracted = {"arm_joint_pos": self._to_numpy(raw_obs["proprio_obs"]["arm_joint_pos"], env_id)}
        for name, value in raw_obs["image_obs"].items():
            extracted[name] = self._to_numpy(value, env_id)
        return extracted

    def _pack_request(self, extracted_obs, instruction):
        raise NotImplementedError

    def _query_server(self, request):
        raise NotImplementedError

    def _unpack_response(self, response):
        raise NotImplementedError


out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
(out / "obs").mkdir()
set_output_dir(str(out))
robolab.constants.RECORD_IMAGE_DATA = a.record_image_data
auto_register_droid_envs(task=[a.task])
result_file, results = init_experiment(str(out))
records = []
env = None
gating = {}
if a.render_policy_steps_only:
    gating = {"render_policy_steps_only": True, "render_warmup_steps": a.render_warmup_steps}
try:
    created = time.perf_counter()
    env, cfg = create_env(a.task, num_envs=a.num_envs, seed=0, device=a.device)
    create_s = time.perf_counter() - created
    client = Client(a.open_loop_horizon)
    for run in range(a.repeats):
        env.reset_eval_state()
        client.begin_episode(run)
        start = time.perf_counter()
        start_unix = time.time()
        env_results, msgs, timing = run_episode(
            env, cfg, run, client, headless=True, save_videos=a.video_mode != "none", video_mode=a.video_mode, **gating
        )
        episode_end = time.perf_counter()
        assert all(r["step"] == env.max_episode_length and r["success"] is False for r in env_results), env_results
        results = summarize_run(
            env_results=env_results, msgs=msgs, env=env, env_cfg=cfg, num_envs=a.num_envs, run_idx=run,
            run_name=f"{a.task}_{run}", task_env=a.task, scene_output_dir=str(out),
            policy="fixed-scripted-chunks-no-model", episode_results=results, episode_results_file=result_file,
            enable_subtask_progress=True, timing=timing,
        )
        finished = time.perf_counter()
        with h5py.File(out / f"run_{run}.hdf5") as f:
            counts = {k: int(v.attrs["num_samples"]) for k, v in f["data"].items()}
            assert len(counts) == a.num_envs and all(n == env.max_episode_length for n in counts.values()), counts
        log = client.log
        np.savez_compressed(
            out / "obs" / f"run_{run}.npz", query_steps=np.array(log["query_steps"]),
            image_small=np.stack(log["image_small"]),
            joint_pos=np.stack(log["joint_pos"]),
        )
        records.append({
            "run": run, "start_unix": start_unix, "end_unix": time.time(), "external_episode_s": episode_end - start,
            "summary_s": finished - episode_end, "evaluation_s": finished - start, "timing": timing,
            "results": env_results, "recorded_samples": counts, "queries": len(log["query_steps"]),
        })
    payload = {
        "task": a.task, "num_envs": a.num_envs, "video_mode": a.video_mode, "open_loop_horizon": a.open_loop_horizon,
        "render_policy_steps_only": a.render_policy_steps_only, "render_warmup_steps": a.render_warmup_steps,
        "record_image_data": a.record_image_data, "create_s": create_s, "runs": records,
        "device": a.device, "torch_num_threads": torch.get_num_threads(),
        "contract": "original task horizon and task checks; scripted joint/gripper chunks; no policy/network",
    }
    (out / "measurement.json").write_text(json.dumps(payload, indent=2))
    print("EVAL_PROBE_RESULT", json.dumps(payload), flush=True)
except BaseException:
    import traceback

    (out / "failure.txt").write_text(traceback.format_exc())
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
