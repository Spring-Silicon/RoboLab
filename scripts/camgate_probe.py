"""Per-camera render cost and render-product gating checks (camgate track probe, not committed).

1. Lists the camera sensors, their resolutions and the camera observation groups.
2. Times env.step and sim.render with each set of render products enabled (all, none, only X, all but X),
   with and without reading the enabled groups back.
3. Freshness: replays the same actions three times from reset: twice rendering every camera on every step
   (reference and RTX noise), once with the image_obs cameras disabled except every 8th step. Compares the
   gated images with the reference images of the same step, the previous step and 8 steps earlier.
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
p.add_argument("--steps", type=int, default=40)
p.add_argument("--fresh-steps", type=int, default=32)
p.add_argument("--actions", default="/workspace/perf/replay_actions.npy")
p.add_argument("--output", required=True)
p.add_argument("--limit-cpu-threads", type=int, default=4)
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.config import parse_env_cfg  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
set_output_dir(str(out))
auto_register_droid_envs(task=[a.task])
(env_name,) = sorted(get_envs(task=[a.task]))
cfg = parse_env_cfg(env_name, device=a.device, seed=0, num_envs=1, env_spacing=None, eye=None, lookat=None,
                    use_fabric=None)
env, cfg = create_env(cfg, policy="camgate-probe")
actions = torch.as_tensor(np.load(a.actions), device=env.device).reshape(-1, env.action_manager.total_action_dim)
om = env.observation_manager
group_sensors = om.camera_group_sensors
cameras = {name: env.scene.sensors[name] for names in group_sensors.values() for name in names}
result = {
    "task": a.task,
    "device": a.device,
    "groups": {g: list(s) for g, s in group_sensors.items()},
    "cameras": {n: {"width": c.cfg.width, "height": c.cfg.height, "type": type(c).__name__} for n, c in cameras.items()},
}
print("CAMERAS", json.dumps(result), flush=True)

render_times = []
_render = env.sim.render


def timed_render(*args, **kwargs):
    start = time.perf_counter()
    ret = _render(*args, **kwargs)
    render_times.append(time.perf_counter() - start)
    return ret


env.sim.render = timed_render
counter = 0


def step(**options):
    global counter
    action = actions[counter % len(actions)].unsqueeze(0)
    counter += 1
    return env.step(action, **options)


def set_enabled(enabled):
    for name, cam in cameras.items():
        cam.set_render_product_enabled(name in enabled)


def readable_groups(enabled):
    return tuple(g for g, names in group_sensors.items() if all(n in enabled for n in names))


def readable(cam):
    try:
        cam.data.output
        return True
    except RuntimeError:
        return False


def time_config(enabled, read, steps):
    set_enabled(enabled)
    groups = readable_groups(enabled) if read else ()
    # Render without reads until the enabled cameras deliver frames (render products enabled again can lag).
    for _ in range(5):
        step(render=True, camera_observations=())
    for _ in range(10):
        if all(readable(cameras[n]) for n in enabled):
            break
        step(render=True, camera_observations=())
    for _ in range(5):
        step(render=True, camera_observations=groups)
    render_times.clear()
    walls = []
    for _ in range(steps):
        start = time.perf_counter()
        step(render=True, camera_observations=groups)
        walls.append(time.perf_counter() - start)
    return {
        "enabled": sorted(enabled),
        "read_groups": list(groups),
        "step_ms_median": 1e3 * float(np.median(walls)),
        "step_ms_mean": 1e3 * float(np.mean(walls)),
        "render_ms_median": 1e3 * float(np.median(render_times)),
        "render_ms_mean": 1e3 * float(np.mean(render_times)),
        "renders": len(render_times),
    }


def delivery_lag(name, max_renders=10):
    """Renders after enabling the render product of `name` again until its annotators deliver a frame."""
    set_enabled(set(names) - {name})
    for _ in range(3):
        step(render=True, camera_observations=())
    set_enabled(set(names))
    for renders in range(1, max_renders + 1):
        step(render=True, camera_observations=())
        if readable(cameras[name]):
            return renders
    return None


env.reset()
env.reset()
names = list(cameras)
lags = {n: [delivery_lag(n) for _ in range(3)] for n in names}
result["delivery_lag_renders"] = lags
print("LAGS", json.dumps(lags), flush=True)
if any(lag is None for v in lags.values() for lag in v):
    raise RuntimeError(f"A render product delivered no frame within 10 renders after enabling: {lags}")
# Enable the gated cameras this many steps before the consumed step (that step renders too).
gate_warmup = max(lag for v in lags.values() for lag in v) - 1
result["gate_warmup_steps"] = gate_warmup
set_enabled(set(names))
configs = [("all", set(names)), ("none", set())]
configs += [(f"only_{n}", {n}) for n in names]
configs += [(f"all_but_{n}", set(names) - {n}) for n in names]
timing = {}
for read in (False, True):
    for label, enabled in configs:
        timing[f"{'read' if read else 'render'}:{label}"] = time_config(enabled, read, a.steps)
        print("TIMING", label, read, json.dumps(timing[f"{'read' if read else 'render'}:{label}"]), flush=True)
# render=False reference (no render at all)
set_enabled(set(names))
for _ in range(5):
    step(render=False, camera_observations=False)
walls = []
for _ in range(a.steps):
    start = time.perf_counter()
    step(render=False, camera_observations=False)
    walls.append(time.perf_counter() - start)
timing["no_render"] = {"step_ms_median": 1e3 * float(np.median(walls)), "step_ms_mean": 1e3 * float(np.mean(walls))}
result["timing"] = timing
set_enabled(set(names))


def images(obs, groups):
    return {
        term: value[0].cpu().numpy().astype(np.int16)
        for g in groups
        for term, value in obs[g].items()
    }


def episode(gated):
    """Replays the first fresh-steps actions from reset; returns per-step images and joint states."""
    global counter
    set_enabled(set(names))
    env.reset()
    env.reset()
    counter = 0
    gated_names = set(group_sensors.get("image_obs", ()))
    imgs, joints, raised = [], [], []
    for t in range(a.fresh_steps):
        consumed = t % 8 == 7
        enabled_now = t % 8 >= 7 - gate_warmup
        if gated:
            set_enabled(set(names) if enabled_now else set(names) - gated_names)
            groups = tuple(group_sensors) if consumed else readable_groups(set(names) - gated_names)
        else:
            groups = tuple(group_sensors)
        obs, *_ = step(render=True, camera_observations=groups)
        imgs.append(images(obs, groups))
        joints.append(env.scene["robot"].data.joint_pos[0].cpu().numpy().copy())
        if gated and not enabled_now:
            try:
                env.scene.sensors[next(iter(gated_names))].data.output["rgb"]
                raised.append(False)
            except RuntimeError:
                raised.append(True)
    set_enabled(set(names))
    return imgs, joints, raised


ref, ref_joints, _ = episode(False)
ref2, ref2_joints, _ = episode(False)
gated, gated_joints, raised = episode(True)


def mad(x, y):
    return float(np.abs(x - y).mean())


fresh = {"steps": [], "disabled_read_raised": raised}
for t in range(a.fresh_steps):
    if t % 8 != 7:
        continue
    row = {"step": t}
    for term, img in gated[t].items():
        row[term] = {
            "vs_ref_same_step": mad(img, ref[t][term]),
            "ref_vs_ref2_same_step": mad(ref2[t][term], ref[t][term]),
            "vs_ref_prev_step": mad(img, ref[t - 1][term]),
            "vs_ref_8_steps_earlier": mad(img, ref[t - 8][term]) if t >= 8 else None,
        }
    fresh["steps"].append(row)
# viewport frames on every gated step (rendered every step in both runs)
vp_terms = [term for g in group_sensors if g != "image_obs" for term in om.active_terms[g]]
fresh["every_step_terms"] = {
    term: {
        "gated_vs_ref_mean": float(np.mean([mad(gated[t][term], ref[t][term]) for t in range(a.fresh_steps)])),
        "ref2_vs_ref_mean": float(np.mean([mad(ref2[t][term], ref[t][term]) for t in range(a.fresh_steps)])),
        "gated_vs_ref_max": float(np.max([mad(gated[t][term], ref[t][term]) for t in range(a.fresh_steps)])),
        "ref2_vs_ref_max": float(np.max([mad(ref2[t][term], ref[t][term]) for t in range(a.fresh_steps)])),
    }
    for term in vp_terms
}
fresh["joint_max_abs_diff_gated_vs_ref"] = float(np.max(np.abs(np.array(gated_joints) - np.array(ref_joints))))
fresh["joint_max_abs_diff_ref2_vs_ref"] = float(np.max(np.abs(np.array(ref2_joints) - np.array(ref_joints))))
result["freshness"] = fresh
(out / "probe.json").write_text(json.dumps(result, indent=2))
print("PROBE_RESULT", json.dumps(result), flush=True)
env.close()
app.close()
