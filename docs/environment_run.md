# Running Environments

Once you have [registered environments](environment_registration.md), this page covers how to create, step, and evaluate them — from basic single-task runs to full multi-task benchmark evaluation.

## Creating an Environment

Use `create_env` to create and initialize an environment from a registered name or a configuration object. This supports running multiple environments sequentially without restarting the simulation.

```python
from robolab.core.environments.runtime import create_env

env, env_cfg = create_env(
    "BananaInBowlTask",       # Registered environment name
    device="cuda:0",
    num_envs=1,
    use_fabric=True,
)

obs, _ = env.reset()
# ... step the environment ...
env.close()
```

**`create_env` parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scene` | `str` or `ManagerBasedEnvCfg` | *(required)* | Registered environment name, or a pre-built env config object |
| `device` | `str` | `"cuda:0"` | Device for simulation |
| `seed` | `int` | `0` | Random seed |
| `num_envs` | `int` | `1` | Number of parallel environments |
| `env_spacing` | `float` | `5.0` | Distance in meters between environment origins (see [Multi-Environment Configuration](#multi-environment-configuration)) |
| `use_fabric` | `bool` | `True` | Use Fabric for physics (see IsaacLab docs) |
| `events` | `dict` or configclass | `None` | Event configuration to merge into the environment (e.g., pose randomization, camera variation) |
| `instruction_type` | `str` | `"default"` | Instruction variant to use when the task defines multiple variants |
| `policy` | `str` | `None` | Policy backend name, stored on `env_cfg` for downstream use |
| `renderer` | `str` | `"realtime"` | RTX renderer: `"realtime"` (RaytracedLighting) or `"pathtracing"` (PathTracing). Recorded in `env_cfg.json` (see [Renderer Selection](#renderer-selection)) |
| `rendering_mode` | `str` | `None` | Realtime quality preset (`"performance"`, `"balanced"`, `"quality"`); `None` leaves IsaacLab's default (`balanced`) |
| `eye` | `tuple` | `None` | Camera eye position override |
| `lookat` | `tuple` | `None` | Camera look-at position override |

## Querying Registered Environments

Use `get_envs` to retrieve environment names by task, tag, or all registered environments:

```python
from robolab.core.environments.factory import get_envs

all_envs = get_envs()                                # All registered environments
task_envs = get_envs(task="BananaInBowlTask")        # All variants of a task
task_envs = get_envs(task=["BananaInBowlTask", "RubiksCubeTask"])  # Multiple tasks
tag_envs = get_envs(tag="pick_place")                # All environments with a tag
tag_envs = get_envs(tag=["spatial", "simple"])        # Multiple tags
```

## Instruction Type Selection

When a task defines multiple instruction variants (see [task.md — Instruction Variants](task.md#3-instruction-variants)), you can select which variant to use at runtime:

```python
env, env_cfg = create_env(
    "BananaInBowlTask",
    device="cuda:0",
    num_envs=1,
    instruction_type="vague",
)

print(env_cfg.instruction)             # e.g., "Put stuff in the bowl"
print(env_cfg._instruction_variants)   # The raw instruction field (dict or str)
```

When `instruction_type` is not `"default"`, the type name is appended to the run name for distinguishable output files (e.g., `MyTask_vague_0.mp4`) and recorded in the episode results as `"instruction_type"`.

## Recorder Manager Patch

RoboLab uses a patched recorder manager for fine-grained control over HDF5 data recording. This must be set up **before** registering environments:

```python
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Patch recorder manager (before registration)
from robolab.core.logging.recorder_manager import patch_recorder_manager
patch_recorder_manager()

# Register environments (after patching)
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
auto_register_droid_envs()
```

At the end of each episode, explicitly call `end_episode` to flush recorded data to the HDF5 file:

```python
from robolab.core.environments.runtime import create_env, end_episode

env, env_cfg = create_env("BananaInBowlTask", device="cuda:0", num_envs=1, use_fabric=True)

# ... run episode ...

end_episode(env)   # Exports recorded data to run_*.hdf5
env.close()
```

## Multi-Environment Configuration

When running with `num_envs > 1`, IsaacLab spawns multiple copies of the scene in a grid layout. The `env_spacing` parameter controls the distance (in meters) between environment origins.

**Setting environment spacing:**

```python
env, env_cfg = create_env(
    "BananaInBowlTask",
    device="cuda:0",
    num_envs=20,
    env_spacing=5.0,        # 5 meters between environment origins (default)
)
```

Or when using `parse_env_cfg` directly:

```python
env_cfg = parse_env_cfg("BananaInBowlTask",
    device="cuda:0", num_envs=20, env_spacing=5.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_envs` | `int` | `1` | Number of parallel environments to spawn |
| `env_spacing` | `float` | `5.0` | Distance in meters between environment origins in the grid layout |

**Choosing `env_spacing`:** The spacing must be large enough that objects in adjacent environments do not overlap or interfere with each other. The default of 5.0m is sufficient for most tabletop manipulation tasks. Reduce it (e.g., 2.0m) for compact scenes to fit more environments in the viewport, or increase it if your scene has large objects or long-reach robot motions.

**Choosing `num_envs`:** The number of parallel environments is limited by GPU VRAM. As a rough guide, a single DROID tabletop environment with one camera uses ~1–2 GB; scale accordingly. For empirically measured per-task ceilings on L40 (48 GB), see the [num_envs VRAM size guide](env_vram_size_guide.md). If you run out of memory, reduce `num_envs` and increase `--num-runs` to reach the same total episode count:

```bash
# 20 episodes total: 20 envs × 1 run (if VRAM allows)
python policies/pi0_family/run.py --headless --num_envs 20

# 20 episodes total: 10 envs × 2 runs (if 20 doesn't fit)
python policies/pi0_family/run.py --headless --num_envs 10 --num-runs 2
```

**Multi-env episode handling:** With `num_envs > 1`, each environment runs an independent episode. The built-in `run_eval.py` handles per-env termination, video recording, and result logging automatically. If you are writing a custom evaluation loop, see `robolab/eval/episode.py` for the multi-env episode runner pattern (`from robolab.eval import run_episode`), which manages per-env video writers, independent termination tracking, and batched policy inference.

## Initial Condition Randomization

To randomize object poses between episodes:

1. Load the env config explicitly using `parse_env_cfg`
2. Add your desired randomization as an `events` parameter
3. Pass the modified config to `create_env`

```python
from robolab.core.environments.runtime import create_env
from robolab.core.environments.config import parse_env_cfg
from robolab.core.events.reset_pose import RandomizeInitPoseUniform

env_cfg = parse_env_cfg("BananaInBowlTask",
    device="cuda:0", seed=42, num_envs=1, use_fabric=True)

env_cfg.events = RandomizeInitPoseUniform.from_params(
    objects=["banana", "bowl"],
    pose_range={"x": (-0.1, 0.1), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
)

env, env_cfg = create_env(env_cfg,
    device="cuda:0", seed=42, num_envs=1, use_fabric=True)

# Between episodes, reset() will re-randomize poses according to the seed
```

You can also pass events directly to `create_env` using the `events` parameter. This is not encouraged, however.

```python
from robolab.core.events.reset_camera import RandomizeCameraPoseUniform

events = RandomizeCameraPoseUniform.from_params(
    cameras=["over_shoulder_left_camera"],
    pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
)
env, env_cfg = create_env("BananaInBowlTask", events=events)
```

## Writing an Evaluation Script

A typical evaluation script follows this pattern: register environments, loop over tasks and episodes, run inference, log results. The example below shows the full pipeline with multi-task evaluation, video recording, trajectory metrics, and result logging.

The registration import at the top determines which environments are available. For DROID with joint-position actions, use the built-in registration directly. If you set up a custom registration (see [Environment Registration](environment_registration.md)), import your own function instead.

```python
# run_eval.py

import argparse
import cv2  # Must import before isaaclab
import os
import re
import sys
import traceback

from isaaclab.app import AppLauncher
from robolab.constants import get_timestamp, DEFAULT_TASK_SUBFOLDERS

# ── CLI args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate my policy on RoboLab benchmark")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs="+", default=None, help="Specific task(s) to evaluate")
parser.add_argument("--tag", nargs="+", default=None, help="Tag(s) of tasks to evaluate")
parser.add_argument("--task-dirs", nargs="+", default=DEFAULT_TASK_SUBFOLDERS)
parser.add_argument("--num-runs", type=int, default=1)
parser.add_argument("--remote-host", type=str, default="localhost")
parser.add_argument("--remote-port", type=int, default=8000)
parser.add_argument("--output-folder-name", type=str, default=None)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports that require IsaacSim runtime ─────────────────────────────────
import torch
from tqdm import tqdm

import robolab.constants
from robolab.constants import PACKAGE_DIR, set_output_dir
from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env, end_episode
from robolab.core.logging.recorder_manager import patch_recorder_manager
from robolab.core.logging.results import (
    check_all_episodes_complete, check_run_complete,
    init_experiment, update_experiment_results, summarize_experiment_results,
)
from robolab.core.observations.observation_utils import unpack_image_obs
from robolab.core.utils.video_utils import VideoWriter

# ── Register environments ─────────────────────────────────────────────────
# Option A: Use the built-in DROID joint-position registration
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)

# Option B: Use your own custom registration (see environment_registration.md)
# from my_policy.register_envs import register_envs
# register_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)

# ── Import YOUR inference client ──────────────────────────────────────────
from my_policy.inference_client import MyPolicyClient

patch_recorder_manager()


def run_episode(env, env_cfg, client, episode, headless=False):
    """Run a single policy-controlled episode (single-env example).

    For multi-env, see robolab/eval/episode.py which handles per-env
    video writers, per-env policy clients, and independent termination.
    """
    obs, _ = env.reset()
    obs, _ = env.reset()
    max_steps = env.max_episode_length
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    instruction = env_cfg.instruction

    cleaned = re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")
    output_dir = robolab.constants.get_output_dir()
    # For multi-env (num_envs > 1), use f"{cleaned}_{episode}_env{env_id}.mp4"
    video_writer = VideoWriter(os.path.join(output_dir, f"{cleaned}_{episode}.mp4"), video_fps)

    success = None
    episode_step = 0

    for step in tqdm(range(max_steps)):
        ret = client.infer(obs, instruction)

        if not headless:
            cv2.imshow(instruction, cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        action = torch.tensor(ret["action"]).unsqueeze(0)
        obs, reward, term, trunc, info = env.step(action)

        frame = unpack_image_obs(obs, scale=0.5).get("combined_image")
        video_writer.write(frame)

        episode_step = step
        if term:
            success = True
            break
        if trunc:
            success = False
            break

    video_writer.release()
    client.reset()
    return success, episode_step


def main():
    if args_cli.output_folder_name is None:
        args_cli.output_folder_name = get_timestamp() + "_my_policy"
    output_dir = os.path.join(PACKAGE_DIR, "output", args_cli.output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    task_envs = get_envs(task=args_cli.task) if args_cli.task else get_envs(tag=args_cli.tag) if args_cli.tag else get_envs()
    episode_results_file, episode_results = init_experiment(output_dir)

    # Create the inference client once (shared across tasks)
    client = MyPolicyClient(remote_host=args_cli.remote_host, remote_port=args_cli.remote_port)

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        total_episodes = args_cli.num_runs * args_cli.num_envs
        if check_all_episodes_complete(episode_results=episode_results, env_name=task_env, num_episodes=total_episodes):
            print(f"[MyPolicy] Task `{task_env}` already done. Skipping.")
            continue

        env, env_cfg = create_env(task_env, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)

        for i in range(args_cli.num_runs):
            if check_run_complete(episode_results=episode_results, env_name=task_env, episode=i):
                continue

            print(f"[MyPolicy] Running {task_env} episode {i}: '{env_cfg.instruction}'")
            success, episode_step = run_episode(env, env_cfg, client, i, headless=args_cli.headless)

            end_episode(env)
            dt = env_cfg.sim.dt * env_cfg.decimation
            run_summary = {
                "env_name": task_env,
                "task_name": env_cfg._task_name,
                "episode": i,
                "instruction": env_cfg.instruction,
                "success": success,
                "episode_step": episode_step,
                "duration": episode_step * dt,
                "dt": dt,
            }
            episode_results = update_experiment_results(
                run_summary=run_summary,
                episode_results=episode_results,
                episode_results_file=episode_results_file,
            )

        env.close()

    summarize_experiment_results(episode_results)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[MyPolicy] Terminated with error: {e}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
```

## `run_eval.py` CLI Reference

The built-in `policies/pi0_family/run.py` supports the full set of evaluation features:

| Flag | Description | Default |
|------|-------------|---------|
| `--task TASK [TASK ...]` | Specific task name(s) to evaluate | all registered tasks |
| `--tag TAG [TAG ...]` | Evaluate all tasks with the given tag(s) | `None` |
| `--task-dirs DIR [DIR ...]` | Task directories to scan for registration | `DEFAULT_TASK_SUBFOLDERS` |
| `--policy BACKEND` | Policy backend (`pi0`, `pi0_fast`, `pi05`, `gr00t`, etc.) | `pi05` |
| `--num_envs N` | Number of parallel environments (= number of episodes per task) | `1` |
| `--num-runs N` | Number of sequential runs (default 1). Only increase if GPU memory limits `num_envs`. Total episodes = `num_runs * num_envs`. | `1` |
| `--instruction-type TYPE` | Instruction variant (`default`, `vague`, `specific`, etc.) | `default` |
| `--remote-host HOST` | Policy server host | `localhost` |
| `--remote-port PORT` | Policy server port | `8000` |
| `--output-folder-name NAME` | Output folder under `output/`. Reusing a previous folder skips completed episodes. | `<timestamp>_<policy>` |
| `--disable-subtask` | Disable subtask progress checking (drops score, reason, subtask log from results; tracking is on by default) | `False` |
| `--enable-gt-state` | Export per-env ground-truth simulator state to the inference client each step (`obs["gt_state"] = {env_id: state}`). See [Ground-Truth State Export](#ground-truth-state-export). | `False` |
| `--record-image-data` | Record image observations to HDF5 | `False` |
| `--video-mode MODE` | Which videos to save: `all` (sensor + viewport), `viewport` only, `sensor` only, or `none` | `all` |
| `--render-policy-steps-only` | Render cameras only on steps whose observation the policy client queries with (action-chunking clients). Requires `--video-mode none`. See [Rendering Only Policy Steps](#rendering-only-policy-steps). | `False` |
| `--render-warmup-steps N` | With `--render-policy-steps-only`, also render (without camera read-back) the `N` steps before each consumed step. See [Rendering Only Policy Steps](#rendering-only-policy-steps). | `0` |
| `--renderer MODE` | RTX renderer: `realtime` (RaytracedLighting) or `pathtracing` (PathTracing). See [Renderer Selection](#renderer-selection). | `realtime` |
| `--rendering-type MODE` | Realtime quality preset: `performance`, `balanced`, or `quality`. No effect under `--renderer pathtracing`. | IsaacLab default (`balanced`) |
| `--randomize-background` | Sample a random non-default background per task at registration time. The chosen texture is recorded in each task's `env_cfg.json`. See [Backgrounds — Per-Run Random Background per Task](background.md#per-run-random-background-per-task). | `False` |
| `--background-seed N` | Seed for reproducible per-task background sampling. Used with `--randomize-background`. | `None` |
| `--limit-cpu-threads N` | Size of Kit's carb.tasking and TBB worker pools (Isaac Sim `limit_cpu_threads`). See [Running Many Evaluations per Machine](#running-many-evaluations-per-machine). | `4` |
| `--policy-query-mode MODE` | How the client sends the policy queries of several envs that need a new action chunk at the same step: `auto`, `serial`, `concurrent` or `batch`. See [Sending Policy Queries Together](#sending-policy-queries-together). | `auto` |
| `--policy-query-workers N` | Maximum number of server connections in `concurrent` mode. | `16` |
| `--headless` | Run without live display window. **Recommended for multi-task runs** — see [GPU VRAM leak in non-headless mode](debug.md#gpu-vram-leak-in-non-headless-mode-across-environment-reloads) | `False` |
| `--enable-verbose` | Verbose output | `False` |
| `--enable-debug` | Debug output | `False` |

### Renderer Selection

Both `--renderer` modes are RTX renderers; the flag selects which one IsaacLab configures via [`RenderCfg`](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.RenderCfg) (it sets the `/rtx/rendermode` carb value).

- **`realtime`** (`RaytracedLighting`) — the default. Fast, deterministic per frame, and compatible with the tiled multi-env cameras the eval pipeline uses. This is what you want for evaluation.
- **`pathtracing`** (`PathTracing`) — a physically-based renderer with accurate global illumination, ambient occlusion, and soft shadows. It accumulates samples over successive frames, converging on a *static* scene.

> **Path tracing does affect the eval cameras.** It applies to the tiled `TiledCamera` sensors used for observations and viewport video, not just standalone captures — verified on a settled `num_envs=1` scene, where the tiled viewport output differs from realtime in a GI-consistent way (contact shadows, ambient occlusion, material response), not as noise. Two caveats remain: (1) PT accumulates samples on a *static* scene, so during a moving rollout — where the arm and objects change every frame — per-frame convergence drops and frames can be noisier than a settled capture; (2) it is substantially slower than realtime. So realtime stays the default and recommended renderer for large-N evaluation, while path tracing suits higher-fidelity or slow/static captures. The cleanest path-traced stills come from the standalone-viewport tools `misc/compare_renderers.py` and `misc/capture_scene_image.py`, which accumulate on a frozen scene.

`--rendering-type` is a quality preset for the realtime renderer only. The resolved renderer is recorded in every run's `env_cfg.json` (top-level `renderer`, plus `sim.render.carb_settings["/rtx/rendermode"]`) for provenance. For the underlying settings, see the [Omniverse RTX Renderer documentation](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer.html).

**Examples:**

```bash
# Run all benchmark tasks
python policies/pi0_family/run.py --headless

# Run specific tasks
python policies/pi0_family/run.py --task BananaInBowlTask RubiksCubeTask

# Run tasks by tag
python policies/pi0_family/run.py --tag pick_place

# Run 20 parallel episodes
python policies/pi0_family/run.py --headless --num_envs 20

# If 20 envs don't fit in GPU memory, split into runs:
python policies/pi0_family/run.py --headless --num_envs 10 --num-runs 2

# Use a different policy backend (each lives under policies/<policy>/run.py)
python policies/gr00t/run.py --remote-host 10.0.0.1 --remote-port 5555

# Use a specific instruction variant
python policies/pi0_family/run.py --task BananaInBowlTask --instruction-type vague

# Render with the path tracer (standalone viewport only; see Renderer Selection)
python policies/pi0_family/run.py --task BananaInBowlTask --renderer pathtracing

# Resume a previous run (skips completed episodes automatically)
python policies/pi0_family/run.py --output-folder-name 2026-01-24_15-35-59_pi05
```

**Runtime estimate:** Each task runs up to 900 steps. With policy inference overhead, expect roughly **~15 min per task per run** on a single GPU (at ~1–2 it/s depending on `num_envs` and policy backend). For the full 120-task benchmark this works out to approximately **~20–30 hours per 100 tasks** on a single machine. Tasks that succeed early terminate faster, so actual runtime is usually on the lower end.

### Rendering Only Policy Steps

Action-chunking clients (`open_loop_horizon > 1`, e.g. pi0 10, GR00T 10, DreamZero 24) query their policy with an observation only once per chunk; the other steps replay cached actions. Without video, `--render-policy-steps-only` (`run_episode(render_policy_steps_only=True)`) renders, reads back the cameras and computes the camera observation groups (`image_obs`, `viewport_cam`) only on steps whose observation the client will query with, as reported by `InferenceClient.controls_until_observation` (any active env counts; `open_loop_horizon=1` clients render every step). On the other steps the camera groups are `UnrenderedObservationGroup` placeholders: reading any of their terms raises, so a policy can never receive an image from an earlier render. Physics, non-image observations, recorders (HDF5) and success checks still run on every step and are bitwise identical to the default. The flag requires `--video-mode none` and is incompatible with `--record-image-data`. The default renders every step.

RTX real-time rendering keeps temporal history (anti-aliasing, lighting cache, denoising), so a frame rendered after skipped steps is not pixel-identical to the same frame rendered every step. `--render-warmup-steps N` also renders the `N` steps before each consumed step (no read-back); `N >= open_loop_horizon - 1` renders every step and gives the default images, while still skipping camera read-back and image observations on unconsumed steps.

Measured on an L40S with `MustardInRightBinTask`, 1 env, CPU PhysX (`OMP_NUM_THREADS=4`), 450 steps, HDF5 on, video off, a scripted chunking client (no model):

| Mode | ms/step | s/episode | Busy cores | GPU util |
|---|---:|---:|---:|---:|
| Default (render every step) | 28.8–30.4 | 13.0–13.7 | 9.7–10.4 | 29–31% |
| `--render-policy-steps-only`, horizon 8 | 13.5–13.8 | 6.1–6.2 | 7.0–8.4 | 10% |
| `--render-policy-steps-only`, horizon 10 | 12.8–13.2 | 5.8–5.9 | 7.7–8.4 | 8–9% |
| `--render-policy-steps-only`, horizon 24 | 11.4–11.9 | 5.1–5.3 | 7.3–7.9 | 5–6% |
| horizon 10, `--render-warmup-steps 2` | 15.7–15.8 | 7.1 | 7.6–8.2 | 17–18% |
| horizon 10, `--render-warmup-steps 4` | 18.6–19.1 | 8.4–8.6 | 8.4–8.5 | 23% |
| horizon 10, `--render-warmup-steps 9` | 26.0–26.3 | 11.7–11.9 | 8.8–9.1 | 31% |

Mean absolute pixel difference (0–255 scale) of the images at the consumed steps versus a fresh process rendering every step, per camera (over-shoulder / wrist / viewport). For reference, two fresh processes rendering every step differ by 0.17 / 0.41 / 0.39, and the same frames rendered in the second or third episode of one process (a different renderer history, as in every multi-run evaluation) differ by 0.59–0.65 / 2.04–2.13 / 0.78–1.10.

| Warm-up steps (horizon 10) | 0 | 1 | 2 | 4 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|
| Over-shoulder | 1.02 | 0.81 | 0.69 | 0.61 | 0.50 | 0.17 |
| Wrist | 2.93 | 2.22 | 2.22 | 2.00 | 1.86 | 0.41 |
| Viewport | 1.33 | 1.10 | 0.93 | 0.74 | 0.50 | 0.40 |

Horizon 24 without warm-up differs by 1.10 / 3.29 / 1.56–1.60. Judge whether the images your policy receives may differ this much before using the mode for results that are compared with default-rendered evaluations.

### Sending Policy Queries Together

With `--num_envs` above 1, several envs often need a new action chunk at the same control step. The historical client loop sends their queries one after the other, so the step waits for the sum of the server latencies. `--policy-query-mode` (`InferenceClient.configure_queries`) selects how `InferenceClient.infer_batch` sends them:

| Mode | Queries | Actions compared with `serial` |
|---|---|---|
| `serial` | One env after the other (the historical loop). | Same. |
| `concurrent` | One request per env, all at the same time over separate connections (at most `--policy-query-workers`). Needs `supports_concurrent_queries` (pi0, Cosmos3, GR00T). | Same requests. A server that draws noise from one shared RNG (pi0, GR00T and Cosmos flow-matching servers) gives each env a different noise sample, because the request order changes. |
| `batch` | One request with all envs. Needs `supports_batched_queries` (GR00T, mock). | Same requests. The server computes all envs in one forward pass, so numerics and noise samples can change. Check that your GR00T server accepts a batch dimension above 1. |
| `auto` (default) | `concurrent` for clients that set `order_independent_server` (the server output depends only on the request), else `serial`. No shipped policy client sets it, so they keep the serial loop. | Same. |

A client that cannot use the selected mode falls back (`batch` to `concurrent` to `serial`). Clients that override `infer` (and VoLo, DreamZero) always use the serial loop. In every mode each env gets the same observation extraction and request as in the serial loop, and `controls_until_observation` and the action order do not change.

`concurrent` reduces the step time only if the server computes several requests at the same time (several replicas or an asynchronous server). A single-process server that computes one request at a time gains only the overlap of transfer and serialization. `batch` replaces the per-env latencies with one batch latency.

If a concurrent query fails, `infer_batch` waits for the other queries, closes all extra connections and raises the error; no env gets a new chunk at that step. The pi0 and Cosmos3 retries in `concurrent` mode reconnect only the failed connection and keep the chunks of the other envs (the serial retry reconnects the main connection and clears all chunks).

`scripts/mock_policy_server.py` is a sleeping mock server (latency 20–80 ms per request or batch, chunk computed from the request), `scripts/benchmark_policy_queries.py` measures the modes on the CPU without Isaac Sim, and `scripts/benchmark_policy_queries_env.py` runs full episodes with the mock policy (`--verify-serial` checks every applied action and `controls_until_observation` against the serial loop on the same observation). The CPU-only tests are in `tests_cpu/` (`python -m pytest tests_cpu`).

Policy wait per control step that queries the policy, mock server, two 720×1280 images per env, `benchmark_policy_queries.py --num-envs 1 4 10 --steps 60` (Ryzen 7 5800X; mean ms; all modes gave the same actions as `serial`):

| Envs | `serial` | `concurrent`, parallel server | `concurrent`, one-at-a-time server | `batch` |
|---|---|---|---|---|
| 1 | 44–50 | 49 | 42 | 42–49 |
| 4 | 217–229 | 80 | 212 | 71–76 |
| 10 | 548–555 | 114 | 543 | 98–112 |

### Running Many Evaluations per Machine

One evaluation process drives a single scene from one main Python thread and keeps the GPU busy only part of each step, so a machine evaluates fastest with several processes per GPU (for example one process per task, each with its own `--output-folder-name`). With `--device cpu` (CPU PhysX), RTX rendering still runs on the GPU.

- Keep `--limit-cpu-threads` at 4 and set `OMP_NUM_THREADS=1`. Isaac Sim's default 32-thread pools spin about four extra cores per process without shortening a step.
- Give each task's episodes to one process with `--num_envs` (for example 10 envs for 10 episodes). Ten envs in one process render with about a third less GPU time per env-step than ten single-env processes and use about 0.9 GiB of GPU memory per env instead of 4.0 GiB per process.
- Size the number of processes per GPU by GPU memory: a 1-env process needs about 4.0 GiB and a 10-env process about 9.2 GiB. Running out of GPU memory crashes the renderer (`ERROR_DEVICE_LOST`).

Measured on one L40S (45 GiB usable) with CPU PhysX, `MustardInRightBinTask`, 450-step episodes with sensor video and HDF5, a scripted client (no model):

| Processes per GPU | Episodes per hour | Busy cores | GPU util | GPU memory |
|---|---:|---:|---:|---:|
| 1 × 1 env | 296 | – | – | 4.0 GiB |
| 8 × 1 env | 677 | 9.8 | 84% | 32.2 GiB |
| 1 × 10 envs | 584 | 2.3 | 40% | 9.2 GiB |
| 4 × 10 envs | 1342 | 5.8 | 92% | 37.0 GiB |

The original tree evaluated 36 such episodes per hour with one process on the same GPU. A real policy adds its inference latency to every queried step.

## Evaluation Features

The built-in `run_eval.py` provides several features out of the box:

- **Resumability** — If you provide `--output-folder-name` pointing to an existing run, completed tasks and episodes are automatically skipped. This makes it safe to restart interrupted evaluations.
- **Trajectory metrics** — At the end of each episode, trajectory metrics (SPARC smoothness, path length, speed, joint tracking error) are computed from the HDF5 data and written directly into `episode_results.jsonl`. See [Data Storage — Episode Results](data.md#episode-results) for the full list.
- **Error event extraction** — Error events (wrong object grabbed, gripper hit table, object dropped, etc.) are extracted from the episode log and recorded in the results.
- **Video recording** — Two videos per episode: observation camera view and viewport camera view, saved to the task output directory.
- **Subtask tracking** — On by default: subtask completion scores and reasons are recorded per episode (`score`/`reason` in `episode_results.jsonl`, synced with the HDF5 `subtask/score` dataset and the per-env event log). Disable with `--disable-subtask`.
- **Result summarization** — After all tasks complete, a summary table is printed. For more detailed analysis, see [Analysis and Results Parsing](analysis.md).

## Ground-Truth State Export

With `--enable-gt-state`, the episode loop attaches privileged simulator state to the observation dict once per environment step (the policy rate — after decimated physics substeps), keyed per env:

```python
obs["gt_state"] = {env_id: state}   # one entry per active env
```

Clients read their own env's entry (the `InferenceClient._get_env_gt_state` helper), so multi-env evaluations stay independent. The per-env `state` is a raw snapshot produced by `GroundTruthStateExporter` (`robolab/eval/gt_state.py`); everything is plain numpy and msgpack-serialisable:

| Key | Contents |
|-----|----------|
| `objects.<name>.pos` | `(3,)` float32, **env-local frame** (world minus env origin), meters |
| `objects.<name>.quat` | `(4,)` float32, **world frame**, `(w, x, y, z)` |
| `objects.<name>.vel` | `(6,)` float32, world-frame linear (m/s) + angular (rad/s) |
| `robot.ee_pos` / `robot.ee_quat` | End-effector pose, same conventions as above |
| `robot.gripper_closedness` | float32 in `[0, 1]`, 0 = open, 1 = closed |
| `robot.objects_in_contact` | `list[str]` — objects touching the gripper per the contact sensor |
| `subtask.completed` / `.total` / `.score` / `.info` | Live subtask-recorder tracking for this env |
| `subtask.conditions` | Per-object condition satisfaction rows for the current subtask |
| `subtask.object_completed` | `{object: bool}` across the current and all past subtasks |
| `subtask.all_subtask_conditions` | `{"subtask_i": bool}` — every subtask's conditions re-evaluated against the live sim each step |
| `scene_objects` | Manipulable object names (the task's `contact_object_list` minus fixtures) |
| `step` | Environment steps since episode start |

One object entry exists per element of the task's `contact_object_list` (minus the `table`/`robot` fixtures). Objects are covered per env — each env's entry reflects that env's own poses, contacts, and subtask progress.

The snapshot is deliberately raw: derived quantities (lift thresholds, grasp heuristics) are the consumer's business, computed client-side from the per-step stream; see [Evaluating a New Policy — Ground-Truth State in Your Client](policy.md#optional-ground-truth-state-in-your-client).

**Ground truth through the observation pipeline.** Independently of `--enable-gt-state`, environments can be registered with `object_state_obs=True` (an argument to `auto_register_droid_envs`, not a CLI flag) to add an `object_state_obs` observation group with `<object>_pos` / `<object>_quat` / `<object>_vel` terms per task — same frame conventions, but batched, recorded to HDF5, and replayable like any other observation. Useful for consumers that want privileged object state inside the standard pipeline rather than on the inference side-channel.

## Robustness Evaluation Scripts

Additional evaluation scripts test policy robustness under visual or physical variations:

| Script | Description |
|--------|-------------|
| `policies/pi0_family/run_lighting.py` | Evaluate with lighting intensity, shadow, and color variations |
| `policies/pi0_family/run_background_variation.py` | Evaluate with different HDR background scenes |
| `policies/pi0_family/run_camera_pose_variation.py` | Evaluate with random camera pose perturbations |
| `policies/pi0_family/run_table_variation.py` | Evaluate with different table materials (oak, maple, bamboo, black) |

## Example Scripts

| Script | Description |
|--------|-------------|
| [`examples/run_empty.py`](../examples/run_empty.py) | Environment initialization test with random actions (no policy server needed) |
| [`policies/pi0_family/run.py`](../policies/pi0_family/run.py) | Full multi-task policy evaluation (supports multi-env) |

## See Also

- [Environment Registration](environment_registration.md) — How to register tasks as runnable environments
- [Evaluating a New Policy](policy.md) — Implementing an inference client for your model
- [Data Storage and Output](data.md) — Output directory structure, HDF5 layout, and result fields
- [Analysis and Results Parsing](analysis.md) — Scripts for summarizing experiment results
