# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled, fixed-action RoboLab simulator SEGMENT benchmark (not full evaluation).

Launch with the IsaacLab 2.2 / Isaac Sim 5.0 Python environment, from the repo root::

    python scripts/benchmark_speed.py --headless --device cuda:0 \
        --task MustardInRightBinTask --seed 1 --num-envs 1 \
        --warmup-steps 20 --steps 100 --termination-safe-horizon 120 \
        --repeats 3 --output /tmp/robolab-benchmark/result.json

No policy, network, video encoding, or end-of-episode export is timed. Production
scene cameras, observations, recorder callbacks, task checks, and physics settings
are preserved by default. Recorder streaming flushes, if configured, ARE timed.
Every repetition has the same seeded double reset, warmup, and action sequence;
a separate discarded priming warmup precedes repetitions. A termination/reset
(including RoboLab's early physics-artifact resets) invalidates the entire run.
The horizon argument is a user-supplied safe segment cap, NOT a timeout override
or a guarantee that a task will not terminate. Shorten the segment on failure.

``policy`` removes viewport camera sensors and observations; ``none`` removes ALL
CameraCfg sensors, including robot-mounted wrist_cam, and their observations.
These are diagnostic workloads, not equivalent production-speed measurements.
``--diagnostic-disable-task-terminations`` explicitly removes non-timeout
termination terms; subtask recording remains enabled and this is labeled diagnostic.

Timing includes lightweight call counters, device-side validity reductions, and
preallocated trajectory copies. There are NO benchmark-added per-step CUDA syncs;
only segment boundaries synchronize. Existing production callbacks may synchronize.
Optional ``--phase-timing`` measures INCLUSIVE HOST wall time (GPU launch/wait time,
not GPU kernel durations); nested phases must not be summed. ``--cprofile`` writes
one profile per measured repetition and also marks timing as diagnostic.

Schema v2: configuration/provenance, create_s, priming, raw runs (reset_s, warmup,
measured counts/timing/validation), summary, verification. NPZ companions contain
fixed actions, every measured state, and start/end observation/image arrays.
Array hashes are over contiguous raw bytes; NPZ file hashes also cover metadata.
Artifacts are written outside timers. Success/failure both write JSON and emit
``ROBOLAB_BENCHMARK_RESULT=<absolute JSON path>``; failures exit 1 and omit the
successful throughput summary. Use raw state/image arrays for tolerance-based comparisons: hashes
alone do not establish numerical or rendering equivalence across GPU executions.
"""

from __future__ import annotations

import argparse
import cProfile
import functools
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PINNED_COMMIT = "9db0aaf09d9fe5d4f37b168320788258c7012463"


def parse_args() -> argparse.Namespace:
    # cv2 must precede Isaac imports; neither this nor AppLauncher imports RoboLab.
    import cv2  # noqa: F401
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="MustardInRightBinTask")
    parser.add_argument("--env-name", help="Exact registered ID; required when task selects multiple environments")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--termination-safe-horizon", type=int, default=120,
                        help="Maximum warmup+measured steps; never changes the production timeout")
    parser.add_argument("--camera-mode", choices=("production", "policy", "none"), default="production")
    parser.add_argument("--diagnostic-disable-task-terminations", action="store_true")
    parser.add_argument("--renderer", choices=("realtime", "pathtracing"), default="realtime")
    parser.add_argument("--rendering-type", choices=("performance", "balanced", "quality"))
    parser.add_argument("--actions", type=Path,
                        help="Fixed .npy replay, shape [warmup+steps, 8] or [warmup+steps, num_envs, 8]")
    parser.add_argument("--motion-amplitude", type=float, default=0.08, help="Joint-space sine amplitude, radians")
    parser.add_argument("--motion-period-s", type=float, default=4.0)
    parser.add_argument("--phase-timing", action="store_true")
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    # Same default as robolab.eval.runner; forwarded to SimulationApp by AppLauncher. Pass 32 for Isaac Sim's default.
    parser.add_argument("--limit-cpu-threads", "--limit_cpu_threads", type=int, default=4)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    for name in ("num_envs", "warmup_steps", "steps", "repeats", "termination_safe_horizon"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_steps + args.steps > args.termination_safe_horizon:
        parser.error("warmup+steps exceeds --termination-safe-horizon")
    if not math.isfinite(args.motion_amplitude) or args.motion_amplitude <= 0:
        parser.error("--motion-amplitude must be finite and positive")
    if not math.isfinite(args.motion_period_s) or args.motion_period_s <= 0:
        parser.error("--motion-period-s must be finite and positive")
    if args.device != "cpu" and not args.device.startswith("cuda"):
        parser.error("--device must be cpu or cuda[:index]; RTX rendering still requires a GPU")
    # Always initialize camera-capable Kit. No-camera ablation is implemented in
    # scene/observation configs, never by relying on a no-RTX launch flag.
    args.enable_cameras = True
    return args


def synchronize(torch: Any, physics_device: Any) -> None:
    # CPU physics is synchronous, but its RTX cameras still use the active GPU.
    torch.cuda.synchronize(physics_device if str(physics_device).startswith("cuda") else None)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        # Some upstream configs use infinite bounds; keep the artifact strict JSON.
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, functools.partial):
        return {"partial": canonical(value.func), "args": canonical(value.args),
                "keywords": canonical(value.keywords)}
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "tolist"):
        return canonical(value.tolist())
    return str(value)


def command_info(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=30)
        return {"command": command, "returncode": proc.returncode, "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def provenance(env: Any, cfg: Any, torch: Any) -> dict[str, Any]:
    import carb
    import isaaclab
    import robolab.constants
    from isaacsim.core.version import get_version

    root = Path(robolab.constants.PACKAGE_DIR)
    source_files = sorted((root / "robolab").rglob("*.py")) + [Path(__file__).resolve()]
    sources = {str(path.relative_to(root)): sha256(path.read_bytes()) for path in source_files}
    upstream = {}
    upstream_types = {type(env).__mro__[1], type(env.scene), type(env.sim), type(env.observation_manager),
                      type(env.termination_manager), type(env.scene["robot"])}
    upstream_types.update(type(sensor) for sensor in env.scene.sensors.values())
    for obj in upstream_types:
        source = inspect.getsourcefile(obj)
        if source:
            upstream[f"{obj.__module__}.{obj.__qualname__}"] = {
                "path": source, "sha256": sha256(Path(source).read_bytes())}
    config = canonical(cfg.to_dict())
    assets = {}

    def asset_paths(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                asset_paths(item)
        elif isinstance(value, list):
            for item in value:
                asset_paths(item)
        elif isinstance(value, str) and value.lower().endswith((".usd", ".usda", ".usdc", ".png", ".jpg", ".hdr", ".exr")):
            path = Path(value)
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                assets[value] = {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}
            else:
                assets[value] = {"local_file": False}

    asset_paths(config)
    gpu_device = env.device if str(env.device).startswith("cuda") else f"cuda:{torch.cuda.current_device()}"
    props = torch.cuda.get_device_properties(gpu_device)
    return {
        "expected_base_commit": PINNED_COMMIT,
        "git_head": command_info(["git", "rev-parse", "HEAD"], root),
        "git_status": command_info(["git", "status", "--porcelain"], root),
        "git_base_is_ancestor": command_info(["git", "merge-base", "--is-ancestor", PINNED_COMMIT, "HEAD"], root),
        "source_sha256": sources,
        "source_manifest_sha256": sha256(json.dumps(sources, sort_keys=True).encode()),
        "upstream_sources": upstream,
        "isaaclab_path": str(Path(isaaclab.__file__).resolve()),
        "isaacsim_version": canonical(get_version()),
        "isaacsim_version_source": "isaacsim.core.version.get_version (runtime API, not distribution metadata)",
        "packages": sorted({(dist.metadata.get("Name", ""), dist.version) for dist in importlib.metadata.distributions()}),
        "python": sys.version, "platform": platform.platform(), "argv": sys.argv,
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
        "gpu": {"device": str(gpu_device), "name": props.name, "total_memory_bytes": props.total_memory,
                "sm_count": props.multi_processor_count, "capability": [props.major, props.minor]},
        "nvidia_smi": command_info(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.limit", "--format=csv"]),
        "runtime_settings": {key: carb.settings.get_settings().get(key) for key in (
            "/plugins/carb.tasking.plugin/threadCount", "/plugins/omni.tbb.globalcontrol/maxThreadCount",
            "/persistent/physics/numThreads", "/app/runLoops/main/rateLimitEnabled",
            "/app/runLoops/main/rateLimitFrequency", "/app/player/useFixedTimeStepping")},
        "thread_environment": {key: os.environ.get(key) for key in (
            "PXR_WORK_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "gpu_processes_at_creation": command_info([
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory", "--format=csv"]),
        "config": config, "config_sha256": sha256(json.dumps(config, sort_keys=True).encode()),
        "direct_asset_files": assets,
        "asset_scope": "Direct config file references only; transitive USD references and remote assets are not hashed",
        "constants": {name: getattr(robolab.constants, name) for name in (
            "DEBUG", "VERBOSE", "VISUALIZE", "ENABLE_SUBTASK_PROGRESS_CHECKING", "RECORD_IMAGE_DATA")},
    }


def configure_cameras(cfg: Any, mode: str) -> dict[str, Any]:
    from isaaclab.sensors import CameraCfg

    cameras = {name for name, value in vars(cfg.scene).items() if isinstance(value, CameraCfg)}
    groups = ("viewport_cam",) if mode == "policy" else ("viewport_cam", "image_obs") if mode == "none" else ()
    removed = set()
    for group_name in groups:
        group = getattr(cfg.observations, group_name, None)
        if group is not None:
            for term in vars(group).values():
                sensor = getattr(term, "params", {}).get("sensor_cfg")
                if sensor is not None:
                    removed.add(sensor.name)
            setattr(cfg.observations, group_name, None)
    if mode == "none":
        removed.update(cameras)
    for name in removed:
        if name not in cameras:
            raise RuntimeError(f"Camera observation refers to unknown camera {name!r}")
        setattr(cfg.scene, name, None)
    retained = cameras - removed
    if mode != "none" and not retained:
        raise RuntimeError("Camera workload contains no cameras")
    return {"configured": sorted(cameras), "removed": sorted(removed), "retained": sorted(retained)}


class Instrumentation:
    """Instance-local counters; optional inclusive host timings, never GPU synchronization."""

    def __init__(self, env: Any, phase_timing: bool):
        self.enabled = False
        self.phase_timing = phase_timing
        self.counts: dict[str, int] = {}
        self.host_s: dict[str, float] = {}
        self.originals = []
        self.segment = "outside_segment"
        self.control_index = -1
        self.wrap(env.sim, "step", "physics_step")
        self.wrap(env.sim, "render", "render")
        self.wrap(env.scene, "update", "scene_update")
        self.wrap(env.observation_manager, "compute", "observations")
        self.wrap(env.termination_manager, "compute", "termination_checks")
        for name in ("record_pre_step", "record_post_step", "record_pre_reset", "record_post_reset"):
            self.wrap(env.recorder_manager, name, name)
        if phase_timing:
            for name, term in env.recorder_manager._terms.items():
                self.wrap(term, "record_post_step", f"recorder_term:{name}")
        original_reset = env._reset_idx

        def guarded_reset(env_ids: Any) -> Any:
            if self.enabled and len(env_ids):
                raise RuntimeError(f"Invalid workload: reset/termination in {self.segment}, control index "
                                   f"{self.control_index}, env IDs {env_ids.tolist()}. Shorten the horizon; "
                                   "no frozen/reset steps are counted as useful work.")
            return original_reset(env_ids)

        self.originals.append((env, "_reset_idx", original_reset))
        env._reset_idx = guarded_reset

    def wrap(self, obj: Any, name: str, label: str) -> None:
        original = getattr(obj, name)
        self.originals.append((obj, name, original))

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not self.enabled:
                return original(*args, **kwargs)
            self.counts[label] = self.counts.get(label, 0) + 1
            if not self.phase_timing:
                return original(*args, **kwargs)
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.host_s[label] = self.host_s.get(label, 0.0) + time.perf_counter() - start

        setattr(obj, name, wrapped)

    def start(self, segment: str) -> None:
        self.counts = {}
        self.host_s = {}
        self.segment = segment
        self.enabled = True

    def stop(self) -> dict[str, Any]:
        self.enabled = False
        return {"calls": dict(self.counts), "inclusive_host_s": dict(self.host_s)}

    def restore(self) -> None:
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)


def flatten_tensors(value: Any, torch: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, dict):
        result = {}
        for name, item in value.items():
            result.update(flatten_tensors(item, torch, f"{prefix}/{name}" if prefix else str(name)))
        return result
    raise RuntimeError(f"Non-tensor observation at {prefix}: {type(value).__name__}")


def state_tensors(env: Any) -> dict[str, Any]:
    state = {}
    for name, asset in env.scene.articulations.items():
        state[f"articulations/{name}/root_state_w"] = asset.data.root_state_w
        state[f"articulations/{name}/joint_pos"] = asset.data.joint_pos
        state[f"articulations/{name}/joint_vel"] = asset.data.joint_vel
    for name, asset in env.scene.rigid_objects.items():
        state[f"rigid_objects/{name}/root_state_w"] = asset.data.root_state_w
    return state


def make_actions(env: Any, args: argparse.Namespace, torch: Any, np: Any) -> Any:
    if list(env.action_manager.active_terms) != ["body", "finger_joint"] or env.action_manager.total_action_dim != 8:
        raise RuntimeError("Benchmark requires DROID absolute joint-position actions [7 arm joints, gripper]")
    robot = env.scene["robot"]
    joint_ids, names = robot.find_joints(["panda_joint.*"], preserve_order=True)
    if len(joint_ids) != 7:
        raise RuntimeError(f"Expected seven arm joints, got {names}")
    total = args.warmup_steps + args.steps
    if args.actions is not None:
        values = np.load(args.actions, allow_pickle=False)
        if values.shape == (total, 8):
            values = np.broadcast_to(values[:, None, :], (total, args.num_envs, 8)).copy()
        if values.shape != (total, args.num_envs, 8):
            raise RuntimeError(f"Replay must have shape [{total}, {args.num_envs}, 8] or [{total}, 8]")
        actions = torch.as_tensor(values, dtype=torch.float32, device=env.device).contiguous()
    else:
        base = robot.data.default_joint_pos[:, joint_ids]
        phases = torch.tensor(np.random.default_rng(args.seed).uniform(-math.pi, math.pi, 7),
                              dtype=torch.float32, device=env.device)
        times = torch.arange(total, dtype=torch.float32, device=env.device) * env.step_dt
        offsets = args.motion_amplitude * (torch.sin(2 * math.pi * times[:, None] / args.motion_period_s + phases)
                                           - torch.sin(phases))
        actions = torch.zeros((total, args.num_envs, 8), device=env.device)
        actions[:, :, :7] = base[None] + offsets[:, None]
    limits = robot.data.soft_joint_pos_limits[:, joint_ids]
    if not torch.isfinite(actions).all().item():
        raise RuntimeError("Actions contain nonfinite values")
    if ((actions[:, :, :7] < limits[None, :, :, 0]) | (actions[:, :, :7] > limits[None, :, :, 1])).any().item():
        raise RuntimeError("Arm targets exceed configured soft joint limits; reduce motion amplitude")
    if ((actions[:, :, 7] < 0) | (actions[:, :, 7] > 1)).any().item():
        raise RuntimeError("Gripper replay must lie in [0, 1]")
    if not (actions[:, :, :7].amax(dim=0) - actions[:, :, :7].amin(dim=0)).amax(dim=1).gt(1e-5).all().item():
        raise RuntimeError("Action replay must move the arm in every environment")
    return actions


def camera_frames(env: Any, names: list[str]) -> dict[str, list[int]]:
    return {name: env.scene.sensors[name].frame.detach().cpu().tolist() for name in names}


def save_arrays(path: Path, arrays: dict[str, Any], np: Any) -> dict[str, Any]:
    # All transfers, compression, summaries and hashing occur outside timers.
    host = {key: np.ascontiguousarray(value.detach().cpu().numpy()) if hasattr(value, "detach")
            else np.ascontiguousarray(value) for key, value in arrays.items()}
    np.savez_compressed(path, **host)
    return {"path": str(path.resolve()), "file_sha256": sha256(path.read_bytes()), "arrays": {
        key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha256(value.tobytes()),
              "min": float(value.min()), "max": float(value.max()),
              "sample": value.reshape(-1)[:32].tolist()} for key, value in host.items()}}


def snapshot_observations(obs: Any, torch: Any, num_envs: int) -> dict[str, Any]:
    result = {}
    for name, value in flatten_tensors(obs, torch).items():
        if value.ndim < 1 or value.shape[0] != num_envs or value.numel() == 0:
            raise RuntimeError(f"Invalid observation shape at {name}: {tuple(value.shape)}")
        if not torch.isfinite(value).all().item():
            raise RuntimeError(f"Nonfinite observation at {name}")
        if name.startswith(("image_obs/", "viewport_cam/")) and value.ndim == 4:
            # Fail on missing-render/zero-image fallbacks instead of timing fake cameras.
            pixels = value.reshape(num_envs, -1)
            if not (pixels.amax(dim=1) > pixels.amin(dim=1)).all().item():
                raise RuntimeError(f"Uniform/empty image at {name}; camera workload is invalid")
        result[name] = value.detach().cpu().clone()
    if not result:
        raise RuntimeError("Environment returned no observations")
    return result


def segment(env: Any, actions: Any, obs: Any, instrumentation: Instrumentation,
            label: str, args: argparse.Namespace, torch: Any, profile_path: Path | None = None) -> tuple[Any, dict, dict]:
    steps = len(actions)
    initial = state_tensors(env)
    trajectory = {key: torch.empty((steps + 1, *value.shape), dtype=value.dtype, device=value.device)
                  for key, value in initial.items()}
    for key, value in initial.items():
        trajectory[key][0].copy_(value)
    lengths = torch.empty((steps, env.num_envs), dtype=env.episode_length_buf.dtype, device=env.device)
    start_lengths = env.episode_length_buf.clone()
    valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    observations = flatten_tensors(obs, torch)
    schema = {key: (tuple(value.shape), value.dtype) for key, value in observations.items()}
    if not schema:
        raise RuntimeError("Empty observation schema")
    if env._frozen_envs.any().item():
        raise RuntimeError("Cannot benchmark frozen environments")
    if not env.sim.is_playing():
        raise RuntimeError("Simulator is not playing")
    profiler = cProfile.Profile() if profile_path else None
    instrumentation.start(label)
    synchronize(torch, env.device)
    start = time.perf_counter()
    if profiler:
        profiler.enable()
    try:
        for index, action in enumerate(actions):
            instrumentation.control_index = index
            obs, reward, terminated, truncated, _ = env.step(action)
            valid.logical_and_(~(terminated | truncated | env._frozen_envs))
            current = flatten_tensors(obs, torch)
            if current.keys() != schema.keys():
                raise RuntimeError("Observation schema changed during segment")
            for key, value in current.items():
                if (tuple(value.shape), value.dtype) != schema[key] or value.ndim < 1 or value.shape[0] != env.num_envs or value.numel() == 0:
                    raise RuntimeError(f"Observation schema/shape changed at {key}")
                if value.is_floating_point() or value.is_complex():
                    valid.logical_and_(torch.isfinite(value).reshape(env.num_envs, -1).all(dim=1))
                if key.startswith(("image_obs/", "viewport_cam/")) and value.ndim == 4:
                    pixels = value.reshape(env.num_envs, -1)
                    valid.logical_and_(pixels.amax(dim=1) > pixels.amin(dim=1))
            valid.logical_and_(torch.isfinite(reward).reshape(env.num_envs, -1).all(dim=1))
            for key, value in state_tensors(env).items():
                trajectory[key][index + 1].copy_(value)
            lengths[index].copy_(env.episode_length_buf)
        synchronize(torch, env.device)
        elapsed = time.perf_counter() - start
    finally:
        if profiler:
            profiler.disable()
        counters = instrumentation.stop()
    if profiler:
        profiler.dump_stats(str(profile_path))
    if not valid.all().item():
        raise RuntimeError(f"Invalid workload in {label}: nonfinite observations/reward, termination, or frozen env")
    for name, value in trajectory.items():
        if not torch.isfinite(value).all().item():
            raise RuntimeError(f"Nonfinite trajectory in {label}: {name}")
    expected_lengths = start_lengths[None] + torch.arange(1, steps + 1, device=env.device)[:, None]
    if not torch.equal(lengths, expected_lengths):
        raise RuntimeError(f"Episode progression shows reset/skipped work in {label}")
    if counters["calls"].get("physics_step", 0) != steps * env.cfg.decimation:
        raise RuntimeError(f"Incorrect actual physics step count in {label}: {counters}")
    if counters["calls"].get("scene_update", 0) != steps * env.cfg.decimation:
        raise RuntimeError(f"Incorrect scene update count in {label}: {counters}")
    if not env.sim.is_playing():
        raise RuntimeError("Simulation stopped during segment")
    return obs, {
        "wall_s": elapsed, "control_steps": steps, "useful_env_steps": steps * env.num_envs,
        "control_step_ms": 1000 * elapsed / steps,
        "aggregate_env_steps_per_s": steps * env.num_envs / elapsed,
        "simulated_seconds_per_wall_second": steps * env.step_dt / elapsed,
        "counts": counters, "valid_observation_env_steps": steps * env.num_envs,
        "observation_schema": {key: {"shape": list(shape), "dtype": str(dtype)} for key, (shape, dtype) in schema.items()},
        "finite_trajectories": True, "terminated_or_frozen_env_steps": 0, "in_segment_resets": 0,
        "episode_length_start": start_lengths.cpu().tolist(), "episode_length_end": lengths[-1].cpu().tolist(),
        "profile_path": str(profile_path.resolve()) if profile_path else None,
    }, trajectory


def reset_repetition(env: Any, seed: int, index: int, torch: Any) -> tuple[Any, float]:
    synchronize(torch, env.device)
    start = time.perf_counter()
    env.reset_eval_state()
    env.recorder_manager.clear()
    # These counters control render cadence and curriculum, not physical state.
    # Align them for every repetition; do not change physics or renderer settings.
    env._sim_step_counter = 0
    env.common_step_counter = 0
    env.reset(seed=seed)
    obs, _ = env.reset()
    env.recorder_manager.set_hdf5_file(f"benchmark_{index}.hdf5")
    for env_id in range(env.num_envs):
        env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])
    synchronize(torch, env.device)
    elapsed = time.perf_counter() - start
    if env._frozen_envs.any().item() or env.episode_length_buf.any().item():
        raise RuntimeError("Reset failed to restore all environments")
    return obs, elapsed


def benchmark(args: argparse.Namespace, result: dict[str, Any]) -> None:
    # All simulator/repository imports are deliberately AFTER AppLauncher startup.
    import numpy as np
    import torch
    from isaaclab.sensors import Camera
    from isaaclab.managers import TerminationTermCfg
    import robolab.constants
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    directory = args.output.parent / (args.output.stem + "_artifacts")
    directory.mkdir(parents=True, exist_ok=False)
    robolab.constants.set_output_dir(str(directory / "recordings"))
    auto_register_droid_envs(task=[args.task])
    names = sorted(get_envs(task=[args.task]))
    if args.env_name:
        if args.env_name not in names:
            raise RuntimeError(f"Requested env {args.env_name!r} not among task registrations: {names}")
        env_name = args.env_name
    elif len(names) == 1:
        env_name = names[0]
    else:
        raise RuntimeError(f"Expected one environment; choose --env-name from {names}")
    cfg = parse_env_cfg(env_name, device=args.device, seed=args.seed, num_envs=args.num_envs,
                        env_spacing=None, eye=None, lookat=None, use_fabric=None)
    camera_config = configure_cameras(cfg, args.camera_mode)
    disabled_terms = []
    if args.diagnostic_disable_task_terminations:
        for name, term in vars(cfg.terminations).items():
            if isinstance(term, TerminationTermCfg) and not term.time_out:
                disabled_terms.append(name)
                setattr(cfg.terminations, name, None)
    total_steps = args.warmup_steps + args.steps
    timeout_steps = math.ceil(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
    if total_steps >= timeout_steps:
        raise RuntimeError(f"warmup+steps={total_steps} reaches production timeout={timeout_steps}; shorten segment")
    diagnostics = []
    if args.camera_mode != "production":
        diagnostics.append(f"camera_ablation:{args.camera_mode}")
    if args.diagnostic_disable_task_terminations:
        diagnostics.append("task_terminations_disabled")
    if args.phase_timing:
        diagnostics.append("inclusive_host_phase_timing")
    if args.cprofile:
        diagnostics.append("cprofile")
    result["workload"] = "fixed_action_segment_not_full_evaluation"
    result["diagnostic_reasons"] = diagnostics
    result["configuration"] = {
        "arguments": canonical(vars(args)), "registered_env": env_name, "cameras": camera_config,
        "disabled_termination_terms": disabled_terms, "production_timeout_steps": timeout_steps,
        "reset_protocol": "reset_eval_state, recorder.clear, cadence counters=0, reset(seed), reset(), recorder file setup",
        "repeat_protocol": "discarded priming warmup; each repeat independently resets, warms up, measures",
        "timing_scope": "env.step plus counters, async validity reductions and preallocated state copies; segment-boundary CUDA sync",
        "excluded": ["policy inference", "network", "video encoding", "sample hashing/writing", "episode-end export"],
        "phase_semantics": "Inclusive host wall time; asynchronous CUDA launch/wait costs, not GPU durations; do not sum nested phases",
    }
    env = None
    instrumentation = None
    try:
        synchronize(torch, args.device)
        start = time.perf_counter()
        env, cfg = create_env(cfg, policy="fixed-action-segment-benchmark", renderer=args.renderer,
                              rendering_mode=args.rendering_type)
        synchronize(torch, env.device)
        result["create_s"] = time.perf_counter() - start
        actual_cameras = sorted(name for name, sensor in env.scene.sensors.items() if isinstance(sensor, Camera))
        if actual_cameras != camera_config["retained"]:
            raise RuntimeError(f"Camera construction mismatch: {actual_cameras} != {camera_config['retained']}")
        if args.camera_mode == "none" and env.sim.has_rtx_sensors():
            raise RuntimeError("No-camera config still has active RTX sensors")
        result["configuration"].update({
            "actual_camera_sensors": actual_cameras, "physics_dt_s": cfg.sim.dt, "decimation": cfg.decimation,
            "render_interval": cfg.sim.render_interval, "control_dt_s": env.step_dt,
            "active_termination_terms": list(env.termination_manager.active_terms),
            "active_recorder_terms": list(env.recorder_manager.active_terms),
            "recorder_flush_interval": env.recorder_manager._flush_interval,
        })
        result["provenance"] = provenance(env, cfg, torch)
        actions = make_actions(env, args, torch, np)
        result["actions"] = save_arrays(directory / "actions.npz", {"actions": actions}, np)
        instrumentation = Instrumentation(env, args.phase_timing)
        obs, reset_s = reset_repetition(env, args.seed, -1, torch)
        obs, warmup, _ = segment(env, actions[:args.warmup_steps], obs, instrumentation, "priming", args, torch)
        snapshot_observations(obs, torch, env.num_envs)
        result["priming"] = {"reset_s": reset_s, "warmup": warmup}
        result["runs"] = []
        for repeat in range(args.repeats):
            obs, reset_s = reset_repetition(env, args.seed, repeat, torch)
            obs, warmup, _ = segment(env, actions[:args.warmup_steps], obs, instrumentation,
                                     f"repeat_{repeat}_warmup", args, torch)
            before_obs = snapshot_observations(obs, torch, env.num_envs)
            before_frames = camera_frames(env, actual_cameras)
            profile = directory / f"repeat_{repeat}.pstats" if args.cprofile else None
            obs, measured, trajectory = segment(env, actions[args.warmup_steps:], obs, instrumentation,
                                                f"repeat_{repeat}_measured", args, torch, profile)
            after_obs = snapshot_observations(obs, torch, env.num_envs)
            after_frames = camera_frames(env, actual_cameras)
            for name in actual_cameras:
                if not all(end > begin for begin, end in zip(before_frames[name], after_frames[name])):
                    raise RuntimeError(f"Camera {name} did not produce new frames for every environment")
            joints = trajectory["articulations/robot/joint_pos"]
            joint_ids, _ = env.scene["robot"].find_joints(["panda_joint.*"], preserve_order=True)
            movement = (joints[:, :, joint_ids].amax(dim=0) - joints[:, :, joint_ids].amin(dim=0)).amax(dim=1)
            if not movement.gt(1e-5).all().item():
                raise RuntimeError("No meaningful measured arm motion in at least one environment")
            measured["max_arm_joint_range_rad_per_env"] = movement.cpu().tolist()
            measured["camera_frames_before"] = before_frames
            measured["camera_frames_after"] = after_frames
            arrays = {f"trajectory/{key}": value for key, value in trajectory.items()}
            arrays.update({f"observation_start/{key}": value for key, value in before_obs.items()})
            arrays.update({f"observation_end/{key}": value for key, value in after_obs.items()})
            verification = save_arrays(directory / f"repeat_{repeat}.npz", arrays, np)
            result["runs"].append({"repeat": repeat, "seed": args.seed, "reset_s": reset_s,
                                   "warmup": warmup, "measured": measured, "verification": verification})
        times = [run["measured"]["wall_s"] for run in result["runs"]]
        median = statistics.median(times)
        result["summary"] = {
            "objective": "median_steady_state_segment_control_step_ms",
            "median_wall_s": median, "median_control_step_ms": median * 1000 / args.steps,
            "min_control_step_ms": min(times) * 1000 / args.steps,
            "max_control_step_ms": max(times) * 1000 / args.steps,
            "median_aggregate_env_steps_per_s": args.steps * env.num_envs / median,
            "median_simulated_seconds_per_wall_second": args.steps * env.step_dt / median,
            "raw_repeat_count": len(times),
        }
        result["verification"] = {
            "status": "passed", "all_measured_observations_finite": True,
            "all_measured_trajectories_finite": True, "no_frozen_or_reset_work": True,
            "all_envs_have_arm_motion": True, "camera_sensor_config_matches": True,
            "camera_frames_advance": bool(actual_cameras), "pixel_samples": "start/end arrays in each repetition NPZ",
            "comparison_status": "not_compared_to_another_configuration",
            "gpu_determinism": "not_assumed; compare stored trajectories and images with application tolerances",
        }
        result["hardware_after"] = command_info(["nvidia-smi", "--query-gpu=index,uuid,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw", "--format=csv"])
    finally:
        if instrumentation:
            instrumentation.restore()
        if env is not None:
            env.close()


def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark result {args.output}")
    result: dict[str, Any] = {"benchmark_version": 2, "status": "running", "started_unix_s": time.time()}
    simulation_app = None
    try:
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        benchmark(args, result)
        result["status"] = "ok"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        result.pop("summary", None)
        # Partial repeats are debugging evidence, never a successful workload result.
        result["verification"] = {"status": "failed", "partial_results_are_not_valid_throughput": True}
    finally:
        result["finished_unix_s"] = time.time()
        args.output.write_text(json.dumps(canonical(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(f"ROBOLAB_BENCHMARK_RESULT={args.output}", flush=True)
        if result["status"] != "ok":
            # Kit fastShutdown may exit(0) from close(), masking a raised exception.
            # Preserve the traceback in both the durable artifact and stderr, then
            # let the OS reclaim this dedicated benchmark process without Kit close.
            print(result["error"]["traceback"], file=sys.stderr, flush=True)
            sys.stdout.flush()
            os._exit(1)
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
