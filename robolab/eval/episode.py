# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy episode runner for RoboLab.

This module contains the run_episode function that executes a single
policy-controlled episode given any :class:`InferenceClient` subclass.
The function stays policy-agnostic — concrete clients live under
``policies/<policy>/client.py``.

Supports multi-env: one PolicyClient per env, per-env video writers,
actions inferred per active env and stacked for env.step().
"""

import logging
import os
import re
import time
from collections import defaultdict
from functools import partial

import cv2
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TimingStats:
    """Episode wall clock plus host-side section timings (not GPU kernel timings)."""

    def __init__(self):
        self.times = defaultdict(list)
        self._start_times = {}
        self._wall_start = time.perf_counter()

    def start(self, name: str):
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str):
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self.times[name].append(elapsed)
            del self._start_times[name]

    def to_dict(self, num_steps: int, *, active_env_steps: int) -> dict:
        """Include reset, bookkeeping, and finalization, not just named sections."""
        elapsed = time.perf_counter() - self._wall_start
        d = {"timing_schema_version": 2}
        for name, times in self.times.items():
            d[f"{name}_s"] = sum(times)
            d[f"{name}_avg_ms"] = sum(times) / len(times) * 1000 if times else 0.0
        d["timed_sections_s"] = sum(sum(t) for t in self.times.values())
        d["wall_total_s"] = elapsed
        d["control_steps"] = num_steps
        d["active_env_steps"] = active_env_steps
        d["it_per_sec"] = num_steps / elapsed if elapsed > 0 else 0.0
        d["active_env_steps_per_sec"] = active_env_steps / elapsed if elapsed > 0 else 0.0
        return d

from robolab.constants import VISUALIZE, get_output_dir
from robolab.core.logging.results import get_all_env_subtask_infos
from robolab.core.observations.observation_utils import (
    camera_frame_sources,
    combine_camera_frames,
    unpack_combined_frames,
    unpack_viewport_cams,
)
from robolab.core.utils.video_utils import VideoWriter
from robolab.core.world.world_state import get_world
from robolab.core.events.basic_recorders import PreStepFlatPolicyObservationsRecorder
from robolab.eval.base_client import InferenceClient
from robolab.eval.camera_gating import CameraRenderGate


def run_episode(
    env,
    env_cfg,
    episode,
    client: InferenceClient,
    *,
    headless=False,
    save_videos=True,
    video_mode="all",
    enable_gt_state=False,
    render_policy_steps_only=False,
    render_warmup_steps=0,
):
    """Run a policy-controlled episode across all parallel envs.

    The policy client is constructed by the caller (typically a per-policy
    runner under ``policies/<policy>/run.py``). This function stays
    policy-agnostic.

    Args:
        env: The environment instance (RobolabEnv with num_envs >= 1)
        env_cfg: Environment configuration
        episode: Run index (each run produces num_envs episodes)
        client: Constructed inference client. One connection shared across envs
            with per-env chunk state keyed by ``env_id``. Clients must not modify
            observation tensors in place: with host (CPU-device) observations,
            sensor-video frames are prepared from them on the video writer threads.
        headless: If True, don't display video
        save_videos: If True, save per-env episode videos
        video_mode: Which videos to save: 'all', 'viewport', 'sensor', or 'none'
        enable_gt_state: Attach per-env ground-truth sim state to the obs dict
            as ``obs["gt_state"] = {env_id: state}`` each step (see
            ``robolab/eval/gt_state.py`` for the schema).
        render_policy_steps_only: Render the cameras that only the policy
            reads, and compute their observation groups, only on steps whose
            observation the client queries its policy with
            (``client.controls_until_observation`` is 0 for any active env).
            The video groups (``image_obs`` for sensor video, ``viewport_cam``
            for viewport video) render and are computed on every step, so
            videos keep all frames. On the other steps, the render products
            of the other cameras are disabled (not rendered), their groups are
            unrendered placeholders that raise if read, and a step without
            video groups does not render. Physics, non-camera observations,
            recorders and success checks run on every step. Recorded policy
            observations hold zero rows for unrendered groups and a
            ``camera_rendered`` mask. Default False: render every step.
        render_warmup_steps: With ``render_policy_steps_only``, also render
            all cameras (without reading the gated cameras) this many steps
            before each consumed step. RTX real-time rendering keeps temporal
            history (for example denoising and anti-aliasing), so this changes
            the consumed images; at ``open_loop_horizon - 1`` or more every
            step renders all cameras. Default 0.

    Returns:
        tuple: (env_results, subtask_status, timing)
            env_results: per-env dicts with {env_id, success, step}
            subtask_status: list of per-step subtask info dicts
            timing: dict with wall-clock timing breakdown
    """
    camera_gate = None
    if render_policy_steps_only:
        if render_warmup_steps < 0:
            raise ValueError("render_warmup_steps must be >= 0")
        every_step_groups = []
        if save_videos and video_mode in ("all", "sensor"):
            every_step_groups.append("image_obs")
        if save_videos and video_mode in ("all", "viewport"):
            every_step_groups.append("viewport_cam")
        camera_gate = CameraRenderGate(env, every_step_groups)
    elif render_warmup_steps:
        raise ValueError("render_warmup_steps requires render_policy_steps_only")
    # Recorded policy observations of unrendered groups become zero rows with a camera_rendered mask.
    obs_recorders = [
        recorder for recorder in getattr(env.recorder_manager, "_terms", {}).values()
        if isinstance(recorder, PreStepFlatPolicyObservationsRecorder)
    ] if camera_gate is not None else []

    cuda_device = torch.device(env.device)
    if cuda_device.type == "cuda":
        torch.cuda.synchronize(cuda_device)
    timer = TimingStats()

    obs, _ = env.reset()
    obs, _ = env.reset()
    max_steps = env.max_episode_length
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt) # Hz
    instruction = env_cfg.instruction
    # Pull action dim from the env's action manager (IsaacLab canonical),
    # falling back to the gym action space if the manager isn't available.
    action_dim = getattr(
        getattr(env, "action_manager", None),
        "total_action_dim",
        None,
    ) or env.action_space.shape[-1]

    subtask_status = []

    client.begin_episode(episode)

    # Set up per-run HDF5 file and per-env demo indices
    if env.recorder_manager is not None and hasattr(env.recorder_manager, 'set_hdf5_file'):
        env.recorder_manager.set_hdf5_file(f"run_{episode}.hdf5")
        for env_id in range(env.num_envs):
            env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])

    # Setup per-env streaming video writers
    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")
    cleaned_instruction = re.sub(r'[^\w\s]', '', instruction).replace(' ', '_')
    # Define unconditionally so the finally clause below can iterate them either way.
    video_writers_obs: list[VideoWriter] = []
    video_writers_viewport: list[VideoWriter] = []
    if save_videos:
        for env_id in range(env.num_envs):
            suffix = f"_{episode}_env{env_id}" if env.num_envs > 1 else f"_{episode}"
            if save_sensor:
                video_path = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}.mp4")
                video_writers_obs.append(VideoWriter(video_path, video_fps))
            if save_viewport:
                video_path_viewport = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}_viewport.mp4")
                video_writers_viewport.append(VideoWriter(video_path_viewport, video_fps))

    gt_exporter = None
    if enable_gt_state:
        try:
            from robolab.eval.gt_state import GroundTruthStateExporter

            gt_exporter = GroundTruthStateExporter(env, env_cfg)
        except Exception:
            logger.exception("Failed to initialize the GT-state exporter")

    import omni.kit.app
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    kit_app = omni.kit.app.get_app()

    actual_steps = 0
    active_env_steps = 0
    for recorder in obs_recorders:
        recorder.record_render_mask = True
    try:
        for step in tqdm(range(max_steps)):

            while not timeline.is_playing():
                kit_app.update()

            # Attach per-env ground-truth state so clients that want privileged
            # sim state can pick out their env's entry (see robolab/eval/gt_state.py).
            if gt_exporter is not None:
                try:
                    obs["gt_state"] = gt_exporter.export_all(list(env.active_env_ids))
                except Exception:
                    logger.exception(
                        "GT-state export failed at step %d; disabling it for the rest of the episode. "
                        "Clients will stop receiving gt_state from this step on.", step,
                    )
                    gt_exporter = None
                    # Drop the previous step's entry rather than forwarding stale state.
                    obs.pop("gt_state", None)

            timer.start("policy_inference")
            # Infer actions for all active (non-frozen) envs in ONE call.
            # Batching-capable clients send a single request for every env
            # needing a replan; the InferenceClient default is a serial
            # loop, so other policies behave exactly as before.
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            last_viz = None
            active_env_ids = list(env.active_env_ids)
            rets = client.infer_batch(
                obs, instruction, env_ids=active_env_ids
            )
            for env_id in sorted(rets):
                ret = rets[env_id]
                actions[env_id] = torch.tensor(ret["action"], device=env.device)
                if env_id == 0 or last_viz is None:
                    last_viz = ret.get("viz")
            timer.stop("policy_inference")

            if not headless and last_viz is not None:
                cv2.imshow(f"{instruction}", cv2.cvtColor(last_viz, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if VISUALIZE:
                get_world(env).visualize()

            timer.start("env_step")
            if camera_gate is not None:
                # Steps until any active env next queries its policy with an observation.
                until_consumed = min(client.controls_until_observation(env_id) for env_id in active_env_ids)
                obs, reward, term, trunc, info = env.step(
                    actions, **camera_gate.step_options(until_consumed, render_warmup_steps)
                )
            else:
                obs, reward, term, trunc, info = env.step(actions)
            timer.stop("env_step")

            # Collect per-env subtask info (list of dicts, one per env)
            per_env_infos = get_all_env_subtask_infos(env)
            subtask_status.append(per_env_infos)

            # Write per-env video frames (skip frozen envs). Frozen envs are exactly the
            # envs removed from the host-side active set, so no device mask is read here.
            if save_videos:
                timer.start("video_write")
                active_set = set(env.active_env_ids)
                still_active = [env_id for env_id in range(env.num_envs) if env_id in active_set]
                if save_sensor and still_active:
                    if cuda_device.type == "cpu":
                        # Each env's writer thread halves and concatenates that env's camera images.
                        # The observation manager returns newly allocated (cloned) tensors each step
                        # and nothing writes to them afterwards, so the writers read them without a copy.
                        for env_id in still_active:
                            video_writers_obs[env_id].write_deferred(
                                partial(combine_camera_frames, camera_frame_sources(obs, env_id), 0.5)
                            )
                    else:
                        # one batched resize and host transfer; each frame is a fresh, never-modified row
                        frames_obs = unpack_combined_frames(obs, still_active, scale=0.5)
                        for frame_obs, env_id in zip(frames_obs, still_active):
                            video_writers_obs[env_id].write(frame_obs, take_ownership=True)
                if save_viewport:
                    for env_id in still_active:
                        frame_vp = unpack_viewport_cams(obs, env_id=env_id).get("combined_image")
                        video_writers_viewport[env_id].write(frame_vp, take_ownership=True)
                timer.stop("video_write")

            actual_steps += 1
            active_env_steps += len(active_env_ids)

            # RobolabEnv freezes terminated envs and exports recordings automatically
            if env.all_terminated:
                break
    finally:
        if camera_gate is not None:
            # Resets read the cameras, so all render products are enabled again.
            try:
                camera_gate.restore()
            except Exception:
                logger.exception("Failed to enable the camera render products again")
        for recorder in obs_recorders:
            recorder.record_render_mask = False
        for vw in video_writers_obs + video_writers_viewport:
            try:
                vw.release()
            except Exception:
                logger.exception("Failed to release video writer")
        try:
            client.reset()
        except Exception:
            logger.exception("Failed to reset client after episode")

    if cuda_device.type == "cuda":
        torch.cuda.synchronize(cuda_device)
    timing = timer.to_dict(actual_steps, active_env_steps=active_env_steps)
    return env.get_env_results(), subtask_status, timing
