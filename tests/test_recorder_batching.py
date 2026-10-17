# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched per-control recording must export exactly the HDF5 that per-step accumulation exported."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from isaaclab.managers import RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm

from robolab.core.logging.recorder_manager import RobolabRecorderManager

NUM_ENVS = 4


class StateRecorder(RecorderTerm):
    """Nested, mixed-dtype post-step payload read from buffers the env overwrites in place."""

    def record_post_step(self):
        state = self._env.state
        return "states", {
            "robot": {"joint_pos": state, "flags": (state[:, :2] * 7).to(torch.uint16)},
            "bbox_mm": (state[:, :3] * 1000).to(torch.int16),
            "centroid": state[:, 3:].to(torch.float16),
        }


class ActionRecorder(RecorderTerm):
    def record_pre_step(self):
        return "actions", self._env.action


class InitialRecorder(RecorderTerm):
    def record_post_reset(self, env_ids):
        return "initial_state", self._env.state[env_ids].clone()


def _run(manager_cls, export_dir, device, flush_interval):
    generator = torch.Generator().manual_seed(0)
    env = SimpleNamespace(
        num_envs=NUM_ENVS, device=device,
        cfg=SimpleNamespace(env_name="RecorderBatching"),
        sim=SimpleNamespace(is_playing=lambda: True),
        _active_env_ids=tuple(range(NUM_ENVS)),
        state=torch.zeros(NUM_ENVS, 5, device=device),
        action=torch.zeros(NUM_ENVS, 2, device=device),
    )
    cfg = RecorderManagerBaseCfg()
    cfg.dataset_export_dir_path = str(export_dir)
    cfg.record_states = RecorderTermCfg(class_type=StateRecorder)
    cfg.record_actions = RecorderTermCfg(class_type=ActionRecorder)
    cfg.record_initial = RecorderTermCfg(class_type=InitialRecorder)
    manager = manager_cls(cfg, env)
    manager.set_flush_interval(flush_interval)
    manager.set_hdf5_file("run_0.hdf5")
    for env_id in range(NUM_ENVS):
        manager.set_episode_index(env_id, env_ids=[env_id])
    manager.record_post_reset(list(range(NUM_ENVS)))
    for step in range(23):
        env.action.copy_(torch.rand(NUM_ENVS, 2, generator=generator).to(device))
        manager.record_pre_step()
        env.state.copy_(torch.rand(NUM_ENVS, 5, generator=generator).to(device) * 3)  # in place, like sim buffers
        manager.record_post_step()
        if step == 9:  # env 2 terminates and freezes: exported mid-run, never recorded again
            manager.set_success_to_episodes([2], torch.tensor([True]))
            manager.export_episodes(env_ids=[2])
            env._active_env_ids = (0, 1, 3)
        if step == 15:
            env._active_env_ids = (1, 3)
            manager.export_episodes(env_ids=[0])
        if step == 17 and not flush_interval:  # the public per-env view includes every recorded control
            for env_id in (1, 3):
                assert manager.get_episode(env_id).data["actions"].shape == (18, 2)
    manager.export_episodes()
    manager._dataset_file_handler.close()
    return export_dir / "run_0.hdf5"


class PerStepRecorderManager(RobolabRecorderManager):
    """Reference: accumulate every control into each env's EpisodeData immediately."""

    def _buffer_step_rows(self, key, value, env_ids):
        self._add_full_batch_to_episodes(key, value, env_ids)


def _datasets(path):
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            attrs = {k: np.asarray(obj.attrs[k]).tobytes() for k in obj.attrs if k != "recorded_at"}
            out[name] = (attrs, (obj.dtype.str, obj.shape, obj[()].tobytes()) if isinstance(obj, h5py.Dataset) else None)
        f.visititems(visit)
    return out


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("flush_interval", [0, 4])
def test_batched_recording_exports_identical_hdf5(tmp_path, device, flush_interval):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    (tmp_path / "batched").mkdir()
    (tmp_path / "per_step").mkdir()
    batched = _datasets(_run(RobolabRecorderManager, tmp_path / "batched", device, flush_interval))
    reference = _datasets(_run(PerStepRecorderManager, tmp_path / "per_step", device, flush_interval))
    assert batched.keys() == reference.keys()
    assert batched == reference
    assert reference["data/demo_2/actions"][1][1] == (10, 2)
    assert reference["data/demo_0/actions"][1][1] == (16, 2)
    assert reference["data/demo_1/states/robot/flags"][1][1] == (23, 2)
