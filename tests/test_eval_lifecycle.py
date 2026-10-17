# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partial termination, artifact resets, and per-run recorder isolation."""

from types import SimpleNamespace

import h5py
import pytest
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm

from robolab.core.environments.env import RobolabEnv
from robolab.core.logging.recorder_manager import RobolabRecorderManager
from robolab.core.world.world_state import clear_world_cache, get_world


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


def test_partial_freeze_artifact_reset_and_next_batch(monkeypatch, device):
    # Isolate the evaluation lifecycle from physics. The base reset changes real
    # per-env state; assertions cover its effects, not just callback forwarding.
    def base_init(env, cfg, **kwargs):
        env.cfg = cfg
        env.scene = SimpleNamespace(num_envs=4)
        env.sim = SimpleNamespace(device=device)
        env._is_closed = True
        env.recorder_manager = None
        env.episode_length_buf = torch.zeros(4, dtype=torch.long, device=device)
        env.reset_generation = torch.zeros(4, dtype=torch.long, device=device)
        env.termination_manager = SimpleNamespace(terminated=torch.zeros(4, dtype=torch.bool, device=device))

    def base_reset(env, env_ids):
        env.episode_length_buf[env_ids] = 0
        env.reset_generation[env_ids] += 1

    def base_step(env, action):
        env.episode_length_buf += 1
        return action

    monkeypatch.setattr(ManagerBasedRLEnv, "__init__", base_init)
    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", base_reset)
    monkeypatch.setattr(ManagerBasedRLEnv, "step", base_step)
    env = RobolabEnv(SimpleNamespace())
    try:
        world = get_world(env)
        latched = torch.ones(4, dtype=torch.bool, device=device)
        predicate_resets = torch.zeros(4, dtype=torch.long, device=device)

        def reset_latches(ids):
            latched[ids] = False
            predicate_resets[ids] += 1

        world.get_or_init_predicate_state("lifecycle", lambda: {"__reset__": reset_latches})
        env._reset_idx(torch.arange(4, device=device))
        assert env.reset_generation.tolist() == [1, 1, 1, 1]
        assert not latched.any()
        latched[:] = True
        action = torch.ones(4, 2, device=device)
        assert env.step(action) is action
        env.episode_length_buf[:] = torch.tensor([6, 2, 9, 4], device=device)
        env.termination_manager.terminated[:] = torch.tensor([True, False, False, True], device=device)
        env._reset_idx(torch.tensor([0, 1, 2], device=device))

        assert env.active_env_ids == [1, 3]
        assert not env.all_terminated
        assert env._frozen_envs.tolist() == [True, False, True, False]
        assert env._pre_step_frozen.tolist() == [False, False, False, False]
        assert env.get_env_results() == [
            {"env_id": 0, "success": True, "step": 6},
            {"env_id": 1, "success": None, "step": None},
            {"env_id": 2, "success": False, "step": 9},
            {"env_id": 3, "success": None, "step": None},
        ]
        # Keep the established artifact individual+batch reset semantics. Frozen
        # environments retain their state and predicate latches.
        assert env.reset_generation.tolist() == [1, 3, 1, 1]
        assert predicate_resets.tolist() == [1, 3, 1, 1]
        assert latched.tolist() == [True, False, True, True]
        env._reset_idx(torch.tensor([0, 2], device=device))
        assert env.reset_generation.tolist() == [1, 3, 1, 1]

        active_copy = env.active_env_ids
        active_copy.clear()
        assert env.active_env_ids == [1, 3]
        masked = env.step(action)
        assert masked.tolist() == [[0, 0], [1, 1], [0, 0], [1, 1]]
        assert action.tolist() == [[1, 1]] * 4
        assert env._pre_step_frozen.tolist() == [True, False, True, False]
        env.episode_length_buf[1] = 3
        env._reset_idx(torch.tensor([1, 3], device=device))
        assert env.all_terminated
        assert env.active_env_ids == []

        env.reset_eval_state()
        assert env.active_env_ids == [0, 1, 2, 3]
        assert not env.all_terminated
        assert not env._frozen_envs.any()
        assert not env._pre_step_frozen.any()
        assert all(result["success"] is None for result in env.get_env_results())
        env._reset_idx(torch.arange(4, device=device))
        assert env.reset_generation.tolist() == [2, 4, 2, 2]
        assert not latched.any()
        assert env.step(action) is action
    finally:
        clear_world_cache()


class LifecycleRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.clear_count = 0

    def record_pre_step(self):
        return "actions", self._env.payload

    def record_post_step(self):
        return "states", {"position": self._env.payload, "status": self._env.payload.to(torch.uint16)}

    def record_pre_reset(self, env_ids):
        return "pre_reset", self._env.payload[env_ids]

    def record_post_reset(self, env_ids):
        return "initial_state", {"position": self._env.payload[env_ids]}

    def record_final_status(self):
        return "final_status", torch.tensor([101, 102, 103], dtype=torch.uint16, device=self._env.device)

    def clear(self):
        self.clear_count += 1


def test_partial_exports_autoflush_and_cross_run_isolation(tmp_path, device):
    env = SimpleNamespace(
        num_envs=3, device=device,
        cfg=SimpleNamespace(env_name="RecorderLifecycle"),
        sim=SimpleNamespace(is_playing=lambda: True),
        _active_env_ids=(0, 1, 2),
        _frozen_envs=torch.zeros(3, dtype=torch.bool, device=device),
        payload=torch.arange(3, dtype=torch.float32, device=device).reshape(3, 1),
        termination_manager=SimpleNamespace(
            active_terms=["success_object"],
            terminated=torch.tensor([True, False, False], device=device),
        ),
    )
    cfg = RecorderManagerBaseCfg()
    cfg.dataset_export_dir_path = str(tmp_path)
    cfg.export_in_record_pre_reset = False
    cfg.lifecycle = RecorderTermCfg(class_type=LifecycleRecorder)
    manager = RobolabRecorderManager(cfg, env)
    manager.set_flush_interval(2)
    term = manager.get_term(LifecycleRecorder)

    def arm(filename):
        manager.set_hdf5_file(filename)
        for eid in range(3):
            manager.set_episode_index(eid, [eid])
        manager.record_post_reset(None)

    def control(value):
        env.payload[:] = torch.arange(3, device=device).reshape(3, 1) + value
        manager.record_pre_step()
        manager.record_post_step()

    def finish(eid):
        # Repeated reset IDs include previously finalized envs, as reset_buf does.
        manager.record_pre_reset(list(range(eid + 1)))
        env._frozen_envs[eid] = True
        env._active_env_ids = tuple(i for i in env._active_env_ids if i != eid)
        manager.export_episodes([eid])
        manager.record_post_reset(list(range(eid + 1)))

    try:
        arm("run_0.hdf5")
        control(10)
        control(20)
        finish(0)
        control(30)
        control(40)
        finish(1)
        control(50)
        finish(2)
        manager.export_episodes()
        manager.record_pre_step()
        manager.record_post_step()
        assert term.clear_count == 5  # Three first flushes, then two active flushes.
        assert all(manager.get_episode(eid).is_empty() for eid in range(3))
        assert manager._step_count == {0: 0, 1: 0, 2: 0}

        # No recorder.clear() between runs: finalized buffers alone must isolate
        # the next file, including term final-status and unsigned leaves.
        env._active_env_ids = (0, 1, 2)
        env._frozen_envs[:] = False
        env.payload[:] = torch.arange(3, device=device).reshape(3, 1) + 1000
        env.termination_manager.terminated[:] = True
        arm("run_1.hdf5")
        control(1010)
        manager.record_pre_reset(None)
        manager.export_episodes()
        manager._dataset_file_handler.close()

        with h5py.File(tmp_path / "run_0.hdf5", "r") as stream:
            assert set(stream["data"]) == {"demo_0", "demo_1", "demo_2"}
            for eid, steps in enumerate((2, 4, 5)):
                demo = stream[f"data/demo_{eid}"]
                expected = [[10 * step + eid] for step in range(1, steps + 1)]
                assert demo["actions"][()].tolist() == expected
                assert demo["states/position"][()].tolist() == expected
                assert demo["states/status"][()].tolist() == expected
                assert demo["final_status"][()].tolist() == [101 + eid]
                assert demo["pre_reset"][()].tolist() == [expected[-1]]
                assert demo["initial_state/position"][()].tolist() == [[eid]]
                assert bool(demo.attrs["success"]) == (eid == 0)
        with h5py.File(tmp_path / "run_1.hdf5", "r") as stream:
            assert set(stream["data"]) == {"demo_0", "demo_1", "demo_2"}
            for eid in range(3):
                demo = stream[f"data/demo_{eid}"]
                assert demo["actions"][()].tolist() == [[1010 + eid]]
                assert demo["states/position"][()].tolist() == [[1010 + eid]]
                assert demo["initial_state/position"][()].tolist() == [[1000 + eid]]
                assert demo["final_status"][()].tolist() == [101 + eid]
                assert bool(demo.attrs["success"])
    finally:
        if manager._dataset_file_handler is not None:
            manager._dataset_file_handler.close()
