# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording mixed active/frozen batches must preserve unsigned payload snapshots."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab.managers import RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm

from robolab.core.logging.recorder_manager import RobolabRecorderManager


class PayloadRecorder(RecorderTerm):
    def record_post_step(self):
        return "subtask", self._env.payload


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64, torch.float32, torch.int64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_recording_partial_batch_preserves_values_dtype_and_ownership(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    payload = torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=dtype, device=device)
    env = SimpleNamespace(
        num_envs=4, device=device,
        cfg=SimpleNamespace(env_name="RecorderLifecycle"),
        sim=SimpleNamespace(is_playing=lambda: True),
        _frozen_envs=torch.tensor([False, True, False, True], device=device),
        payload={"status": payload},
    )
    cfg = RecorderManagerBaseCfg()
    cfg.payload = RecorderTermCfg(class_type=PayloadRecorder)
    manager = RobolabRecorderManager(cfg, env)
    manager.set_flush_interval(0)

    # This generic env has no host active-ID cache: exercise the device-mask fallback.
    manager.record_post_step()
    payload.fill_(0)
    for env_id, expected in ((0, [[10, 11]]), (2, [[30, 31]])):
        out = manager.get_episode(env_id).data["subtask"]["status"]
        assert out.dtype == dtype
        assert out.cpu().tolist() == expected
    assert manager.get_episode(1).is_empty()
    assert manager.get_episode(3).is_empty()

    # A second control must append only active rows, without aliasing the first.
    manager.record_post_step()
    assert manager.get_episode(0).data["subtask"]["status"].cpu().tolist() == [[10, 11], [0, 0]]
    assert manager.get_episode(2).data["subtask"]["status"].cpu().tolist() == [[30, 31], [0, 0]]
