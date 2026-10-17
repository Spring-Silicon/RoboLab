# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab environment runtime class.

This module contains the RobolabEnv class which extends ManagerBasedRLEnv
with custom recorder manager support and eval-specific behavior:
- Terminated envs are frozen (no auto-reset) so they hold their final state
- Per-env success/failure tracking
- Per-env recording export on termination
"""

import logging

import torch
from isaaclab.envs import ManagerBasedRLEnv

from robolab.core.logging.recorder_manager import RobolabRecorderManager
from robolab.core.world.world_state import get_world

logger = logging.getLogger(__name__)


class RobolabEnv(ManagerBasedRLEnv):
    """Environment for RoboLab evaluation.

    Extends ManagerBasedRLEnv with:
    - Custom recorder manager (RobolabRecorderManager)
    - Frozen terminated envs: when an env terminates, it holds its final state
      instead of auto-resetting. Actions for frozen envs are zeroed out.
    - Per-env result tracking (success/truncated, termination step)
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._frozen_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pre_step_frozen = self._frozen_envs.clone()  # snapshot before each step()
        # Host membership changes only at freeze/reset boundaries. Keep it in
        # lockstep with the device mask instead of reading CUDA scalars per step.
        self._active_env_ids: tuple[int, ...] = tuple(range(self.num_envs))
        self._pre_step_active_env_ids: tuple[int, ...] = self._active_env_ids  # host mirror of _pre_step_frozen
        self._pre_step_active_count = self.num_envs
        self._env_results: dict[int, bool] = {}      # env_id -> True (success) / False (truncated)
        self._env_term_step: dict[int, int] = {}      # env_id -> episode step when terminated
        self._has_stepped = False                     # tracks whether step() has been called

    def load_managers(self):
        """Load managers; replace upstream RecorderManager with the streaming-
        capable RobolabRecorderManager.

        super().load_managers() builds an upstream RecorderManager whose
        constructor eagerly creates the HDF5 file at ``cfg.dataset_filename``
        (default ``data.hdf5``) — we don't want that, because we replace the
        manager below with RobolabRecorderManager (which opens the real
        per-run file lazily via ``set_hdf5_file("run_N.hdf5")``). To suppress
        the eager file creation, temporarily set ``cfg.recorders`` to None
        so upstream's RecorderManager.__init__ takes its
        ``if not cfg: return`` early-exit path; then restore cfg before
        instantiating our manager.
        """
        recorders_cfg = self.cfg.recorders
        self.cfg.recorders = None
        try:
            super().load_managers()
        finally:
            self.cfg.recorders = recorders_cfg

        self.recorder_manager = RobolabRecorderManager(self.cfg.recorders, self)

    def step(self, action, **step_options):
        """Step the environment. Zero out actions for frozen (terminated) envs.

        ``step_options`` (``render``, ``camera_observations``) are forwarded to
        :meth:`ManagerBasedRLEnv.step`; see ``run_episode(render_policy_steps_only=True)``.
        """
        self._has_stepped = True
        # The snapshot changes only after new terminations, not every control.
        active_count = len(self._active_env_ids)
        if self._pre_step_active_count != active_count:
            self._pre_step_frozen.copy_(self._frozen_envs)
            self._pre_step_active_env_ids = self._active_env_ids
            self._pre_step_active_count = active_count
        if active_count < self.num_envs:
            action = action.clone()
            action[self._frozen_envs] = 0.0
        return super().step(action, **step_options)

    def _reset_idx(self, env_ids):
        """Override to freeze terminated envs instead of resetting them.

        On initial reset (before any stepping), all envs are reset normally.
        During stepping, when an env terminates:
        1. Mark it as frozen
        2. Record success/failure and termination step
        3. Export its recording data
        4. Skip the actual reset (env holds its final state)
        """
        if not self._has_stepped:
            # Initial reset — let all envs reset normally
            super()._reset_idx(env_ids)
            get_world(self).reset_predicate_state(env_ids)
            return

        # Frozen IDs can recur in reset_buf on every subsequent control. Filter
        # them on the host before reading any episode state from the device.
        pending_ids = [eid for eid in env_ids.tolist() if eid not in self._env_results]
        if not pending_ids:
            return
        # A single batched readback replaces per-env length/success scalar reads.
        terminal_state = torch.stack((
            self.episode_length_buf[pending_ids],
            self.termination_manager.terminated[pending_ids],
        ), dim=-1).tolist()
        artifact_ids = []
        for eid, (ep_len, succeeded) in zip(pending_ids, terminal_state):
            if eid in self._env_results:
                continue
            if ep_len <= 2:
                # Preserve the existing individual artifact reset followed by
                # the active-ID batch reset below, including its RNG/callbacks.
                ids = torch.tensor([eid], device=self.device, dtype=env_ids.dtype)
                super()._reset_idx(ids)
                get_world(self).reset_predicate_state(ids)
                artifact_ids.append(eid)
                continue
            self._frozen_envs[eid] = True
            self._env_results[eid] = bool(succeeded)
            self._env_term_step[eid] = ep_len
            self._active_env_ids = tuple(i for i in self._active_env_ids if i != eid)
            if self.recorder_manager is not None:
                try:
                    self.recorder_manager.export_episodes(env_ids=[eid])
                except Exception:
                    logger.exception(
                        "Failed to export recording for env_id=%d at step=%d; episode data may be incomplete.",
                        eid, ep_len,
                    )

        if artifact_ids:
            active_ids = torch.tensor(artifact_ids, device=self.device, dtype=env_ids.dtype)
            super()._reset_idx(active_ids)
            get_world(self).reset_predicate_state(active_ids)

    @property
    def all_terminated(self) -> bool:
        """True when all envs have terminated."""
        return not self._active_env_ids

    @property
    def active_env_ids(self) -> list[int]:
        """List of env_ids that are still running."""
        return list(self._active_env_ids)

    def get_env_results(self) -> list[dict]:
        """Get per-env results after termination."""
        results = []
        for eid in range(self.num_envs):
            results.append({
                'env_id': eid,
                'success': self._env_results.get(eid),
                'step': self._env_term_step.get(eid),
            })
        return results

    def reset_eval_state(self):
        """Reset frozen state for next episode batch."""
        self._frozen_envs[:] = False
        self._pre_step_frozen[:] = False
        self._active_env_ids = tuple(range(self.num_envs))
        self._pre_step_active_env_ids = self._active_env_ids
        self._pre_step_active_count = self.num_envs
        self._env_results.clear()
        self._env_term_step.clear()
        self._has_stepped = False
