# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-camera render gating for ``run_episode(render_policy_steps_only=True)``.

On steps whose observation no client queries, only the cameras of the camera
observation groups that are needed on every step (the video groups) render.
The other camera render products are disabled, so the renderer does not
update them, and their groups are unrendered placeholders for that step.
"""

from collections.abc import Iterable


class CameraRenderGate:
    """Chooses the rendered cameras and computed camera groups of each step.

    Args:
        env: The environment (a ``ManagerBasedRLEnv``).
        every_step_groups: Camera observation groups to render and compute on
            every step, for example the groups that videos are written from.
            Groups that the environment does not have are ignored.
    """

    def __init__(self, env, every_step_groups: Iterable[str]):
        group_sensors = env.observation_manager.camera_group_sensors
        self.every_step_groups = tuple(group for group in dict.fromkeys(every_step_groups) if group in group_sensors)
        every_step_sensors = {name for group in self.every_step_groups for name in group_sensors[group]}
        # Cameras that only the policy-step groups read; they are disabled on the other steps. Without
        # every-step groups the other steps do not render at all, so no render product is toggled.
        gated = () if not self.every_step_groups else dict.fromkeys(
            name for names in group_sensors.values() for name in names if name not in every_step_sensors
        )
        self._env = env
        sensors = env.scene.sensors
        unsupported = [name for name in gated if not hasattr(sensors[name], "set_render_product_enabled")]
        if unsupported:
            raise ValueError(
                f"Cameras {unsupported} cannot disable their render products (TiledCamera required) to render"
                f" only the groups {list(self.every_step_groups)} on unconsumed steps."
            )
        self._gated = [sensors[name] for name in gated]

    def step_options(self, controls_until_observation: int, warmup_steps: int) -> dict:
        """``env.step`` options for a step whose observation is consumed in ``controls_until_observation`` steps.

        A consumed step (0) renders all cameras and computes all camera groups. A step within ``warmup_steps``
        of a consumed step renders all cameras and computes the every-step groups. Other steps render only the
        cameras of the every-step groups and compute only those groups; without every-step groups they do not
        render.
        """
        if controls_until_observation == 0:
            self._set_gated_enabled(True)
            return {"render": True, "camera_observations": True}
        if controls_until_observation <= warmup_steps:
            self._set_gated_enabled(True)
            return {"render": True, "camera_observations": self.every_step_groups}
        if not self.every_step_groups:
            # Nothing renders this step, so the render products keep their state.
            return {"render": False, "camera_observations": False}
        self._set_gated_enabled(False)
        return {"render": True, "camera_observations": self.every_step_groups}

    def restore(self):
        """Enable all render products again and render until they hold a frame (for resets and later episodes).

        Resets read the cameras (for example the initial-state recorder reads their poses) before they render,
        so the frames are rendered here without stepping the simulation; they show the final state of the episode.

        Raises:
            RuntimeError: If a camera delivers no frame within ``env.max_frame_delivery_renders`` renders.
        """
        self._set_gated_enabled(True)
        for _ in range(self._env.max_frame_delivery_renders):
            if not any(sensor.frame_pending for sensor in self._gated):
                return
            self._env.sim.render()
        pending = [sensor.cfg.prim_path for sensor in self._gated if sensor.frame_pending]
        if pending:
            raise RuntimeError(
                f"Cameras {pending} delivered no frame within {self._env.max_frame_delivery_renders} renders after"
                " their render products were enabled."
            )

    def _set_gated_enabled(self, enabled: bool):
        for sensor in self._gated:
            sensor.set_render_product_enabled(enabled)
