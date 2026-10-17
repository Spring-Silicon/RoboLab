# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render-on-demand evaluation: ``run_episode(render_policy_steps_only=True)``.

Runs the real episode loop, ``InferenceClient`` chunking, ``RobolabEnv.step``,
``ManagerBasedRLEnv.step`` and ``ObservationManager`` against a minimal
simulated world (no Kit stage, physics or RTX): the "camera" holds the state
captured at the last render, so a stale image is directly observable.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationManager
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.utils import configclass

from robolab.core.environments.env import RobolabEnv
from robolab.eval.base_client import InferenceClient
from robolab.eval.episode import run_episode

NUM_ENVS = 2
STEPS = 13
DECIMATION = 2


class FramebufferCamera(Camera):
    """A camera sensor whose image is the physics state at its last render."""

    def __init__(self, num_envs):
        self.framebuffer = torch.full((num_envs,), -1.0)
        self.readbacks = 0

    def __del__(self):
        pass


def camera_pixels(env, sensor_cfg: SceneEntityCfg):
    camera = env.scene.sensors[sensor_cfg.name]
    camera.readbacks += 1
    return camera.framebuffer.clone().unsqueeze(-1)


def joint_state(env):
    return env.physics_state.clone().unsqueeze(-1)


@configclass
class ImageObsCfg(ObsGroup):
    pixels = ObsTerm(func=camera_pixels, params={"sensor_cfg": SceneEntityCfg("cam")})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class ProprioObsCfg(ObsGroup):
    state = ObsTerm(func=joint_state)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class ObservationsCfg:
    image_obs: ImageObsCfg = ImageObsCfg()
    proprio_obs: ProprioObsCfg = ProprioObsCfg()


class Scene:
    def __init__(self, camera):
        self.num_envs = NUM_ENVS
        self.sensors = {"cam": camera}

    def keys(self):
        return list(self.sensors)

    def __getitem__(self, name):
        return self.sensors[name]

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass


class Sim:
    device = "cpu"

    def __init__(self, env):
        self.env = env

    def is_playing(self):
        return True

    def has_gui(self):
        return False

    def has_rtx_sensors(self):
        return True

    def step(self, render=False):
        self.env.physics_state += self.env.joint_target

    def render(self):
        self.env.render_counters.append(self.env._sim_step_counter)
        self.env.scene.sensors["cam"].framebuffer = self.env.physics_state.clone()


class SimulatedRobolabEnv(RobolabEnv):
    """RobolabEnv over a one-joint integrator world; the managers' step/observation logic is Isaac Lab's."""

    max_episode_length = STEPS

    def __init__(self):
        self._is_closed = True
        self.cfg = SimpleNamespace(
            decimation=DECIMATION, rerender_on_reset=True, sim=SimpleNamespace(dt=0.01, render_interval=DECIMATION)
        )
        self.sim = Sim(self)
        self.scene = Scene(FramebufferCamera(NUM_ENVS))
        self.physics_state = torch.zeros(NUM_ENVS)
        self.joint_target = torch.zeros(NUM_ENVS)
        self.render_counters = []
        self.extras = {}
        self._sim_step_counter = 0
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
        no_done = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.termination_manager = SimpleNamespace(
            compute=lambda: no_done.clone(), terminated=no_done, time_outs=no_done
        )
        self.reward_manager = SimpleNamespace(compute=lambda dt: torch.zeros(NUM_ENVS))
        self.command_manager = SimpleNamespace(compute=lambda dt: None)
        self.event_manager = SimpleNamespace(available_modes=[])
        self.recorder_manager = SimpleNamespace(active_terms=[], record_pre_step=lambda: None)
        processed = {}
        self.action_manager = SimpleNamespace(
            total_action_dim=1,
            process_action=lambda action: processed.update(action=action),
            apply_action=lambda: self.joint_target.copy_(processed["action"][:, 0]),
        )
        self.observation_manager = ObservationManager(ObservationsCfg(), self)
        self._frozen_envs = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self._pre_step_frozen = self._frozen_envs.clone()
        self._active_env_ids = tuple(range(NUM_ENVS))
        self._pre_step_active_env_ids = self._active_env_ids
        self._pre_step_active_count = NUM_ENVS
        self._env_results = {}
        self._env_term_step = {}
        self._has_stepped = False
        self.trajectory = []

    def reset(self, seed=None, options=None):
        self.sim.render()
        self.obs_buf = self.observation_manager.compute()
        return self.obs_buf, self.extras

    def step(self, action, **step_options):
        ret = super().step(action, **step_options)
        self.trajectory.append((self.physics_state.clone(), ret[0]["proprio_obs"]["state"].clone()))
        return ret

    @property
    def rendered_controls(self):
        """Control index of each render during stepping (reset renders excluded)."""
        return [counter // DECIMATION - 1 for counter in self.render_counters if counter > 0]


class ChunkClient(InferenceClient):
    """Chunking client that records what it sees whenever it queries its policy."""

    def __init__(self, horizon, replan=None):
        super().__init__()
        self.open_loop_horizon = horizon
        self.replan = replan or {}  # step -> env_id whose cached chunk is dropped after that step
        self.step = 0
        self.queries = []

    def infer_batch(self, obs, instruction, *, env_ids):
        rets = super().infer_batch(obs, instruction, env_ids=env_ids)
        if self.step in self.replan:
            self.reset(env_id=self.replan[self.step])
        self.step += 1
        return rets

    def _extract_observation(self, raw_obs, *, env_id=0):
        return {
            "image": float(raw_obs["image_obs"]["pixels"][env_id]),
            "state": float(raw_obs["proprio_obs"]["state"][env_id]),
        }

    def _pack_request(self, extracted_obs, instruction):
        return extracted_obs

    def _query_server(self, request):
        self.queries.append((self.step, request["image"], request["state"]))
        return request

    def _unpack_response(self, response):
        return np.array([[0.25 + 0.5 * self.step + 0.125 * i] for i in range(self.open_loop_horizon)])


def run(client, **options):
    env = SimulatedRobolabEnv()
    env_cfg = SimpleNamespace(sim=env.cfg.sim, instruction="move", recorders=None)
    kwargs = {"headless": True, "save_videos": False, "video_mode": "none", **options}
    run_episode(env, env_cfg, 0, client, **kwargs)
    return env


@pytest.fixture(autouse=True)
def playing_timeline(monkeypatch):
    import omni.timeline

    timeline = SimpleNamespace(is_playing=lambda: True)
    monkeypatch.setattr(omni.timeline, "get_timeline_interface", lambda: timeline)


def consumed_controls(client):
    """Controls whose observations fed a query (the reset observation is control -1)."""
    return sorted({step - 1 for step, _, _ in client.queries})


@pytest.mark.parametrize(
    "replan,warmup,consumed",
    [
        # env 1 drops its chunk after step 5, so the two envs query on different steps
        ({5: 1}, 0, [-1, 3, 5, 7, 9, 11]),
        ({}, 0, [-1, 3, 7, 11]),
        ({}, 1, [-1, 3, 7, 11]),
        ({}, 2, [-1, 3, 7, 11]),
    ],
)
def test_renders_exactly_the_consumed_steps_and_never_delivers_stale_images(replan, warmup, consumed):
    default_client, client = ChunkClient(4, replan=replan), ChunkClient(4, replan=replan)
    default_env = run(default_client)
    env = run(client, render_policy_steps_only=True, render_warmup_steps=warmup)

    assert consumed_controls(client) == consumed
    # every queried image shows the state of the observation it came with: nothing stale
    assert client.queries and all(image == state for _, image, state in client.queries)
    assert client.queries == default_client.queries
    # renders: each consumed control plus `warmup` controls before it; cameras read back only when consumed
    expected = sorted({c - k for c in consumed if c >= 0 for k in range(warmup + 1)})
    assert env.rendered_controls == expected
    assert env.scene.sensors["cam"].readbacks == default_env.scene.sensors["cam"].readbacks - STEPS + len(consumed) - 1
    # physics and non-camera observations are identical to rendering every step
    assert len(env.trajectory) == len(default_env.trajectory) == STEPS
    for (state, proprio), (ref_state, ref_proprio) in zip(env.trajectory, default_env.trajectory):
        assert torch.equal(state, ref_state) and torch.equal(proprio, ref_proprio)


def test_default_and_single_action_clients_render_every_step():
    default_env = run(ChunkClient(4))
    assert default_env.rendered_controls == list(range(STEPS))
    gated_env = run(ChunkClient(1), render_policy_steps_only=True)
    assert gated_env.rendered_controls == list(range(STEPS))


def test_unrendered_camera_observations_fail_loudly():
    env = SimulatedRobolabEnv()
    env.reset()
    action = torch.ones(NUM_ENVS, 1)
    obs, *_ = env.step(action, render=False, camera_observations=False)
    assert "pixels" in obs["image_obs"]
    with pytest.raises(RuntimeError, match="not rendered"):
        obs["image_obs"]["pixels"]
    with pytest.raises(RuntimeError, match="not rendered"):
        InferenceClient._find_obs_term(obs, "pixels")
    assert InferenceClient._find_obs_term(obs, "state") is obs["proprio_obs"]["state"]
    with pytest.raises(ValueError, match="require rendering"):
        env.step(action, render=False)
    with pytest.raises(ValueError, match="video"):
        run(ChunkClient(4), render_policy_steps_only=True, save_videos=True, video_mode="sensor")
