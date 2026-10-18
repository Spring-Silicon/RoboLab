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
    """A camera sensor whose image is the physics state at its last render with its render product enabled."""

    def __init__(self, num_envs):
        self.framebuffer = torch.full((num_envs,), -1.0)
        self.readbacks = 0
        self.render_product_enabled = True
        self.rendered_counters = []
        # A render with the render product enabled updates the framebuffer at once: no frame is ever pending.
        self.frame_pending = False

    def set_render_product_enabled(self, enabled):
        self.render_product_enabled = enabled

    def __del__(self):
        pass


def camera_pixels(env, sensor_cfg: SceneEntityCfg):
    camera = env.scene.sensors[sensor_cfg.name]
    if not camera.render_product_enabled:
        raise RuntimeError("render product disabled")
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


@configclass
class ViewportObsCfg(ObsGroup):
    vp_pixels = ObsTerm(func=camera_pixels, params={"sensor_cfg": SceneEntityCfg("vp")})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class ViewportObservationsCfg(ObservationsCfg):
    viewport_cam: ViewportObsCfg = ViewportObsCfg()


class Scene:
    def __init__(self, cameras):
        self.num_envs = NUM_ENVS
        self.sensors = cameras

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
        for camera in self.env.scene.sensors.values():
            if camera.render_product_enabled:
                camera.framebuffer = self.env.physics_state.clone()
                camera.rendered_counters.append(self.env._sim_step_counter)


class SimulatedRobolabEnv(RobolabEnv):
    """RobolabEnv over a one-joint integrator world; the managers' step/observation logic is Isaac Lab's."""

    max_episode_length = STEPS

    def __init__(self, viewport=False):
        self._is_closed = True
        self.cfg = SimpleNamespace(
            decimation=DECIMATION, rerender_on_reset=True, sim=SimpleNamespace(dt=0.01, render_interval=DECIMATION)
        )
        self.sim = Sim(self)
        cameras = {"cam": FramebufferCamera(NUM_ENVS)}
        if viewport:
            cameras["vp"] = FramebufferCamera(NUM_ENVS)
        self.scene = Scene(cameras)
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
        self.observation_manager = ObservationManager(ViewportObservationsCfg() if viewport else ObservationsCfg(), self)
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
        return controls(self.render_counters)

    def camera_rendered_controls(self, name):
        return controls(self.scene.sensors[name].rendered_counters)


def controls(render_counters):
    return [counter // DECIMATION - 1 for counter in render_counters if counter > 0]


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


def run(client, viewport=False, setup=None, **options):
    env = SimulatedRobolabEnv(viewport=viewport)
    if setup is not None:
        setup(env)
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
    with pytest.raises(ValueError, match="Not camera observation groups"):
        env.step(action, camera_observations=("proprio_obs",))


@pytest.fixture
def video_frames(monkeypatch):
    """Video frames written by run_episode, per video file name; each frame is the image value of one env."""
    import robolab.eval.episode as episode

    frames = {}

    class Writer:
        def __init__(self, path, fps):
            self.frames = frames.setdefault(path, [])

        def write(self, frame, take_ownership=False):
            self.frames.append(float(frame))

        def write_deferred(self, make_frame):
            self.frames.append(float(make_frame()))

        def release(self):
            pass

    monkeypatch.setattr(episode, "VideoWriter", Writer)
    monkeypatch.setattr(episode, "get_output_dir", lambda: "out")
    monkeypatch.setattr(episode, "camera_frame_sources", lambda obs, env_id: obs["image_obs"]["pixels"][env_id])
    monkeypatch.setattr(episode, "combine_camera_frames", lambda sources, scale: sources)
    monkeypatch.setattr(
        episode, "unpack_viewport_cams", lambda obs, env_id: {"combined_image": obs["viewport_cam"]["vp_pixels"][env_id]}
    )
    return frames


@pytest.mark.parametrize("video_mode", ["viewport", "sensor", "all"])
@pytest.mark.parametrize("warmup", [0, 1])
def test_video_cameras_render_every_step_and_other_cameras_only_for_consumed_steps(video_frames, video_mode, warmup):
    default_client, client = ChunkClient(4), ChunkClient(4)
    options = {"viewport": True, "save_videos": True, "video_mode": video_mode}
    default_env = run(default_client, **options)
    default_frames = dict(video_frames)
    video_frames.clear()
    env = run(client, render_policy_steps_only=True, render_warmup_steps=warmup, **options)

    consumed = [c for c in consumed_controls(client) if c >= 0]
    assert consumed == [3, 7, 11]
    assert client.queries == default_client.queries
    assert all(image == state for _, image, state in client.queries)
    video_cameras = {"sensor": {"cam"}, "viewport": {"vp"}, "all": {"cam", "vp"}}[video_mode]
    gated_controls = sorted({c - k for c in consumed for k in range(warmup + 1)})
    for name in ("cam", "vp"):
        expected = list(range(STEPS)) if name in video_cameras else gated_controls
        assert env.camera_rendered_controls(name) == expected, name
        # all render products are enabled again after the episode
        assert env.scene.sensors[name].render_product_enabled
    # every video frame shows the physics state after its step, exactly as when rendering every step
    assert video_frames == default_frames and len(video_frames) == NUM_ENVS * len(video_cameras)
    for path, frames in video_frames.items():
        env_id = int(path.split("_env")[-1].split("_")[0].split(".")[0])
        assert frames == [float(state[env_id]) for state, _ in env.trajectory]
    for (state, proprio), (ref_state, ref_proprio) in zip(env.trajectory, default_env.trajectory):
        assert torch.equal(state, ref_state) and torch.equal(proprio, ref_proprio)


def test_recorded_policy_observations_mask_unrendered_groups(video_frames):
    from robolab.core.events.basic_recorders import (
        PreStepFlatPolicyObservationsRecorder,
        PreStepFlatPolicyObservationsRecorderCfg,
    )

    rows = []
    recorders = []

    def add_recorder(env):
        recorder = PreStepFlatPolicyObservationsRecorder(PreStepFlatPolicyObservationsRecorderCfg(), env)
        env.recorder_manager._terms = {"record_policy_observations": recorder}
        env.recorder_manager.record_pre_step = lambda: rows.append(recorder.record_pre_step()[1])
        recorders.append(recorder)

    client = ChunkClient(4)
    env = run(
        client, viewport=True, setup=add_recorder, render_policy_steps_only=True, save_videos=True, video_mode="viewport"
    )

    assert len(rows) == STEPS and not recorders[0].record_render_mask
    consumed = set(consumed_controls(client))
    states = [torch.zeros(NUM_ENVS)] + [state for state, _ in env.trajectory]
    for step, row in enumerate(rows):
        # the pre-step row of control `step` holds the observation returned by control `step - 1`
        state = states[step]
        rendered = step - 1 in consumed
        assert row["camera_rendered"]["image_obs"].tolist() == [rendered] * NUM_ENVS
        assert row["camera_rendered"]["viewport_cam"].tolist() == [True] * NUM_ENVS
        assert torch.equal(row["image_obs"]["pixels"][:, 0], state if rendered else torch.zeros(NUM_ENVS))
        assert torch.equal(row["viewport_cam"]["vp_pixels"][:, 0], state)
        assert torch.equal(row["proprio_obs"]["state"][:, 0], state)

    # without the mask (outside render_policy_steps_only), an unrendered group is an error, not a silent row
    with pytest.raises(RuntimeError, match="not rendered"):
        recorders[0].record_pre_step()
