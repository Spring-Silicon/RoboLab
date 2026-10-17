# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image terms preserve values while leaving snapshot ownership to the manager."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import modifiers, noise

from robolab.core.observations.observation_utils import camera_image, image_safe


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("normalize", [False, True])
def test_camera_image_manager_snapshot_ownership(device, normalize):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pixels = torch.arange(72, device=device, dtype=torch.uint8).reshape(2, 3, 4, 3)
    sensor = SimpleNamespace(data=SimpleNamespace(output={"rgb": pixels}))
    env = SimpleNamespace(
        num_envs=2,
        device=device,
        scene=SimpleNamespace(sensors={"camera": sensor}),
        sim=SimpleNamespace(is_playing=lambda: True),
    )
    # A pre-resolved name avoids asset-name discovery; this test uses real
    # ObservationManager processing, not a camera renderer or an output mock.
    group = ObservationGroupCfg(concatenate_terms=False)
    group.image = ObservationTermCfg(
        func=camera_image,
        params={"sensor_cfg": SimpleNamespace(name="camera"), "normalize": normalize},
    )
    manager = ObservationManager({"images": group}, env)
    observation = manager.compute()["images"]["image"]
    expected = pixels.float() / 255 if normalize else pixels.clone()
    if normalize:
        expected -= expected.mean(dim=(1, 2), keepdim=True)
    torch.testing.assert_close(observation, expected)
    assert observation.data_ptr() != pixels.data_ptr()
    assert observation.dtype == (torch.float32 if normalize else torch.uint8)
    pixels.zero_()
    torch.testing.assert_close(observation, expected)
    observation.fill_(1)
    assert torch.count_nonzero(pixels) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_image_safe_retains_owned_output_and_initial_depth_shape(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    sensor = SimpleNamespace(
        data=SimpleNamespace(output={}),
        cfg=SimpleNamespace(height=3, width=4),
    )
    env = SimpleNamespace(num_envs=2, device=device, scene=SimpleNamespace(sensors={"camera": sensor}))
    cfg = SceneEntityCfg("camera")
    initial = image_safe(env, sensor_cfg=cfg, data_type="depth", normalize=False)
    assert initial.shape == (2, 3, 4, 1)
    assert initial.dtype == torch.float32
    assert torch.count_nonzero(initial) == 0
    term_initial = camera_image(env, sensor_cfg=cfg, data_type="depth", normalize=False, allow_missing=True)
    torch.testing.assert_close(term_initial, initial)
    depth = torch.ones_like(initial)
    sensor.data.output["depth"] = depth
    snapshot = image_safe(env, sensor_cfg=cfg, data_type="depth", normalize=False)
    assert snapshot.data_ptr() != depth.data_ptr()
    term_value = camera_image(env, sensor_cfg=cfg, data_type="depth", normalize=False, allow_missing=True)
    assert term_value.data_ptr() == depth.data_ptr()
    torch.testing.assert_close(term_value, snapshot)
    depth.zero_()
    torch.testing.assert_close(snapshot, torch.ones_like(snapshot))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_camera_image_history_partial_reset_and_recording_snapshot(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pixels = torch.ones((2, 3, 4, 3), device=device, dtype=torch.uint8)
    env = SimpleNamespace(
        num_envs=2,
        device=device,
        scene=SimpleNamespace(sensors={"camera": SimpleNamespace(data=SimpleNamespace(output={"rgb": pixels}))}),
        sim=SimpleNamespace(is_playing=lambda: True),
    )
    group = ObservationGroupCfg(concatenate_terms=False)
    group.image = ObservationTermCfg(
        func=camera_image,
        params={"sensor_cfg": SimpleNamespace(name="camera"), "normalize": False},
        history_length=3,
        flatten_history_dim=False,
    )
    manager = ObservationManager({"images": group}, env)
    initial = manager.compute(update_history=True)["images"]["image"]
    assert initial.shape == (2, 3, 3, 4, 3)
    pixels.fill_(2)
    recording = manager.compute()["images"]["image"]
    torch.testing.assert_close(recording, initial)
    stepped = manager.compute(update_history=True)["images"]["image"]
    torch.testing.assert_close(stepped[:, -1], pixels)
    torch.testing.assert_close(stepped[:, :-1], torch.ones_like(stepped[:, :-1]))

    manager.reset(env_ids=[0])
    pixels[0].fill_(3)
    reset = manager.compute(update_history=True)["images"]["image"]
    torch.testing.assert_close(reset[0], torch.full_like(reset[0], 3))
    torch.testing.assert_close(reset[1, 0], torch.ones_like(reset[1, 0]))
    torch.testing.assert_close(reset[1, 1:], torch.full_like(reset[1, 1:], 2))
    torch.testing.assert_close(initial, torch.ones_like(initial))
    torch.testing.assert_close(recording, torch.ones_like(recording))
    reset.zero_()
    torch.testing.assert_close(pixels[0], torch.full_like(pixels[0], 3))
    torch.testing.assert_close(pixels[1], torch.full_like(pixels[1], 2))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_camera_image_matches_owned_term_with_modifiers_noise_and_history(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    depth = torch.ones((2, 3, 4, 1), device=device)
    env = SimpleNamespace(
        num_envs=2,
        device=device,
        scene=SimpleNamespace(sensors={"camera": SimpleNamespace(data=SimpleNamespace(output={"depth": depth}))}),
        sim=SimpleNamespace(is_playing=lambda: True),
    )

    def run(term_func):
        group = ObservationGroupCfg(concatenate_terms=False)
        group.image = ObservationTermCfg(
            func=term_func,
            params={"sensor_cfg": SimpleNamespace(name="camera"), "data_type": "depth", "normalize": False},
            modifiers=[modifiers.IntegratorCfg(dt=0.1)],
            noise=noise.GaussianNoiseCfg(std=0.05),
            clip=(0.0, 2.0),
            scale=0.5,
            history_length=2,
        )
        manager = ObservationManager({"images": group}, env)
        torch.manual_seed(83)
        outputs = []
        for step in range(4):
            # Preserve both the pre-reset recorder query and final history update.
            outputs.append(manager.compute()["images"]["image"])
            if step == 2:
                manager.reset(env_ids=[0])
            outputs.append(manager.compute(update_history=True)["images"]["image"])
        return outputs, torch.rand(5, device=device)

    reference, reference_rng = run(image_safe)
    actual, actual_rng = run(camera_image)
    for expected, observed in zip(reference, actual):
        torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(actual_rng, reference_rng)
    torch.testing.assert_close(depth, torch.ones_like(depth))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_camera_rgb_view_observation_values_and_ownership(device):
    """An rgb view of an rgba buffer yields the same contiguous, owned observation values."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = torch.Generator().manual_seed(7)
    rgba = torch.randint(0, 256, (2, 6, 10, 4), dtype=torch.uint8, generator=generator).to(device)
    rgb = rgba[..., :3]
    env = SimpleNamespace(
        num_envs=2,
        device=device,
        scene=SimpleNamespace(sensors={"camera": SimpleNamespace(data=SimpleNamespace(output={"rgba": rgba, "rgb": rgb}))}),
        sim=SimpleNamespace(is_playing=lambda: True),
    )
    group = ObservationGroupCfg(concatenate_terms=False)
    group.image = ObservationTermCfg(
        func=camera_image, params={"sensor_cfg": SimpleNamespace(name="camera"), "normalize": False}
    )
    manager = ObservationManager({"images": group}, env)
    expected = rgb.clone()
    observation = manager.compute()["images"]["image"]
    assert torch.equal(observation, expected)
    assert observation.dtype == torch.uint8 and observation.is_contiguous()
    assert observation.data_ptr() != rgba.data_ptr()
    rgba.fill_(9)
    assert torch.equal(observation, expected)
    assert torch.equal(manager.compute()["images"]["image"], torch.full_like(expected, 9))