# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensor-video frames: exact half-scale resize and ordered, complete background encoding."""

import time

import imageio.v2 as imageio
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from robolab.core.observations.observation_utils import (
    camera_frame_sources,
    combine_camera_frames,
    unpack_combined_frames,
    unpack_image_obs,
    unpack_viewport_cams,
)
from robolab.core.utils.video_utils import VideoWriter


def _interpolate_reference(image: torch.Tensor, scale: float) -> np.ndarray:
    resized = F.interpolate(image.permute(2, 0, 1).unsqueeze(0).float(), scale_factor=scale,
                            mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).to(image.dtype).cpu().numpy()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("shape", [(720, 1280, 3), (480, 864, 3), (6, 10, 3), (7, 10, 3), (6, 9, 1)])
def test_half_scale_frames_match_bilinear_interpolation(device, shape):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = torch.Generator().manual_seed(sum(shape))
    images = [torch.randint(0, 256, (2, *shape), dtype=torch.uint8, generator=generator) for _ in range(2)]
    images[0][0].fill_(255)
    images[1][1].fill_(0)
    images = [image.to(device) for image in images]
    obs = {"image_obs": {"left_camera": images[0], "wrist_cam": images[1]},
           "viewport_cam": {"left_camera": images[0], "wrist_cam": images[1]}}
    for env_id in range(2):
        expected = [_interpolate_reference(image[env_id], 0.5) for image in images]
        for unpack in (unpack_image_obs, unpack_viewport_cams):
            unpacked = unpack(obs, scale=0.5, env_id=env_id)
            np.testing.assert_array_equal(unpacked["left_camera"], expected[0])
            np.testing.assert_array_equal(unpacked["wrist_cam"], expected[1])
            np.testing.assert_array_equal(unpacked["combined_image"], np.concatenate(expected, axis=1))
            assert unpacked["combined_image"].dtype == np.uint8
        frames = unpack_combined_frames(obs, [env_id, 1 - env_id], scale=0.5)
        np.testing.assert_array_equal(frames[0], np.concatenate(expected, axis=1))
        np.testing.assert_array_equal(frames[1], unpack_image_obs(obs, scale=0.5, env_id=1 - env_id)["combined_image"])
    np.testing.assert_array_equal(
        unpack_combined_frames(obs, [0, 1], scale=0.5)[1], unpack_image_obs(obs, scale=0.5, env_id=1)["combined_image"]
    )


@pytest.mark.parametrize("shape", [(720, 1280, 3), (6, 10, 3), (7, 10, 3), (6, 9, 1), (6, 10, 1)])
@pytest.mark.parametrize("scale", [0.5, 1.0, 0.25])
def test_writer_thread_frames_match_unpacked_combined_image(shape, scale):
    generator = torch.Generator().manual_seed(sum(shape))
    images = [torch.randint(0, 256, (3, *shape), dtype=torch.uint8, generator=generator) for _ in range(2)]
    images[0][0].fill_(255)
    images[1][2].fill_(0)
    obs = {"image_obs": {"left_camera": images[0], "depth_of_field": images[1][..., :1], "wrist_cam": images[1]}}
    originals = [image.clone() for image in images]
    for env_id in range(3):
        sources = camera_frame_sources(obs, env_id)
        assert len(sources) == 2  # only camera-suffixed terms, as in unpack_image_obs
        frame = combine_camera_frames(sources, scale)
        expected = unpack_image_obs(obs, scale=scale, env_id=env_id)["combined_image"]
        assert frame.dtype == expected.dtype == np.uint8
        np.testing.assert_array_equal(frame, expected)
    for image, original in zip(images, originals):
        torch.testing.assert_close(image, original, rtol=0, atol=0)


@pytest.mark.parametrize("take_ownership", [False, True])
def test_video_writer_encodes_every_frame_in_order(tmp_path, take_ownership):
    path = tmp_path / "video.mp4"
    writer = VideoWriter(str(path), 15, max_pending=2)
    levels = [16 * (index % 15) + 8 for index in range(30)]
    frame = np.empty((32, 48, 3), dtype=np.uint8)
    for level in levels:
        if take_ownership:
            writer.write(np.full((32, 48, 3), level, dtype=np.uint8), take_ownership=True)
        else:
            # the writer copies caller-owned buffers before queueing them
            frame.fill(level)
            writer.write(frame)
    writer.release()
    decoded = [np.asarray(image) for image in imageio.get_reader(str(path))]
    assert len(decoded) == len(levels)
    assert all(image.shape == (32, 48, 3) for image in decoded)
    np.testing.assert_allclose([image.mean() for image in decoded], levels, atol=3)


def test_video_writer_encodes_deferred_and_direct_frames_in_call_order(tmp_path):
    path = tmp_path / "video.mp4"
    writer = VideoWriter(str(path), 15, max_pending=2)
    levels = [16 * (index % 15) + 8 for index in range(30)]
    for index, level in enumerate(levels):
        if index % 3:
            writer.write_deferred(lambda level=level: np.full((32, 48, 3), level, dtype=np.uint8))
        else:
            writer.write(np.full((32, 48, 3), level, dtype=np.uint8), take_ownership=True)
    writer.release()
    decoded = [np.asarray(image) for image in imageio.get_reader(str(path))]
    assert len(decoded) == len(levels)
    np.testing.assert_allclose([image.mean() for image in decoded], levels, atol=3)


def test_video_writer_reports_deferred_frame_failure(tmp_path):
    def broken_frame():
        raise ValueError("no camera image")

    writer = VideoWriter(str(tmp_path / "video.mp4"), 15)
    writer.write_deferred(broken_frame)
    # the failure surfaces from a later write once the encoder thread has run the frame
    with pytest.raises(RuntimeError, match="Video encoding failed") as raised:
        for _ in range(500):
            writer.write_deferred(lambda: np.zeros((8, 8, 3), dtype=np.uint8))
            time.sleep(0.01)
    assert isinstance(raised.value.__cause__, ValueError)
    writer.release()  # the error was reported once; releasing finishes cleanly

    writer = VideoWriter(str(tmp_path / "video2.mp4"), 15)
    writer.write_deferred(broken_frame)
    with pytest.raises(RuntimeError, match="Video encoding failed") as raised:
        writer.release()
    assert isinstance(raised.value.__cause__, ValueError)


def test_video_writer_reports_encoding_failure(tmp_path):
    writer = VideoWriter(str(tmp_path / "video.mp4"), 15)
    writer.write(np.zeros((8, 8, 3), dtype=np.uint8))
    writer.write(np.zeros((16, 8, 3), dtype=np.uint8))  # frame size changes: the encoder thread fails
    with pytest.raises(RuntimeError, match="Video encoding failed") as raised:
        writer.release()
    assert isinstance(raised.value.__cause__, ValueError)
