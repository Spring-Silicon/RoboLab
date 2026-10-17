# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import Any, List

import isaaclab.envs.mdp as mdp
import numpy as np
import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


def _image_observation_func():
    """Resolve the IsaacLab image observation function across versions.

    IsaacLab 2.2 (IsaacSim 5.0) exposes it as ``mdp.observations.image``;
    IsaacLab 2.3 (IsaacSim 5.1) also exposes it directly as ``mdp.image``.
    Prefer the direct attribute, then fall back to the nested module.
    """
    image_func = getattr(mdp, "image", None)
    if image_func is not None:
        return image_func
    observations = getattr(mdp, "observations", None)
    image_func = getattr(observations, "image", None) if observations is not None else None
    if image_func is None:
        raise AttributeError("IsaacLab image observation function not found")
    return image_func


def camera_image(
    env: Any,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "rgb",
    normalize: bool = True,
    allow_missing: bool = False,
) -> torch.Tensor:
    """Image observation term; unnormalized images borrow the sensor buffer.

    ObservationManager copies term outputs before publishing or modifying them,
    so an additional full image clone inside the term is unnecessary. Callers
    using this function outside the manager must copy before retaining or
    modifying an unnormalized image. Normalization uses Isaac Lab's image
    implementation unchanged. ``allow_missing`` retains :func:`image_safe`'s
    correctly shaped initial output before a depth annotator is populated.

    On CPU, a camera's ``rgb`` output is a strided view of its ``rgba`` buffer;
    that view is packed into a new contiguous tensor with the same values, which
    is several times faster than the manager's element-wise strided copy.
    """
    if allow_missing and data_type not in env.scene.sensors[sensor_cfg.name].data.output:
        return image_safe(env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=normalize)
    if not normalize:
        output = env.scene.sensors[sensor_cfg.name].data.output
        image = output[data_type]
        if data_type == "rgb" and image.device.type == "cpu" and not image.is_contiguous():
            packed = _pack_rgb_view(output.get("rgba"), image)
            if packed is not None:
                return packed
        return image
    return _image_observation_func()(
        env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=normalize
    )


def _pack_rgb_view(rgba: torch.Tensor | None, rgb: torch.Tensor) -> torch.Tensor | None:
    """Contiguous copy of ``rgb`` when it is exactly ``rgba[..., :3]`` of a contiguous uint8 CPU buffer."""
    if (
        rgba is None
        or rgba.dtype != torch.uint8
        or rgb.dtype != torch.uint8
        or rgba.device.type != "cpu"
        or not rgba.is_contiguous()
        or rgba.ndim != 4
        or rgba.shape[-1] != 4
        or rgb.shape != rgba.shape[:-1] + (3,)
        or rgb.data_ptr() != rgba.data_ptr()
        or rgb.stride() != rgba.stride()
    ):
        return None
    import cv2

    packed = torch.empty(rgb.shape, dtype=torch.uint8)
    for env_index in range(rgba.shape[0]):
        cv2.cvtColor(rgba[env_index].numpy(), cv2.COLOR_RGBA2RGB, dst=packed[env_index].numpy())
    return packed


def image_safe(
    env: Any,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("over_shoulder_left_camera"),
    data_type: str = "rgb",
    normalize: bool = True,
) -> torch.Tensor:
    """Read a camera image, tolerating the first pre-render manager query.

    With lazy sensor updates disabled, annotators such as ``depth`` are only
    populated after the first render. Return a correctly shaped-and-typed zero
    image until then so the observation manager can size its buffers.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    if data_type not in sensor.data.output:
        channels = 3 if data_type == "rgb" else 1
        dtype = torch.float32 if (normalize or data_type != "rgb") else torch.uint8
        return torch.zeros(
            (env.num_envs, sensor.cfg.height, sensor.cfg.width, channels),
            device=env.device,
            dtype=dtype,
        )
    return _image_observation_func()(
        env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=normalize
    )


def camera_pos(env: Any, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Camera position, env-local frame (world minus env origin), meters. Shape (num_envs, 3)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.pos_w - env.scene.env_origins


def camera_quat(env: Any, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Camera orientation, world frame, OpenGL convention, (w, x, y, z). Shape (num_envs, 4)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.quat_w_opengl


def camera_intrinsics(env: Any, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Camera intrinsic matrix K in pixels. Shape (num_envs, 3, 3)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    matrices = getattr(sensor.data, "intrinsic_matrices", None)
    if matrices is None:
        return torch.zeros((env.num_envs, 3, 3), device=env.device, dtype=torch.float32)
    return matrices


def object_pos(env: Any, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Object root position, env-local frame (world minus env origin), meters. Shape (num_envs, 3)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def object_quat(env: Any, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Object root orientation, world frame, (w, x, y, z). Shape (num_envs, 4)."""
    return env.scene[asset_cfg.name].data.root_quat_w


def object_vel(env: Any, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Object root velocity, world frame, linear (m/s) + angular (rad/s). Shape (num_envs, 6)."""
    return env.scene[asset_cfg.name].data.root_vel_w


def generate_object_state_obs(object_names: List[str]):
    """Create an observation group with ground-truth state terms per object.

    Each object gets ``<name>_pos`` (env-local meters), ``<name>_quat``
    (world-frame w, x, y, z), and ``<name>_vel`` (world-frame linear+angular)
    terms, evaluated once per environment step like any other observation.
    Opt in at registration time via ``object_state_obs=True`` on
    ``auto_register_droid_envs`` / ``generate_task_env_cfg`` — the object
    list then comes from the task's ``contact_object_list``.

    Args:
        object_names: Scene entity names to observe (rigid objects or
            articulations with a root state).

    Returns:
        A dynamically generated observation group class (ObsGroup subclass).
    """
    obs_terms = {}
    for name in object_names:
        for suffix, func in (("pos", object_pos), ("quat", object_quat), ("vel", object_vel)):
            obs_terms[f"{name}_{suffix}"] = ObsTerm(
                func=func,
                params={"asset_cfg": SceneEntityCfg(name)},
            )

    @configclass
    class DynamicObjectStateObsCfg(ObsGroup):
        """Dynamically generated ground-truth object-state observations."""

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    for term_name, obs_term in obs_terms.items():
        setattr(DynamicObjectStateObsCfg, term_name, obs_term)

    return DynamicObjectStateObsCfg


def generate_image_obs_from_cameras(camera_cfgs: List[Any] | Any):
    """
    Dynamically create an image observation group configuration from one or more camera configs.

    Example usage:
        from robolab.variations.camera import EgocentricWideAngleCameraCfg, EgocentricMirroredWideAngleCameraCfg, EgocentricMirroredCameraCfg

        # Create a list of camera configuration classes
        camera_cfgs = [EgocentricWideAngleCameraCfg, EgocentricMirroredWideAngleCameraCfg, EgocentricMirroredCameraCfg]

        # Generate dynamic observation group configuration
        DynamicImageObsCfg = generate_image_obs_from_cameras(camera_cfgs)

        # Instantiate the configuration
        image_obs_cfg = DynamicImageObsCfg()

        # The image_obs_cfg will now contain observation terms for all cameras found in the camera configs:
        # - egocentric_wide_angle_camera
        # - egocentric_mirrored_wide_angle_camera
        # - egocentric_mirrored_camera

    Args:
        camera_cfgs: List of camera configuration classes (e.g., [EgocentricWideAngleCameraCfg, ...])

    Returns:
        A dynamically generated observation group configuration class (ObsGroup)
    """
    # Create a dictionary to store observation terms
    obs_terms = {}

    if not isinstance(camera_cfgs, list):
        camera_cfgs = [camera_cfgs]

    # Iterate through camera configs and extract camera names
    for camera_cfg in camera_cfgs:
        # Create an instance of the camera config to access its attributes
        camera_cfg_instance = camera_cfg()

        # Get all camera attributes from the config class instance
        for attr_name in dir(camera_cfg_instance):
            if not attr_name.startswith('_'):
                attr_value = getattr(camera_cfg_instance, attr_name)
                # Check if this attribute is a CameraCfg instance (has prim_path)
                if isinstance(attr_value, CameraCfg):
                # if hasattr(attr_value, 'prim_path'):
                    camera_name = attr_name
                    obs_terms[camera_name] = ObsTerm(
                        func=camera_image,
                        params={
                            "sensor_cfg": SceneEntityCfg(camera_name),
                            "data_type": "rgb",
                            "normalize": False,
                        }
                    )
                    # Cameras that render depth (see robolab.variations.camera.with_depth)
                    # also get <camera>_depth plus pose/intrinsics metadata terms, so
                    # calibration flows through the standard observation pipeline
                    # (per-env, recorded like any other term).
                    if "depth" in attr_value.data_types:
                        obs_terms[f"{camera_name}_depth"] = ObsTerm(
                            func=camera_image,
                            params={
                                "sensor_cfg": SceneEntityCfg(camera_name),
                                "data_type": "depth",
                                "normalize": False,
                                "allow_missing": True,
                            }
                        )
                        for term_suffix, term_func in (
                            ("pos", camera_pos),
                            ("quat", camera_quat),
                            ("K", camera_intrinsics),
                        ):
                            obs_terms[f"{camera_name}_{term_suffix}"] = ObsTerm(
                                func=term_func,
                                params={"sensor_cfg": SceneEntityCfg(camera_name)},
                            )

    # Create the dynamic image observation group class
    @configclass
    class DynamicImageObsCfg(ObsGroup):
        """Dynamically generated image observations for policy."""

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # Add observation terms to the class
    for camera_name, obs_term in obs_terms.items():
        setattr(DynamicImageObsCfg, camera_name, obs_term)

    return DynamicImageObsCfg

def generate_obs_cfg(obs_groups: dict[str, ObsGroup]):
    """
    Dynamically create an observation configuration from multiple observation groups.

    Example usage:
        from robolab.variations.camera import EgocentricWideAngleCameraCfg, EgocentricMirroredWideAngleCameraCfg
        from robolab.core.observations.image_obs import generate_image_obs_from_cameras, generate_obs_cfg

        # Generate image observation group
        ImageObsCfg = generate_image_obs_from_cameras([EgocentricWideAngleCameraCfg, EgocentricMirroredWideAngleCameraCfg])

        # Create main observation configuration with multiple groups
        ObservationCfg = generate_obs_cfg({
            "image_obs": ImageObsCfg(),
            # "policy": other_obs_group,  # Could add other observation groups
        })

        # Instantiate the configuration
        obs_cfg = ObservationCfg()

    Args:
        obs_groups: Dictionary mapping group names to ObsGroup instances
                   e.g., {"image_obs": ImageObsCfg(), "policy": PolicyObsCfg()}

    Returns:
        A dynamically generated observation configuration class
    """
    # Create the main configuration class
    @configclass
    class DynamicObservationCfg:
        """Dynamically generated observation configuration."""
        pass

    # Add observation groups to the class
    for group_name, obs_group in obs_groups.items():
        setattr(DynamicObservationCfg, group_name, obs_group)

    return DynamicObservationCfg

def unpack_image_obs(obs, obs_group_name="image_obs", camera_suffix=["_camera", "_cam", "_img", "_image"], scale: float = 1.0, env_id: int = 0):
    """
    Unpack image observations from an observation dictionary.

    Args:
        obs: Observation dictionary
        obs_group_name: Name of the observation group to unpack, default is "image_obs"
        camera_suffix: Suffix of the camera to unpack, default is ["_camera", "_cam", "_img", "_image"]
        scale: Scale factor for resizing on GPU before CPU transfer (0.5 = half size).
               Resizing on GPU is faster than transferring full resolution then resizing on CPU.

    Returns:
        Dictionary containing the unpacked image observations
    """
    images = []
    image_dict = {}
    for key, value in obs[obs_group_name].items():
        if any(key.endswith(suffix) for suffix in camera_suffix):
            # Resize on GPU before CPU transfer (much faster for small scale values)
            tensor = _resize_image(value[env_id].detach(), scale)
            image = tensor.cpu().numpy()
            image_dict[key] = image
            images.append(image)
    combined_image = np.concatenate(images, axis=1)
    image_dict["combined_image"] = combined_image
    return image_dict

def unpack_proprio_obs(obs, obs_group_name="proprio_obs", env_id: int = 0):
    """
    Unpack proprioceptive observations from an observation dictionary.

    Args:
        obs: Observation dictionary
        obs_group_name: Name of the observation group to unpack, default is "proprio_obs"
        env_id: Environment index to unpack, default is 0

    Returns:
        Dictionary containing the unpacked proprioceptive observations
    """
    proprio_dict = {}
    for key, value in obs[obs_group_name].items():
        proprio_dict[key] = value[env_id].clone().detach().cpu().numpy()
    return proprio_dict

def _resize_image(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Bilinear resize of (..., H, W, C) images, as ``F.interpolate(align_corners=False)`` then a cast back.

    For uint8 images halved along even dimensions, every output pixel samples source
    coordinate 2i + 0.5 with weights 0.5/0.5, so the float result is exactly the mean of a
    2x2 block (all intermediates are multiples of 0.25 below 256) and the cast truncates it.
    That case is computed with integer adds and a shift, bit-identical to the float path.
    """
    if scale == 1.0:
        return tensor
    height, width = tensor.shape[-3], tensor.shape[-2]
    if scale == 0.5 and tensor.dtype == torch.uint8 and height % 2 == 0 and width % 2 == 0:
        total = tensor[..., 0::2, 0::2, :].to(torch.int16)
        total += tensor[..., 0::2, 1::2, :]
        total += tensor[..., 1::2, 0::2, :]
        total += tensor[..., 1::2, 1::2, :]
        return total.bitwise_right_shift_(2).to(torch.uint8)
    import torch.nn.functional as F

    # tensor shape is (..., H, W, C), need (N, C, H, W) for interpolate
    # Convert to float for interpolate, then back to the original dtype
    original_dtype = tensor.dtype
    images = tensor.reshape(-1, *tensor.shape[-3:])
    resized = F.interpolate(images.permute(0, 3, 1, 2).float(), scale_factor=scale,
                            mode='bilinear', align_corners=False)
    resized = resized.permute(0, 2, 3, 1).to(original_dtype)
    return resized.reshape(*tensor.shape[:-3], *resized.shape[-3:])


def unpack_combined_frames(obs, env_ids: list[int], obs_group_name="image_obs",
                           camera_suffix=["_camera", "_cam", "_img", "_image"], scale: float = 1.0) -> np.ndarray:
    """``unpack_image_obs(obs, ...)["combined_image"]`` for several envs at once.

    Resizes all selected envs together and transfers them to the host once. Returns a
    new (len(env_ids), H, W_total, C) array whose row ``i`` equals the combined image of
    ``env_ids[i]``.
    """
    images = []
    for key, value in obs[obs_group_name].items():
        if any(key.endswith(suffix) for suffix in camera_suffix):
            selected = value if list(env_ids) == list(range(value.shape[0])) else value[env_ids]
            images.append(_resize_image(selected.detach(), scale))
    return torch.cat(images, dim=2).cpu().numpy()


def camera_frame_sources(obs, env_id: int, obs_group_name="image_obs",
                         camera_suffix=["_camera", "_cam", "_img", "_image"]) -> list[torch.Tensor]:
    """The camera images that ``unpack_image_obs(obs, ..., env_id=env_id)`` combines, in the same order.

    Returns views of the observation tensors, not copies. Pass them to
    :func:`combine_camera_frames` to obtain the same ``combined_image``.
    """
    return [
        value[env_id].detach()
        for key, value in obs[obs_group_name].items()
        if any(key.endswith(suffix) for suffix in camera_suffix)
    ]


def _half_scale_concat(images: List[torch.Tensor]) -> torch.Tensor:
    """Exact 2x2 integer mean of each (H, W, C) uint8 image (even H, W), concatenated along the width.

    Same arithmetic as :func:`_resize_image`'s integer path: the 2x2 sum in int16, shifted right by 2.
    """
    halves: List[torch.Tensor] = []
    for image in images:
        height = image.shape[0] // 2
        width = image.shape[1] // 2
        channels = image.shape[2]
        pixel_pairs = image.reshape(height, 2, width, 2 * channels)
        row_sums = pixel_pairs[:, 0].to(torch.int16) + pixel_pairs[:, 1]
        block_sums = row_sums[:, :, :channels] + row_sums[:, :, channels:]
        halves.append(block_sums.bitwise_right_shift(2).to(torch.uint8))
    return torch.cat(halves, dim=1)


_half_scale_concat_scripted = None
_half_scale_concat_lock = threading.Lock()


def _scripted_half_scale_concat():
    """:func:`_half_scale_concat` compiled once with TorchScript, which runs it without holding the GIL."""
    global _half_scale_concat_scripted
    with _half_scale_concat_lock:
        if _half_scale_concat_scripted is None:
            _half_scale_concat_scripted = torch.jit.script(_half_scale_concat)
        return _half_scale_concat_scripted


def combine_camera_frames(images: list[torch.Tensor], scale: float = 1.0) -> np.ndarray:
    """``unpack_image_obs(...)["combined_image"]`` from one env's (H, W, C) camera images.

    Returns a new array; the inputs are only read. Intended for video writer threads:
    halving host uint8 images with even dimensions (the sensor-video case) is a single
    TorchScript call that does not hold the GIL, so it does not stall the simulation
    thread. Other inputs use :func:`_resize_image` per image.
    """
    if scale == 0.5 and images and all(
        image.dtype == torch.uint8
        and image.device.type == "cpu"
        and image.ndim == 3
        and image.shape[0] % 2 == 0
        and image.shape[1] % 2 == 0
        for image in images
    ):
        return _scripted_half_scale_concat()(images).numpy()
    return torch.cat([_resize_image(image, scale) for image in images], dim=1).cpu().numpy()


def unpack_viewport_cams(obs, obs_group_name="viewport_cam", camera_suffix=["_camera", "_cam", "_img", "_image"], scale: float = 1.0, env_id: int = 0):
    """
    Unpack viewport camera observations from an observation dictionary.

    Args:
        obs: Observation dictionary
        obs_group_name: Name of the observation group to unpack, default is "viewport_cam"
        camera_suffix: Suffix of the camera to unpack, default is ["_camera", "_cam", "_img", "_image"]
        scale: Scale factor for resizing on GPU before CPU transfer (0.5 = half size).
               Resizing on GPU is faster than transferring full resolution then resizing on CPU.
        env_id: Environment index to unpack, default is 0

    Returns:
        Dictionary containing the unpacked viewport camera observations
    """
    images = []
    viewport_dict = {}
    for key, value in obs[obs_group_name].items():
        if any(key.endswith(suffix) for suffix in camera_suffix):
            # Resize on GPU before CPU transfer (much faster for small scale values)
            tensor = _resize_image(value[env_id].detach(), scale)
            image = tensor.cpu().numpy()
            viewport_dict[key] = image
            images.append(image)
    combined_image = np.concatenate(images, axis=1)
    viewport_dict["combined_image"] = combined_image
    return viewport_dict