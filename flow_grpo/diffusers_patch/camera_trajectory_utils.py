"""
Camera trajectory and latent-initialization utilities for World-R1.

Source note:
- The noise-wrapping idea is adapted from Go-with-the-Flow.
- The released file keeps the parts required by the World-R1 paper to
  synthesize camera trajectories and inject them into the latent rollout.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Union
from einops import rearrange
import rp
import os
import re


# Camera movement definitions
layout_info = {
    "push_in": {"scenenum": 1, "prompt": "push in", "layout_type": "intra"},
    "pull_out": {"scenenum": 1, "prompt": "pull out", "layout_type": "inter"},
    "move_left": {"scenenum": 1, "prompt": "move left", "layout_type": "inter"},
    "move_right": {"scenenum": 1, "prompt": "move right", "layout_type": "inter"},
    "orbit_left": {"scenenum": 1, "prompt": "orbit left", "layout_type": "intra"},
    "orbit_right": {"scenenum": 1, "prompt": "orbit right", "layout_type": "intra"},
    "pan_left": {"scenenum": 1, "prompt": "pan left", "layout_type": "inter"},
    "pan_right": {"scenenum": 1, "prompt": "pan right", "layout_type": "inter"},
    "pull_left": {"scenenum": 1, "prompt": "move left, pull out, then pan left", "layout_type": "inter"},
    "pull_right": {"scenenum": 1, "prompt": "move right, pull out, then pan right", "layout_type": "inter"},
    "fixed": {"scenenum": 1, "prompt": "fixed", "layout_type": "camera fix"}
}

primitive_camera_movements = [
    "push_in",
    "pull_out",
    "move_left",
    "move_right",
    "orbit_left",
    "orbit_right",
    "pan_left",
    "pan_right",
    "fixed",
]


class TrajectoryGenerator:
    """Generate camera trajectories for different camera movements."""

    def __init__(
        self,
        start_pos: List[float],
        num_frames: int = 81,
        motion_profile: Optional[Dict[str, float]] = None,
    ):
        self.num_frames = num_frames
        self.start_pos = np.array(start_pos, dtype=float)
        self.default_rot = np.eye(3)
        self.motion_profile = motion_profile or {}

    def get_identity_4x4(self) -> np.ndarray:
        return np.eye(4)

    def format_matrix(self, mat: np.ndarray) -> str:
        """
        Format 4x4 matrix to string format.
        Example: "[r00 r10 r20 0] [r01 r11 r21 0] [r02 r12 r22 0] [tx ty tz 1]"
        """
        cols = []
        for i in range(4):
            col = mat[:, i]
            col_str = f"[{col[0]:.6g} {col[1]:.6g} {col[2]:.6g} {col[3]:.6g}]"
            cols.append(col_str)
        return " ".join(cols) + " "

    def get_translation_matrix(self, offset: np.ndarray) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, 3] = offset
        return mat

    def get_rotation_y_matrix(self, angle_deg: float) -> np.ndarray:
        """Rotate around Y axis (Pan)"""
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        rot = np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
        return rot

    def get_rotation_x_matrix(self, angle_deg: float) -> np.ndarray:
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        rot = np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ])
        return rot

    def get_rotation_z_matrix(self, angle_deg: float) -> np.ndarray:
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        rot = np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return rot

    def get_compound_rotation_matrix(self, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
        return (
            self.get_rotation_y_matrix(yaw_deg)
            @ self.get_rotation_x_matrix(pitch_deg)
            @ self.get_rotation_z_matrix(roll_deg)
        )

    def generate(self, layout_name: str) -> Dict[str, str]:
        """Generate camera trajectory for given layout."""
        trajectories = {}

        step_count = max(self.num_frames - 1, 1)
        move_total = float(self.motion_profile.get("move_total", 1800.0))
        zoom_total = float(self.motion_profile.get("zoom_total", 2200.0))
        yaw_total = float(self.motion_profile.get("yaw_total_deg", 200.0))
        pitch_total = float(self.motion_profile.get("pitch_total_deg", 20.0))
        roll_total = float(self.motion_profile.get("roll_total_deg", 12.0))
        orbit_radius = float(self.motion_profile.get("orbit_radius", 600.0))
        orbit_arc = float(self.motion_profile.get("orbit_arc_deg", max(180.0, abs(yaw_total))))

        speed_move = abs(move_total) / step_count
        speed_zoom = abs(zoom_total) / step_count
        speed_rot = abs(yaw_total) / step_count

        for i in range(self.num_frames):
            frame_key = f"frame{i}"
            progress = i / step_count
            pitch_i = pitch_total * progress
            roll_i = roll_total * progress

            # Initialize current frame matrix
            current_mat = np.eye(4)
            current_mat[:3, 3] = self.start_pos

            # Apply transformation based on layout_name
            if layout_name == "fixed":
                pass  # No movement

            elif layout_name == "push_in":
                offset = np.array([0, 0, speed_zoom * i])
                current_mat = current_mat @ self.get_translation_matrix(offset)

            elif layout_name == "pull_out":
                offset = np.array([0, 0, -speed_zoom * i])
                current_mat = current_mat @ self.get_translation_matrix(offset)

            elif layout_name == "move_left":
                offset = np.array([-speed_move * i, 0, 0])
                current_mat = current_mat @ self.get_translation_matrix(offset)

            elif layout_name == "move_right":
                offset = np.array([speed_move * i, 0, 0])
                current_mat = current_mat @ self.get_translation_matrix(offset)

            elif layout_name == "pan_left":
                rot_mat = self.get_compound_rotation_matrix(-speed_rot * i, pitch_i, roll_i)
                pos = current_mat[:3, 3].copy()
                current_mat[:3, 3] = 0
                current_mat = current_mat @ rot_mat
                current_mat[:3, 3] = pos

            elif layout_name == "pan_right":
                rot_mat = self.get_compound_rotation_matrix(speed_rot * i, pitch_i, roll_i)
                pos = current_mat[:3, 3].copy()
                current_mat[:3, 3] = 0
                current_mat = current_mat @ rot_mat
                current_mat[:3, 3] = pos

            elif layout_name.startswith("orbit"):
                center = self.start_pos + np.array([0, 0, -orbit_radius])
                angle = orbit_arc * progress if "left" in layout_name else -orbit_arc * progress
                rad = np.radians(angle)

                new_x = center[0] + orbit_radius * np.sin(rad)
                new_z = center[2] + orbit_radius * np.cos(rad)
                current_mat[:3, 3] = [new_x, self.start_pos[1], new_z]

                rot_mat = self.get_compound_rotation_matrix(-angle, pitch_i, roll_i)
                current_mat[:3, :3] = rot_mat[:3, :3]

            elif layout_name == "pull_left":
                offset = np.array([-speed_move * i * 0.5, 0, -speed_zoom * i * 0.5])
                current_mat = current_mat @ self.get_translation_matrix(offset)

                rot_mat = self.get_compound_rotation_matrix(-speed_rot * i, pitch_i, roll_i)
                pos = current_mat[:3, 3].copy()
                current_mat[:3, 3] = 0
                current_mat = current_mat @ rot_mat
                current_mat[:3, 3] = pos

            elif layout_name == "pull_right":
                offset = np.array([speed_move * i * 0.5, 0, -speed_zoom * i * 0.5])
                current_mat = current_mat @ self.get_translation_matrix(offset)

                rot_mat = self.get_compound_rotation_matrix(speed_rot * i, pitch_i, roll_i)
                pos = current_mat[:3, 3].copy()
                current_mat[:3, 3] = 0
                current_mat = current_mat @ rot_mat
                current_mat[:3, 3] = pos

            if layout_name in {"push_in", "pull_out", "move_left", "move_right"}:
                rot_sign = 1.0
                if layout_name in {"pull_out", "move_left"}:
                    rot_sign = -1.0
                current_mat[:3, :3] = self.get_compound_rotation_matrix(
                    rot_sign * speed_rot * i,
                    pitch_i,
                    roll_i,
                )[:3, :3]

            trajectories[frame_key] = self.format_matrix(current_mat)

        return trajectories


def sample_motion_profiles(
    trajectory_names: List[str],
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, float]]:
    del rng
    profiles = []
    for name in trajectory_names:
        profile = {
            "move_total": 0.0,
            "zoom_total": 0.0,
            "yaw_total_deg": 0.0,
            "pitch_total_deg": 0.0,
            "roll_total_deg": 0.0,
            "orbit_radius": 520.0,
            "orbit_arc_deg": 0.0,
        }

        if name == "fixed":
            pass
        elif name in {"orbit_left", "orbit_right"}:
            profile["orbit_arc_deg"] = 24.0
            profile["orbit_radius"] = 600.0
        elif name in {"pan_left", "pan_right"}:
            profile["yaw_total_deg"] = 14.0 if name == "pan_left" else -14.0
        elif name == "push_in":
            profile["zoom_total"] = 600.0
        elif name == "pull_out":
            profile["zoom_total"] = 600.0
        elif name == "move_left":
            profile["move_total"] = 220.0
        elif name == "move_right":
            profile["move_total"] = 220.0
        elif name == "pull_left":
            profile["move_total"] = 160.0
            profile["zoom_total"] = 300.0
            profile["yaw_total_deg"] = 8.0
        elif name == "pull_right":
            profile["move_total"] = 160.0
            profile["zoom_total"] = 300.0
            profile["yaw_total_deg"] = -8.0
        else:
            raise ValueError(f"Unsupported trajectory name: {name}")

        profiles.append(profile)

    return profiles


def detect_camera_movements(prompts: Union[str, List[str]]) -> List[str]:
    """
    Detect camera movements in prompts based on layout_info.

    Args:
        prompts: Single prompt string or list of prompts

    Returns:
        List of detected camera movement names (e.g., ['push_in', 'pan_left'])
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    detected_movements = []

    for prompt in prompts:
        prompt_lower = prompt.lower()
        prompt_matches = []

        # Only detect primitive motions automatically.
        # Composite prompts such as "move left, pull out, then pan left"
        # should be expanded into the ordered primitive sequence instead of
        # being matched both as components and as the synthetic "pull_left".
        for movement_name in primitive_camera_movements:
            movement_prompt = layout_info[movement_name]["prompt"].lower()
            pattern = rf"(?<![a-z]){re.escape(movement_prompt)}(?![a-z])"
            for match in re.finditer(pattern, prompt_lower):
                prompt_matches.append(
                    (match.start(), -(match.end() - match.start()), movement_name)
                )

        prompt_matches.sort()

        last_start = None
        for start, _, movement_name in prompt_matches:
            if last_start == start:
                continue
            detected_movements.append(movement_name)
            last_start = start

    return detected_movements


def detect_camera_movements_for_batch(
    prompts: Union[str, List[str]],
    force_camera_movement: Optional[str] = None,
) -> List[str]:
    """
    Detect camera movements for a batch using the same logic as noise wrapping.
    """
    if force_camera_movement is not None:
        # If force_camera_movement is specified, use it directly.
        print(f"Force camera movement: {force_camera_movement}")
        return [force_camera_movement] if force_camera_movement != "fixed" else []

    if isinstance(prompts, list):
        per_prompt_movements = [detect_camera_movements(p) for p in prompts]
        non_empty = [m for m in per_prompt_movements if len(m) > 0]
        detected_movements = non_empty[0] if len(non_empty) > 0 else []

        # Warn if batch contains different camera movements.
        unique_non_empty = {tuple(m) for m in non_empty}
        if len(unique_non_empty) > 1:
            print(
                f"Warning: batch prompts contain multiple camera movement sequences {sorted(unique_non_empty)}; "
                f"using the first detected sequence: {detected_movements}"
            )
        return detected_movements

    return detect_camera_movements(prompts)


def expand_prompts_for_batch(
    prompts: Union[str, List[str]],
    batch_size: Optional[int] = None,
) -> List[str]:
    """
    Expand prompts so there is one prompt string for each batch item.

    This keeps prompt-to-latent / prompt-to-trajectory mapping stable when a
    batch contains multiple different prompts or repeated samples per prompt.
    """
    if isinstance(prompts, str):
        prompt_list = [prompts]
    else:
        prompt_list = list(prompts)

    if batch_size is None or batch_size <= len(prompt_list):
        return prompt_list[:batch_size] if batch_size is not None else prompt_list

    if len(prompt_list) == 1:
        return prompt_list * batch_size

    repeats = (batch_size + len(prompt_list) - 1) // len(prompt_list)
    return (prompt_list * repeats)[:batch_size]


def remove_camera_keywords_from_prompts(prompts: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Remove all camera movement keywords from prompts.

    This is useful for testing whether noise wrapping is responsible for video motion.
    By removing camera keywords while keeping noise wrapping active, you can isolate
    the effect of noise wrapping alone.

    Args:
        prompts: Single prompt string or list of prompts

    Returns:
        Cleaned prompts with camera keywords removed
    """
    is_single = isinstance(prompts, str)
    if is_single:
        prompts = [prompts]

    cleaned_prompts = []

    for prompt in prompts:
        cleaned = prompt
        # Remove all camera movement keyword phrases
        for info in layout_info.values():
            movement_prompt = info['prompt']
            # Case-insensitive replacement
            import re
            cleaned = re.sub(re.escape(movement_prompt), '', cleaned, flags=re.IGNORECASE)

        # Clean up extra spaces
        cleaned = ' '.join(cleaned.split())
        cleaned_prompts.append(cleaned)

    return cleaned_prompts[0] if is_single else cleaned_prompts


def concatenate_camera_trajectories(
    trajectory_names: List[str],
    frames_per_trajectory: int = 81,
    start_position: List[float] = [3390, 1380, 240],
    motion_profiles: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, str]:
    """
    Generate and concatenate multiple camera trajectories into one.

    Args:
        trajectory_names: List of trajectory names (e.g., ['push_in', 'pan_left'])
        frames_per_trajectory: Number of frames for each trajectory segment
        start_position: Starting camera position [x, y, z]

    Returns:
        Dictionary mapping frame keys to camera matrix strings
    """
    if not trajectory_names:
        # No camera movements detected, return identity trajectory
        generator = TrajectoryGenerator(start_position, num_frames=frames_per_trajectory)
        return generator.generate("fixed")

    if motion_profiles is None:
        motion_profiles = sample_motion_profiles(trajectory_names)
    if len(motion_profiles) != len(trajectory_names):
        raise ValueError(
            f"motion_profiles length {len(motion_profiles)} must match trajectory_names length {len(trajectory_names)}"
        )

    concatenated_trajectory = {}
    current_start_pose = np.eye(4)
    current_start_pose[:3, 3] = np.array(start_position, dtype=float)

    for traj_idx, (traj_name, motion_profile) in enumerate(zip(trajectory_names, motion_profiles)):
        # Generate the segment in the local camera frame, then compose it onto the
        # full 4x4 pose from the previous segment. This preserves orientation and
        # avoids catastrophic flow spikes at segment boundaries.
        generator = TrajectoryGenerator(
            [0.0, 0.0, 0.0],
            num_frames=frames_per_trajectory,
            motion_profile=motion_profile,
        )
        segment_trajectory = generator.generate(traj_name)
        segment_frame_keys = sorted(segment_trajectory.keys(), key=lambda x: int(x.replace('frame', '')))

        frame_offset = traj_idx * frames_per_trajectory
        for frame_key in segment_frame_keys:
            matrix_str = segment_trajectory[frame_key]
            frame_num = int(frame_key.replace('frame', ''))
            new_frame_key = f"frame{frame_offset + frame_num}"
            local_pose = parse_camera_matrix(matrix_str)
            global_pose = current_start_pose @ local_pose
            concatenated_trajectory[new_frame_key] = TrajectoryGenerator([0.0, 0.0, 0.0]).format_matrix(global_pose)

        last_local_pose = parse_camera_matrix(segment_trajectory[segment_frame_keys[-1]])
        current_start_pose = current_start_pose @ last_local_pose

    return concatenated_trajectory


def get_camera_trajectory_for_prompts(
    prompts: Union[str, List[str]],
    frames_per_trajectory: int = 81,
    force_camera_movement: Optional[str] = None,
    motion_profiles: Optional[List[Dict[str, float]]] = None,
) -> Tuple[Optional[Dict[str, str]], List[str]]:
    """
    Return the camera trajectory and detected movements for prompts.
    """
    detected_movements = detect_camera_movements_for_batch(
        prompts, force_camera_movement=force_camera_movement
    )
    if not detected_movements:
        return None, detected_movements
    segment_frames = max(2, int(math.ceil(frames_per_trajectory / len(detected_movements))))
    trajectory = concatenate_camera_trajectories(
        detected_movements,
        frames_per_trajectory=segment_frames,
        motion_profiles=motion_profiles,
    )
    return trajectory, detected_movements


def get_camera_trajectories_for_batch(
    prompts: Union[str, List[str]],
    batch_size: Optional[int] = None,
    frames_per_trajectory: int = 81,
    force_camera_movement: Optional[str] = None,
) -> Tuple[List[Optional[Dict[str, str]]], List[List[str]], List[str], List[Optional[List[Dict[str, float]]]]]:
    """
    Return one camera trajectory per batch item.

    Unlike ``get_camera_trajectory_for_prompts`` this keeps trajectories aligned
    with individual prompts instead of collapsing a whole batch to the first
    detected camera motion sequence.
    """
    expanded_prompts = expand_prompts_for_batch(prompts, batch_size=batch_size)
    trajectories = []
    detected_movements_batch = []
    motion_profiles_batch = []
    cached_prompt_data: Dict[str, Tuple[Optional[Dict[str, str]], List[str], Optional[List[Dict[str, float]]]]] = {}

    for prompt in expanded_prompts:
        if prompt in cached_prompt_data:
            trajectory, detected_movements, motion_profiles = cached_prompt_data[prompt]
        else:
            detected_movements = detect_camera_movements_for_batch(
                prompt,
                force_camera_movement=force_camera_movement,
            )
            motion_profiles = (
                sample_motion_profiles(detected_movements)
                if detected_movements
                else None
            )
            trajectory, detected_movements = get_camera_trajectory_for_prompts(
                prompt,
                frames_per_trajectory=frames_per_trajectory,
                force_camera_movement=force_camera_movement,
                motion_profiles=motion_profiles,
            )
            cached_prompt_data[prompt] = (trajectory, detected_movements, motion_profiles)
        trajectories.append(trajectory)
        detected_movements_batch.append(detected_movements)
        motion_profiles_batch.append(motion_profiles)

    return trajectories, detected_movements_batch, expanded_prompts, motion_profiles_batch


def parse_camera_matrix(matrix_str: str) -> np.ndarray:
    """
    Parse camera matrix string to 4x4 numpy array.

    Args:
        matrix_str: Camera matrix in string format

    Returns:
        4x4 numpy array
    """
    cols_str = matrix_str.strip().split('] [')
    cols = []
    for col_str in cols_str:
        col_str = col_str.replace('[', '').replace(']', '').strip()
        values = [float(x) for x in col_str.split()]
        cols.append(values)
    matrix = np.array(cols).T
    return matrix


def parse_camera_matrix_torch(
    matrix_str: str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    cols_str = matrix_str.strip().split('] [')
    cols = []
    for col_str in cols_str:
        col_str = col_str.replace('[', '').replace(']', '').strip()
        cols.append([float(x) for x in col_str.split()])
    matrix = torch.tensor(cols, device=device, dtype=dtype).transpose(0, 1).contiguous()
    return matrix


_camera_flow_grid_cache: Dict[Tuple[int, int, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_centered_meshgrid_torch(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_key = (height, width, str(device), str(dtype))
    if cache_key in _camera_flow_grid_cache:
        return _camera_flow_grid_cache[cache_key]

    cy = height / 2
    cx = width / 2
    y_coords = torch.arange(height, device=device, dtype=dtype) - cy
    x_coords = torch.arange(width, device=device, dtype=dtype) - cx
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
    _camera_flow_grid_cache[cache_key] = (x_grid, y_grid)
    return x_grid, y_grid


def camera_motion_to_flow(
    cam_prev: torch.Tensor,
    cam_curr: torch.Tensor,
    height: int,
    width: int,
    *,
    focal_length: Optional[float] = None,
    depth: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = cam_prev.device

    cam_prev = cam_prev.to(device=device, dtype=dtype)
    cam_curr = cam_curr.to(device=device, dtype=dtype)

    if focal_length is None:
        focal_length = width * 1.2

    cam_prev_inv = torch.linalg.inv(cam_prev)
    cam_curr_inv = torch.linalg.inv(cam_curr)
    relative_transform = cam_curr_inv @ cam_prev

    rotation = relative_transform[:3, :3]
    translation = relative_transform[:3, 3]

    x_coords, y_coords = _get_centered_meshgrid_torch(height, width, device=device, dtype=dtype)

    x_norm = x_coords / focal_length
    y_norm = y_coords / focal_length
    z_norm = torch.ones_like(x_norm)

    depth_value = torch.as_tensor(depth, device=device, dtype=dtype)
    x_world = x_norm * depth_value
    y_world = y_norm * depth_value
    z_world = z_norm * depth_value

    points_3d = torch.stack([x_world, y_world, z_world], dim=0)
    points_flat = points_3d.reshape(3, -1)
    transformed_flat = rotation @ points_flat + translation.view(3, 1)
    transformed_3d = transformed_flat.reshape(3, height, width)

    x_new, y_new, z_new = transformed_3d
    z_new = torch.clamp(z_new, min=0.01)

    x_proj = (x_new / z_new) * focal_length
    y_proj = (y_new / z_new) * focal_length

    dx = x_proj - x_coords
    dy = y_proj - y_coords

    flow_mag = torch.sqrt(dx * dx + dy * dy)
    if torch.isfinite(flow_mag).all():
        max_allowed = max(height, width) * 0.25
        max_observed = flow_mag.max()
        if max_observed.item() > max_allowed:
            scale = max_allowed / (max_observed + 1e-8)
            dx = dx * scale
            dy = dy * scale

    return dx, dy


# ============================================================================
# Noise Warping Utilities (Adapted from Go-with-the-Flow)
# ============================================================================

def unique_pixels(image):
    """
    Find unique pixel values in an image tensor and return their RGB values, counts, and inverse indices.

    Args:
        image (torch.Tensor): Image tensor of shape [c, h, w], where c is the number of channels (e.g., 3 for RGB),
                              h is the height, and w is the width of the image.

    Returns:
        tuple: A tuple containing three tensors:
            - unique_colors (torch.Tensor): Tensor of shape [u, c] representing the unique RGB values found in the image,
                                            where u is the number of unique colors.
            - counts (torch.Tensor): Tensor of shape [u] representing the counts of each unique color.
            - index_matrix (torch.Tensor): Tensor of shape [h, w] representing the inverse indices of each pixel,
                                           mapping each pixel to its corresponding unique color index.
    """
    c, h, w = image.shape

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Find unique RGB values, counts, and inverse indices
    unique_colors, inverse_indices, counts = torch.unique(flattened_pixels, dim=0, return_inverse=True, return_counts=True, sorted=False)

    # Get the number of unique indices
    u = unique_colors.shape[0]

    # Reshape the inverse indices back to the original image dimensions [h, w] using einops
    index_matrix = rearrange(inverse_indices, "(h w) -> h w", h=h, w=w)

    # Assert the shapes of the output tensors
    assert unique_colors.shape == (u, c)
    assert counts.shape == (u,)
    assert index_matrix.shape == (h, w)
    assert index_matrix.min() == 0
    assert index_matrix.max() == u - 1

    return unique_colors, counts, index_matrix


def sum_indexed_values(image, index_matrix):
    """
    Sum the values in the CHW image tensor based on the indices specified in the HW index matrix.

    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W], where C is the number of channels,
                              H is the height, and W is the width of the image.
        index_matrix (torch.Tensor): Index matrix tensor of shape [H, W] containing indices
                                     specifying the mapping of each pixel to its corresponding
                                     unique value.
                                     Indices range [0, U), where U is the number of unique indices

    Returns:
        torch.Tensor: Tensor of shape [U, C] representing the sum of values in the image tensor
                      based on the indices in the index matrix, where U is the number of unique
                      indices in the index matrix.
    """
    c, h, w = image.shape
    u = index_matrix.max() + 1

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Create an output tensor of shape [u, c] initialized with zeros
    output = torch.zeros((u, c), dtype=flattened_pixels.dtype, device=flattened_pixels.device)

    # Scatter sum the flattened pixel values using the index matrix
    output.index_add_(0, index_matrix.view(-1), flattened_pixels)

    # Assert the shapes of the input and output tensors
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"
    assert index_matrix.shape == (h, w), f"Expected index_matrix shape: ({h}, {w}), but got: {index_matrix.shape}"
    assert output.shape == (u, c), f"Expected output shape: ({u}, {c}), but got: {output.shape}"

    return output


def indexed_to_image(index_matrix, unique_colors):
    """
    Create a CHW image tensor from an HW index matrix and a UC unique_colors matrix.

    Args:
        index_matrix (torch.Tensor): Index matrix tensor of shape [H, W] containing indices
                                     specifying the mapping of each pixel to its corresponding
                                     unique color.
        unique_colors (torch.Tensor): Unique colors matrix tensor of shape [U, C] containing
                                      the unique color values, where U is the number of unique
                                      colors and C is the number of channels.

    Returns:
        torch.Tensor: Image tensor of shape [C, H, W] representing the reconstructed image
                      based on the index matrix and unique colors matrix.
    """
    h, w = index_matrix.shape
    u, c = unique_colors.shape

    # Assert the shapes of the input tensors
    assert index_matrix.max() < u, f"Index matrix contains indices ({index_matrix.max()}) greater than the number of unique colors ({u})"

    # Gather the colors based on the index matrix
    flattened_image = unique_colors[index_matrix.view(-1)]

    # Reshape the flattened image to [h, w, c]
    image = rearrange(flattened_image, "(h w) c -> h w c", h=h, w=w)

    # Rearrange the image tensor from [h, w, c] to [c, h, w] using einops
    image = rearrange(image, "h w c -> c h w")

    # Assert the shape of the output tensor
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"

    return image


def regaussianize(noise):
    """
    Regaussianize warped noise to maintain Gaussian distribution properties.

    This is critical for noise warping - after warping, some pixels may be duplicated
    or stretched, breaking the Gaussian distribution. This function fixes that by:
    1. Finding groups of identical pixels (from warping)
    2. Generating fresh random noise
    3. Averaging the fresh noise within each group
    4. Adjusting variance based on group sizes
    5. Adding zero-mean noise to maintain randomness

    Args:
        noise: Tensor of shape [C, H, W]

    Returns:
        output: Regaussianized noise of shape [C, H, W]
        counts_image: Weight/count image of shape [1, H, W]
    """
    c, hs, ws = noise.shape

    # Find unique pixel values, their indices, and counts in the pixelated noise image
    unique_colors, counts, index_matrix = unique_pixels(noise[:1])
    u = len(unique_colors)
    assert unique_colors.shape == (u, 1)
    assert counts.shape == (u,)
    assert index_matrix.max() == u - 1
    assert index_matrix.min() == 0
    assert index_matrix.shape == (hs, ws)

    foreign_noise = torch.randn_like(noise)
    assert foreign_noise.shape == noise.shape == (c, hs, ws)

    summed_foreign_noise_colors = sum_indexed_values(foreign_noise, index_matrix)
    assert summed_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise_colors = summed_foreign_noise_colors / rearrange(counts, "u -> u 1")
    assert meaned_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise = indexed_to_image(index_matrix, meaned_foreign_noise_colors)
    assert meaned_foreign_noise.shape == (c, hs, ws)

    zeroed_foreign_noise = foreign_noise - meaned_foreign_noise
    assert zeroed_foreign_noise.shape == (c, hs, ws)

    counts_as_colors = rearrange(counts, "u -> u 1")
    counts_image = indexed_to_image(index_matrix, counts_as_colors)
    assert counts_image.shape == (1, hs, ws)

    # To upsample noise, we must first divide by the area then add zero-sum-noise
    output = noise
    output = output / counts_image ** .5
    output = output + zeroed_foreign_noise

    assert output.shape == noise.shape == (c, hs, ws)

    return output, counts_image


# Use rp package functions directly
torch_remap_image = rp.torch_remap_image
torch_scatter_add_image = rp.torch_scatter_add_image
torch_resize_image = rp.torch_resize_image


def resize_noise(noise: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Variance-preserving noise resize (shrink-only), matching ref_code/noise_wrap.py:resize_noise.

    Why: using bilinear/area interpolation for noise introduces cross-correlation and changes statistics.
    This uses scatter-add binning + sqrt(count) normalization to keep std≈1 without adding correlation.

    Args:
        noise: CHW tensor
        size: (new_height, new_width), must be <= (old_height, old_width)
    """
    assert isinstance(size, tuple) and len(size) == 2
    assert noise.ndim == 3, "resize_noise expects CHW noise"
    c, old_h, old_w = noise.shape
    new_h, new_w = int(size[0]), int(size[1])
    if (new_h, new_w) == (old_h, old_w):
        return noise.clone()
    assert new_h <= old_h, "resize_noise is shrink-only (new_h must be <= old_h)"
    assert new_w <= old_w, "resize_noise is shrink-only (new_w must be <= old_w)"

    # Cache coordinate grids; they are constant for a given (in_h,in_w,out_h,out_w,device).
    # This is important because resize_noise is called once per frame.
    global _resize_noise_grid_cache
    try:
        _resize_noise_grid_cache
    except NameError:
        _resize_noise_grid_cache = {}

    cache_key = (old_h, old_w, new_h, new_w, str(noise.device))
    if cache_key in _resize_noise_grid_cache:
        x, y = _resize_noise_grid_cache[cache_key]
    else:
        # Use float32 coordinates to avoid precision issues in indexing/scatter.
        import torch as _torch

        x, y = rp.xy_torch_matrices(
            old_h,
            old_w,
            max_x=new_w,
            max_y=new_h,
            device=noise.device,
            dtype=_torch.float32,
        )
        _resize_noise_grid_cache[cache_key] = (x, y)
    resized = rp.torch_scatter_add_image(
        noise,
        x,
        y,
        height=new_h,
        width=new_w,
        interp="floor",
        prepend_ones=True,
    )
    total, resized = resized[:1], resized[1:]
    total = total.clamp_min(1.0)
    adjusted = resized / total.sqrt()
    assert adjusted.shape == (c, new_h, new_w)
    return adjusted


def xy_meshgrid_like_image(image):
    """
    Create a meshgrid of x,y coordinates matching image dimensions.

    Args:
        image: [C, H, W] tensor

    Returns:
        [2, H, W] tensor with x,y coordinates
    """
    assert image.ndim == 3, "image is in CHW form"
    c, h, w = image.shape
    device, dtype = image.device, image.dtype

    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing='ij'
    )
    return torch.stack([x, y], dim=0)


def noise_to_xyωc(noise):
    """
    Convert noise to state format [dx, dy, weight, *noise_channels].

    Args:
        noise: [C, H, W] noise tensor

    Returns:
        [3+C, H, W] state tensor with [dx=0, dy=0, weight=1, *noise]
    """
    assert noise.ndim == 3, "noise is in CHW form"
    zeros = torch.zeros_like(noise[0][None])
    ones = torch.ones_like(noise[0][None])

    # Prepend [dx=0, dy=0, weights=1] channels
    output = torch.concat([zeros, zeros, ones, noise])
    return output


def xyωc_to_noise(xyωc):
    """
    Extract noise from state format.

    Args:
        xyωc: [3+C, H, W] state tensor

    Returns:
        [C, H, W] noise tensor
    """
    assert xyωc.ndim == 3, "xyωc is in [ω x y c]·h·w form"
    assert xyωc.shape[0] > 3, 'xyωc should have at least one noise channel'
    noise = xyωc[3:]
    return noise


def warp_xyωc(I, F, xy_mode="none", expand_only=False):
    """
    Warp noise state using optical flow.

    This is the core algorithm from Go-with-the-Flow that maintains Gaussian
    properties during warping by:
    1. Separating expansion (new regions) and shrinkage (compressed regions)
    2. Regaussianizing to maintain noise distribution
    3. Weighted averaging for overlapping regions

    Args:
        I: Input state [ω x y c]·h·w where ω=weights, x,y are offsets, c is noise
        F: Flow field [x y]·h·w
        xy_mode: 'none' or 'float' for position tracking
        expand_only: If True, only do expansion (for ablation)

    Returns:
        Warped state with same shape as I
    """
    # Input assertions
    assert F.device == I.device
    assert F.ndim == 3, str(F.shape) + ' F stands for flow, and its in [x y]·h·w form'
    assert I.ndim == 3, str(I.shape) + ' I stands for input, in [ω x y c]·h·w form where ω=weights, x and y are offsets, and c is num noise channels'
    xyωc, h, w = I.shape
    assert F.shape == (2, h, w)  # Should be [x y]·h·w
    device = I.device

    # How I'm going to address the different channels:
    x = 0        # index of Δx channel
    y = 1        # index of Δy channel
    xy = 2       # I[:xy]
    xyω = 3      # I[:xyω]
    ω = 2        # I[ω]     // index of weight channel
    c = xyωc - xyω  # I[-c:]   // num noise channels
    ωc = xyωc - xy  # I[-ωc:]
    w_dim = 2
    assert c, 'I has no noise channels. There is nothing to warp.'
    assert (I[ω] > 0).all(), 'All weights should be greater than 0'

    # Compute the grid of xy indices
    grid = xy_meshgrid_like_image(I)
    assert grid.shape == (2, h, w)  # Shape is [x y]·h·w

    # The default values we initialize to
    init = torch.empty_like(I)
    init[:xy] = 0
    init[ω] = 1
    init[-c:] = 0

    # Calculate initial pre-expand
    pre_expand = torch.empty_like(I)

    # ABLATION STUFF
    interp = 'nearest' if not isinstance(expand_only, str) else expand_only
    regauss = not isinstance(expand_only, str)
    F_index = F
    if interp == 'nearest':
        F_index = F_index.round()

    pre_expand[:xy] = torch_remap_image(I[:xy], *-F, relative=True, interp=interp)
    pre_expand[-ωc:] = torch_remap_image(I[-ωc:], *-F, relative=True, interp=interp)
    pre_expand[ω][pre_expand[ω] == 0] = 1  # Give new noise regions a weight of 1

    if expand_only:
        if regauss:
            # This is an ablation option - simple warp + regaussianize
            pre_expand[-c:] = regaussianize(pre_expand[-c:])[0]
        else:
            # Turn zeroes to noise
            pre_expand[-c:] = torch.randn_like(pre_expand[-c:]) * (pre_expand[-c:] == 0) + pre_expand[-c:]
        return pre_expand

    # Calculate initial pre-shrink
    pre_shrink = I.clone()
    pre_shrink[:xy] += F

    # Pre-Shrink mask - discard out-of-bounds pixels
    pos = (grid + pre_shrink[:xy]).round()
    in_bounds = (0 <= pos[x]) & (pos[x] < w) & (0 <= pos[y]) & (pos[y] < h)
    in_bounds = in_bounds[None]  # Match the shape of the input
    out_of_bounds = ~in_bounds
    assert out_of_bounds.dtype == torch.bool
    assert out_of_bounds.shape == (1, h, w)
    assert pre_shrink.shape == init.shape
    pre_shrink = torch.where(out_of_bounds, init, pre_shrink)

    # Deal with shrink positions offsets
    scat_xy = pre_shrink[:xy].round()
    pre_shrink[:xy] -= scat_xy

    # FLOATING POINT POSITIONS
    assert xy_mode in ['float', 'none'] or isinstance(xy_mode, int)
    if xy_mode == 'none':
        pre_shrink[:xy] = 0

    if isinstance(xy_mode, int):
        # XY quantization
        quant = xy_mode
        pre_shrink[:xy] = (pre_shrink[:xy] * quant).round() / quant

    scat = lambda tensor: torch_scatter_add_image(tensor, *scat_xy, relative=True)

    # Where mask==True, we output shrink. Where mask==0, we output expand.
    shrink_mask = torch.ones(1, h, w, dtype=bool, device=device)
    shrink_mask = scat(shrink_mask)
    assert shrink_mask.dtype == torch.bool

    # Remove the expansion points where we'll use shrink
    pre_expand = torch.where(shrink_mask, init, pre_expand)

    # Horizontally Concat
    concat_dim = w_dim
    concat = torch.concat([pre_shrink, pre_expand], dim=concat_dim)

    # Regaussianize
    concat[-c:], counts_image = regaussianize(concat[-c:])
    assert counts_image.shape == (1, h, 2 * w)

    # Distribute Weights
    concat[ω] /= counts_image[0]
    concat[ω] = concat[ω].nan_to_num()

    pre_shrink, expand = torch.chunk(concat, chunks=2, dim=concat_dim)
    assert pre_shrink.shape == expand.shape == (3 + c, h, w)

    shrink = torch.empty_like(pre_shrink)
    shrink[ω] = scat(pre_shrink[ω][None])[0]
    shrink[:xy] = scat(pre_shrink[:xy] * pre_shrink[ω][None]) / shrink[ω][None]
    shrink[-c:] = scat(pre_shrink[-c:] * pre_shrink[ω][None]) / scat(pre_shrink[ω][None] ** 2).sqrt()

    output = torch.where(shrink_mask, shrink, expand)
    output[ω] = output[ω] / output[ω].mean()  # Don't let them get too big or too small
    ε = .00001
    output[ω] += ε  # Don't let it go too low

    assert (output[ω] > 0).all()

    output[ω] **= .9999  # Make it tend towards 1

    return output


def blend_noise(noise_background, noise_foreground, alpha):
    """Variance-preserving blend"""
    return (noise_foreground * alpha + noise_background * (1 - alpha)) / (alpha ** 2 + (1 - alpha) ** 2) ** .5


def lowpass_latent_delta(delta: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Low-pass filter latent deltas in space, preserving BCTHW layout."""
    if kernel_size <= 1:
        return delta
    if delta.ndim != 5:
        raise ValueError(f"Expected latent delta with shape [B, C, T, H, W], got {delta.shape}")
    batch, channels, frames, height, width = delta.shape
    x = delta.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    padding = kernel_size // 2
    x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    return x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


def per_sample_mean_std(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reduce_dims = tuple(range(1, tensor.ndim))
    mean = tensor.mean(dim=reduce_dims, keepdim=True)
    std = tensor.std(dim=reduce_dims, keepdim=True)
    return mean, std


def build_stepwise_delta_callback(
    delta_low: torch.Tensor,
    wrap_strength: float,
    guidance_steps: int,
):
    if guidance_steps <= 0 or wrap_strength <= 0:
        return None

    weights = torch.linspace(1.0, 0.1, guidance_steps, device=delta_low.device, dtype=delta_low.dtype)
    weights = weights / weights.sum()
    _, delta_std = per_sample_mean_std(delta_low)
    delta_unit = delta_low / (delta_std + 1e-6)

    def _callback(pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        if step >= guidance_steps:
            return callback_kwargs

        original_mean, original_std = per_sample_mean_std(latents)
        guided_latents = latents + wrap_strength * weights[step] * original_std * delta_unit
        guided_mean, guided_std = per_sample_mean_std(guided_latents)
        guided_latents = (guided_latents - guided_mean) / (guided_std + 1e-6)
        guided_latents = guided_latents * original_std + original_mean
        callback_kwargs["latents"] = guided_latents
        return callback_kwargs

    return _callback


def apply_wrap_strength_to_latents(
    base_latents: torch.Tensor,
    wrapped_latents: torch.Tensor,
    wrap_strength: float,
    injection_mode: str = "lowpass_delta",
    delta_lowpass_kernel: int = 9,
) -> torch.Tensor:
    """Inject camera-warped structure into random latents using the same semantics as ablation."""
    if wrap_strength <= 0:
        return base_latents

    wrapped_latents = wrapped_latents.float()
    base_latents = base_latents.float()

    if injection_mode == "blend":
        if wrap_strength >= 1:
            return wrapped_latents
        return blend_noise(base_latents, wrapped_latents, float(wrap_strength))

    if injection_mode == "lowpass_delta":
        delta = wrapped_latents - base_latents
        delta_low = lowpass_latent_delta(delta, delta_lowpass_kernel)
        mixed = base_latents + float(wrap_strength) * delta_low
        return (mixed - mixed.mean()) / (mixed.std() + 1e-6)

    raise ValueError(
        f"Unsupported wrap injection mode {injection_mode!r}; expected 'blend' or 'lowpass_delta'."
    )


def mix_new_noise(noise, alpha):
    """As alpha --> 1, noise is destroyed"""
    if isinstance(noise, torch.Tensor):
        return blend_noise(noise, torch.randn_like(noise), alpha)
    elif isinstance(noise, np.ndarray):
        return blend_noise(noise, np.random.randn(*noise.shape), alpha)
    else:
        raise TypeError(f"Unsupported input type: {type(noise)}. Expected PyTorch Tensor or NumPy array.")


class NoiseWarper:
    """
    Complete noise warping implementation from Go-with-the-Flow.

    This warper maintains Gaussian distribution properties during camera movement
    by properly handling pixel expansion, shrinkage, and regaussianization.
    """

    def __init__(
        self,
        c, h, w,
        device,
        dtype=torch.float32,
        scale_factor=1,
        post_noise_alpha=0,
        progressive_noise_alpha=0,
        warp_kwargs=dict(),
    ):
        # Some non-exhaustive input assertions
        assert isinstance(c, int) and c > 0
        assert isinstance(h, int) and h > 0
        assert isinstance(w, int) and w > 0
        assert isinstance(scale_factor, int) and scale_factor >= 1

        # Record arguments
        self.c = c
        self.h = h
        self.w = w
        self.device = device
        self.dtype = dtype
        self.scale_factor = scale_factor
        self.progressive_noise_alpha = progressive_noise_alpha
        self.post_noise_alpha = post_noise_alpha
        self.warp_kwargs = warp_kwargs

        # Initialize the state
        self._state = self._noise_to_state(
            noise=torch.randn(
                c,
                h * scale_factor,
                w * scale_factor,
                dtype=dtype,
                device=device,
            )
        )

    @property
    def noise(self):
        """Get the current noise, properly weighted and downsampled"""
        noise = self._state_to_noise(self._state)
        weights = self._state[2][None]  # xyωc
        noise = (
            torch_resize_image(noise * weights, (self.h, self.w), interp="area")
            / torch_resize_image(weights ** 2, (self.h, self.w), interp="area").sqrt()
        )
        noise = noise * self.scale_factor

        if self.post_noise_alpha:
            noise = mix_new_noise(noise, self.post_noise_alpha)

        return noise

    def __call__(self, dx, dy):
        """Apply optical flow to warp the noise"""
        if isinstance(dx, np.ndarray):
            dx = torch.tensor(dx).to(self.device, self.dtype)
        if isinstance(dy, np.ndarray):
            dy = torch.tensor(dy).to(self.device, self.dtype)

        flow = torch.stack([dx, dy]).to(self.device, self.dtype)
        _, oflowh, ofloww = flow.shape  # Original height and width of the flow

        assert flow.ndim == 3 and flow.shape[0] == 2, "Flow is in [x y]·h·w form"
        flow = torch_resize_image(
            flow,
            (
                self.h * self.scale_factor,
                self.w * self.scale_factor,
            ),
        )

        _, flowh, floww = flow.shape

        # Multiply the flow values by the size change
        flow[0] *= flowh / oflowh * self.scale_factor
        flow[1] *= floww / ofloww * self.scale_factor

        self._state = self._warp_state(self._state, flow)
        return self

    # The following three methods can be overridden in subclasses:

    @staticmethod
    def _noise_to_state(noise):
        return noise_to_xyωc(noise)

    @staticmethod
    def _state_to_noise(state):
        return xyωc_to_noise(state)

    def _warp_state(self, state, flow):
        if self.progressive_noise_alpha:
            state[3:] = mix_new_noise(state[3:], self.progressive_noise_alpha)

        return warp_xyωc(state, flow, **self.warp_kwargs)


# ============================================================================
# End of Noise Warping Utilities
# ============================================================================


def generate_camera_warped_latents(
    trajectory: Dict[str, str],
    batch_size: int = 1,
    num_channels_latents: int = 16,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    spatial_compression: int = 8,
    temporal_compression: int = 4,
    noise_downtemp_interp: str = "nearest",
    noise_downspatial_mode: str = "area",
    noise_degradation: float = 0.3,
    flow_scale: int = 16,
    focal_length: Optional[float] = None,
    scene_depth: float = 1000.0,
    debug_precompress_vis_dir: Optional[str] = None,
    debug_precompress_vis_fps: int = 12,
    debug_precompress_vis_max_frames: int = 82,
    debug_precompress_vis_upsample_size: Optional[Tuple[int, int]] = None,
    debug_precompress_vis_upsample_interp: str = "nearest",
    debug_precompress_vis_scale: float = 5.0,  # Match reference visualization: /5+.5
) -> torch.Tensor:
    """
    Generate camera-warped latents from a camera trajectory.

    Args:
        trajectory: Dictionary mapping frame keys to camera matrix strings
        batch_size: Batch size
        num_channels_latents: Number of latent channels (16 for CogVideoX)
        height: Video height
        width: Video width
        num_frames: Number of video frames
        dtype: Tensor dtype
        device: Computation device
        spatial_compression: Spatial compression ratio (8 for CogVideoX)
        temporal_compression: Temporal compression ratio (4 for CogVideoX)
        noise_downtemp_interp: How to map output-frame noise to latent-time noise when temporal_compression > 1.
            - "nearest": take frames [0, 4, 8, ...] (matches ref_code/cut_and_drag_inference.py default)
            - "blend": mean within each temporal-compression block (can visually wash out motion cues)
        noise_downspatial_mode: How to downsample warped noise from warping resolution to latent resolution.
            - "area": use torch area resize and multiply by flow_scale to restore std (fast, but can add correlation)
            - "resize_noise": use ref variance-preserving scatter-add binning (recommended for stability)
        noise_degradation: Amount of fresh noise to mix in after temporal downsampling (0..1).
            Higher values reduce over-correlation but can weaken the camera-motion effect.
        flow_scale: High-resolution noise warping scale factor
        focal_length: Camera focal length
        scene_depth: Assumed scene depth

    Returns:
        Latent tensor of shape [B, C, T, H, W]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dtype is None:
        dtype = torch.float32

    # Calculate latent space dimensions
    latent_height = height // spatial_compression
    latent_width = width // spatial_compression

    # Calculate warping resolution (higher resolution for better quality)
    warp_height = latent_height * flow_scale
    warp_width = latent_width * flow_scale

    # Parse camera matrices (sample in *output-frame* space first)
    camera_matrices = []
    frame_keys = sorted(trajectory.keys(), key=lambda x: int(x.replace('frame', '')))


    # 均匀采样 trajectory，使camera_matrices长度为num_frames
    total_frames = len(frame_keys)
    if num_frames >= total_frames:
        selected_indices = list(range(total_frames))
    else:
        # 均匀采样num_frames个索引
        selected_indices = [
            int(round(i * (total_frames - 1) / (num_frames - 1))) if num_frames > 1 else 0
            for i in range(num_frames)
        ]
    for idx in selected_indices:
        frame_key = frame_keys[idx]
        matrix_str = trajectory[frame_key]
        matrix = parse_camera_matrix_torch(matrix_str, device=device, dtype=torch.float32)
        camera_matrices.append(matrix)

    if len(camera_matrices) < num_frames:
        print(f"Warning: Only {len(camera_matrices)} frames in trajectory, need {num_frames}")
        # Extend with last frame
        while len(camera_matrices) < num_frames:
            camera_matrices.append(camera_matrices[-1].clone())

    # If the VAE uses temporal compression (e.g. 4), the denoiser/VAE operate on *latent* time:
    #   num_latent_frames = (num_frames - 1) // temporal_compression + 1
    #
    # Important: the noise warper is non-linear (regaussianization, weighting, etc.), so "warp using a single
    # big flow between frame 0 and frame 4" is not equivalent to "warp 4 times with small per-frame flows".
    # Doing only big jumps can collapse temporal variation after downsampling/normalization and look static.
    #
    # So we always warp *sequentially at output-frame rate*, but only *record* latent frames at the temporal
    # boundaries: frame 0, then every `temporal_compression` frames (4, 8, ...). The recording strategy is
    # controlled by `noise_downtemp_interp` to match the reference behavior.
    if temporal_compression > 1 and num_frames > 1:
        if noise_downtemp_interp not in {"nearest", "blend"}:
            raise ValueError(
                f"Unsupported noise_downtemp_interp={noise_downtemp_interp!r}; expected 'nearest' or 'blend'."
            )
        remainder = (num_frames - 1) % temporal_compression
        if remainder != 0:
            pad_frames = temporal_compression - remainder
            for _ in range(pad_frames):
                camera_matrices.append(camera_matrices[-1].copy())
            num_frames += pad_frames

        block_size = temporal_compression
        expected_latent_frames = (num_frames - 1) // block_size + 1
    else:
        block_size = 1
        expected_latent_frames = num_frames

    # Initialize noise warper with complete Go-with-the-Flow implementation
    warper = NoiseWarper(
        c=num_channels_latents,
        h=warp_height,
        w=warp_width,
        device=device,
        dtype=dtype,
        scale_factor=1,  # Use scale_factor=1 as in reference implementation
    )

    # Generate warped latents for all frames
    all_latents = []
    precompress_latents = [] if debug_precompress_vis_dir is not None else None

    # Reference rp.resize_list uses: indices = [round(i * step) for i in range(length)]
    # with step = (len-1)/(length-1). When `num_frames` is padded to match the temporal compression,
    # this becomes exactly indices [0, block_size, 2*block_size, ...].
    capture_indices = None
    if block_size > 1 and noise_downtemp_interp == "nearest":
        if expected_latent_frames > 1 and num_frames > 1:
            step = (num_frames - 1) / (expected_latent_frames - 1)
        else:
            step = 0.0
        capture_indices = {round(i * step) for i in range(expected_latent_frames)}

    # First frame: initial noise, downsampled to latent resolution
    # Note: warper.noise is already [C, H, W] after proper weighting
    first_noise = warper.noise
    if noise_downspatial_mode == "resize_noise":
        first_latent = resize_noise(first_noise, (latent_height, latent_width))
    elif noise_downspatial_mode == "area":
        first_latent = torch_resize_image(
            first_noise,
            (latent_height, latent_width),
            interp="area",
        ) * flow_scale
    else:
        raise ValueError(
            f"Unsupported noise_downspatial_mode={noise_downspatial_mode!r}; expected 'area' or 'resize_noise'."
        )
    all_latents.append(first_latent)
    if precompress_latents is not None:
        precompress_latents.append(first_latent)

    # Accumulate warped latents inside each temporal-compression block (only used for "blend")
    block_latents = []

    # Process subsequent frames
    for i in range(1, num_frames):
        cam_prev = camera_matrices[i - 1]
        cam_curr = camera_matrices[i]

        # Calculate optical flow at high resolution
        dx, dy = camera_motion_to_flow(
            cam_prev, cam_curr,
            warp_height, warp_width,
            focal_length=focal_length,
            depth=scene_depth,
            device=device,
            dtype=torch.float32,
        )

        # Warp noise using complete Go-with-the-Flow algorithm
        # This maintains Gaussian properties through regaussianization
        warper(dx, dy)
        noise = warper.noise

        # Downsample to latent resolution
        if noise_downspatial_mode == "resize_noise":
            latent = resize_noise(noise, (latent_height, latent_width))
        elif noise_downspatial_mode == "area":
            latent = torch_resize_image(
                noise,
                (latent_height, latent_width),
                interp="area",
            ) * flow_scale
        else:
            raise ValueError(
                f"Unsupported noise_downspatial_mode={noise_downspatial_mode!r}; expected 'area' or 'resize_noise'."
            )
        if precompress_latents is not None:
            precompress_latents.append(latent)

        if block_size == 1:
            all_latents.append(latent)
        elif noise_downtemp_interp == "nearest":
            # Reference default: take frame 0, then every `temporal_compression` frames.
            # This preserves motion cues much better than averaging shifted noise.
            if capture_indices is None:
                raise RuntimeError("Internal error: capture_indices was not computed for nearest mode.")
            if i in capture_indices:
                all_latents.append(latent)
        else:  # "blend"
            block_latents.append(latent)
            # At the boundary of a compression window (or final frame), downsample with mean + sqrt(scale)
            if len(block_latents) == block_size or i == num_frames - 1:
                block = torch.stack(block_latents, dim=0)
                block = block.mean(dim=0) * (len(block_latents) ** 0.5)
                all_latents.append(block)
                block_latents = []

    if len(all_latents) != expected_latent_frames:
        raise ValueError(
            f"Temporal sampling bug: expected {expected_latent_frames} latent frames, got {len(all_latents)} "
            f"(num_frames={num_frames}, temporal_compression={temporal_compression})."
        )

    # Optional: save pre-temporal-compression visualization (matches reference workflow more closely).
    # This dumps the per-output-frame latent noise before any temporal downsampling/averaging.
    if debug_precompress_vis_dir is not None:
        try:
            from flow_grpo.diffusers_patch.noise_visualizer import visualize_latents_as_video

            os.makedirs(debug_precompress_vis_dir, exist_ok=True)

            if precompress_latents is None:
                raise RuntimeError("Internal error: precompress_latents was not collected.")
            pre = torch.stack(precompress_latents, dim=0)  # [F, C, H, W]
            pre = pre.permute(1, 0, 2, 3).contiguous()  # [C, F, H, W]

            # Time subsample for visualization
            if pre.shape[1] > debug_precompress_vis_max_frames:
                idx = torch.linspace(0, pre.shape[1] - 1, debug_precompress_vis_max_frames, device=pre.device).long()
                pre_vis = pre.index_select(1, idx)
            else:
                pre_vis = pre

            pre_path = os.path.join(debug_precompress_vis_dir, "precompress_latents.mp4")
            visualize_latents_as_video(
                pre_vis,
                pre_path,
                fps=debug_precompress_vis_fps,
                upsample_size=debug_precompress_vis_upsample_size,
                upsample_interp=debug_precompress_vis_upsample_interp,
                scale=debug_precompress_vis_scale,
            )

            np.save(
                os.path.join(debug_precompress_vis_dir, "precompress_latents.npy"),
                pre_vis.detach().float().cpu().numpy().astype(np.float16),
            )

            if pre_vis.shape[1] > 1:
                delta = pre_vis[:, 1:] - pre_vis[:, :-1]
                delta_path = os.path.join(debug_precompress_vis_dir, "precompress_latents_delta.mp4")
                visualize_latents_as_video(
                    delta,
                    delta_path,
                    fps=debug_precompress_vis_fps,
                    upsample_size=debug_precompress_vis_upsample_size,
                    upsample_interp=debug_precompress_vis_upsample_interp,
                    scale=debug_precompress_vis_scale,
                )
                np.save(
                    os.path.join(debug_precompress_vis_dir, "precompress_latents_delta.npy"),
                    delta.detach().float().cpu().numpy().astype(np.float16),
                )
        except Exception as e:
            print(f"Warning: failed to save precompress visualization: {e}")

    # Stack all frames: [T, C, H, W]
    latents_tensor = torch.stack(all_latents, dim=0)

    # Rearrange to [C, T, H, W]
    latents_tensor = latents_tensor.permute(1, 0, 2, 3)

    # Add batch dimension: [C, T, H, W] -> [B, C, T, H, W]
    latents_tensor = latents_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

    # Reduce over-correlation by mixing in fresh Gaussian noise (variance-preserving blend).
    # Mirrors ref_code/cut_and_drag_inference.py:213 (degradation).
    if noise_degradation is not None and noise_degradation > 0:
        if not (0.0 <= float(noise_degradation) <= 1.0):
            raise ValueError(f"noise_degradation must be in [0, 1], got {noise_degradation}.")
        latents_tensor = mix_new_noise(latents_tensor, float(noise_degradation))

    # Convert to desired dtype
    if dtype is not None:
        latents_tensor = latents_tensor.to(dtype)

    return latents_tensor


def prepare_latents_with_camera(
    prompt: Union[str, List[str]],
    batch_size: int,
    num_channels_latents: int = 16,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    vae_scale_factor_temporal: int = 4,
    frames_per_trajectory: int = 81,
    force_camera_movement: Optional[str] = None,
    remove_camera_keywords_from_prompt: bool = False,
    noise_wrap_compute_dtype: str = "fp32",
    noise_downtemp_interp: str = "nearest",
    noise_downspatial_mode: str = "area",
    noise_degradation: float = 0.35,
    noise_wrap_flow_scale: int = 16,
    wrap_strength: Optional[float] = None,
    wrap_injection_mode: str = "lowpass_delta",
    delta_lowpass_kernel: int = 9,
    debug_precompress_vis_dir: Optional[str] = None,
    debug_precompress_vis_fps: int = 12,
    debug_precompress_vis_max_frames: int = 81,
    debug_precompress_vis_upsample_size: Optional[Tuple[int, int]] = None,
    debug_precompress_vis_upsample_interp: str = "nearest",
    debug_precompress_vis_scale: float = 5.0,  # Match reference visualization: /5+.5
    camera_trajectories: Optional[Union[Dict[str, str], List[Optional[Dict[str, str]]]]] = None,
    detected_movements_batch: Optional[List[List[str]]] = None,
    return_base_latents: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Prepare latents with optional camera trajectory warping.

    If camera movements are detected in prompts, generates camera-warped latents.
    Otherwise, generates random latents.

    Args:
        prompt: Prompt or list of prompts
        batch_size: Batch size
        num_channels_latents: Number of latent channels
        height: Video height
        width: Video width
        num_frames: Number of video frames
        dtype: Tensor dtype
        device: Computation device
        generator: Random generator (used only if no camera movements detected)
        latents: Pre-generated latents (if provided, returns as-is)
        vae_scale_factor_temporal: Temporal scale factor for VAE
        frames_per_trajectory: Frames per trajectory segment
        force_camera_movement: If specified, force this camera movement instead of detecting from prompt
                              (e.g., "push_in", "pan_left", "orbit_left", etc.)
        remove_camera_keywords_from_prompt: If True, removes camera movement keywords from the original prompt
                                           (used to test if noise wrapping is the cause of video motion)

    Returns:
        Latent tensor of shape [B, C, T, H, W], or a tuple of
        `(latents, base_latents)` when `return_base_latents=True`.
    """
    from diffusers.utils.torch_utils import randn_tensor
    # Noise-wrapping relies on exact-value grouping (e.g. torch.unique in regaussianize).
    # Running it in bf16 can introduce many accidental value collisions and break the algorithm.
    requested_dtype = dtype
    if noise_wrap_compute_dtype not in {"fp32", "bf16"}:
        raise ValueError(
            f"noise_wrap_compute_dtype must be 'fp32' or 'bf16', got {noise_wrap_compute_dtype!r}."
        )
    warp_dtype = torch.float32 if noise_wrap_compute_dtype == "fp32" else torch.bfloat16

    # Match Wan/CogVideoX temporal convention: `num_frames - 1` must be divisible by `vae_scale_factor_temporal`.
    # The pipeline will do a similar rounding, so we mirror it here to keep the provided `latents` aligned.
    if num_frames % vae_scale_factor_temporal != 1:
        num_frames = num_frames // vae_scale_factor_temporal * vae_scale_factor_temporal + 1

    # If latents are already provided, return them
    if latents is not None:
        if return_base_latents:
            raise ValueError("`return_base_latents=True` is not supported when explicit `latents` are provided.")
        return latents.to(device=device, dtype=dtype)

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    expanded_prompts = expand_prompts_for_batch(prompt, batch_size=batch_size)
    if camera_trajectories is None:
        trajectories, detected_movements_batch, expanded_prompts, _ = get_camera_trajectories_for_batch(
            prompt,
            batch_size=batch_size,
            frames_per_trajectory=frames_per_trajectory,
            force_camera_movement=force_camera_movement,
        )
    else:
        if isinstance(camera_trajectories, list):
            trajectories = list(camera_trajectories)
        else:
            trajectories = [camera_trajectories] * batch_size

        if len(trajectories) != batch_size:
            raise ValueError(
                f"camera_trajectories length {len(trajectories)} must match batch_size {batch_size}"
            )
        if detected_movements_batch is None:
            detected_movements_batch = [
                detect_camera_movements(prompt_item) if trajectory is not None else []
                for prompt_item, trajectory in zip(expanded_prompts, trajectories)
            ]

    num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
    random_shape = (
        1,
        num_channels_latents,
        num_latent_frames,
        int(height) // 8,
        int(width) // 8,
    )

    batch_latents = []
    batch_base_latents = [] if return_base_latents else None
    for idx, (prompt_item, trajectory, detected_movements) in enumerate(
        zip(expanded_prompts, trajectories, detected_movements_batch)
    ):
        item_generator = generator[idx] if isinstance(generator, list) else generator

        base_item_latents = randn_tensor(
            random_shape,
            generator=item_generator,
            device=device,
            dtype=torch.float32,
        )

        if not detected_movements:
            print(f"No camera movements detected for batch item {idx}, generating random latents")
            item_latents = base_item_latents
        else:
            print(f"Detected camera movements for batch item {idx}: {detected_movements} | prompt={prompt_item}")
            total_frames = len(detected_movements) * frames_per_trajectory
            item_latents = generate_camera_warped_latents(
                trajectory=trajectory,
                batch_size=1,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=min(num_frames, total_frames),
                temporal_compression=vae_scale_factor_temporal,
                noise_downtemp_interp=noise_downtemp_interp,
                noise_downspatial_mode=noise_downspatial_mode,
                noise_degradation=noise_degradation,
                flow_scale=noise_wrap_flow_scale,
                dtype=warp_dtype,
                device=device,
                debug_precompress_vis_dir=debug_precompress_vis_dir,
                debug_precompress_vis_fps=debug_precompress_vis_fps,
                debug_precompress_vis_max_frames=debug_precompress_vis_max_frames,
                debug_precompress_vis_upsample_size=debug_precompress_vis_upsample_size,
                debug_precompress_vis_upsample_interp=debug_precompress_vis_upsample_interp,
                debug_precompress_vis_scale=debug_precompress_vis_scale,
            )

            if wrap_strength is not None:
                item_latents = apply_wrap_strength_to_latents(
                    base_latents=base_item_latents,
                    wrapped_latents=item_latents,
                    wrap_strength=float(wrap_strength),
                    injection_mode=wrap_injection_mode,
                    delta_lowpass_kernel=delta_lowpass_kernel,
                )

        if requested_dtype is not None:
            item_latents = item_latents.to(dtype=requested_dtype)
            if return_base_latents:
                base_item_latents = base_item_latents.to(dtype=requested_dtype)
        elif return_base_latents:
            base_item_latents = base_item_latents.to(dtype=item_latents.dtype)

        batch_latents.append(item_latents)
        if batch_base_latents is not None:
            batch_base_latents.append(base_item_latents)

    latents = torch.cat(batch_latents, dim=0)

    print(f"Prepared camera-aware latents shape: {latents.shape}")
    print(f"Prepared camera-aware latents mean: {latents.mean().item():.6f}, std: {latents.std().item():.6f}")
    print(f"Prepared camera-aware latents min: {latents.min().item():.6f}, max: {latents.max().item():.6f}")
    if batch_base_latents is not None:
        base_latents = torch.cat(batch_base_latents, dim=0)
        return latents, base_latents
    return latents

if __name__ == '__main__':
    import os
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video


    from typing import Any, Callable, Dict, List, Optional, Union, Tuple
    import torch
    from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from diffusers.utils.torch_utils import randn_tensor
    import math
    import numpy as np
    dtype = torch.bfloat16
    device = "cuda"

    model_id = os.environ.get("WORLD_R1_WAN_MODEL", "")
    if not model_id:
        raise RuntimeError("Set WORLD_R1_WAN_MODEL to a Wan checkpoint before running this module directly.")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
    # pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=3)
    pipe.to(device)
    # import pdb; pdb.set_trace()
    height = 480
    width = 832
    num_frames = 81
    num_inference_steps = 20
    guidance_scale = 5.0


    prompt = ["Camera push in, then pan left. In the Family Breakfast scene, morning sunlight gently illuminates the cozy kitchen, as the aroma of freshly brewed coffee wafts from the coffee maker while the barstools stand invitingly by the kitchen counter."]
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    from wan_pipeline_with_logprob import wan_pipeline_with_logprob
    print("prepare latents with camera")
    latents = prepare_latents_with_camera(
        prompt=prompt,
        batch_size=1,
        num_channels_latents=16,  # WanModel VAE channel count
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=dtype,
        device=device,
        vae_scale_factor_temporal=4, # WanModel temporal compression
        frames_per_trajectory=81
    )
    # import pdb; pdb.set_trace()
    print("prepare latents with camera done")
    with torch.no_grad():
        output = wan_pipeline_with_logprob(
            pipe,
            prompt=prompt,
            latents = latents,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator().manual_seed(1),
            return_dict=False,
            use_camera_trajectory=True,
        )
    # import pdb; pdb.set_trace()
    export_to_video([frame for frame in output[0].squeeze(0)], "out_nonorm.mp4", fps=15)
