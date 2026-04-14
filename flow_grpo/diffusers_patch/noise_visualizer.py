"""
Noise Visualization Utilities for Training

Lightweight utilities to visualize and save noise during training.
"""

import os
from typing import Optional, List
import numpy as np
import torch
import torch.nn.functional as F


def noise_to_rgb(
    noise: np.ndarray,
    channels: List[int] = [0, 1, 2],
    scale: float = 6.0,
) -> np.ndarray:
    """
    Convert multi-channel noise to RGB visualization.

    Args:
        noise: Noise array of shape [H, W, C] or [C, H, W]
        channels: Which channels to use for R, G, B

    Returns:
        RGB image array [H, W, 3] in range [0, 1]
    """
    # Handle both [H, W, C] and [C, H, W] formats
    if noise.ndim == 3:
        if noise.shape[0] <= 16 and noise.shape[2] > 16:
            # Likely [C, H, W] format
            noise = noise.transpose(1, 2, 0)
        # else: already [H, W, C]

    h, w, c = noise.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    for i, ch in enumerate(channels[:3]):
        if ch < c:
            rgb[..., i] = np.clip(noise[..., ch] / scale + 0.5, 0, 1)

    return rgb


def flow_to_rgb(dx: np.ndarray, dy: np.ndarray, max_flow: Optional[float] = None) -> np.ndarray:
    """
    Convert optical flow to RGB visualization.

    Args:
        dx: Horizontal flow component [H, W]
        dy: Vertical flow component [H, W]
        max_flow: Maximum flow for normalization

    Returns:
        RGB image [H, W, 3] in range [0, 1]
    """
    import colorsys

    h, w = dx.shape
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    if max_flow is None:
        max_flow = magnitude.max()
    if max_flow > 0:
        magnitude = magnitude / max_flow
    magnitude = np.clip(magnitude, 0, 1)

    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi)
    hsv[..., 1] = 1.0
    hsv[..., 2] = magnitude

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            rgb[i, j] = colorsys.hsv_to_rgb(hsv[i, j, 0], hsv[i, j, 1], hsv[i, j, 2])

    return rgb


def save_video_mp4(frames: List[np.ndarray], output_path: str, fps: int = 12):
    """
    Save frames as MP4 video.

    Args:
        frames: List of frames [H, W, 3] as uint8 or float32
        output_path: Output video path
        fps: Frames per second
    """
    try:
        import cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open; falling back to imageio.")

        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        raise RuntimeError("cv2 produced empty video; falling back to imageio.")
    except Exception:
        try:
            import imageio
            frames_uint8 = []
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                frames_uint8.append(frame)
            imageio.mimsave(output_path, frames_uint8, fps=fps, codec='libx264')
            return True
        except ImportError:
            print("Warning: Neither cv2 nor imageio available. Cannot save video.")
            return False


def visualize_latents_as_video(
    latents: torch.Tensor,
    output_path: str,
    fps: int = 12,
    channels: List[int] = [0, 1, 2],
    upsample_size: Optional[tuple] = None,
    upsample_interp: str = "nearest",
    scale: float = 6.0,
) -> bool:
    """
    Visualize latent tensor as a video.

    Args:
        latents: Latent tensor of shape [B, C, T, H, W] or [C, T, H, W]
        output_path: Output video file path
        fps: Frames per second
        channels: Which channels to visualize as RGB
        upsample_size: Optional (height, width) to upsample frames

    Returns:
        True if successful, False otherwise
    """
    # Handle batch dimension
    if latents.ndim == 5:
        latents = latents[0]  # Take first batch

    # latents: [C, T, H, W]
    C, T, H, W = latents.shape

    frames = []
    for t in range(T):
        # Get frame: [C, H, W]
        frame = latents[:, t, :, :].float().cpu().numpy()

        # Convert to [H, W, C]
        frame = frame.transpose(1, 2, 0)

        # Convert to RGB
        frame_rgb = noise_to_rgb(frame, channels=channels, scale=scale)

        # Upsample if requested
        if upsample_size is not None:
            try:
                import cv2
                interp = cv2.INTER_NEAREST if upsample_interp == "nearest" else cv2.INTER_LINEAR
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (upsample_size[1], upsample_size[0]),
                    interpolation=interp,
                )
            except ImportError:
                pass

        # Convert to uint8
        frame_uint8 = (frame_rgb * 255).astype(np.uint8)
        frames.append(frame_uint8)

    # Save video
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    return save_video_mp4(frames, output_path, fps=fps)


def visualize_noise_sequence(
    noise_list: List[np.ndarray],
    output_path: str,
    fps: int = 12,
    add_flow: bool = False,
    flow_list: Optional[List[np.ndarray]] = None,
) -> bool:
    """
    Visualize a sequence of noise frames and optionally their flows.

    Args:
        noise_list: List of noise arrays, each of shape [H, W, C]
        output_path: Output video file path
        fps: Frames per second
        add_flow: Whether to add flow visualization side-by-side
        flow_list: Optional list of flows [dx, dy] for each frame

    Returns:
        True if successful
    """
    frames = []

    for i, noise in enumerate(noise_list):
        # Convert noise to RGB
        noise_rgb = noise_to_rgb(noise)

        # Add flow if available
        if add_flow and flow_list is not None and i < len(flow_list):
            flow = flow_list[i]
            dx, dy = flow[0], flow[1]
            flow_rgb = flow_to_rgb(dx, dy)

            # Concatenate horizontally
            noise_uint8 = (noise_rgb * 255).astype(np.uint8)
            flow_uint8 = (flow_rgb * 255).astype(np.uint8)
            frame = np.concatenate([noise_uint8, flow_uint8], axis=1)
        else:
            frame = (noise_rgb * 255).astype(np.uint8)

        frames.append(frame)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    return save_video_mp4(frames, output_path, fps=fps)


def save_latents_snapshot(
    latents: torch.Tensor,
    output_dir: str,
    step: int,
    max_frames: int = 16,
    fps: int = 12,
):
    """
    Save a snapshot of latents during training.

    This is a lightweight function designed to be called during training.

    Args:
        latents: Latent tensor [B, C, T, H, W]
        output_dir: Directory to save snapshots
        step: Training step number
        max_frames: Maximum frames to save (to limit file size)
        fps: Video frame rate
    """
    os.makedirs(output_dir, exist_ok=True)

    # Take first batch item
    if latents.ndim == 5:
        latent = latents[0]  # [C, T, H, W]
    else:
        latent = latents

    # Limit number of frames
    if latent.shape[1] > max_frames:
        # Sample evenly
        indices = torch.linspace(0, latent.shape[1] - 1, max_frames).long()
        latent = latent[:, indices]

    # Save as video
    video_path = os.path.join(output_dir, f'latents_step_{step:06d}.mp4')
    success = visualize_latents_as_video(
        latent,
        video_path,
        fps=fps,
        upsample_size=(480, 720),  # Upsample for better visibility
    )

    if success:
        print(f"Saved latents snapshot: {video_path}")

    # Also save as numpy for inspection
    npy_path = os.path.join(output_dir, f'latents_step_{step:06d}.npy')
    np.save(npy_path, latent.cpu().numpy().astype(np.float16))

    return video_path if success else None
