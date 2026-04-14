#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from typing import List, Optional

import numpy as np
import torch
from peft import PeftModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from flow_grpo.diffusers_patch.camera_trajectory_utils import (
    blend_noise,
    camera_motion_to_flow,
    get_camera_trajectory_for_prompts,
    parse_camera_matrix_torch,
    prepare_latents_with_camera,
)
from flow_grpo.diffusers_patch.noise_visualizer import (
    flow_to_rgb,
    noise_to_rgb,
    save_video_mp4,
    visualize_latents_as_video,
)


def parse_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def slugify(text: str, max_len: int = 60) -> str:
    ascii_text = text.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").lower()
    if not cleaned:
        cleaned = "prompt"
    return cleaned[:max_len]


def build_pipeline(
    model_id: str,
    lora_path: str,
    device: torch.device,
    dtype: torch.dtype,
    show_progress: bool,
) -> WanPipeline:
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=not show_progress)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    if lora_path:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)
        pipe.transformer.set_adapter("default")

    pipe.text_encoder.to(device, dtype=dtype)
    pipe.transformer.to(device, dtype=dtype)
    pipe.vae.to(device, dtype=torch.float32)
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    return pipe


def sample_trajectory_matrices(
    trajectory: dict,
    num_frames: int,
    *,
    device: torch.device,
) -> List[torch.Tensor]:
    frame_keys = sorted(trajectory.keys(), key=lambda x: int(x.replace("frame", "")))
    total_frames = len(frame_keys)
    if num_frames >= total_frames:
        selected_indices = list(range(total_frames))
    else:
        selected_indices = [
            int(round(i * (total_frames - 1) / (num_frames - 1))) if num_frames > 1 else 0
            for i in range(num_frames)
        ]
    return [
        parse_camera_matrix_torch(trajectory[frame_keys[idx]], device=device, dtype=torch.float32)
        for idx in selected_indices
    ]


def compute_flow_sequence(
    trajectory: Optional[dict],
    num_frames: int,
    height: int,
    width: int,
    scene_depth: float,
    *,
    device: torch.device,
) -> tuple[list[np.ndarray], dict]:
    if trajectory is None:
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        return [blank for _ in range(num_frames)], {
            "flow_mean_per_step": [],
            "flow_peak_per_step": [],
            "max_flow_peak": 0.0,
        }

    matrices = sample_trajectory_matrices(trajectory, num_frames, device=device)
    preview_frames = []
    mean_magnitudes = []
    peak_magnitudes = []

    blank = np.zeros((height, width, 3), dtype=np.uint8)
    preview_frames.append(blank)

    for idx in range(1, len(matrices)):
        dx, dy = camera_motion_to_flow(
            matrices[idx - 1],
            matrices[idx],
            height=height,
            width=width,
            depth=scene_depth,
            device=device,
            dtype=torch.float32,
        )
        magnitude = torch.sqrt(dx * dx + dy * dy)
        mean_magnitudes.append(float(magnitude.mean().item()))
        peak_magnitudes.append(float(magnitude.max().item()))
        preview_frames.append(
            (flow_to_rgb(dx.float().cpu().numpy(), dy.float().cpu().numpy()) * 255).astype(np.uint8)
        )

    while len(preview_frames) < num_frames:
        preview_frames.append(preview_frames[-1].copy())

    return preview_frames, {
        "flow_mean_per_step": mean_magnitudes,
        "flow_peak_per_step": peak_magnitudes,
        "max_flow_peak": max(peak_magnitudes) if peak_magnitudes else 0.0,
    }


def latents_to_preview_frames(
    latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
) -> List[np.ndarray]:
    if latents.ndim == 5:
        latents = latents[0]
    _, latent_t, _, _ = latents.shape

    preview_frames = []
    for frame_index in range(num_frames):
        if num_frames > 1 and latent_t > 1:
            source_index = round(frame_index * (latent_t - 1) / (num_frames - 1))
        else:
            source_index = 0
        frame = latents[:, source_index].float().cpu().numpy().transpose(1, 2, 0)
        frame_rgb = noise_to_rgb(frame)
        frame_uint8 = (np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8)
        try:
            import cv2

            frame_uint8 = cv2.resize(frame_uint8, (width, height), interpolation=cv2.INTER_NEAREST)
        except ImportError:
            pass
        preview_frames.append(frame_uint8)
    return preview_frames


def normalize_video_frames(video) -> List[np.ndarray]:
    if isinstance(video, torch.Tensor):
        tensor = video.detach().float().cpu()
        if tensor.ndim == 4 and tensor.shape[1] in {1, 3}:
            tensor = tensor.permute(0, 2, 3, 1)
        frames = tensor.numpy()
        return [(np.clip(frame, 0, 1) * 255).astype(np.uint8) for frame in frames]

    frames = []
    for frame in video:
        if hasattr(frame, "convert"):
            frame = np.array(frame.convert("RGB"))
        elif isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")
        frames.append(frame)
    return frames


def add_panel_label(frame: np.ndarray, label: str) -> np.ndarray:
    try:
        import cv2

        output = frame.copy()
        cv2.rectangle(output, (0, 0), (output.shape[1], 28), (0, 0, 0), thickness=-1)
        cv2.putText(
            output,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return output
    except ImportError:
        return frame


def build_comparison_frames(
    flow_frames: List[np.ndarray],
    latent_frames: List[np.ndarray],
    generated_frames: List[np.ndarray],
    baseline_frames: Optional[List[np.ndarray]] = None,
) -> List[np.ndarray]:
    frame_count = min(len(flow_frames), len(latent_frames), len(generated_frames))
    if baseline_frames is not None:
        frame_count = min(frame_count, len(baseline_frames))

    comparison_frames = []
    for idx in range(frame_count):
        panels = [
            add_panel_label(flow_frames[idx], "Flow"),
            add_panel_label(latent_frames[idx], "Wrapped Latent"),
            add_panel_label(generated_frames[idx], "Generated"),
        ]
        if baseline_frames is not None:
            panels.append(add_panel_label(baseline_frames[idx], "Baseline"))
        comparison_frames.append(np.concatenate(panels, axis=1))
    return comparison_frames


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lowpass_latent_delta(delta: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return delta
    assert delta.ndim == 5, delta.shape
    batch, channels, frames, height, width = delta.shape
    x = delta.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    padding = kernel_size // 2
    x = torch.nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
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


def load_prompt(args) -> str:
    if args.prompt:
        return args.prompt.strip()
    with open(args.prompt_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise ValueError("No valid prompt found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise-wrap ablation and visualization for World-R1.")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--out-dir", default="outputs/noise_wrap_ablation")
    parser.add_argument(
        "--model",
        default=os.environ.get("WORLD_R1_WAN_MODEL", ""),
        help="Wan Diffusers checkpoint path or Hugging Face repo id. Can also be set via WORLD_R1_WAN_MODEL.",
    )
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-steps", type=int, default=20, help="Inference steps for visualization. Very small values like 2 are only for smoke tests and will look noisy.")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--force-camera-movement", default=None)
    parser.add_argument("--scene-depth", type=float, default=1000.0)
    parser.add_argument("--noise-wrap-compute-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--noise-downtemp-interp", choices=["nearest", "blend"], default="nearest")
    parser.add_argument("--noise-downspatial-mode", choices=["resize_noise", "area"], default="resize_noise")
    parser.add_argument("--noise-degradation", type=float, default=0.35)
    parser.add_argument("--noise-wrap-flow-scale", type=int, default=16)
    parser.add_argument(
        "--wrap-strength",
        type=float,
        default=0.5,
        help="Injection strength for wrapped latent guidance. For Wan 2.1 14B, 0.45-0.5 is a reasonable range.",
    )
    parser.add_argument("--injection-mode", choices=["blend", "lowpass_delta", "stepwise_delta"], default="lowpass_delta")
    parser.add_argument("--delta-lowpass-kernel", type=int, default=9)
    parser.add_argument("--stepwise-guidance-steps", type=int, default=8)
    args = parser.parse_args()

    if not args.prompt and not args.prompt_file:
        raise ValueError("Provide either --prompt or --prompt-file.")
    if not args.model:
        raise ValueError("Provide --model or set WORLD_R1_WAN_MODEL to a Wan Diffusers checkpoint.")

    prompt = load_prompt(args)
    prompt_slug = slugify(prompt)
    output_dir = os.path.join(args.out_dir, prompt_slug)
    os.makedirs(output_dir, exist_ok=True)

    dtype = parse_dtype(args.dtype)
    device = resolve_device(args.device)
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32
    if device.type == "cuda":
        torch.cuda.set_device(device)

    trajectory, detected_movements = get_camera_trajectory_for_prompts(
        prompt,
        frames_per_trajectory=args.num_frames,
        force_camera_movement=args.force_camera_movement,
    )

    flow_frames, flow_stats = compute_flow_sequence(
        trajectory=trajectory,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        scene_depth=args.scene_depth,
        device=device,
    )
    save_video_mp4(flow_frames, os.path.join(output_dir, "flow_preview.mp4"), fps=args.fps)

    seed_everything(args.seed)
    wrapped_latents = prepare_latents_with_camera(
        prompt=[prompt],
        batch_size=1,
        num_channels_latents=16,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        dtype=dtype,
        device=device,
        vae_scale_factor_temporal=4,
        frames_per_trajectory=args.num_frames,
        force_camera_movement=args.force_camera_movement,
        noise_wrap_compute_dtype=args.noise_wrap_compute_dtype,
        noise_downtemp_interp=args.noise_downtemp_interp,
        noise_downspatial_mode=args.noise_downspatial_mode,
        noise_degradation=args.noise_degradation,
        noise_wrap_flow_scale=args.noise_wrap_flow_scale,
        debug_precompress_vis_dir=os.path.join(output_dir, "latent_precompress"),
    )

    base_generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    pipe_for_latents = build_pipeline(
        args.model,
        args.lora_path,
        device,
        dtype,
        args.show_progress,
    )
    base_latents = pipe_for_latents.prepare_latents(
        batch_size=1,
        num_channels_latents=16,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        dtype=torch.float32,
        device=device,
        generator=base_generator,
        latents=None,
    )

    stepwise_callback = None
    if args.wrap_strength <= 0:
        latents = base_latents
    elif args.wrap_strength >= 1 and args.injection_mode == "blend":
        latents = wrapped_latents.float()
    else:
        wrapped_latents = wrapped_latents.float()
        if args.injection_mode == "blend":
            latents = blend_noise(base_latents, wrapped_latents, args.wrap_strength)
        elif args.injection_mode == "lowpass_delta":
            delta = wrapped_latents - base_latents
            delta_low = lowpass_latent_delta(delta, args.delta_lowpass_kernel)
            latents = base_latents + args.wrap_strength * delta_low
            latents = (latents - latents.mean()) / (latents.std() + 1e-6)
        else:
            delta = wrapped_latents - base_latents
            delta_low = lowpass_latent_delta(delta, args.delta_lowpass_kernel)
            latents = base_latents
            stepwise_callback = build_stepwise_delta_callback(
                delta_low=delta_low,
                wrap_strength=args.wrap_strength,
                guidance_steps=args.stepwise_guidance_steps,
            )

    visualize_latents_as_video(
        latents,
        os.path.join(output_dir, "latent_injected.mp4"),
        fps=args.fps,
        upsample_size=(args.height, args.width),
    )
    latent_preview_frames = latents_to_preview_frames(latents, args.num_frames, args.height, args.width)
    visualize_latents_as_video(
        base_latents,
        os.path.join(output_dir, "latent_random.mp4"),
        fps=args.fps,
        upsample_size=(args.height, args.width),
    )
    visualize_latents_as_video(
        wrapped_latents,
        os.path.join(output_dir, "latent_wrapped.mp4"),
        fps=args.fps,
        upsample_size=(args.height, args.width),
    )

    summary = {
        "prompt": prompt,
        "detected_movements": detected_movements,
        "trajectory_frame_count": 0 if trajectory is None else len(trajectory),
        "latent_shape": list(latents.shape),
        "latent_mean": float(latents.float().mean().item()),
        "latent_std": float(latents.float().std().item()),
        "wrap_strength": args.wrap_strength,
        "injection_mode": args.injection_mode,
        "flow_stats": flow_stats,
        "config": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "noise_wrap_compute_dtype": args.noise_wrap_compute_dtype,
            "noise_downtemp_interp": args.noise_downtemp_interp,
            "noise_downspatial_mode": args.noise_downspatial_mode,
            "noise_degradation": args.noise_degradation,
            "noise_wrap_flow_scale": args.noise_wrap_flow_scale,
            "wrap_strength": args.wrap_strength,
            "injection_mode": args.injection_mode,
            "delta_lowpass_kernel": args.delta_lowpass_kernel,
            "stepwise_guidance_steps": args.stepwise_guidance_steps,
            "scene_depth": args.scene_depth,
        },
    }

    if trajectory is not None:
        with open(os.path.join(output_dir, "camera_trajectory.json"), "w", encoding="utf-8") as handle:
            json.dump(trajectory, handle, ensure_ascii=False, indent=2)

    generated_frames = None
    baseline_frames = None
    if not args.skip_generation:
        with torch.no_grad():
            wrapped_video = pipe_for_latents(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=None,
                latents=latents.clone().to(device=device, dtype=torch.float32),
                callback_on_step_end=stepwise_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            ).frames[0]
        export_to_video(wrapped_video, os.path.join(output_dir, "generated_wrapped.mp4"), fps=args.fps)
        generated_frames = normalize_video_frames(wrapped_video)

        if args.save_baseline:
            with torch.no_grad():
                baseline_video = pipe_for_latents(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    generator=None,
                    latents=base_latents.clone().to(device=device, dtype=torch.float32),
                ).frames[0]
            export_to_video(baseline_video, os.path.join(output_dir, "generated_baseline.mp4"), fps=args.fps)
            baseline_frames = normalize_video_frames(baseline_video)

        comparison_frames = build_comparison_frames(
            flow_frames=flow_frames,
            latent_frames=latent_preview_frames,
            generated_frames=generated_frames,
            baseline_frames=baseline_frames,
        )
        save_video_mp4(comparison_frames, os.path.join(output_dir, "comparison.mp4"), fps=args.fps)

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
