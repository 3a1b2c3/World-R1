#!/usr/bin/env python3
"""Compare train-path and inference-path noise-wrap behavior."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from flow_grpo.diffusers_patch.camera_trajectory_utils import (  # noqa: E402
    apply_wrap_strength_to_latents,
    get_camera_trajectory_for_prompts,
    prepare_latents_with_camera,
)
from flow_grpo.diffusers_patch.noise_visualizer import visualize_latents_as_video  # noqa: E402
from scripts.noise_wrap_ablation import (  # noqa: E402
    build_pipeline,
    normalize_video_frames,
    parse_dtype,
    resolve_device,
    seed_everything,
    slugify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--out-dir",
        default="outputs/compare_train_infer_wrap",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("WORLD_R1_WAN_MODEL", ""),
    )
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--num-frames", type=int, default=21)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--force-camera-movement", default=None)
    parser.add_argument("--noise-wrap-compute-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--noise-downtemp-interp", choices=["nearest", "blend"], default="nearest")
    parser.add_argument("--noise-downspatial-mode", choices=["resize_noise", "area"], default="resize_noise")
    parser.add_argument("--noise-degradation", type=float, default=0.35)
    parser.add_argument("--noise-wrap-flow-scale", type=int, default=16)
    parser.add_argument("--wrap-strength", type=float, default=0.45)
    parser.add_argument("--wrap-injection-mode", choices=["blend", "lowpass_delta"], default="lowpass_delta")
    parser.add_argument("--delta-lowpass-kernel", type=int, default=9)
    parser.add_argument("--generate-videos", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def latent_stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach().float().cpu()
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "l2": float(torch.linalg.vector_norm(t).item()),
    }


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    diff = a - b
    mse = torch.mean(diff * diff).item()
    mae = torch.mean(torch.abs(diff)).item()
    max_abs = torch.max(torch.abs(diff)).item()
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0).item()
    return {
        "mse": float(mse),
        "rmse": float(mse ** 0.5),
        "mae": float(mae),
        "max_abs": float(max_abs),
        "cosine": float(cosine),
    }


def video_stats(video_a, video_b) -> Dict[str, float]:
    frames_a = normalize_video_frames(video_a)
    frames_b = normalize_video_frames(video_b)
    count = min(len(frames_a), len(frames_b))
    mses = []
    maes = []
    psnrs = []
    for fa, fb in zip(frames_a[:count], frames_b[:count]):
        a = fa.astype(np.float32) / 255.0
        b = fb.astype(np.float32) / 255.0
        mse = float(np.mean((a - b) ** 2))
        mae = float(np.mean(np.abs(a - b)))
        mses.append(mse)
        maes.append(mae)
        psnrs.append(99.0 if mse == 0 else float(10 * np.log10(1.0 / mse)))
    return {
        "frame_count": count,
        "mse_mean": float(np.mean(mses)),
        "mae_mean": float(np.mean(maes)),
        "psnr_mean": float(np.mean(psnrs)),
    }


def build_base_latents_with_pipe(pipe, args, device: torch.device) -> torch.Tensor:
    base_generator = torch.Generator(device=device).manual_seed(args.seed)
    return pipe.prepare_latents(
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


def build_base_latents_with_randn(args, device: torch.device, num_latent_frames: int) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(args.seed)
    return randn_tensor(
        (
            1,
            16,
            num_latent_frames,
            args.height // 8,
            args.width // 8,
        ),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def save_video(pipe, latents: torch.Tensor, args, out_path: str):
    with torch.no_grad():
        video = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            generator=None,
            latents=latents.clone().to(device=latents.device, dtype=torch.float32),
        ).frames[0]
    export_to_video(video, out_path, fps=12)
    return video


def main() -> None:
    args = parse_args()
    if not args.model:
        raise ValueError("Provide --model or set WORLD_R1_WAN_MODEL to a Wan Diffusers checkpoint.")
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    slug = slugify(args.prompt)
    out_dir = Path(args.out_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(
        args.model,
        args.lora_path,
        device,
        dtype,
        args.show_progress,
    )

    trajectory, detected_movements = get_camera_trajectory_for_prompts(
        args.prompt,
        frames_per_trajectory=args.num_frames,
        force_camera_movement=args.force_camera_movement,
    )
    num_latent_frames = (args.num_frames - 1) // pipe.vae_scale_factor_temporal + 1

    seed_everything(args.seed)
    train_actual = prepare_latents_with_camera(
        prompt=[args.prompt],
        batch_size=1,
        num_channels_latents=16,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        dtype=dtype,
        device=device,
        vae_scale_factor_temporal=pipe.vae_scale_factor_temporal,
        frames_per_trajectory=args.num_frames,
        force_camera_movement=args.force_camera_movement,
        noise_wrap_compute_dtype=args.noise_wrap_compute_dtype,
        noise_downtemp_interp=args.noise_downtemp_interp,
        noise_downspatial_mode=args.noise_downspatial_mode,
        noise_degradation=args.noise_degradation,
        noise_wrap_flow_scale=args.noise_wrap_flow_scale,
        wrap_strength=args.wrap_strength,
        wrap_injection_mode=args.wrap_injection_mode,
        delta_lowpass_kernel=args.delta_lowpass_kernel,
    ).float()

    seed_everything(args.seed)
    wrapped_only = prepare_latents_with_camera(
        prompt=[args.prompt],
        batch_size=1,
        num_channels_latents=16,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        dtype=dtype,
        device=device,
        vae_scale_factor_temporal=pipe.vae_scale_factor_temporal,
        frames_per_trajectory=args.num_frames,
        force_camera_movement=args.force_camera_movement,
        noise_wrap_compute_dtype=args.noise_wrap_compute_dtype,
        noise_downtemp_interp=args.noise_downtemp_interp,
        noise_downspatial_mode=args.noise_downspatial_mode,
        noise_degradation=args.noise_degradation,
        noise_wrap_flow_scale=args.noise_wrap_flow_scale,
    ).float()

    base_pipe = build_base_latents_with_pipe(pipe, args, device)
    base_randn = build_base_latents_with_randn(args, device, num_latent_frames)

    infer_actual = apply_wrap_strength_to_latents(
        base_latents=base_pipe,
        wrapped_latents=wrapped_only,
        wrap_strength=args.wrap_strength,
        injection_mode=args.wrap_injection_mode,
        delta_lowpass_kernel=args.delta_lowpass_kernel,
    ).float()

    train_seeded_generator = apply_wrap_strength_to_latents(
        base_latents=base_randn,
        wrapped_latents=wrapped_only,
        wrap_strength=args.wrap_strength,
        injection_mode=args.wrap_injection_mode,
        delta_lowpass_kernel=args.delta_lowpass_kernel,
    ).float()

    visualize_latents_as_video(train_actual, str(out_dir / "train_actual_latents.mp4"), fps=12, upsample_size=(args.height, args.width))
    visualize_latents_as_video(infer_actual, str(out_dir / "infer_actual_latents.mp4"), fps=12, upsample_size=(args.height, args.width))
    visualize_latents_as_video(train_seeded_generator, str(out_dir / "train_seeded_generator_latents.mp4"), fps=12, upsample_size=(args.height, args.width))
    visualize_latents_as_video(wrapped_only, str(out_dir / "wrapped_only_latents.mp4"), fps=12, upsample_size=(args.height, args.width))
    visualize_latents_as_video(base_pipe, str(out_dir / "base_pipe_latents.mp4"), fps=12, upsample_size=(args.height, args.width))
    visualize_latents_as_video(base_randn, str(out_dir / "base_randn_latents.mp4"), fps=12, upsample_size=(args.height, args.width))

    summary: Dict[str, object] = {
        "prompt": args.prompt,
        "detected_movements": detected_movements,
        "trajectory_frames": 0 if trajectory is None else len(trajectory),
        "config": {
            "model": args.model,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "dtype": args.dtype,
            "wrap_strength": args.wrap_strength,
            "wrap_injection_mode": args.wrap_injection_mode,
            "delta_lowpass_kernel": args.delta_lowpass_kernel,
            "noise_wrap_compute_dtype": args.noise_wrap_compute_dtype,
            "noise_downtemp_interp": args.noise_downtemp_interp,
            "noise_downspatial_mode": args.noise_downspatial_mode,
            "noise_degradation": args.noise_degradation,
            "noise_wrap_flow_scale": args.noise_wrap_flow_scale,
        },
        "latent_stats": {
            "train_actual": latent_stats(train_actual),
            "wrapped_only": latent_stats(wrapped_only),
            "base_pipe": latent_stats(base_pipe),
            "base_randn": latent_stats(base_randn),
            "infer_actual": latent_stats(infer_actual),
            "train_seeded_generator": latent_stats(train_seeded_generator),
        },
        "latent_diffs": {
            "train_actual_vs_infer_actual": diff_stats(train_actual, infer_actual),
            "train_actual_vs_train_seeded_generator": diff_stats(train_actual, train_seeded_generator),
            "infer_actual_vs_train_seeded_generator": diff_stats(infer_actual, train_seeded_generator),
            "base_pipe_vs_base_randn": diff_stats(base_pipe, base_randn),
        },
    }

    torch.save(train_actual.cpu(), out_dir / "train_actual_latents.pt")
    torch.save(infer_actual.cpu(), out_dir / "infer_actual_latents.pt")
    torch.save(train_seeded_generator.cpu(), out_dir / "train_seeded_generator_latents.pt")
    torch.save(base_pipe.cpu(), out_dir / "base_pipe_latents.pt")
    torch.save(base_randn.cpu(), out_dir / "base_randn_latents.pt")
    torch.save(wrapped_only.cpu(), out_dir / "wrapped_only_latents.pt")

    if args.generate_videos:
        train_video = save_video(pipe, train_actual.to(device), args, str(out_dir / "train_actual.mp4"))
        infer_video = save_video(pipe, infer_actual.to(device), args, str(out_dir / "infer_actual.mp4"))
        seeded_video = save_video(pipe, train_seeded_generator.to(device), args, str(out_dir / "train_seeded_generator.mp4"))
        summary["video_diffs"] = {
            "train_actual_vs_infer_actual": video_stats(train_video, infer_video),
            "train_actual_vs_train_seeded_generator": video_stats(train_video, seeded_video),
            "infer_actual_vs_train_seeded_generator": video_stats(infer_video, seeded_video),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved comparison outputs to: {out_dir}")


if __name__ == "__main__":
    main()
