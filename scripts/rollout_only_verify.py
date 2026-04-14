#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from peft import PeftModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
'''
单次运行示例

  1.3B：

  python scripts/rollout_only_verify.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "Camera push in. A massive waterfall pours over a towering cliff in a continuous white torrent, striking dark rock
  ledges before plunging into the basin below." \
    --out-dir Logs/rollout_only_fix_045 \
    --device cuda:0 \
    --dtype bf16 \
    --num-steps 50 \
    --wrap-strength 0.45 \
    --wrap-injection-mode stepwise_delta \
    --stepwise-guidance-steps 8 \
    --noise-wrap-flow-scale 16 \
    --noise-degradation 0.35 \
    --save-noise-vis

  14B：

  python scripts/rollout_only_verify.py \
    --model /path/to/Wan2.1-T2V-14B-Diffusers \
    --prompt "Camera push in. A massive waterfall pours over a towering cliff in a continuous white torrent, striking dark rock
  ledges before plunging into the basin below." \
    --out-dir Logs/rollout_only_fix_045_14b \
    --device cuda:0 \
    --dtype bf16 \
    --num-steps 50 \
    --wrap-strength 0.45 \
    --wrap-injection-mode stepwise_delta \
    --stepwise-guidance-steps 8 \
    --noise-wrap-flow-scale 16 \
    --noise-degradation 0.35 \
    --save-noise-vis

  三档强度批量验证

  for strength in 0.05 0.35 0.45; do
    tag=$(python3 - <<PY
  s = float("${strength}")
  print(f"{int(round(s * 100)):03d}")
  PY
  )
    python scripts/rollout_only_verify.py \
      --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
      --prompt "Camera push in. A massive waterfall pours over a towering cliff in a continuous white torrent, striking dark rock
  ledges before plunging into the basin below." \
      --out-dir "Logs/rollout_only_strength_${tag}" \
      --device cuda:0 \
      --dtype bf16 \
      --num-steps 50 \
      --wrap-strength "${strength}" \
      --wrap-injection-mode stepwise_delta \
      --stepwise-guidance-steps 8 \
      --noise-wrap-flow-scale 16 \
      --noise-degradation 0.35 \
      --save-noise-vis
  done

  重点看这些输出

  - Logs/.../rollout.mp4
  - Logs/.../summary.json
  - Logs/.../base_latents.pt
  - Logs/.../wrapped_latents.pt
  - Logs/.../delta_low.pt
  - Logs/.../noise_vis/prepared_latents/
'''

from flow_grpo.diffusers_patch.camera_trajectory_utils import (
    build_stepwise_delta_callback,
    get_camera_trajectories_for_batch,
    lowpass_latent_delta,
    prepare_latents_with_camera,
    remove_camera_keywords_from_prompts,
)
from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import wan_pipeline_with_logprob
from flow_grpo.diffusers_patch.wan_prompt_embedding import encode_prompt


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


def slugify(text: str, max_len: int = 80) -> str:
    ascii_text = text.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").lower()
    if not cleaned:
        cleaned = "prompt"
    return cleaned[:max_len]


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt.strip()
    with open(args.prompt_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise ValueError("No valid prompt found.")


def build_pipeline(
    model_id: str,
    lora_path: str,
    device: torch.device,
    dtype: torch.dtype,
    show_progress: bool,
    sampler_mode: str,
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
    if sampler_mode == "standard":
        pipe.transformer.to(device, dtype=dtype)
    else:
        pipe.transformer.to(device)
    pipe.vae.to(device, dtype=torch.float32)
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    return pipe


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_text_embeddings(
    prompts,
    text_encoder,
    tokenizer,
    device: torch.device,
    max_sequence_length: int,
) -> torch.Tensor:
    with torch.no_grad():
        prompt_embeds = encode_prompt(
            [text_encoder],
            [tokenizer],
            prompts,
            max_sequence_length=max_sequence_length,
        )
    return prompt_embeds.to(device)


def per_tensor_stats(tensor: Optional[torch.Tensor]) -> Optional[dict]:
    if tensor is None:
        return None
    tensor = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def prepare_rollout_latents_and_callback(
    pipeline,
    prompt,
    batch_size,
    num_channels_latents,
    height,
    width,
    num_frames,
    dtype,
    device,
    vae_scale_factor_temporal,
    frames_per_trajectory,
    force_camera_movement,
    noise_wrap_compute_dtype,
    noise_downtemp_interp,
    noise_downspatial_mode,
    noise_degradation,
    noise_wrap_flow_scale,
    wrap_strength,
    wrap_injection_mode,
    delta_lowpass_kernel,
    stepwise_guidance_steps,
    camera_trajectories,
    detected_movements_batch,
    debug_precompress_vis_dir=None,
):
    common_kwargs = dict(
        prompt=prompt,
        batch_size=batch_size,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=dtype,
        device=device,
        vae_scale_factor_temporal=vae_scale_factor_temporal,
        frames_per_trajectory=frames_per_trajectory,
        force_camera_movement=force_camera_movement,
        remove_camera_keywords_from_prompt=False,
        noise_wrap_compute_dtype=noise_wrap_compute_dtype,
        noise_downtemp_interp=noise_downtemp_interp,
        noise_downspatial_mode=noise_downspatial_mode,
        noise_degradation=noise_degradation,
        noise_wrap_flow_scale=noise_wrap_flow_scale,
        camera_trajectories=camera_trajectories,
        detected_movements_batch=detected_movements_batch,
        debug_precompress_vis_dir=debug_precompress_vis_dir,
    )

    if wrap_injection_mode != "stepwise_delta":
        latents = prepare_latents_with_camera(
            **common_kwargs,
            wrap_strength=wrap_strength,
            wrap_injection_mode=wrap_injection_mode,
            delta_lowpass_kernel=delta_lowpass_kernel,
        )
        debug_tensors = {
            "base_latents": latents.detach().float().cpu(),
            "wrapped_latents": latents.detach().float().cpu(),
            "delta_low": None,
        }
        return latents, None, debug_tensors

    wrapped_latents, base_latents = prepare_latents_with_camera(
        **common_kwargs,
        return_base_latents=True,
    )
    wrapped_latents = wrapped_latents.float()
    base_latents = base_latents.float()

    if wrap_strength is None or wrap_strength <= 0:
        debug_tensors = {
            "base_latents": base_latents.detach().float().cpu(),
            "wrapped_latents": wrapped_latents.detach().float().cpu(),
            "delta_low": None,
        }
        return base_latents.to(dtype=dtype), None, debug_tensors

    delta = wrapped_latents - base_latents
    delta_low = lowpass_latent_delta(delta, delta_lowpass_kernel)
    callback = build_stepwise_delta_callback(
        delta_low=delta_low,
        wrap_strength=float(wrap_strength),
        guidance_steps=stepwise_guidance_steps,
    )
    debug_tensors = {
        "base_latents": base_latents.detach().float().cpu(),
        "wrapped_latents": wrapped_latents.detach().float().cpu(),
        "delta_low": delta_low.detach().float().cpu(),
    }
    return base_latents.to(dtype=dtype), callback, debug_tensors


def save_video_tensor(video, output_path: str, fps: int) -> None:
    if isinstance(video, torch.Tensor):
        video = video.detach().float().cpu()
        if video.ndim == 4 and video.shape[1] in {1, 3}:
            frames = [img for img in video.numpy().transpose(0, 2, 3, 1)]
        else:
            frames = [img for img in video.numpy()]
    else:
        frames = []
        for frame in video:
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().float().cpu().numpy()
                if frame.ndim == 3 and frame.shape[0] in {1, 3}:
                    frame = frame.transpose(1, 2, 0)
            elif hasattr(frame, "convert"):
                frame = np.array(frame.convert("RGB"))
            elif not isinstance(frame, np.ndarray):
                raise TypeError(f"Unsupported frame type: {type(frame)}")
            frames.append(frame)

    frames = [(frame * 255).clip(0, 255).astype("uint8") for frame in frames]
    imageio.mimsave(output_path, frames, fps=fps, codec="libx264", format="FFMPEG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout-only verification for World-R1 stepwise noise wrap.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--frames-per-trajectory", type=int, default=81)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--remove-camera-keywords", action="store_true")
    parser.add_argument("--force-camera-movement", default=None)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--noise-wrap-compute-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--noise-downtemp-interp", choices=["nearest", "blend"], default="nearest")
    parser.add_argument("--noise-downspatial-mode", choices=["resize_noise", "area"], default="resize_noise")
    parser.add_argument("--noise-degradation", type=float, default=0.35)
    parser.add_argument("--noise-wrap-flow-scale", type=int, default=16)
    parser.add_argument("--wrap-strength", type=float, default=0.45)
    parser.add_argument("--wrap-injection-mode", choices=["blend", "lowpass_delta", "stepwise_delta"], default="stepwise_delta")
    parser.add_argument("--delta-lowpass-kernel", type=int, default=9)
    parser.add_argument("--stepwise-guidance-steps", type=int, default=8)
    parser.add_argument("--sampler-mode", choices=["rollout", "standard"], default="rollout")
    parser.add_argument("--save-noise-vis", action="store_true")
    args = parser.parse_args()

    if not args.prompt and not args.prompt_file:
        raise ValueError("Provide either --prompt or --prompt-file.")

    prompt = load_prompt(args)
    rollout_prompt = prompt
    if args.remove_camera_keywords:
        rollout_prompt = remove_camera_keywords_from_prompts([prompt])[0]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = parse_dtype(args.dtype)
    device = resolve_device(args.device)
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32
    if device.type == "cuda":
        torch.cuda.set_device(device)

    seed_everything(args.seed)

    pipeline = build_pipeline(
        args.model,
        args.lora_path,
        device,
        dtype,
        args.show_progress,
        args.sampler_mode,
    )

    prompt_embeds = None
    negative_prompt_embeds = None
    if args.sampler_mode == "rollout":
        prompt_embeds = compute_text_embeddings(
            [prompt],
            pipeline.text_encoder,
            pipeline.tokenizer,
            device,
            args.max_sequence_length,
        )
        negative_prompt_embeds = compute_text_embeddings(
            [args.negative_prompt],
            pipeline.text_encoder,
            pipeline.tokenizer,
            device,
            args.max_sequence_length,
        )

    camera_trajectories, detected_movements_batch, _, _ = get_camera_trajectories_for_batch(
        [rollout_prompt],
        batch_size=1,
        frames_per_trajectory=args.frames_per_trajectory,
        force_camera_movement=args.force_camera_movement,
    )

    noise_vis_base = out_dir / "noise_vis" if args.save_noise_vis else None
    latents, latent_callback, debug_tensors = prepare_rollout_latents_and_callback(
        pipeline=pipeline,
        prompt=[rollout_prompt],
        batch_size=1,
        num_channels_latents=pipeline.transformer.config.in_channels,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        dtype=dtype,
        device=device,
        vae_scale_factor_temporal=pipeline.vae_scale_factor_temporal,
        frames_per_trajectory=args.frames_per_trajectory,
        force_camera_movement=args.force_camera_movement,
        noise_wrap_compute_dtype=args.noise_wrap_compute_dtype,
        noise_downtemp_interp=args.noise_downtemp_interp,
        noise_downspatial_mode=args.noise_downspatial_mode,
        noise_degradation=args.noise_degradation,
        noise_wrap_flow_scale=args.noise_wrap_flow_scale,
        wrap_strength=args.wrap_strength,
        wrap_injection_mode=args.wrap_injection_mode,
        delta_lowpass_kernel=args.delta_lowpass_kernel,
        stepwise_guidance_steps=args.stepwise_guidance_steps,
        camera_trajectories=camera_trajectories,
        detected_movements_batch=detected_movements_batch,
        debug_precompress_vis_dir=str(noise_vis_base / "precompress") if noise_vis_base else None,
    )

    if device.type == "cuda" and dtype != torch.float32:
        autocast_context = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        autocast_context = contextlib.nullcontext()

    if args.sampler_mode == "standard" and args.save_noise_vis:
        print("Warning: `--save-noise-vis` is only supported in rollout mode; ignoring it for standard mode.")

    with torch.no_grad():
        with autocast_context:
            if args.sampler_mode == "rollout":
                videos, all_latents, log_probs, _ = wan_pipeline_with_logprob(
                    pipeline,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    output_type="pt",
                    return_dict=False,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    kl_reward=0.0,
                    determistic=True,
                    save_latents_vis=args.save_noise_vis,
                    vis_output_dir=str(noise_vis_base / "prepared_latents") if noise_vis_base else None,
                    callback_on_step_end=latent_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
                rollout_video = videos[0]
            else:
                rollout_video = pipeline(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    latents=latents.clone().to(device=device, dtype=torch.float32),
                    output_type="pt",
                    callback_on_step_end=latent_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).frames[0]
                all_latents = []
                log_probs = []

    save_video_tensor(rollout_video, str(out_dir / "rollout.mp4"), args.fps)

    torch.save(debug_tensors["base_latents"], out_dir / "base_latents.pt")
    torch.save(debug_tensors["wrapped_latents"], out_dir / "wrapped_latents.pt")
    if debug_tensors["delta_low"] is not None:
        torch.save(debug_tensors["delta_low"], out_dir / "delta_low.pt")
    if all_latents:
        torch.save(torch.stack(all_latents, dim=1).cpu(), out_dir / "all_latents.pt")
    if log_probs:
        torch.save(torch.stack(log_probs, dim=1).cpu(), out_dir / "log_probs.pt")

    summary = {
        "prompt": prompt,
        "rollout_prompt": rollout_prompt,
        "detected_movements_batch": detected_movements_batch,
        "camera_trajectories_present": [item is not None for item in camera_trajectories],
        "config": {
            "model": args.model,
            "lora_path": args.lora_path,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "frames_per_trajectory": args.frames_per_trajectory,
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "dtype": args.dtype,
            "device": str(device),
            "sampler_mode": args.sampler_mode,
            "noise_wrap_compute_dtype": args.noise_wrap_compute_dtype,
            "noise_downtemp_interp": args.noise_downtemp_interp,
            "noise_downspatial_mode": args.noise_downspatial_mode,
            "noise_degradation": args.noise_degradation,
            "noise_wrap_flow_scale": args.noise_wrap_flow_scale,
            "wrap_strength": args.wrap_strength,
            "wrap_injection_mode": args.wrap_injection_mode,
            "delta_lowpass_kernel": args.delta_lowpass_kernel,
            "stepwise_guidance_steps": args.stepwise_guidance_steps,
            "remove_camera_keywords": args.remove_camera_keywords,
            "force_camera_movement": args.force_camera_movement,
        },
        "tensor_stats": {
            "base_latents": per_tensor_stats(debug_tensors["base_latents"]),
            "wrapped_latents": per_tensor_stats(debug_tensors["wrapped_latents"]),
            "delta_low": per_tensor_stats(debug_tensors["delta_low"]),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "prompts.json").write_text(json.dumps([prompt], ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved rollout outputs to: {out_dir}")


if __name__ == "__main__":
    main()
