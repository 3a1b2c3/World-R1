#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
import re
import sys

import torch
from peft import PeftModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video


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


def parse_devices(devices_arg: str) -> list[str]:
    if not devices_arg:
        return []
    return [item.strip() for item in devices_arg.split(",") if item.strip()]


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
    return prompts


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


def run_inference(device_str: str, prompt_items: list[tuple[int, str]], args_dict: dict) -> None:
    device = torch.device(device_str)
    dtype = parse_dtype(args_dict["dtype"])
    if device.type == "cpu" and dtype != torch.float32:
        print("CPU does not support fp16/bf16 well; falling back to fp32.")
        dtype = torch.float32
    if device.type == "cuda":
        torch.cuda.set_device(device)

    os.makedirs(args_dict["out_dir"], exist_ok=True)
    pipe = build_pipeline(
        args_dict["model"],
        args_dict["lora_path"],
        device,
        dtype,
        args_dict["show_progress"],
    )

    for idx, prompt in prompt_items:
        print(f"[{device_str}] [{idx + 1}/{args_dict['total_prompts']}] {prompt}")
        generator = None
        if args_dict["seed"] is not None:
            generator = torch.Generator(device=device).manual_seed(args_dict["seed"] + idx)

        with torch.no_grad():
            video = pipe(
                prompt=prompt,
                negative_prompt=args_dict["negative_prompt"],
                height=args_dict["height"],
                width=args_dict["width"],
                num_frames=args_dict["num_frames"],
                num_inference_steps=args_dict["num_steps"],
                guidance_scale=args_dict["guidance_scale"],
                generator=generator,
            ).frames[0]

        slug = slugify(prompt)
        output_path = os.path.join(args_dict["out_dir"], f"{idx:04d}_{slug}.mp4")
        export_to_video(video, output_path, fps=args_dict["fps"])
        print(f"[{device_str}] Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic WAN LoRA inference script for World-R1 release.")
    parser.add_argument(
        "--model",
        default=os.environ.get("WORLD_R1_WAN_MODEL", ""),
        help="Base Wan Diffusers model path or Hugging Face repo id. Can also be set via WORLD_R1_WAN_MODEL.",
    )
    parser.add_argument(
        "--lora-path",
        default="",
        help="Optional LoRA path. Use empty string to run the base model.",
    )
    parser.add_argument("--prompt-file", required=True, help="Text file with one prompt per line.")
    parser.add_argument("--out-dir", default="outputs/infer_wan")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--devices",
        default="",
        help="Comma-separated devices for prompt-level parallelism, e.g. cuda:0,cuda:1.",
    )
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.model:
        raise ValueError("Provide --model or set WORLD_R1_WAN_MODEL to a Wan Diffusers checkpoint.")
    prompts = load_prompts(args.prompt_file)
    if not prompts:
        raise ValueError("No prompts found in the prompt file.")

    devices = parse_devices(args.devices)
    prompt_items = list(enumerate(prompts))
    args_dict = vars(args)
    args_dict["total_prompts"] = len(prompts)

    if devices:
        buckets = [[] for _ in range(len(devices))]
        for idx, item in enumerate(prompt_items):
            buckets[idx % len(devices)].append(item)

        ctx = mp.get_context("spawn")
        workers = []
        for device_str, bucket in zip(devices, buckets):
            if not bucket:
                continue
            proc = ctx.Process(target=run_inference, args=(device_str, bucket, args_dict))
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"Worker on {proc.name} failed with exit code {proc.exitcode}.")
    else:
        device = resolve_device(args.device)
        run_inference(str(device), prompt_items, args_dict)


if __name__ == "__main__":
    main()
