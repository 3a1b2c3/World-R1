#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATION_SCRIPT = PROJECT_ROOT / "scripts" / "noise_wrap_ablation.py"
WAN_PYTHON = Path(os.environ.get("WAN_PYTHON", sys.executable))

PRESETS = [
    {"name": "baseline", "wrap_strength": 0.0},
    {"name": "strength_005", "wrap_strength": 0.05},
    {"name": "strength_010", "wrap_strength": 0.10},
    {"name": "strength_020", "wrap_strength": 0.20},
    {"name": "strength_035", "wrap_strength": 0.35},
    {"name": "strength_040", "wrap_strength": 0.40},
    {"name": "strength_0425", "wrap_strength": 0.425},
    {"name": "strength_045", "wrap_strength": 0.45},
    {"name": "strength_0475", "wrap_strength": 0.475},
    {"name": "strength_050", "wrap_strength": 0.50},
    {"name": "strength_065", "wrap_strength": 0.65},
    {"name": "strength_080", "wrap_strength": 0.80},
    {"name": "strength_100", "wrap_strength": 1.00},
]


def parse_gpus(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_command(args, preset, out_dir: Path) -> list[str]:
    command = [
        str(WAN_PYTHON),
        str(ABLATION_SCRIPT),
        "--prompt",
        args.prompt,
        "--out-dir",
        str(out_dir),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num-frames",
        str(args.num_frames),
        "--num-steps",
        str(args.num_steps),
        "--dtype",
        args.dtype,
        "--device",
        "cuda:0",
        "--noise-degradation",
        str(args.noise_degradation),
        "--noise-wrap-flow-scale",
        str(args.noise_wrap_flow_scale),
        "--wrap-strength",
        str(preset["wrap_strength"]),
        "--injection-mode",
        args.injection_mode,
        "--delta-lowpass-kernel",
        str(args.delta_lowpass_kernel),
        "--stepwise-guidance-steps",
        str(args.stepwise_guidance_steps),
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.lora_path:
        command.extend(["--lora-path", args.lora_path])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full-resolution wrap-strength sweep.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="", help="Override model path passed to noise_wrap_ablation.py")
    parser.add_argument("--lora-path", default="", help="Optional LoRA path passed to noise_wrap_ablation.py")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "outputs" / "noise_wrap_strength_sweep"),
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--noise-degradation", type=float, default=0.35)
    parser.add_argument("--noise-wrap-flow-scale", type=int, default=16)
    parser.add_argument("--injection-mode", choices=["blend", "lowpass_delta", "stepwise_delta"], default="lowpass_delta")
    parser.add_argument("--delta-lowpass-kernel", type=int, default=9)
    parser.add_argument("--stepwise-guidance-steps", type=int, default=8)
    parser.add_argument("--preset-names", default="", help="Comma-separated preset names to run. Empty means run all.")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()

    gpu_pool = parse_gpus(args.gpus)
    if not gpu_pool:
        raise ValueError("No GPUs provided.")

    root_out_dir = Path(args.out_dir).resolve()
    root_out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_steps": args.num_steps,
        "model": args.model,
        "lora_path": args.lora_path,
        "dtype": args.dtype,
        "noise_degradation": args.noise_degradation,
        "noise_wrap_flow_scale": args.noise_wrap_flow_scale,
        "injection_mode": args.injection_mode,
        "delta_lowpass_kernel": args.delta_lowpass_kernel,
        "stepwise_guidance_steps": args.stepwise_guidance_steps,
        "presets": PRESETS,
    }
    (root_out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    pending = PRESETS.copy()
    if args.preset_names:
        selected_names = {item.strip() for item in args.preset_names.split(",") if item.strip()}
        pending = [preset for preset in PRESETS if preset["name"] in selected_names]
        missing = selected_names - {preset["name"] for preset in PRESETS}
        if missing:
            raise ValueError(f"Unknown preset names: {sorted(missing)}")
    running: dict[str, dict] = {}
    completed = []

    while pending or running:
        while pending and len(running) < len(gpu_pool):
            preset = pending.pop(0)
            gpu_id = gpu_pool[len(running)]
            preset_out_dir = root_out_dir / preset["name"]
            preset_out_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_command(args, preset, preset_out_dir)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            log_path = preset_out_dir / "run.log"
            log_handle = open(log_path, "w", encoding="utf-8")
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[preset["name"]] = {
                "gpu_id": gpu_id,
                "process": process,
                "log_handle": log_handle,
                "log_path": str(log_path),
                "out_dir": str(preset_out_dir),
            }
            print(f"started {preset['name']} on gpu {gpu_id}: pid={process.pid}")

        time.sleep(5)

        finished_names = []
        for name, job in running.items():
            retcode = job["process"].poll()
            if retcode is None:
                continue
            job["log_handle"].close()
            completed.append(
                {
                    "name": name,
                    "retcode": retcode,
                    "gpu_id": job["gpu_id"],
                    "out_dir": job["out_dir"],
                    "log_path": job["log_path"],
                }
            )
            print(f"finished {name}: retcode={retcode}")
            finished_names.append(name)

        for name in finished_names:
            del running[name]

    (root_out_dir / "results.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
    failed = [item for item in completed if item["retcode"] != 0]
    if failed:
        print(json.dumps(failed, indent=2))
        sys.exit(1)

    print(f"all presets completed: {root_out_dir}")


if __name__ == "__main__":
    main()
