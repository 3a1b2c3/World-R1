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
    {"name": "aggressive", "noise_degradation": 0.15, "noise_wrap_flow_scale": 32},
    {"name": "moderate_high", "noise_degradation": 0.25, "noise_wrap_flow_scale": 24},
    {"name": "balanced", "noise_degradation": 0.35, "noise_wrap_flow_scale": 16},
    {"name": "conservative", "noise_degradation": 0.50, "noise_wrap_flow_scale": 12},
]


def parse_gpus(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_command(args, preset, out_dir: Path) -> list[str]:
    return [
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
        str(preset["noise_degradation"]),
        "--noise-wrap-flow-scale",
        str(preset["noise_wrap_flow_scale"]),
        "--save-baseline",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full-resolution noise-wrap parameter sweep.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "noise_wrap_sweep"))
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
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
        "dtype": args.dtype,
        "presets": PRESETS,
    }
    (root_out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    pending = PRESETS.copy()
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
                "preset": preset,
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
            print(f"finished {name}: retcode={retcode}")
            completed.append(
                {
                    "name": name,
                    "retcode": retcode,
                    "gpu_id": job["gpu_id"],
                    "out_dir": job["out_dir"],
                    "log_path": job["log_path"],
                }
            )
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
