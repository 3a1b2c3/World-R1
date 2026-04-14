#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_SCRIPT = PROJECT_ROOT / "scripts" / "noise_wrap_strength_sweep.py"
WAN_PYTHON = Path(os.environ.get("WAN_PYTHON", sys.executable))

PROMPT_SPECS = [
    {
        "name": "push_in_waterfall",
        "motion": "push_in",
        "prompt": "Camera push in. A massive waterfall pours over a towering cliff in a continuous white torrent, striking dark rock ledges before plunging into the basin below. Thick mist billows upward from the impact and drifts across the cliff face, softening the rugged stone and catching the light in a cool haze. Wet surfaces glisten with constant spray, and narrow streams split off along the rock wall, feeding the larger fall. The scale feels immense and elemental, with roaring water, churning foam, and dense vapor turning the whole cliffside into a thunderous wall of motion.",
    },
    {
        "name": "pull_out_music_festival",
        "motion": "pull_out",
        "prompt": "Camera pull out. A huge crowd erupts with excitement at a music festival, people cheering, jumping, and throwing their hands into the air in unison as the energy ripples outward through the packed audience. Faces glow with exhilaration beneath colorful stage lights, and waves of movement pass across the mass of bodies like a living pulse. Dust and mist catch the light above the crowd while flags and raised phones bob overhead. The moment feels loud, ecstatic, and communal, capturing the scale of thousands of fans fully absorbed in the peak of a live performance.",
    },
    {
        "name": "move_right_subway",
        "motion": "move_right",
        "prompt": "Camera move right. A subway train rushes past the platform at high speed, its metallic body streaking by in a blur of windows, doors, and reflected station lights. Bright interior illumination flashes in repeating intervals, briefly revealing seated passengers before each carriage is replaced by the next. The platform edge, warning strip, and tiled wall catch sharp bursts of light as the train barrels through, while air and grit are kicked along the concrete. The sense of momentum is urgent and urban, with the train dominating the frame as it tears through the station.",
    },
    {
        "name": "pan_left_soldiers",
        "motion": "pan_left",
        "prompt": "Camera pan left. Soldiers march in tight synchronization across a dusty field, their boots striking the ground in unison and sending up low clouds of pale dirt with each measured step. Their formation holds clean, straight lines as arms swing in the same rhythm and uniforms move with disciplined precision. The dry earth beneath them looks worn and powdery, and the haze kicked up by the march hangs briefly around their legs before trailing behind the column. The repeated cadence of their movement gives the moment a formal, controlled intensity shaped by order and collective force.",
    },
    {
        "name": "orbit_right_tornado",
        "motion": "orbit_right",
        "prompt": "Camera orbit right. A violent tornado rips through a wooden barn with relentless force, its rotating column of wind splintering beams, tearing away planks, and hurling fragments into the air. The structure buckles unevenly as the roof peels back and sections of wall collapse inward, exposing the framing for a moment before it too is shredded. Dust, straw, and broken wood spiral around the funnel in a chaotic storm of debris. The sky is dark and heavy, and the tornado's power feels immediate and catastrophic, turning a familiar rural building into wreckage in seconds under the sheer force of the twisting wind.",
    },
]

PRESET_NAMES = "baseline,strength_045,strength_050"


def run_sweep(
    *,
    prompt_name: str,
    prompt: str,
    model_path: str,
    model_tag: str,
    gpu_id: str,
    out_root: Path,
):
    prompt_root = out_root / model_tag / prompt_name
    prompt_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(WAN_PYTHON),
        str(SWEEP_SCRIPT),
        "--prompt",
        prompt,
        "--model",
        model_path,
        "--out-dir",
        str(prompt_root),
        "--height",
        "480",
        "--width",
        "832",
        "--num-frames",
        "81",
        "--num-steps",
        "50",
        "--dtype",
        "bf16",
        "--noise-degradation",
        "0.35",
        "--noise-wrap-flow-scale",
        "16",
        "--injection-mode",
        "stepwise_delta",
        "--delta-lowpass-kernel",
        "9",
        "--stepwise-guidance-steps",
        "8",
        "--gpus",
        gpu_id,
        "--preset-names",
        PRESET_NAMES,
    ]

    log_path = prompt_root / "suite.log"
    with open(log_path, "w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc, str(log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-prompt noise-wrap suite for one model.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "outputs" / "noise_wrap_multi_prompt_suite"),
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_tag": args.model_tag,
        "model_path": args.model_path,
        "gpu": args.gpu,
        "preset_names": PRESET_NAMES.split(","),
        "prompts": PROMPT_SPECS,
    }
    model_root = out_root / args.model_tag
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results = []
    for spec in PROMPT_SPECS:
        print(f"starting {args.model_tag}:{spec['name']} on gpu {args.gpu}")
        proc, log_path = run_sweep(
            prompt_name=spec["name"],
            prompt=spec["prompt"],
            model_path=args.model_path,
            model_tag=args.model_tag,
            gpu_id=args.gpu,
            out_root=out_root,
        )
        retcode = proc.wait()
        results.append(
            {
                "prompt_name": spec["name"],
                "motion": spec["motion"],
                "retcode": retcode,
                "log_path": log_path,
            }
        )
        (model_root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        if retcode != 0:
            print(json.dumps(results[-1], indent=2))
            sys.exit(retcode)
        time.sleep(2)

    print(f"completed model suite: {model_root}")


if __name__ == "__main__":
    main()
