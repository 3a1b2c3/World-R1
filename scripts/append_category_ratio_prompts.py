#!/usr/bin/env python3
"""Append appendix-ratio prompts to the enhanced World-R1 datasets."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENHANCED_DIR = PROJECT_ROOT / "dataset" / "enhanced"
DEFAULT_STATE_DIR = DEFAULT_ENHANCED_DIR / ".append_ratio_state"

CAMERA_PREFIX_BY_LOGIC = {
    "push_in": "Camera push in.",
    "pull_out": "Camera pull out.",
    "move_left": "Camera move left.",
    "move_right": "Camera move right.",
    "orbit_left": "Camera orbit left.",
    "orbit_right": "Camera orbit right.",
    "pan_left": "Camera pan left.",
    "pan_right": "Camera pan right.",
    "pull_left": "Camera move left, pull out, then pan left.",
    "pull_right": "Camera move right, pull out, then pan right.",
    "fixed": "Camera fixed.",
}

MAIN_CATEGORY_COUNTS = {
    "Natural Landscapes": 33,
    "Urban and Architectural": 26,
    "Micro and Still Life": 65,
    "Fantasy and Surrealism": 46,
    "Artistic Styles": 13,
}

DYNAMIC_CATEGORY_COUNTS = {
    "Dynamic Data Subset": 15,
}

CATEGORY_GUIDANCE = {
    "Natural Landscapes": {
        "description": (
            "Large-scale rigid geometry, terrain depth, water behavior, atmosphere, "
            "weather, and time-of-day changes. Favor physically grounded natural scenes."
        ),
        "subthemes": [
            "landforms and geological structures",
            "water features, reflections, transparency, spray, shorelines",
            "weather, seasonal atmosphere, lighting transitions, sky conditions",
        ],
        "camera_pool": [
            "push_in",
            "pull_out",
            "move_left",
            "move_right",
            "orbit_left",
            "orbit_right",
            "pan_left",
            "pan_right",
            "pull_left",
            "pull_right",
        ],
    },
    "Urban and Architectural": {
        "description": (
            "Urban, indoor, and infrastructure scenes that stress perspective correctness, "
            "straight lines, scale, vanishing points, and spatial layout."
        ),
        "subthemes": [
            "urban streets and skylines",
            "indoor spaces and structured interiors",
            "transport and infrastructure, stations, bridges, terminals, industrial facilities",
        ],
        "camera_pool": [
            "push_in",
            "pull_out",
            "move_left",
            "move_right",
            "orbit_left",
            "orbit_right",
            "pan_left",
            "pan_right",
            "pull_left",
            "pull_right",
        ],
    },
    "Micro and Still Life": {
        "description": (
            "Macro observation, desktop still life, micro world detail, and tactile "
            "material representation with strong texture fidelity and depth cues."
        ),
        "subthemes": [
            "desktop still life and arranged objects",
            "microscopic or magnified biological and inorganic structures",
            "materials, surfaces, translucency, weave, scratches, cracks, fluids",
        ],
        "camera_pool": [
            "push_in",
            "pull_out",
            "move_left",
            "move_right",
            "orbit_left",
            "orbit_right",
            "pan_left",
            "pan_right",
        ],
    },
    "Fantasy and Surrealism": {
        "description": (
            "Physics-defying yet spatially coherent worlds, non-Euclidean geometry, "
            "floating structures, dreamscapes, and surreal architectural logic."
        ),
        "subthemes": [
            "floating cities, islands, citadels, observatories",
            "surreal landscapes, canyons, forests, voids, symbolic light",
            "dreamlike worlds with coherent geometry despite impossible concepts",
        ],
        "camera_pool": [
            "push_in",
            "pull_out",
            "move_left",
            "move_right",
            "orbit_left",
            "orbit_right",
            "pan_left",
            "pan_right",
            "pull_left",
            "pull_right",
        ],
    },
    "Artistic Styles": {
        "description": (
            "Strongly stylized scenes that preserve a specific visual medium or art "
            "direction instead of drifting toward generic photorealism."
        ),
        "subthemes": [
            "watercolor, matte painting, storybook, pastel, black and white photography",
            "surreal illustration, fantasy illustration, cyberpunk illustration, steampunk illustration",
            "clear palette and mark-making consistent with the named artistic style",
        ],
        "camera_pool": [
            "push_in",
            "pull_out",
            "move_left",
            "move_right",
            "orbit_left",
            "orbit_right",
            "pan_left",
            "pan_right",
        ],
    },
    "Dynamic Data Subset": {
        "description": (
            "High-entropy scenes, deformable or non-rigid motion, fluid dynamics, "
            "crowd-like motion, particles, impacts, and strong temporal change."
        ),
        "subthemes": [
            "animals, people, vehicles, crowds, sports, performance",
            "fire, smoke, splashes, weather, eruptions, impacts, breakage",
            "time-lapse growth, flowing material, launches, transformations",
        ],
        "camera_pool": [
            "push_in",
            "move_left",
            "move_right",
            "orbit_left",
            "orbit_right",
            "pan_left",
            "pan_right",
            "pull_left",
            "pull_right",
            "fixed",
            "pull_out",
        ],
    },
}

FORBIDDEN_BODY_PATTERNS = [
    r"(?<![a-z])push in(?![a-z])",
    r"(?<![a-z])pull out(?![a-z])",
    r"(?<![a-z])move left(?![a-z])",
    r"(?<![a-z])move right(?![a-z])",
    r"(?<![a-z])orbit left(?![a-z])",
    r"(?<![a-z])orbit right(?![a-z])",
    r"(?<![a-z])pan left(?![a-z])",
    r"(?<![a-z])pan right(?![a-z])",
    r"(?<![a-z])fixed(?![a-z])",
]

BATCH_SYSTEM_PROMPT = """You are an expert cinematographer and 3D set designer.

Your job is to generate high-fidelity text-to-video prompt bodies that match a requested semantic category and a requested camera trajectory.

Rules for every item:
- Produce a detailed English prompt body around 600 characters.
- The result must feel individually authored, not templated or formulaic.
- The scene must be geometrically coherent and physically grounded unless the category explicitly requests surreal or fantasy content.
- Match the requested category and subtheme guidance closely.
- Match the requested camera logic with a scene layout that benefits from that motion.
- Return only the body text, not the camera prefix. The caller will add the exact prefix.
- Do not write any camera wording in the body at all.
- Do not use the phrases push in, pull out, move left, move right, orbit left, orbit right, pan left, pan right, or fixed inside the body.
- Do not mention the words camera, cinematography, shot, viewpoint, lens, zoom, panning, orbiting, dolly, trucking, or framing.
- Do not copy or paraphrase the appendix examples.
- Return strict JSON only in the schema requested by the user."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enhanced-dir",
        type=Path,
        default=DEFAULT_ENHANCED_DIR,
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
    )
    parser.add_argument(
        "--train-add",
        type=int,
        default=sum(MAIN_CATEGORY_COUNTS.values()),
        help="Number of prompts to append to train.txt",
    )
    parser.add_argument(
        "--test-add",
        type=int,
        default=30,
        help="Number of prompts to append to test.txt",
    )
    parser.add_argument(
        "--dynamic-add",
        type=int,
        default=sum(DYNAMIC_CATEGORY_COUNTS.values()),
        help="Number of prompts to append to dynamic.txt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=360,
    )
    return parser.parse_args()


def load_codex_config() -> tuple[str, str, str]:
    config_path = Path.home() / ".codex" / "config.toml"
    auth_path = Path.home() / ".codex" / "auth.json"

    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    auth = json.loads(auth_path.read_text(encoding="utf-8"))

    provider_name = config["model_provider"]
    provider = config["model_providers"][provider_name]
    base_url = provider["base_url"].rstrip("/")
    model = config["model"]
    api_key = auth["OPENAI_API_KEY"]
    return base_url, model, api_key


def largest_remainder(total: int, weights: dict[str, int]) -> dict[str, int]:
    weight_sum = sum(weights.values())
    raw = {key: total * value / weight_sum for key, value in weights.items()}
    base = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(base.values())
    remainders = sorted(
        ((raw[key] - base[key], key) for key in weights),
        key=lambda x: (-x[0], x[1]),
    )
    for _, key in remainders[:remaining]:
        base[key] += 1
    return base


def round_robin_camera_logic(category: str, index: int) -> str:
    pool = CATEGORY_GUIDANCE[category]["camera_pool"]
    return pool[index % len(pool)]


def build_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    train_quota = largest_remainder(args.train_add, MAIN_CATEGORY_COUNTS)
    test_quota = largest_remainder(args.test_add, MAIN_CATEGORY_COUNTS)
    dynamic_quota = largest_remainder(args.dynamic_add, DYNAMIC_CATEGORY_COUNTS)

    for split, quota in (("train", train_quota), ("test", test_quota), ("dynamic", dynamic_quota)):
        for category, count in quota.items():
            subthemes = CATEGORY_GUIDANCE[category]["subthemes"]
            for idx in range(count):
                item_id = f"{split}:{category}:{idx}"
                items.append(
                    {
                        "id": item_id,
                        "split": split,
                        "target_file": f"{split}.txt" if split != "dynamic" else "dynamic.txt",
                        "category": category,
                        "subtheme": subthemes[idx % len(subthemes)],
                        "camera_logic": round_robin_camera_logic(category, idx),
                    }
                )
    return items


def clean_body(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip('"').strip("'")
    return text


def extract_text_from_response(data: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    return "".join(texts).strip()


def parse_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def request_batch(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": BATCH_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        "reasoning": {"effort": "none"},
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
    }

    response = client.post(
        f"{base_url}/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return data


def build_batch_prompt(batch: list[dict[str, Any]]) -> str:
    lines = [
        "Generate prompt bodies for the following items.",
        'Return strict JSON only in this shape: {"items":[{"id":"...","body":"..."}]}',
        "Every requested id must appear exactly once.",
        "Each body should usually be between 520 and 700 characters.",
        "For test split items, make them slightly more compositionally demanding and evaluation-like than train split items.",
    ]

    for item in batch:
        category_info = CATEGORY_GUIDANCE[item["category"]]
        lines.extend(
            [
                "",
                f"Item id: {item['id']}",
                f"Split: {item['split']}",
                f"Category: {item['category']}",
                f"Category guidance: {category_info['description']}",
                f"Subtheme emphasis: {item['subtheme']}",
                f"Required camera logic: {item['camera_logic']}",
                f"Exact camera prefix will be added separately as: {CAMERA_PREFIX_BY_LOGIC[item['camera_logic']]}",
                "Output requirement: body only, no camera wording.",
            ]
        )
    return "\n".join(lines)


def validate_body(body: str) -> tuple[bool, str]:
    body = clean_body(body)
    if not body:
        return False, "empty body"
    if len(body) < 520:
        return False, f"too short: {len(body)}"
    if len(body) > 740:
        return False, f"too long: {len(body)}"
    if body.startswith("Camera "):
        return False, "body starts with camera prefix"
    if '"' in body or "“" in body or "”" in body:
        return False, "contains quotes"
    lowered = body.lower()
    for pattern in FORBIDDEN_BODY_PATTERNS:
        if re.search(pattern, lowered):
            return False, f"contains forbidden phrase: {pattern}"
    if re.search(r"(?<![a-z])camera(?![a-z])", lowered):
        return False, "contains camera word"
    return True, "ok"


def generate_batch_items(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    batch: list[dict[str, Any]],
    used_prompts: set[str],
    args: argparse.Namespace,
) -> dict[str, str]:
    last_error = "unknown"

    for attempt in range(1, args.max_attempts + 1):
        prompt = build_batch_prompt(batch)
        if attempt > 1:
            prompt += f"\nPrevious batch failed because: {last_error}\nFix the issue and return valid JSON only."

        try:
            data = request_batch(
                client=client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                user_prompt=prompt,
                temperature=args.temperature,
                max_output_tokens=min(4000, args.max_output_tokens * len(batch) + 300),
            )
            payload = parse_json_payload(extract_text_from_response(data))
        except Exception as exc:
            last_error = f"request failed: {exc}"
            time.sleep(min(8.0, 1.2 * attempt))
            continue

        outputs = payload.get("items")
        if not isinstance(outputs, list):
            last_error = "missing items list"
            continue

        output_map: dict[str, str] = {}
        invalid_reasons: list[str] = []

        for item in batch:
            match = next(
                (
                    out
                    for out in outputs
                    if isinstance(out, dict)
                    and out.get("id") == item["id"]
                    and isinstance(out.get("body"), str)
                ),
                None,
            )
            if match is None:
                invalid_reasons.append(f"missing {item['id']}")
                continue

            body = clean_body(match["body"])
            ok, reason = validate_body(body)
            if not ok:
                invalid_reasons.append(f"{item['id']}: {reason}")
                continue

            full_prompt = f"{CAMERA_PREFIX_BY_LOGIC[item['camera_logic']]} {body}"
            if full_prompt in used_prompts or full_prompt in output_map.values():
                invalid_reasons.append(f"{item['id']}: duplicate prompt")
                continue

            output_map[item["id"]] = full_prompt

        if len(output_map) == len(batch):
            return output_map

        last_error = "; ".join(invalid_reasons[:8]) or "batch validation failed"

    raise RuntimeError(f"failed to generate batch after {args.max_attempts} attempts: {last_error}")


def append_prompts(
    enhanced_dir: Path,
    generated_by_file: dict[str, list[str]],
) -> None:
    for name, prompts in generated_by_file.items():
        if not prompts:
            continue
        path = enhanced_dir / name
        existing = path.read_text(encoding="utf-8")
        suffix = "" if existing.endswith("\n") else "\n"
        path.write_text(existing + suffix + "\n".join(prompts) + "\n", encoding="utf-8")


def summarize_file(path: Path) -> tuple[int, float, float, int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    lengths = [len(line) for line in lines]
    return len(lines), statistics.mean(lengths), statistics.median(lengths), min(lengths), max(lengths)


def main() -> None:
    args = parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)

    base_url, model, api_key = load_codex_config()

    plan_items = build_items(args)
    generated_by_file: dict[str, list[str]] = defaultdict(list)

    used_prompts = set()
    for name in ("train.txt", "test.txt", "dynamic.txt"):
        used_prompts.update((args.enhanced_dir / name).read_text(encoding="utf-8").splitlines())

    with httpx.Client(timeout=httpx.Timeout(180.0, connect=30.0)) as client:
        for start in range(0, len(plan_items), args.batch_size):
            batch = plan_items[start : start + args.batch_size]
            result = generate_batch_items(
                client=client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                batch=batch,
                used_prompts=used_prompts,
                args=args,
            )

            batch_log = []
            for item in batch:
                prompt = result[item["id"]]
                generated_by_file[item["target_file"]].append(prompt)
                used_prompts.add(prompt)
                batch_log.append(
                    {
                        "id": item["id"],
                        "target_file": item["target_file"],
                        "category": item["category"],
                        "camera_logic": item["camera_logic"],
                        "prompt": prompt,
                        "length": len(prompt),
                    }
                )

            (args.state_dir / f"batch_{start:04d}.json").write_text(
                json.dumps(batch_log, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"saved batch {start // args.batch_size + 1}: "
                f"{len(batch)} prompts, last_id={batch[-1]['id']}"
            )

    append_prompts(args.enhanced_dir, generated_by_file)

    quota_summary = defaultdict(lambda: defaultdict(int))
    for item in plan_items:
        quota_summary[item["target_file"]][item["category"]] += 1
    print("append quotas:")
    for file_name, counts in quota_summary.items():
        print(file_name, dict(counts))

    print("appended counts:")
    for file_name, prompts in generated_by_file.items():
        lengths = [len(p) for p in prompts]
        print(
            file_name,
            f"added={len(prompts)}",
            f"mean={statistics.mean(lengths):.1f}" if lengths else "mean=0",
            f"median={statistics.median(lengths):.1f}" if lengths else "median=0",
        )

    print("final file stats:")
    for name in ("train.txt", "test.txt", "dynamic.txt"):
        total, mean_len, median_len, min_len, max_len = summarize_file(args.enhanced_dir / name)
        print(
            name,
            f"count={total}",
            f"mean={mean_len:.1f}",
            f"median={median_len:.1f}",
            f"min={min_len}",
            f"max={max_len}",
        )


if __name__ == "__main__":
    main()
