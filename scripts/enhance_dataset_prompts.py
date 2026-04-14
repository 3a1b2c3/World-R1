#!/usr/bin/env python3
"""Enhance World-R1 dataset prompts with per-line LLM rewrites."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import sys
import time
import tomllib
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "dataset" / "final"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "enhanced"
DEFAULT_STATE_DIR = DEFAULT_OUTPUT_DIR / ".llm_state"

CAMERA_PREFIX_RE = re.compile(r"^(Camera[^.]*\.)\s*(.+)$")

CAMERA_LAYOUT_INFO = {
    "push_in": "push in",
    "pull_out": "pull out",
    "move_left": "move left",
    "move_right": "move right",
    "orbit_left": "orbit left",
    "orbit_right": "orbit right",
    "pan_left": "pan left",
    "pan_right": "pan right",
    "fixed": "fixed",
}

FORBIDDEN_STOCK_PHRASES = [
    "foreground detail, mid-range structure",
    "the scene reads as a complete environment",
    "the richer wording",
    "stable background",
    "spatial continuity",
    "the setting is expanded",
    "the prompt describes a fully built scene",
]

ADDED_CAMERA_PREFIXES = [
    "Camera fixed.",
    "Camera push in.",
    "Camera pull out.",
    "Camera move left.",
    "Camera move right.",
    "Camera pan left.",
    "Camera pan right.",
    "Camera orbit left.",
    "Camera orbit right.",
    "Camera push in, then pan left.",
    "Camera push in, then pan right.",
    "Camera pull out, then pan left.",
    "Camera pull out, then pan right.",
    "Camera orbit left, then pull out.",
    "Camera orbit right, then push in.",
    "Camera move left, pull out, then pan left.",
    "Camera move right, pull out, then pan right.",
]

SYSTEM_PROMPT = """You rewrite one video-generation prompt at a time.

Requirements:
- Preserve the original meaning, subject, action, setting, and any named style.
- Expand the prompt into a naturally written, specific, detailed English paragraph of about 600 characters.
- Write each output as if it were individually authored for that exact source prompt, not from a reusable template.
- Vary sentence structure across items and avoid stock boilerplate.
- If the source prompt begins with a camera-control prefix such as `Camera push in, then pan right.`, copy that prefix exactly character-for-character at the very beginning of the output.
- Do not alter, paraphrase, omit, duplicate, or append camera-control instructions.
- Outside the preserved camera prefix, do not use camera-motion phrases like push in, pull out, move left, move right, orbit left, orbit right, pan left, pan right, or fixed.
- If the source prompt has no camera prefix, do not invent any camera instruction.
- Infer only conservative details that strongly fit the original prompt. Do not change the event or introduce unrelated objects.
- Return exactly one line of plain text. No quotes. No bullet points. No explanations."""

BATCH_SYSTEM_PROMPT = """You rewrite a small batch of video-generation prompts.

Requirements for every item:
- Preserve the original meaning, subject, action, setting, and any named style.
- Expand the prompt into a naturally written, specific, detailed English paragraph of about 600 characters.
- Write each item as if it were independently authored for that exact source prompt, not from a reusable template.
- Vary sentence structure across items and avoid stock boilerplate.
- Respect the exact required camera prefix for each item when one is provided.
- Outside the required camera prefix, do not use camera-motion phrases.
- If an item is marked as having no camera prefix, do not add any camera wording.
- Infer only conservative details that strongly fit the original prompt. Do not change the event or introduce unrelated objects.
- Return strict JSON only, with no markdown fences and no extra commentary."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing train.txt, test.txt, and dynamic.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write enhanced dataset files into",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help="Checkpoint directory for per-file progress and logs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N prompts from each file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the configured model name",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=520,
        help="Minimum acceptable output length",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=740,
        help="Maximum acceptable output length",
    )
    parser.add_argument(
        "--target-guidance",
        type=str,
        default="around 600 characters, usually between 560 and 680 characters",
        help="Length instruction passed to the model",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="Maximum generation attempts per prompt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous outputs and restart from scratch",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=260,
        help="Maximum output tokens per response",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=12,
        help="Number of concurrent per-prompt API calls",
    )
    parser.add_argument(
        "--keep-no-camera-ratio",
        type=float,
        default=0.02,
        help="Fraction of originally no-camera prompts to keep without any camera prefix",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Number of prompts to send in each batch request",
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


def split_camera_prefix(prompt: str) -> tuple[str, str]:
    match = CAMERA_PREFIX_RE.match(prompt.strip())
    if match:
        return match.group(1), match.group(2).strip()
    return "", prompt.strip()


def stable_fraction(text: str) -> float:
    value = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)
    return value / float(16**12 - 1)


def stable_choice(options: list[str], key: str) -> str:
    value = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16)
    return options[value % len(options)]


def detect_camera_movements(prompt: str) -> list[str]:
    prompt_lower = prompt.lower()
    prompt_matches: list[tuple[int, int, str]] = []
    for movement_name, movement_prompt in CAMERA_LAYOUT_INFO.items():
        pattern = rf"(?<![a-z]){re.escape(movement_prompt.lower())}(?![a-z])"
        for match in re.finditer(pattern, prompt_lower):
            prompt_matches.append((match.start(), -(match.end() - match.start()), movement_name))

    prompt_matches.sort()
    detected: list[str] = []
    last_start: int | None = None
    for start, _, movement_name in prompt_matches:
        if start == last_start:
            continue
        detected.append(movement_name)
        last_start = start
    return detected


def extract_text_from_response(data: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    return "".join(texts).strip()


def clean_single_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ").replace("\r", " ")).strip()


def opening_signature(text: str) -> str:
    prefix, body = split_camera_prefix(text)
    reference = body if body else prefix
    words = re.findall(r"[A-Za-z']+", reference)
    return " ".join(words[:10]).lower()


def choose_added_camera_prefix(source_prompt: str, source_name: str) -> str | None:
    lower = source_prompt.lower()
    landscape_pool = [
        "Camera push in.",
        "Camera pan right.",
        "Camera orbit left.",
        "Camera pull out, then pan right.",
        "Camera move right, pull out, then pan right.",
    ]
    macro_pool = [
        "Camera fixed.",
        "Camera push in.",
        "Camera orbit right.",
        "Camera push in, then pan left.",
    ]
    action_pool = [
        "Camera push in.",
        "Camera move right.",
        "Camera orbit left.",
        "Camera push in, then pan right.",
        "Camera move left, pull out, then pan left.",
    ]
    urban_pool = [
        "Camera move left.",
        "Camera pan right.",
        "Camera pull out, then pan left.",
        "Camera orbit right.",
        "Camera move right, pull out, then pan right.",
    ]

    if any(word in lower for word in ["city", "street", "town", "station", "cathedral", "factory", "metro", "skyscraper"]):
        pool = urban_pool
    elif any(word in lower for word in ["flower", "whiskey", "tea", "vase", "circuit board", "glass", "curtain", "gear", "microchip", "orb", "gem", "crystal"]):
        pool = macro_pool
    elif source_name == "dynamic.txt" or any(word in lower for word in ["lion", "cyclist", "chef", "spaceship", "train", "tornado", "shattering", "blooming", "roaring"]):
        pool = action_pool
    else:
        pool = landscape_pool

    return stable_choice(pool, f"{source_name}::{source_prompt}")


def planned_camera_prefix(source_prompt: str, source_name: str, keep_no_camera_ratio: float) -> str | None:
    existing_prefix, _ = split_camera_prefix(source_prompt)
    if existing_prefix:
        return existing_prefix

    keep_without_camera = stable_fraction(f"keep-no-camera::{source_name}::{source_prompt}") < keep_no_camera_ratio
    if keep_without_camera:
        return None

    return choose_added_camera_prefix(source_prompt, source_name)


def build_user_prompt(
    source_prompt: str,
    source_name: str,
    target_camera_prefix: str | None,
    target_guidance: str,
    recent_openings: list[str],
) -> str:
    instructions = [
        "Rewrite the following source prompt into a more detailed prompt for video generation.",
        f"Target length: {target_guidance}.",
        "Do not sound formulaic and do not use generic scaffolding language.",
        f"Avoid these stock phrases entirely: {', '.join(FORBIDDEN_STOCK_PHRASES)}.",
    ]

    if recent_openings:
        instructions.append(
            "Avoid starting in a way that closely echoes these recent openings: "
            + " | ".join(recent_openings)
            + "."
        )

    if target_camera_prefix:
        instructions.append(
            f"Use this exact camera prefix at the beginning, unchanged: {target_camera_prefix}"
        )
        instructions.append(
            "Outside that exact prefix, do not use any camera movement wording."
        )
    else:
        instructions.append("Do not add any camera instruction or camera wording.")

    instructions.append(f"Dataset file: {source_name}")
    instructions.append(f"Source prompt: {source_prompt}")
    instructions.append("Return only the rewritten prompt.")
    return "\n".join(instructions)


def build_batch_user_prompt(
    items: list[dict[str, Any]],
    target_guidance: str,
    recent_openings: list[str],
) -> str:
    lines = [
        "Rewrite every item below into a more detailed prompt for video generation.",
        f"Target length for each item: {target_guidance}.",
        "Do not sound formulaic and do not use generic scaffolding language.",
        f"Avoid these stock phrases entirely: {', '.join(FORBIDDEN_STOCK_PHRASES)}.",
        'Return strict JSON only in this shape: {"items":[{"index":123,"rewritten":"..."}, ...]}',
        "Every input index must appear exactly once in the output.",
    ]

    if recent_openings:
        lines.append(
            "Avoid reusing openings that closely echo these recent outputs: "
            + " | ".join(recent_openings)
            + "."
        )

    for item in items:
        camera_requirement = item["target_camera_prefix"] if item["target_camera_prefix"] else "NONE"
        lines.append("")
        lines.append(f"Item {item['index']}:")
        lines.append(f"- dataset file: {item['source_name']}")
        lines.append(f"- required camera prefix: {camera_requirement}")
        lines.append(f"- source prompt: {item['source_prompt']}")

    return "\n".join(lines)


def validate_output(
    source_prompt: str,
    source_name: str,
    target_camera_prefix: str | None,
    candidate: str,
    min_length: int,
    max_length: int,
) -> tuple[bool, str]:
    candidate = clean_single_line(candidate)
    if not candidate:
        return False, "empty output"

    if len(candidate) < min_length:
        return False, f"too short: {len(candidate)}"
    if len(candidate) > max_length:
        return False, f"too long: {len(candidate)}"

    cand_prefix, _ = split_camera_prefix(candidate)

    if bool(target_camera_prefix) != bool(cand_prefix):
        return False, "camera prefix presence changed"
    if target_camera_prefix and cand_prefix != target_camera_prefix:
        return False, "camera prefix changed"

    source_prefix, _ = split_camera_prefix(source_prompt)
    if source_prefix:
        expected_movements = detect_camera_movements(source_prompt)
    elif target_camera_prefix:
        expected_movements = detect_camera_movements(target_camera_prefix)
    else:
        expected_movements = []

    if expected_movements != detect_camera_movements(candidate):
        return False, "camera movement parsing changed"

    if candidate.count('"') or candidate.count("“") or candidate.count("”"):
        return False, "contains quotes"

    return True, "ok"


def request_rewrite(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
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


def rewrite_prompt(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    source_prompt: str,
    source_name: str,
    target_camera_prefix: str | None,
    recent_openings: list[str],
    args: argparse.Namespace,
) -> str:
    last_error = "unknown"
    for attempt in range(1, args.max_attempts + 1):
        user_prompt = build_user_prompt(
            source_prompt=source_prompt,
            source_name=source_name,
            target_camera_prefix=target_camera_prefix,
            target_guidance=args.target_guidance,
            recent_openings=recent_openings,
        )
        if attempt > 1:
            user_prompt += (
                f"\nPrevious attempt failed validation because: {last_error}."
                "\nFix that issue while keeping the same meaning."
            )

        try:
            data = request_rewrite(
                client=client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )
        except Exception as exc:
            last_error = f"request failed: {exc}"
            time.sleep(min(8.0, 1.5 * attempt))
            continue

        candidate = clean_single_line(extract_text_from_response(data))
        ok, reason = validate_output(
            source_prompt=source_prompt,
            source_name=source_name,
            target_camera_prefix=target_camera_prefix,
            candidate=candidate,
            min_length=args.min_length,
            max_length=args.max_length,
        )
        if ok:
            return candidate
        last_error = reason
        time.sleep(min(2.0, 0.3 * attempt))

    raise RuntimeError(f"failed after {args.max_attempts} attempts: {last_error}")


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


def rewrite_batch(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    items: list[dict[str, Any]],
    recent_openings: list[str],
    args: argparse.Namespace,
) -> dict[int, str]:
    last_error = "unknown"
    for attempt in range(1, max(2, args.max_attempts) + 1):
        user_prompt = build_batch_user_prompt(
            items=items,
            target_guidance=args.target_guidance,
            recent_openings=recent_openings,
        )
        if attempt > 1:
            user_prompt += (
                f"\nPrevious batch attempt failed because: {last_error}."
                "\nFix the issues and return valid JSON."
            )

        try:
            data = request_rewrite(
                client=client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                system_prompt=BATCH_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_output_tokens=min(4000, args.max_output_tokens * len(items) + 300),
            )
            payload = parse_json_payload(extract_text_from_response(data))
        except Exception as exc:
            last_error = f"batch request failed: {exc}"
            time.sleep(min(8.0, 1.5 * attempt))
            continue

        outputs = payload.get("items")
        if not isinstance(outputs, list):
            last_error = "missing items list"
            continue

        rewritten_by_index: dict[int, str] = {}
        invalid: list[dict[str, Any]] = []

        for item in items:
            match = None
            for out in outputs:
                if isinstance(out, dict) and out.get("index") == item["index"]:
                    match = out
                    break
            if match is None or not isinstance(match.get("rewritten"), str):
                invalid.append(item)
                continue

            candidate = clean_single_line(match["rewritten"])
            ok, reason = validate_output(
                source_prompt=item["source_prompt"],
                source_name=item["source_name"],
                target_camera_prefix=item["target_camera_prefix"],
                candidate=candidate,
                min_length=args.min_length,
                max_length=args.max_length,
            )
            if ok:
                rewritten_by_index[item["index"]] = candidate
            else:
                invalid.append({**item, "validation_error": reason})

        if not invalid:
            return rewritten_by_index

        last_error = f"{len(invalid)} items failed validation or were missing"

    rewritten_by_index = {}
    for item in items:
        rewritten_by_index[item["index"]] = rewrite_prompt(
            client=client,
            base_url=base_url,
            api_key=api_key,
            model=model,
            source_prompt=item["source_prompt"],
            source_name=item["source_name"],
            target_camera_prefix=item["target_camera_prefix"],
            recent_openings=recent_openings,
            args=args,
        )
    return rewritten_by_index


def load_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def process_file(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    source_path: Path,
    output_path: Path,
    state_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompts = load_lines(source_path)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    partial_path = state_dir / f"{source_path.name}.partial"
    log_path = state_dir / f"{source_path.name}.jsonl"
    state_path = state_dir / f"{source_path.name}.state.json"

    done_lines: list[str] = []
    recent_openings: deque[str] = deque(maxlen=8)

    if partial_path.exists():
        done_lines = load_lines(partial_path)
        for line in done_lines[-8:]:
            recent_openings.append(opening_signature(line))

    start_index = len(done_lines)
    if start_index > len(prompts):
        raise RuntimeError(
            f"partial file {partial_path} has {start_index} lines but source only has {len(prompts)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=len(prompts), initial=start_index, desc=source_path.name, dynamic_ncols=True)

    def submit_job(executor: ThreadPoolExecutor, batch_indices: list[int]) -> Future[dict[int, str]]:
        items = []
        for idx in batch_indices:
            items.append(
                {
                    "index": idx,
                    "source_prompt": prompts[idx],
                    "source_name": source_path.name,
                    "target_camera_prefix": planned_camera_prefix(
                        prompts[idx], source_path.name, args.keep_no_camera_ratio
                    ),
                }
            )
        return executor.submit(
            rewrite_batch,
            client,
            base_url,
            api_key,
            model,
            items,
            list(recent_openings),
            args,
        )

    with partial_path.open("a", encoding="utf-8") as out_handle, log_path.open("a", encoding="utf-8") as log_handle:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            pending: dict[int, Future[dict[int, str]]] = {}
            ready: dict[int, str] = {}
            next_submit = start_index
            next_write = start_index

            while next_submit < len(prompts) and len(pending) < args.concurrency:
                batch_end = min(len(prompts), next_submit + max(1, args.batch_size))
                batch_indices = list(range(next_submit, batch_end))
                pending[next_submit] = submit_job(executor, batch_indices)
                next_submit = batch_end

            while next_write < len(prompts):
                if not pending:
                    raise RuntimeError("no pending tasks while work remains")

                done, _ = wait(list(pending.values()), return_when=FIRST_COMPLETED)
                done_ids = {id(item) for item in done}
                completed_indices = [idx for idx, future in pending.items() if id(future) in done_ids]

                for idx in completed_indices:
                    future = pending.pop(idx)
                    for item_idx, rewritten in future.result().items():
                        ready[item_idx] = rewritten

                while next_write in ready:
                    rewritten = ready.pop(next_write)
                    source_prompt = prompts[next_write]

                    out_handle.write(rewritten + "\n")
                    out_handle.flush()

                    log_handle.write(
                        json.dumps(
                            {
                                "index": next_write,
                                "source": source_prompt,
                                "rewritten": rewritten,
                                "source_length": len(source_prompt),
                                "rewritten_length": len(rewritten),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    log_handle.flush()

                    recent_openings.append(opening_signature(rewritten))
                    progress.update(1)
                    next_write += 1

                    state_path.write_text(
                        json.dumps(
                            {
                                "completed": next_write,
                                "total": len(prompts),
                                "source_file": str(source_path),
                                "output_file": str(output_path),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                    while next_submit < len(prompts) and len(pending) < args.concurrency:
                        batch_end = min(len(prompts), next_submit + max(1, args.batch_size))
                        batch_indices = list(range(next_submit, batch_end))
                        pending[next_submit] = submit_job(executor, batch_indices)
                        next_submit = batch_end

    progress.close()
    partial_path.replace(output_path)
    if state_path.exists():
        state_path.unlink()

    output_lines = load_lines(output_path)
    lengths = [len(line) for line in output_lines]
    return {
        "count": len(output_lines),
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "mean": statistics.mean(lengths) if lengths else 0.0,
        "median": statistics.median(lengths) if lengths else 0.0,
    }


def main() -> int:
    args = parse_args()
    base_url, configured_model, api_key = load_codex_config()
    model = args.model or configured_model

    if args.overwrite:
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        if args.state_dir.exists():
            shutil.rmtree(args.state_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    with httpx.Client(timeout=httpx.Timeout(180.0, connect=30.0)) as client:
        for name in ("train.txt", "test.txt", "dynamic.txt"):
            source_path = args.source_dir / name
            output_path = args.output_dir / name
            summaries[name] = process_file(
                client=client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                source_path=source_path,
                output_path=output_path,
                state_dir=args.state_dir,
                args=args,
            )
            stats = summaries[name]
            print(
                f"{name}: count={stats['count']} min={stats['min']} max={stats['max']} "
                f"mean={stats['mean']:.1f} median={stats['median']:.1f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
