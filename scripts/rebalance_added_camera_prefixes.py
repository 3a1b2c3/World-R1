#!/usr/bin/env python3
"""Rebalance added camera prefixes in the enhanced World-R1 dataset."""

from __future__ import annotations

import argparse
import hashlib
import re
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "dataset" / "final"
DEFAULT_ENHANCED_DIR = PROJECT_ROOT / "dataset" / "enhanced"

CAMERA_PREFIX_RE = re.compile(r"^(Camera[^.]*\.)\s*(.*)$")

TARGET_PREFIXES = [
    "Camera push in.",
    "Camera pan left.",
    "Camera pan right.",
    "Camera orbit left.",
    "Camera orbit right.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
    )
    parser.add_argument(
        "--enhanced-dir",
        type=Path,
        default=DEFAULT_ENHANCED_DIR,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser.parse_args()


def split_prefix(text: str) -> tuple[str | None, str]:
    match = CAMERA_PREFIX_RE.match(text.strip())
    if not match:
        return None, text.strip()
    return match.group(1), match.group(2).strip()


def stable_pick(options: list[str], key: str) -> str:
    value = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16)
    return options[value % len(options)]


def choose_balanced_prefix(body: str, counts: dict[str, int]) -> str:
    minimum = min(counts.values())
    candidates = [prefix for prefix in TARGET_PREFIXES if counts[prefix] == minimum]
    chosen = stable_pick(candidates, body)
    counts[chosen] += 1
    return chosen


def summarize(lines: list[str]) -> tuple[dict[str, int], int, list[int]]:
    counts: dict[str, int] = {}
    no_camera = 0
    lengths: list[int] = []
    for line in lines:
        lengths.append(len(line))
        prefix, _ = split_prefix(line)
        if prefix is None:
            no_camera += 1
            continue
        counts[prefix] = counts.get(prefix, 0) + 1
    return counts, no_camera, lengths


def main() -> None:
    args = parse_args()

    global_changed = 0
    global_added_changed = 0

    for name in ("train.txt", "test.txt", "dynamic.txt"):
        source_lines = (args.source_dir / name).read_text(encoding="utf-8").splitlines()
        enhanced_path = args.enhanced_dir / name
        enhanced_lines = enhanced_path.read_text(encoding="utf-8").splitlines()

        if len(source_lines) != len(enhanced_lines):
            raise RuntimeError(f"line count mismatch for {name}")

        added_indices: list[int] = []
        bodies: list[str] = []
        for idx, (src, dst) in enumerate(zip(source_lines, enhanced_lines)):
            src_prefix, _ = split_prefix(src)
            dst_prefix, body = split_prefix(dst)
            if src_prefix is None and dst_prefix is not None:
                added_indices.append(idx)
                bodies.append(body)

        target_counts = {prefix: 0 for prefix in TARGET_PREFIXES}
        new_lines = list(enhanced_lines)
        changed = 0

        for idx, body in zip(added_indices, bodies):
            new_prefix = choose_balanced_prefix(body, target_counts)
            new_line = f"{new_prefix} {body}"
            if new_line != new_lines[idx]:
                new_lines[idx] = new_line
                changed += 1

        before_counts, before_no_camera, before_lengths = summarize(enhanced_lines)
        after_counts, after_no_camera, after_lengths = summarize(new_lines)

        print(f"== {name} ==")
        print(f"added_camera_prompts={len(added_indices)} changed={changed}")
        print(
            "before:",
            f"no_camera={before_no_camera}",
            f"mean={statistics.mean(before_lengths):.1f}",
            f"median={statistics.median(before_lengths):.1f}",
        )
        print(
            "after: ",
            f"no_camera={after_no_camera}",
            f"mean={statistics.mean(after_lengths):.1f}",
            f"median={statistics.median(after_lengths):.1f}",
        )
        print("after added target counts:", sorted(target_counts.items(), key=lambda x: x[0]))

        if not args.dry_run:
            enhanced_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        global_changed += changed
        global_added_changed += len(added_indices)

    print("total_changed", global_changed)
    print("total_added_camera_prompts", global_added_changed)


if __name__ == "__main__":
    main()
