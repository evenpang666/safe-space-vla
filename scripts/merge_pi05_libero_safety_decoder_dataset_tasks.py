#!/usr/bin/env python3
"""Merge per-task PI05 LIBERO safety decoder dataset shards."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_pi05_libero_safety_decoder_dataset import (  # noqa: E402
    DEFAULT_OUTPUT,
    TASK_SUITE_MAX_STEPS,
    merge_dataset_shards,
    resolve_task_shard_dir,
    task_shard_paths,
)


TASK_SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-suite", default="libero_spatial", choices=sorted(TASK_SUITE_MAX_STEPS))
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=["all"],
        help="Task ids to merge. Use 'all' to merge the whole task suite.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing per-task shards. Defaults to <output stem>_tasks next to --output.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Merged output .npz path.")
    return parser.parse_args()


def resolve_merge_task_ids(*, task_suite: str, task_ids: list[str] | tuple[str, ...]) -> list[int]:
    raw_ids = [str(item) for item in task_ids]
    lowered = [item.lower() for item in raw_ids]
    if "all" in lowered:
        if len(raw_ids) != 1:
            raise ValueError("--task-ids all cannot be combined with explicit task ids")
        try:
            return list(range(TASK_SUITE_TASK_COUNTS[task_suite]))
        except KeyError as exc:
            raise ValueError(f"Unknown task suite: {task_suite}") from exc
    try:
        resolved = [int(item) for item in raw_ids]
    except ValueError as exc:
        raise ValueError("--task-ids entries must be integers or 'all'") from exc
    max_task_id = TASK_SUITE_TASK_COUNTS[task_suite] - 1
    for task_id in resolved:
        if not 0 <= task_id <= max_task_id:
            raise ValueError(f"task id {task_id} must be in [0, {max_task_id}]")
    if len(set(resolved)) != len(resolved):
        raise ValueError("--task-ids must not contain duplicates")
    return resolved


def main() -> None:
    args = parse_args()
    task_ids = resolve_merge_task_ids(task_suite=args.task_suite, task_ids=args.task_ids)
    shard_dir = args.input_dir or resolve_task_shard_dir(output=args.output, per_task_output_dir=None)
    shard_paths = task_shard_paths(shard_dir=shard_dir, task_suite=args.task_suite, task_ids=task_ids)
    merged_count = merge_dataset_shards(args.output, shard_paths)
    print(f"[done] merged {merged_count} samples from {len(shard_paths)} task shards to {args.output}")


if __name__ == "__main__":
    main()
