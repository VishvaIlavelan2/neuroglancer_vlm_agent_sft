# fine tuning done with tinker.
# essentially only implemented functionality corresponds to trajectory collection / data formatting
# all finetuning logic handled by tinker

from __future__ import annotations

import argparse
from pathlib import Path

from .trajectory_dataset import (
    DEFAULT_INFO_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH,
    build_conversations,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the compact-state Tinker conversation dataset from the processed trajectory manifest."
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--info-path", type=Path, default=DEFAULT_INFO_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conversations, info = build_conversations(args.manifest_path.resolve())

    write_jsonl(args.output_path.resolve(), conversations)
    write_json(args.info_path.resolve(), info)

    print(f"Wrote {len(conversations)} conversations to {args.output_path.resolve()}")
    print(f"Wrote dataset info to {args.info_path.resolve()}")


if __name__ == "__main__":
    main()
