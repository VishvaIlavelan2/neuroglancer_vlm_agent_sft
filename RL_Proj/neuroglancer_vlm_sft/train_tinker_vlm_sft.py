# fine tuning done with tinker.
# essentially only implemented functionality corresponds to trajectory collection / data formatting
# all finetuning logic handled by tinker



from __future__ import annotations

import argparse
import asyncio

from pathlib import Path

from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from .tinker_sft import (
    NeuroglancerConversationDatasetBuilder,
    get_recommended_lora_lr,
    resolve_renderer_name,
)
from .trajectory_dataset import DEFAULT_OUTPUT_PATH

QWEN3_MAX_LENGTH = 262_144


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run supervised VLM finetuning on Neuroglancer trajectories with Tinker."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Tinker/HuggingFace base model name for the VLM you want to finetune.",
    )
    parser.add_argument(
        "--conversations-file",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Conversation JSONL produced by build_tinker_sft_dataset.py.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("neuroglancer_vlm_sft") / "training_runs" / "default_run",
        help="Local or cloud log path for Tinker outputs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="LoRA learning rate. Defaults to Tinker's recommended value for the model.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--eval-holdout-size",
        type=int,
        default=0,
        help="Number of examples to hold out for eval. Use 0 to disable.",
    )
    parser.add_argument("--save-every", type=int, default=150)
    parser.add_argument("--eval-every", type=int, default=150)
    return parser.parse_args()


def count_examples(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_eval_holdout_size(total_examples: int, requested_holdout: int, batch_size: int) -> int:
    requested_holdout = max(requested_holdout, 0)
    max_holdout = max(total_examples - batch_size, 0)
    clamped_holdout = min(requested_holdout, max_holdout)
    return (clamped_holdout // batch_size) * batch_size


def main() -> None:
    args = parse_args()

    conversations_file = args.conversations_file.resolve()
    log_path = args.log_path.resolve()
    if not conversations_file.exists():
        raise FileNotFoundError(
            f"Conversation file not found: {conversations_file}. "
            "Run build_tinker_sft_dataset.py first."
        )

    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")

    total_examples = count_examples(conversations_file)
    if total_examples < args.batch_size:
        raise ValueError(
            f"Need at least {args.batch_size} examples for one full training batch, "
            f"but found {total_examples} in {conversations_file}."
        )

    eval_examples = resolve_eval_holdout_size(
        total_examples=total_examples,
        requested_holdout=args.eval_holdout_size,
        batch_size=args.batch_size,
    )
    train_examples = total_examples - eval_examples
    effective_train_examples = (train_examples // args.batch_size) * args.batch_size
    effective_eval_examples = (eval_examples // args.batch_size) * args.batch_size
    dropped_train_examples = train_examples - effective_train_examples
    dropped_eval_examples = eval_examples - effective_eval_examples

    renderer_name = resolve_renderer_name(args.model_name)
    learning_rate = args.learning_rate or get_recommended_lora_lr(args.model_name)

    dataset_builder = NeuroglancerConversationDatasetBuilder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=args.model_name,
            renderer_name=renderer_name,
            max_length=QWEN3_MAX_LENGTH,
            batch_size=args.batch_size,
        ),
        file_path=str(conversations_file),
        test_size=eval_examples,
        shuffle_seed=0,
    )

    config = train.Config(
        log_path=str(log_path),
        model_name=args.model_name,
        dataset_builder=dataset_builder,
        learning_rate=learning_rate,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
    )

    print(f"Model:        {args.model_name}")
    print(f"Renderer:     {renderer_name}")
    print(f"Data:         {conversations_file}")
    print(f"Log path:     {log_path}")
    print(f"Learning rate:{learning_rate}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Max length:   {QWEN3_MAX_LENGTH}")
    print(f"LoRA rank:    {args.lora_rank}")
    print(f"Eval holdout: requested={args.eval_holdout_size}, actual={eval_examples}")
    print(f"Train size:   split={train_examples}, used={effective_train_examples}")
    print(f"Eval size:    split={eval_examples}, used={effective_eval_examples}")
    if dropped_train_examples or dropped_eval_examples:
        print(
            "Dropped by batching: "
            f"train={dropped_train_examples}, eval={dropped_eval_examples}"
        )

    asyncio.run(train.main(config))

    checkpoints_path = log_path / "checkpoints.jsonl"
    if checkpoints_path.exists():
        print(f"Checkpoint log: {checkpoints_path}")
    else:
        print(
            "Checkpoint log: unavailable. "
            "Tinker only writes checkpoints.jsonl after a checkpoint save event."
        )


if __name__ == "__main__":
    main()
