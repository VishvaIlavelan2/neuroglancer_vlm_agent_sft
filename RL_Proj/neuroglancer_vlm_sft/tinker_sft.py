
# fine tuning done with tinker.
# essentially only implemented functionality corresponds to trajectory collection / data formatting
# all finetuning logic handled by tinker

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import chz
import datasets
from tinker_cookbook import hyperparam_utils
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.model_info import get_recommended_renderer_names
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
    SupervisedDatasetBuilder,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_file_uri(path_or_uri: str) -> str:
    parsed = urlparse(path_or_uri)
    if parsed.scheme and not (len(parsed.scheme) == 1 and path_or_uri[1:3] == ":\\"):
        return path_or_uri
    return Path(path_or_uri).expanduser().resolve().as_uri()


def normalize_messages(messages: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            normalized.append(message)
            continue

        normalized_parts = []
        for part in content:
            if part.get("type") == "image":
                normalized_parts.append({**part, "image": to_file_uri(part["image"])})
            else:
                normalized_parts.append(part)
        normalized.append({**message, "content": normalized_parts})
    return normalized


def infer_renderer_name(model_name: str) -> str:
    short_name = model_name.split("/", 1)[-1]
    if "Qwen3-VL" in short_name:
        return "qwen3_vl_instruct" if "Instruct" in short_name else "qwen3_vl"
    if "Qwen3.5" in short_name:
        return "qwen3_5"
    if "Qwen3" in short_name:
        return "qwen3_instruct" if "Instruct" in short_name else "qwen3"
    raise ValueError(f"Could not infer a renderer for {model_name}.")


def resolve_renderer_name(model_name: str) -> str:
    try:
        return list(get_recommended_renderer_names(model_name))[0]
    except Exception:
        return infer_renderer_name(model_name)


def build_renderer(model_name: str, renderer_name: str):
    return get_renderer(
        renderer_name,
        get_tokenizer(model_name),
        image_processor=get_image_processor(model_name),
        model_name=model_name,
    )


@chz.chz
class NeuroglancerConversationDatasetBuilder(SupervisedDatasetBuilder):
    common_config: ChatDatasetBuilderCommonConfig
    file_path: str
    test_size: int = 0
    shuffle_seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        conversations = load_jsonl(Path(self.file_path))
        # Arrow cannot store mixed string/list message content cleanly, so store each
        # conversation as one JSON string and decode it right before rendering.
        hf_dataset = datasets.Dataset.from_list(
            [{"messages_json": json.dumps(row["messages"], ensure_ascii=False)} for row in conversations]
        )

        renderer = build_renderer(
            model_name=self.common_config.model_name_for_tokenizer,
            renderer_name=self.common_config.renderer_name,
        )
        train_on_what = self.common_config.train_on_what or TrainOnWhat.LAST_ASSISTANT_MESSAGE

        def row_to_datum(row: dict):
            messages = normalize_messages(json.loads(row["messages_json"]))
            return conversation_to_datum(
                messages,
                renderer,
                max_length=self.common_config.max_length,
                train_on_what=train_on_what,
            )

        if self.test_size:
            split = hf_dataset.train_test_split(test_size=self.test_size, seed=self.shuffle_seed)
            train_split = split["train"]
            eval_split = split["test"]
        else:
            train_split = hf_dataset
            eval_split = None

        train_ds = SupervisedDatasetFromHFDataset(
            train_split,
            batch_size=self.common_config.batch_size,
            map_fn=row_to_datum,
        )
        eval_ds = (
            SupervisedDatasetFromHFDataset(
                eval_split,
                batch_size=self.common_config.batch_size,
                map_fn=row_to_datum,
            )
            if eval_split is not None
            else None
        )
        return train_ds, eval_ds


def get_recommended_lora_lr(model_name: str) -> float:
    return float(hyperparam_utils.get_lr(model_name, is_lora=True))
