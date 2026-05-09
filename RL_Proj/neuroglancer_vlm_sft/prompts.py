# fine tuning done with tinker.
# essentially only implemented functionality corresponds to trajectory collection / data formatting
# all finetuning logic handled by tinker

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neuroglancer_vlm_agent.shared_prompts import (
    STEP_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)


DEFAULT_PROMPT_SOURCE = (
    Path(__file__).resolve().parent.parent
    / "neuroglancer_vlm_agent"
    / "shared_prompts.py"
)


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    step_prompt_template: str
    source_path: Path


def load_runtime_prompts() -> PromptBundle:
    return PromptBundle(
        system_prompt=SYSTEM_PROMPT,
        step_prompt_template=STEP_PROMPT_TEMPLATE,
        source_path=DEFAULT_PROMPT_SOURCE.resolve(),
    )
