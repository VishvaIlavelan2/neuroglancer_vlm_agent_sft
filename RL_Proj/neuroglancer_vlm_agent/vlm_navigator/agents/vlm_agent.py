"""
VLM Agent for Neuroglancer navigation.

Uses the original LiteLLM-style interface for standard models and adds the
minimal Tinker-backed path needed for Qwen comparison runs.
"""

import base64
import io
import os

from PIL import Image

from vlm_navigator.utils.action_utils import parse_vlm_response, vlm_json_to_action_vector


class VLMAgent:
    def __init__(self, model_config: dict, action_mode: str = "position_only"):
        """
        Args:
            model_config: Dict with keys:
                - "model": model string
                - "backend": "litellm" (default) or "tinker"
                - "max_tokens": Max tokens for VLM response (default 300)
                - "temperature": Sampling temperature (default 0.0)
                - "history_mode": "accumulate" (default) or "single_turn"
            action_mode: "position_only" (Mode B) or "full" (Mode A)
        """
        self.backend = model_config.get("backend", "litellm")
        self.model = model_config["model"]
        self.max_tokens = model_config.get("max_tokens", 300)
        self.temperature = model_config.get("temperature", 0.0)
        self.top_p = model_config.get("top_p", 1.0)
        self.top_k = model_config.get("top_k", -1)
        self.action_mode = action_mode
        self.base_model = model_config.get("base_model", self.model)
        self.model_path = model_config.get("model_path")
        self.base_url = model_config.get("base_url")
        self.renderer_name = model_config.get("renderer_name")
        self.history_mode = model_config.get("history_mode", "accumulate")

        self.messages = []
        self.step_count = 0
        self.parse_failures = 0

        self._litellm_completion = None
        self._tinker = None
        self._tinker_sampling_client = None
        self._tinker_renderer = None
        self._tinker_stop_sequences = None
        self._format_content_as_string = None

        if self.backend == "litellm":
            try:
                from litellm import completion
            except ImportError as exc:
                raise RuntimeError(
                    "LiteLLM is required for backend='litellm'. "
                    "Install litellm or use a Tinker-backed model preset."
                ) from exc
            self._litellm_completion = completion
        elif self.backend == "tinker":
            self._init_tinker()
        else:
            raise ValueError(f"Unsupported backend '{self.backend}'")

        if self.history_mode not in {"accumulate", "single_turn"}:
            raise ValueError(
                "history_mode must be 'accumulate' or 'single_turn', "
                f"got {self.history_mode!r}"
            )

    def reset(self, system_prompt: str):
        """Reset agent state for a new episode."""
        self.messages = [{"role": "system", "content": system_prompt}]
        self.step_count = 0
        self.parse_failures = 0

    def get_action(
        self,
        screenshot: Image.Image,
        position: list,
        orientation: list,
        cross_section_scale: float,
        projection_scale: float,
        prev_z_delta: float = 0.0,
        step_prompt_template: str = None,
    ) -> tuple[list, dict, str]:
        """Run one step: send screenshot + state to VLM, return action vector."""
        self.step_count += 1

        if step_prompt_template:
            step_text = step_prompt_template
        else:
            step_text = (
                f"Step {self.step_count}. "
                f"Position: {position}, Orientation: {orientation}, "
                f"CrossSectionScale: {cross_section_scale:.4f}, "
                f"ProjectionScale: {projection_scale:.2f}. "
                f"Last Z delta: {prev_z_delta:.1f}. "
                f"Current Z: {position[2]:.1f}. "
                f"Respond with a JSON action."
            )

        self._prepare_messages_for_step(self._build_user_message(step_text, screenshot))
        raw_text, assistant_message = self._complete()
        self.messages.append(assistant_message)

        parsed = parse_vlm_response(raw_text)
        if parsed == {"delta_x": 0, "delta_y": 0, "delta_z": 5}:
            self.parse_failures += 1

        action_vector = vlm_json_to_action_vector(parsed, mode=self.action_mode)
        return action_vector, parsed, raw_text

    def trim_history(self, keep_last_n: int = 10):
        """Trim conversation history to avoid context overflow."""
        if len(self.messages) <= 1:
            return
        non_system = self.messages[1:]
        if len(non_system) > keep_last_n * 2:
            self.messages = [self.messages[0]] + non_system[-(keep_last_n * 2):]

    def _prepare_messages_for_step(self, user_message: dict) -> None:
        if self.history_mode == "single_turn" and self.messages:
            self.messages = [self.messages[0]]
        self.messages.append(user_message)

    def _build_user_message(self, step_text: str, screenshot: Image.Image) -> dict:
        if self.backend == "tinker":
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": step_text},
                    {"type": "image", "image": self._resize_image(screenshot)},
                ],
            }

        image_b64 = self._encode_image(screenshot)
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": step_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }

    def _complete(self) -> tuple[str, dict]:
        if self.backend == "tinker":
            return self._complete_with_tinker()
        return self._complete_with_litellm()

    def _complete_with_litellm(self) -> tuple[str, dict]:
        is_reasoning = any(tag in self.model for tag in ["gpt-5", "o1", "o3"])
        call_kwargs = dict(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            timeout=60.0,
        )
        if is_reasoning:
            call_kwargs["max_completion_tokens"] = self.max_tokens
        else:
            call_kwargs["max_tokens"] = self.max_tokens

        response = self._litellm_completion(**call_kwargs)
        msg = response.choices[0].message
        raw_text = msg.content or getattr(msg, "reasoning_content", None) or ""
        return raw_text, {"role": "assistant", "content": raw_text}

    def _complete_with_tinker(self) -> tuple[str, dict]:
        sampling_params = self._tinker.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop=self._tinker_stop_sequences,
        )
        model_input = self._tinker_renderer.build_generation_prompt(self.messages)
        response = self._tinker_sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()
        parsed_message, _ = self._tinker_renderer.parse_response(response.sequences[0].tokens)
        raw_text = self._format_content_as_string(parsed_message["content"])

        assistant_message = {
            "role": parsed_message.get("role", "assistant"),
            "content": parsed_message["content"],
        }
        if "tool_calls" in parsed_message:
            assistant_message["tool_calls"] = parsed_message["tool_calls"]
        return raw_text, assistant_message

    def _init_tinker(self):
        os.environ.setdefault("TINKER_SUBPROCESS_SAMPLING", "1")

        try:
            import tinker
            from tinker_cookbook.image_processing_utils import get_image_processor
            from tinker_cookbook.model_info import get_recommended_renderer_names
            from tinker_cookbook.renderers import format_content_as_string, get_renderer
        except ImportError as exc:
            raise RuntimeError(
                "Tinker dependencies are required for backend='tinker'. "
                "Install tinker and tinker-cookbook in the active environment."
            ) from exc

        service_kwargs = {}
        if self.base_url:
            service_kwargs["base_url"] = self.base_url
        service_client = tinker.ServiceClient(**service_kwargs)
        sampling_client = service_client.create_sampling_client(
            model_path=self.model_path,
            base_model=self.base_model,
        )

        resolved_base_model = self.base_model or sampling_client.get_base_model()
        renderer_name = self.renderer_name
        if renderer_name is None:
            try:
                renderer_name = list(get_recommended_renderer_names(resolved_base_model))[0]
            except Exception:
                renderer_name = self._infer_tinker_renderer_name(resolved_base_model)

        tokenizer = sampling_client.get_tokenizer()
        image_processor = get_image_processor(resolved_base_model)

        self._tinker = tinker
        self._tinker_sampling_client = sampling_client
        self._tinker_renderer = get_renderer(
            renderer_name,
            tokenizer,
            image_processor=image_processor,
            model_name=resolved_base_model,
        )
        self._tinker_stop_sequences = self._tinker_renderer.get_stop_sequences()
        self._format_content_as_string = format_content_as_string
        self.renderer_name = renderer_name
        self.model = resolved_base_model

    @staticmethod
    def _infer_tinker_renderer_name(model_name: str) -> str:
        short_name = model_name.split("/", 1)[-1]
        if "Qwen3-VL" in short_name:
            return "qwen3_vl_instruct" if "Instruct" in short_name else "qwen3_vl"
        if "Qwen3.5" in short_name:
            return "qwen3_5"
        if "Qwen3" in short_name:
            return "qwen3_instruct" if "Instruct" in short_name else "qwen3"
        raise ValueError(f"Could not infer a Tinker renderer for {model_name}.")

    @staticmethod
    def _resize_image(image: Image.Image, max_size: tuple = (960, 540)) -> Image.Image:
        if image.size != max_size:
            image = image.resize(max_size, Image.LANCZOS)
        return image

    @staticmethod
    def _encode_image(image: Image.Image, max_size: tuple = (960, 540)) -> str:
        """Resize and encode a PIL Image as base64 JPEG string."""
        if image.size != max_size:
            image = image.resize(max_size, Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
