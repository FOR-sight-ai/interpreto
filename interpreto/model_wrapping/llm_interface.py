# MIT License
#
# Copyright (c) 2025 IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL and FOR are research programs operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMInterface(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **generation_kwargs) -> str | None:
        pass

    @abstractmethod
    def batch_generate(self, system_prompt: str, user_prompts: list[str], **generation_kwargs) -> list[str | None]:
        pass


class OpenAILLM(LLMInterface):
    def __init__(self, api_key: str, model: str = "gpt-4.1-nano", default_generation_kwargs: dict | None = None):
        try:
            import openai  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("Install openai to use OpenAI API.") from e

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.default_generation_kwargs = default_generation_kwargs or {
            "temperature": 0.0,
            "max_output_tokens": 20,
        }

    def generate(self, system_prompt: str, user_prompt: str, **generation_kwargs) -> str | None:
        kwargs = {**self.default_generation_kwargs, **generation_kwargs}
        try:
            response = self.client.responses.create(
                model=self.model,
                prompt_cache_key="shared-system-prompt-v1",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs,
            )
            return response.output_text
        except Exception:
            return None

    def batch_generate(self, system_prompt: str, user_prompts: list[str], **generation_kwargs) -> list[str | None]:
        return [self.generate(system_prompt, p, **generation_kwargs) for p in user_prompts]


class HuggingFaceLLM(LLMInterface):
    def __init__(self, model: str, batch_size: int = 8, device: str = "auto"):
        self.model_name = model
        self.batch_size = batch_size
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model,
        #     torch_dtype="auto",
        #     device_map=device,
        # )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                dtype="auto",
                device_map=device,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype="auto",
                device_map=device,
            )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if device == "auto":
            # model_device = getattr(self.model, "device", None)
            model_device = self._resolve_model_device()
            if model_device is None:
                model_device = next(self.model.parameters()).device
            self._inputs_device = model_device
        else:
            self._inputs_device = torch.device(device)

    def _resolve_model_device(self) -> torch.device | None:
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for target in hf_device_map.values():
                if isinstance(target, int):
                    return torch.device(f"cuda:{target}")
                if isinstance(target, str):
                    if target in {"disk", "meta"}:
                        continue
                    return torch.device(target)

        model_device = getattr(self.model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            return torch.device(model_device)

        return None

    def _compute_tokenizer_max_length(self, generation_kwargs: dict) -> int | None:
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if max_positions is None:
            return None

        requested_new_tokens = generation_kwargs.get("max_new_tokens")
        if requested_new_tokens is None:
            requested_new_tokens = 32

        safe_input_length = max_positions - int(requested_new_tokens)
        if safe_input_length <= 0:
            return max_positions
        return safe_input_length

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # return self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"

    def generate(self, system_prompt: str, user_prompt: str, **generation_kwargs) -> str | None:
        return self.batch_generate(system_prompt, [user_prompt], **generation_kwargs)[0]

    def batch_generate(
        self,
        system_prompt: str,
        user_prompts: list[str],
        **generation_kwargs,
    ) -> list[str | None]:
        formatted_prompts = [self._format_prompt(system_prompt, p) for p in user_prompts]
        outputs: list[str | None] = []

        for i in range(0, len(formatted_prompts), self.batch_size):
            batch_prompts = formatted_prompts[i : i + self.batch_size]

            try:
                tokenizer_max_length = self._compute_tokenizer_max_length(generation_kwargs)
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=tokenizer_max_length,
                ).to(self._inputs_device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        pad_token_id=self.tokenizer.pad_token_id,
                        **generation_kwargs,
                    )

                input_lengths = inputs["attention_mask"].sum(dim=1)

                for j, output_ids in enumerate(generated_ids):
                    new_tokens = output_ids[input_lengths[j] :]
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    outputs.append(text)
            except Exception:
                outputs.extend([None] * len(batch_prompts))

        return outputs
