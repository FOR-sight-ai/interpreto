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
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from interpreto.concepts.splitters import BaseSplitter


class LLMInterface(ABC):
    """Generate a response from a system prompt and a user prompt."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **generation_kwargs: Any) -> str:
        """Generate one response."""

    def batch_generate(self, system_prompt: str, user_prompts: list[str], **generation_kwargs: Any) -> list[str]:
        """Sequential fallback for backends without native batching."""
        return [self.generate(system_prompt, prompt, **generation_kwargs) for prompt in user_prompts]


def _render_prompt(
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
    user_prompt: str,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Render a system/user pair using the tokenizer's chat template when available."""
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )  # type: ignore
    return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"


def _decode_responses(tokenizer: PreTrainedTokenizerBase, generated: Any, prompt_length: int) -> list[str]:
    """Decode continuations; prompt_length is zero for encoder-decoder models."""
    sequences = getattr(generated, "sequences", generated)
    return [text.strip() for text in tokenizer.batch_decode(sequences[:, prompt_length:], skip_special_tokens=True)]


class HuggingFaceLLM(LLMInterface):
    """Generate locally from a repository ID or a preloaded model and tokenizer.

    Models are put in eval mode. Tokenizer padding settings are configured in
    place and left that way. Preloaded models require an explicit tokenizer.
    ``device`` applies only to loading; kwargs set overridable generation defaults.
    """

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        batch_size: int = 8,
        device: str = "cuda",
        generation_kwargs: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tqdm_bar: bool = False,
    ) -> None:
        if isinstance(model_or_repo_id, str):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_or_repo_id,
                torch_dtype="auto",
                device_map=device,
            )
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_or_repo_id)
        else:
            self.model = model_or_repo_id

        if tokenizer is None:
            raise ValueError("Provide a tokenizer with a preloaded model.")

        self.model.eval()
        self.tokenizer = tokenizer

        if not self.model.config.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs or {}
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.tqdm_bar = tqdm_bar

    def generate(self, system_prompt: str, user_prompt: str, **generation_kwargs: Any) -> str:
        return self.batch_generate(system_prompt, [user_prompt], **generation_kwargs)[0]

    def batch_generate(self, system_prompt: str, user_prompts: list[str], **generation_kwargs: Any) -> list[str]:
        """Generate in batches, preserving prompt order."""
        prompts = [
            _render_prompt(self.tokenizer, system_prompt, prompt, self.chat_template_kwargs) for prompt in user_prompts
        ]
        kwargs = {"pad_token_id": self.tokenizer.pad_token_id, **self.generation_kwargs, **generation_kwargs}
        responses: list[str] = []
        for start in tqdm(range(0, len(prompts), self.batch_size), desc="Generating", disable=not self.tqdm_bar):
            inputs = self.tokenizer(
                prompts[start : start + self.batch_size],
                padding=True,
                return_tensors="pt",
                add_special_tokens=not bool(self.tokenizer.chat_template),
            ).to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(**inputs, **kwargs)  # type: ignore
            prompt_length = 0 if self.model.config.is_encoder_decoder else inputs["input_ids"].shape[1]
            responses.extend(_decode_responses(self.tokenizer, generated, prompt_length))
        return responses


class _SplitterLLM(LLMInterface):
    """Reuse a splitter for batched generation, dispatching weights only when needed.

    The model is left in eval mode and tokenizer padding is configured in place.
    ``batch_size`` controls generation batches independently of the splitter's
    activation-extraction batch size.
    """

    def __init__(
        self,
        splitter: BaseSplitter,
        *,
        generation_kwargs: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.splitter = splitter
        self.tokenizer = splitter.tokenizer
        # if not splitter.pipeline.model.config.is_encoder_decoder:
        #     self.tokenizer.padding_side = "left"
        # if self.tokenizer.pad_token_id is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        self.batch_size = splitter.batch_size
        self.generation_kwargs = {"max_new_tokens": 200, **(generation_kwargs or {})}
        self.chat_template_kwargs = {"enable_thinking": False, **(chat_template_kwargs or {})}

    def generate(self, system_prompt: str, user_prompt: str, **generation_kwargs: Any) -> str:
        return self.batch_generate(system_prompt, [user_prompt], **generation_kwargs)[0]

    def batch_generate(self, system_prompt: str, user_prompts: list[str], **generation_kwargs: Any) -> list[str]:
        """Generate in batches through the splitter, preserving prompt order."""
        prompts = [
            _render_prompt(self.tokenizer, system_prompt, prompt, self.chat_template_kwargs) for prompt in user_prompts
        ]
        kwargs = {"pad_token_id": self.tokenizer.pad_token_id, **self.generation_kwargs, **generation_kwargs}
        responses: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            inputs = self.tokenizer(
                prompts[start : start + self.batch_size],
                padding=True,
                return_tensors="pt",
                add_special_tokens=not bool(self.tokenizer.chat_template),
            )
            with torch.no_grad():
                self.splitter.dispatch()
                self.splitter.pipeline.model.eval()
                generated = self.splitter.generate(inputs, **kwargs)
            prompt_length = (
                0 if self.splitter.pipeline.model.config.is_encoder_decoder else inputs["input_ids"].shape[1]
            )
            responses.extend(_decode_responses(self.tokenizer, generated, prompt_length))
        return responses


def _resolve_llm_interface(
    llm_interface: str | tuple[PreTrainedModel, PreTrainedTokenizerBase] | LLMInterface | None,
    *,
    fallback_model: BaseSplitter | None = None,
) -> LLMInterface:
    """Resolve an adapter, repository ID, (model, tokenizer), or splitter fallback."""
    if isinstance(llm_interface, LLMInterface):
        return llm_interface
    if isinstance(llm_interface, str):
        return HuggingFaceLLM(
            llm_interface, generation_kwargs={"max_new_tokens": 200}, chat_template_kwargs={"enable_thinking": False}
        )
    if isinstance(llm_interface, tuple):
        model, tokenizer = llm_interface
        return HuggingFaceLLM(model, tokenizer=tokenizer)
    if llm_interface is None:
        if fallback_model is None or not fallback_model.pipeline.model.can_generate():
            raise ValueError("The splitter cannot generate text. Provide llm_interface explicitly.")
        return _SplitterLLM(fallback_model)
    raise TypeError("Expected a repository ID, (model, tokenizer), LLMInterface, or None.")


class OpenAILLM(LLMInterface):
    """Generate with Chat Completions; num_try maps to SDK max_retries + 1."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-nano",
        num_try: int = 5,
        **generation_kwargs: Any,
    ) -> None:
        import openai  # noqa: PLC0415  # type: ignore

        self.client = openai.OpenAI(api_key=api_key, max_retries=num_try - 1)
        self.model = model
        self.generation_kwargs = generation_kwargs

    def generate(self, system_prompt: str, user_prompt: str, **generation_kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **{**self.generation_kwargs, **generation_kwargs},
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned no text.")
        return content.strip()
