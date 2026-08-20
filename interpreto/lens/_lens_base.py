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

"""Shared implementation for lens methods."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from numbers import Integral
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

from interpreto.concepts.splitters.model_with_split_points import ModelWithSplitPoints
from interpreto.typing import LensResults, LensTopKOutput

LensInputs = str | list[str] | BatchEncoding
LensTask = Literal["language_model", "sequence_classification"]
PoolingStrategy = Literal["cls", "mean", "last"]
LanguageModelMode = Literal["causal", "masked"]
LensTargets = int | list[int] | torch.Tensor | None
_SUPERVISION_INPUT_KEYS = {
    "end_positions",
    "label",
    "label_ids",
    "labels",
    "next_sentence_label",
    "sentence_order_label",
    "start_positions",
}


def _slice_batch_encoding(inputs: BatchEncoding, start_index: int, end_index: int) -> BatchEncoding:
    return BatchEncoding(
        {
            key: value[start_index:end_index] if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
    )


@dataclass(frozen=True)
class ProjectionSpec:
    """Modules used to turn a split activation into model logits."""

    head_name: str
    pre_head_name: str | None = None
    pooling_strategy: PoolingStrategy | None = None


class BaseLens:
    """Capture one split activation and project it through the model output head."""

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        head_name: str | None = None,
        pre_head_name: str | None = None,
        pooling_strategy: PoolingStrategy | None = None,
        top_k: int = 5,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(model_with_split_points, ModelWithSplitPoints):
            raise TypeError(
                f"`model_with_split_points` must be a ModelWithSplitPoints, got {type(model_with_split_points)}."
            )
        if model_with_split_points.tokenizer is None:
            raise ValueError("The wrapped model must expose a tokenizer.")

        self._validate_positive_integer(top_k, "top_k")
        self._validate_positive_integer(model_with_split_points.batch_size, "model_with_split_points.batch_size")

        self.model_with_split_points = model_with_split_points
        self.model = model_with_split_points._model
        self.split_point = model_with_split_points.split_point
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = model_with_split_points.tokenizer
        self._synchronize_padding_token()
        self.task: LensTask = self._infer_task()
        self._validate_task_configuration()
        self.language_model_mode: LanguageModelMode | None = (
            self._infer_language_model_mode() if self.task == "language_model" else None
        )
        self.top_k = int(top_k)
        self.model_forward_keys = set(inspect.signature(self.model.forward).parameters)
        self.preferred_padding_side = self._get_preferred_padding_side()

        projection = self._resolve_projection_spec(head_name, pre_head_name, pooling_strategy)
        self.head_name = projection.head_name
        self.pre_head_name = projection.pre_head_name
        self.pooling_strategy = projection.pooling_strategy
        self.model_head, self.model_pre_head = self._resolve_projection_modules()
        self._validate_projection()

        projection_module = self.model_pre_head if self.model_pre_head is not None else self.model_head
        projection_device, _ = self._get_module_placement(projection_module, self.input_device)
        self.device = torch.device(device) if device is not None else projection_device

    @staticmethod
    def _validate_positive_integer(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"`{name}` must be an integer.")
        if value < 1:
            raise ValueError(f"`{name}` must be a strictly positive integer.")

    def _infer_task(self) -> LensTask:
        model_class_name = type(self.model).__name__
        if "ForTokenClassification" in model_class_name:
            raise NotImplementedError("Token classification is not supported by lens methods.")
        if "ForSequenceClassification" in model_class_name:
            return "sequence_classification"
        if "ForMaskedLM" in model_class_name or "ForCausalLM" in model_class_name:
            return "language_model"
        if model_class_name.endswith("LMHeadModel"):
            return "language_model"
        raise ValueError(
            "Unsupported model type for lens methods. Use a causal language model, masked language model, "
            f"or sequence-classification model with a compatible projection; got {model_class_name}."
        )

    def _infer_language_model_mode(self) -> LanguageModelMode:
        return "masked" if "MaskedLM" in type(self.model).__name__ else "causal"

    def _validate_task_configuration(self) -> None:
        if self.task == "language_model":
            self._validate_language_model_output_contract()
            return
        config = self.model.config
        problem_type = getattr(config, "problem_type", None)
        num_labels = getattr(config, "num_labels", None)
        invalid_label_count = isinstance(num_labels, bool) or not isinstance(num_labels, Integral) or num_labels < 2
        if problem_type in {"regression", "multi_label_classification"} or invalid_label_count:
            raise NotImplementedError("Lens methods only support single-label sequence classification.")

    def _validate_language_model_output_contract(self) -> None:
        config = self.model.config
        active_transforms = []
        for attribute_name in ("final_logit_softcapping", "logits_soft_cap"):
            if getattr(config, attribute_name, None) is not None:
                active_transforms.append(attribute_name)
        for attribute_name in ("logit_scale", "logits_scaling"):
            value = getattr(config, attribute_name, None)
            if value is not None and value != 1:
                active_transforms.append(attribute_name)
        if active_transforms:
            transform_names = ", ".join(f"`{name}`" for name in active_transforms)
            raise NotImplementedError(
                "Lens methods do not support functional transformations applied outside the language-model head: "
                f"{transform_names}."
            )

    def _get_preferred_padding_side(self) -> str:
        return "right"

    @property
    def input_device(self) -> torch.device:
        """Return the current input-embedding device."""
        return self._get_input_device()

    @property
    def model_device(self) -> torch.device:
        """Return the device on which model inputs should be placed."""
        return self.input_device

    def _get_input_device(self) -> torch.device:
        get_input_embeddings = getattr(self.model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            input_embeddings = get_input_embeddings()
            if isinstance(input_embeddings, nn.Module):
                device, _ = self._get_module_placement(input_embeddings, fallback=None)
                if device is not None:
                    return device

        for tensor in chain(self.model.parameters(), self.model.buffers()):
            if tensor.device.type != "meta":
                return tensor.device
        raise RuntimeError(
            "The wrapped model only exposes meta tensors. Build ModelWithSplitPoints from a fully loaded model."
        )

    @staticmethod
    def _get_module_placement(
        module: nn.Module,
        fallback: torch.device | None,
    ) -> tuple[torch.device | None, torch.dtype | None]:
        device: torch.device | None = None
        floating_dtype: torch.dtype | None = None
        for tensor in chain(module.parameters(), module.buffers()):
            if tensor.device.type == "meta":
                continue
            if device is None:
                device = tensor.device
            if floating_dtype is None and (tensor.is_floating_point() or tensor.is_complex()):
                floating_dtype = tensor.dtype
            if device is not None and floating_dtype is not None:
                break
        return device if device is not None else fallback, floating_dtype

    @contextmanager
    def _model_in_evaluation_mode(self) -> Iterator[None]:
        training_states = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            yield
        finally:
            for module, training in training_states:
                module.training = training

    def _validate_bound_split_point(self) -> None:
        self._validate_task_configuration()
        if self.model_with_split_points.split_point != self.split_point:
            raise RuntimeError(
                "The ModelWithSplitPoints split point changed after this lens was created. "
                "Create a new lens for the new split point."
            )

    def _resolve_projection_spec(
        self,
        head_name: str | None,
        pre_head_name: str | None,
        pooling_strategy: PoolingStrategy | None,
    ) -> ProjectionSpec:
        if pooling_strategy not in {None, "cls", "mean", "last"}:
            raise ValueError("`pooling_strategy` must be one of `None`, `'cls'`, `'mean'`, or `'last'`.")
        if self.task == "language_model" and pooling_strategy is not None:
            raise ValueError("`pooling_strategy` is only supported for sequence classification.")

        if head_name is not None:
            return ProjectionSpec(head_name, pre_head_name, pooling_strategy)

        inferred = (
            self._get_language_model_projection()
            if self.task == "language_model"
            else self._get_sequence_classification_projection()
        )
        if pre_head_name is None and pooling_strategy is None:
            return inferred

        # An explicit pooling strategy replaces an automatically inferred pooler.
        return ProjectionSpec(inferred.head_name, pre_head_name, pooling_strategy)

    def _resolve_projection_modules(self) -> tuple[nn.Module, nn.Module | None]:
        model_head = self._get_module(self.head_name)
        self._raise_on_meta_module(model_head, self.head_name)

        model_pre_head = None
        if self.pre_head_name is not None:
            model_pre_head = self._get_module(self.pre_head_name)
            self._raise_on_meta_module(model_pre_head, self.pre_head_name)
        return model_head, model_pre_head

    def _get_language_model_projection(self) -> ProjectionSpec:
        normalized_candidates = [
            ProjectionSpec("lm_head", "transformer.ln_f"),
            ProjectionSpec("lm_head", "model.norm"),
            ProjectionSpec("lm_head", "model.decoder.final_layer_norm"),
            ProjectionSpec("lm_head", "decoder.final_layer_norm"),
            ProjectionSpec("embed_out", "gpt_neox.final_layer_norm"),
            ProjectionSpec("embed_out", "model.norm"),
            ProjectionSpec("generator_lm_head", "generator_predictions"),
        ]
        valid_normalized_candidates = [
            candidate
            for candidate in normalized_candidates
            if self._module_exists(candidate.head_name)
            and candidate.pre_head_name is not None
            and self._module_exists(candidate.pre_head_name)
            and self._projection_candidate_is_complete(candidate)
        ]
        if len(valid_normalized_candidates) == 1:
            return valid_normalized_candidates[0]
        if len(valid_normalized_candidates) > 1:
            candidates = ", ".join(
                f"{candidate.pre_head_name} -> {candidate.head_name}" for candidate in valid_normalized_candidates
            )
            raise ValueError(
                "Several known language-model projections match this model. "
                f"Provide `head_name` and `pre_head_name` explicitly. Candidates: {candidates}."
            )

        head_candidates = ["lm_head", "embed_out", "cls", "generator_lm_head"]
        valid_head_candidates = [
            name
            for name in head_candidates
            if self._module_exists(name) and not isinstance(self._get_module(name), nn.Linear)
        ]
        if len(valid_head_candidates) == 1:
            return ProjectionSpec(valid_head_candidates[0])
        if not valid_head_candidates:
            raise ValueError(
                "Could not resolve a faithful language-model projection. "
                "Provide explicit compatible `head_name` and `pre_head_name` paths."
            )
        raise ValueError(
            "Several language-model heads match this model. Provide `head_name` explicitly. "
            f"Candidates: {', '.join(valid_head_candidates)}."
        )

    def _projection_candidate_is_complete(self, candidate: ProjectionSpec) -> bool:
        if candidate.pre_head_name is None:
            return True
        parent_name, _, module_name = candidate.pre_head_name.rpartition(".")
        if module_name == "final_layer_norm" and self._module_exists(f"{parent_name}.project_out"):
            return False

        pre_head = self._get_module(candidate.pre_head_name)
        head = self._get_module(candidate.head_name)
        if isinstance(pre_head, nn.LayerNorm) and isinstance(head, nn.Linear):
            normalized_shape = pre_head.normalized_shape
            output_width = normalized_shape[-1] if isinstance(normalized_shape, tuple) else normalized_shape
            return output_width == head.in_features
        return True

    def _get_sequence_classification_projection(self) -> ProjectionSpec:
        if self._module_exists("classifier"):
            classifier = self._get_module("classifier")
            base_model_prefix = getattr(self.model, "base_model_prefix", None)
            pooler_name = f"{base_model_prefix}.pooler"
            if (
                isinstance(base_model_prefix, str)
                and self._module_exists(pooler_name)
                and not isinstance(self._get_module(pooler_name), nn.Linear)
            ):
                return ProjectionSpec("classifier", pooler_name)
            if not isinstance(classifier, nn.Linear):
                return ProjectionSpec("classifier")
        raise ValueError(
            "Could not resolve a faithful sequence-classification projection. "
            "Provide explicit `head_name`, `pre_head_name`, and `pooling_strategy` values."
        )

    def _validate_projection(self) -> None:
        if (
            self.task == "sequence_classification"
            and self.pooling_strategy is None
            and self.model_pre_head is None
            and isinstance(self.model_head, nn.Linear)
        ):
            raise ValueError(
                "The resolved classifier is a vector head and does not pool token states. "
                "Provide `pooling_strategy` as `'cls'`, `'mean'`, or `'last'`, or provide a compatible "
                "`pre_head_name`."
            )

    @staticmethod
    def _raise_on_meta_module(module: nn.Module, module_name: str) -> None:
        if any(tensor.device.type == "meta" for tensor in chain(module.parameters(), module.buffers())):
            raise RuntimeError(
                f"The projection module `{module_name}` contains meta tensors and cannot be executed directly."
            )

    def _module_exists(self, module_name: str) -> bool:
        try:
            self._get_module(module_name)
        except (AttributeError, IndexError, TypeError, ValueError):
            return False
        return True

    def _get_module(self, module_name: str) -> nn.Module:
        current: Any = self.model
        for path_element in module_name.split("."):
            current = current[int(path_element)] if path_element.isdigit() else getattr(current, path_element)
            if current is None:
                raise ValueError(f"Module `{module_name}` resolves to `None`.")
        if not isinstance(current, nn.Module):
            raise TypeError(f"Module `{module_name}` is not a torch.nn.Module.")
        return current

    def _synchronize_padding_token(self) -> bool:
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            return False

        pad_token_id = self.tokenizer.pad_token_id
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, Integral):
            raise TypeError("The tokenizer padding token id must be an integer.")
        pad_token_id = int(pad_token_id)
        vocab_size = getattr(self.model.config, "vocab_size", None)
        if pad_token_id < 0 or (isinstance(vocab_size, Integral) and pad_token_id >= vocab_size):
            raise ValueError("The tokenizer padding token id must belong to the model vocabulary.")
        self.model.config.pad_token_id = pad_token_id
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.pad_token_id = pad_token_id
        return True

    def _ensure_padding_token(self) -> None:
        if not self._synchronize_padding_token():
            raise ValueError(
                "Raw text inputs require a tokenizer with a padding or end-of-sequence token. "
                "Configure the tokenizer before using the lens."
            )

    def _validate_batch_encoding(self, inputs: BatchEncoding) -> BatchEncoding:
        if "input_ids" not in inputs:
            raise ValueError("BatchEncoding inputs must contain `input_ids`.")
        input_ids = inputs["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError('BatchEncoding inputs must be tensor-backed. Tokenize with `return_tensors="pt"`.')
        if input_ids.ndim != 2:
            raise ValueError("`input_ids` must have shape `(batch_size, sequence_length)`.")
        if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("Empty BatchEncoding inputs are not supported.")

        batch_size = input_ids.shape[0]
        for name, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(f"Tensor field `{name}` must share the `input_ids` batch dimension.")
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]
            if isinstance(attention_mask, torch.Tensor) and attention_mask.shape != input_ids.shape:
                raise ValueError("`attention_mask` must have the same shape as `input_ids`.")
        return inputs

    @staticmethod
    def _normalize_text_inputs(inputs: str | list[str]) -> list[str]:
        if isinstance(inputs, str):
            return [inputs]
        if not inputs:
            raise ValueError("Empty input lists are not supported.")
        if any(not isinstance(text, str) for text in inputs):
            raise TypeError("Input lists must only contain strings.")
        return inputs

    def _tokenize_texts(self, texts: list[str]) -> BatchEncoding:
        self._ensure_padding_token()
        original_padding_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = self.preferred_padding_side
            model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        finally:
            self.tokenizer.padding_side = original_padding_side
        return self._validate_batch_encoding(model_inputs)

    def _prepare_inputs(self, inputs: LensInputs) -> BatchEncoding:
        if isinstance(inputs, BatchEncoding):
            return self._validate_batch_encoding(inputs)
        if not isinstance(inputs, str | list):
            raise TypeError(f"Expected `str`, `list[str]`, or `BatchEncoding` inputs, got {type(inputs)}.")
        return self._tokenize_texts(self._normalize_text_inputs(inputs))

    def _get_input_size(self, inputs: LensInputs) -> int:
        if isinstance(inputs, BatchEncoding):
            return self._validate_batch_encoding(inputs)["input_ids"].shape[0]
        if not isinstance(inputs, str | list):
            raise TypeError(f"Expected `str`, `list[str]`, or `BatchEncoding` inputs, got {type(inputs)}.")
        return len(self._normalize_text_inputs(inputs))

    def _prepare_model_forward_inputs(self, inputs: BatchEncoding) -> dict[str, Any]:
        forward_inputs: dict[str, Any] = {
            key: value.to(self.input_device)
            for key, value in inputs.items()
            if key in self.model_forward_keys
            and key not in _SUPERVISION_INPUT_KEYS
            and isinstance(value, torch.Tensor)
        }
        if "attention_mask" in self.model_forward_keys and "attention_mask" not in forward_inputs:
            forward_inputs["attention_mask"] = torch.ones_like(forward_inputs["input_ids"])
        attention_mask = forward_inputs.get("attention_mask")
        if attention_mask is not None and torch.any(attention_mask[:, 1:] > attention_mask[:, :-1]):
            raise ValueError("BatchEncoding inputs must use right padding.")
        if "use_cache" in self.model_forward_keys:
            forward_inputs["use_cache"] = False
        return forward_inputs

    def _iter_prepared_batches(
        self,
        inputs: BatchEncoding,
        batch_size: int,
    ) -> Iterator[tuple[int, int, BatchEncoding]]:
        self._validate_positive_integer(batch_size, "batch_size")
        nb_samples = inputs["input_ids"].shape[0]
        for start_index in range(0, nb_samples, batch_size):
            end_index = min(start_index + batch_size, nb_samples)
            yield start_index, end_index, _slice_batch_encoding(inputs, start_index, end_index)

    def _iter_input_batches(
        self,
        inputs: LensInputs,
        batch_size: int,
    ) -> Iterator[tuple[int, int, BatchEncoding]]:
        self._validate_positive_integer(batch_size, "batch_size")
        if isinstance(inputs, BatchEncoding):
            yield from self._iter_prepared_batches(self._validate_batch_encoding(inputs), batch_size)
            return
        if not isinstance(inputs, str | list):
            raise TypeError(f"Expected `str`, `list[str]`, or `BatchEncoding` inputs, got {type(inputs)}.")

        texts = self._normalize_text_inputs(inputs)
        for start_index in range(0, len(texts), batch_size):
            end_index = min(start_index + batch_size, len(texts))
            yield start_index, end_index, self._tokenize_texts(texts[start_index:end_index])

    def _get_attention_mask(self, model_inputs: BatchEncoding, hidden_states: torch.Tensor) -> torch.Tensor:
        if "attention_mask" in model_inputs:
            return model_inputs["attention_mask"].to(hidden_states.device)
        return torch.ones(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)

    def _get_special_tokens_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        if not self.tokenizer.all_special_ids:
            return torch.zeros_like(token_ids, dtype=torch.bool)
        special_token_ids = torch.tensor(self.tokenizer.all_special_ids, device=token_ids.device)
        return torch.isin(token_ids, special_token_ids)

    def _pool_hidden_states(self, hidden_states: torch.Tensor, model_inputs: BatchEncoding) -> torch.Tensor:
        if self.pooling_strategy is None:
            return hidden_states
        if hidden_states.ndim != 3:
            raise ValueError("Pooling requires hidden states with shape `(batch, sequence, hidden_size)`.")

        attention_mask = self._get_attention_mask(model_inputs, hidden_states).bool()
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("The attention mask shape must match the first two hidden-state dimensions.")
        if not torch.all(attention_mask.any(dim=1)):
            raise ValueError("Pooling requires at least one attended token in every sample.")

        if self.pooling_strategy == "mean":
            weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1)

        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device).expand_as(attention_mask)
        if self.pooling_strategy == "cls":
            selected_indices = positions.masked_fill(~attention_mask, hidden_states.shape[1]).amin(dim=1)
        elif self.pooling_strategy == "last":
            selected_indices = positions.masked_fill(~attention_mask, -1).amax(dim=1)
        else:  # pragma: no cover - validated at initialization
            raise ValueError("Unknown pooling strategy.")
        gather_index = selected_indices[:, None, None].expand(-1, 1, hidden_states.shape[-1])
        return hidden_states.gather(1, gather_index).squeeze(1)

    def _prepare_projection_input(
        self,
        hidden_states: torch.Tensor,
        model_inputs: BatchEncoding,
    ) -> torch.Tensor:
        if self.task == "sequence_classification":
            return self._pool_hidden_states(hidden_states, model_inputs)
        return hidden_states

    def _capture_split_activation(self, model_inputs: BatchEncoding) -> torch.Tensor:
        split_module = self.model_with_split_points.get(self.split_point)
        output_name = "nns_output" if hasattr(split_module, "nns_output") else "output"
        forward_inputs = self._prepare_model_forward_inputs(model_inputs)
        with self.model_with_split_points.trace(forward_inputs) as tracer:
            output = getattr(split_module, output_name).save()
            tracer.stop()
        hidden_states = self.model_with_split_points._manage_output_tuple(output, self.split_point)
        return self._prepare_projection_input(hidden_states, model_inputs)

    def _capture_projection_input(
        self,
        model_inputs: BatchEncoding,
        include_reference_logits: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not include_reference_logits:
            return self._capture_split_activation(model_inputs), None

        activation: torch.Tensor | None = None

        def capture_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            nonlocal activation
            hidden_states = self.model_with_split_points._manage_output_tuple(output, self.split_point)
            activation = self._prepare_projection_input(hidden_states, model_inputs)

        handle = self._get_module(self.split_point).register_forward_hook(capture_hook)
        try:
            outputs = self.model(**self._prepare_model_forward_inputs(model_inputs))
        finally:
            handle.remove()

        if activation is None:
            raise RuntimeError(f"Failed to capture the split activation at `{self.split_point}`.")
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("The wrapped model must return tensor logits.")
        return activation, logits

    @staticmethod
    def _move_to_module(hidden_states: torch.Tensor, module: nn.Module) -> torch.Tensor:
        device, dtype = BaseLens._get_module_placement(module, hidden_states.device)
        if hidden_states.is_floating_point() and dtype is not None:
            return hidden_states.to(device=device, dtype=dtype)
        return hidden_states.to(device=device)

    def _project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected_hidden_states = hidden_states
        if self.model_pre_head is not None:
            projected_hidden_states = self.model_pre_head(
                self._move_to_module(projected_hidden_states, self.model_pre_head)
            )
            if not isinstance(projected_hidden_states, torch.Tensor):
                raise TypeError("The resolved pre-head module must return a tensor.")

        logits = self.model_head(self._move_to_module(projected_hidden_states, self.model_head))
        if isinstance(logits, tuple):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("The resolved output head must return a tensor.")
        return logits

    def _transform_activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def _project_activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._project_hidden_states(self._transform_activation(hidden_states))

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_values = values.masked_select(valid_mask)
        if valid_values.numel() == 0:
            raise ValueError("No valid targets remain after applying token and padding masks.")
        return valid_values.mean()

    @staticmethod
    def _upcast_for_scores(logits: torch.Tensor) -> torch.Tensor:
        if logits.dtype in {torch.float16, torch.bfloat16}:
            return logits.float()
        return logits

    @staticmethod
    def _to_metric_value(value: torch.Tensor, differentiable: bool) -> torch.Tensor | float:
        return value if differentiable else float(value.detach().cpu())

    def _prepare_language_model_metric_tensors(
        self,
        projected_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        model_inputs: BatchEncoding,
        targets: LensTargets,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        reference_logits = reference_logits.detach().to(
            device=projected_logits.device,
            dtype=projected_logits.dtype,
        )
        if projected_logits.shape != reference_logits.shape:
            raise ValueError(
                "Lens logits must have the same shape as final model logits. "
                f"Got {tuple(projected_logits.shape)} and {tuple(reference_logits.shape)}."
            )

        input_ids = model_inputs["input_ids"].to(projected_logits.device)
        attention_mask = self._get_attention_mask(model_inputs, projected_logits).bool()
        if self.language_model_mode == "causal":
            aligned_projected_logits = projected_logits[:, :-1]
            aligned_reference_logits = reference_logits[:, :-1]
            target_ids = input_ids[:, 1:]
            valid_mask = attention_mask[:, :-1] & attention_mask[:, 1:]
            target_source = "next_token"
        else:
            aligned_projected_logits = projected_logits
            aligned_reference_logits = reference_logits
            target_ids = input_ids
            valid_mask = attention_mask
            target_source = "token_identity"

        if targets is not None:
            if not isinstance(targets, torch.Tensor):
                raise TypeError("Language-model targets must be a tensor.")
            if targets.dtype == torch.bool or targets.is_floating_point() or targets.is_complex():
                raise TypeError("Language-model targets must contain integer token ids.")
            target_tensor = targets.to(projected_logits.device).long()
            if self.language_model_mode == "causal" and target_tensor.shape == input_ids.shape:
                target_tensor = target_tensor[:, 1:]
            if target_tensor.shape != target_ids.shape:
                raise ValueError(
                    "Language-model targets must align with tokenized inputs. "
                    f"Expected {tuple(target_ids.shape)}, got {tuple(target_tensor.shape)}."
                )
            target_ids = target_tensor
            target_source = "provided_targets"
        else:
            valid_mask = valid_mask & ~self._get_special_tokens_mask(target_ids)

        valid_mask = valid_mask & target_ids.ne(-100)
        valid_target_ids = target_ids.masked_select(valid_mask)
        if torch.any((valid_target_ids < 0) | (valid_target_ids >= projected_logits.shape[-1])):
            raise ValueError("Language-model targets contain token ids outside the model vocabulary.")
        safe_target_ids = target_ids.masked_fill(~valid_mask, 0)
        return (
            aligned_projected_logits,
            aligned_reference_logits,
            safe_target_ids,
            valid_mask,
            target_source,
        )

    def _prepare_sequence_classification_targets(
        self,
        targets: LensTargets,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if targets is None:
            return None
        if isinstance(targets, bool):
            raise TypeError("Sequence-classification targets must be integer labels.")
        if isinstance(targets, Integral):
            target_tensor = torch.tensor([int(targets)], dtype=torch.long, device=device)
        elif isinstance(targets, list):
            if any(isinstance(target, bool) or not isinstance(target, Integral) for target in targets):
                raise TypeError("Sequence-classification target lists must contain integers.")
            target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        elif isinstance(targets, torch.Tensor):
            if targets.dtype == torch.bool or targets.is_floating_point() or targets.is_complex():
                raise TypeError("Sequence-classification target tensors must contain integer labels.")
            target_tensor = targets.to(device).long()
        else:
            raise TypeError("Sequence-classification targets must be an integer, list of integers, or tensor.")

        if target_tensor.ndim == 2 and target_tensor.shape[-1] == 1:
            target_tensor = target_tensor.squeeze(-1)
        if target_tensor.ndim != 1:
            raise ValueError("Sequence-classification targets must have shape `(batch_size,)`.")
        if target_tensor.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} sequence-classification targets, got {target_tensor.shape[0]}.")
        return target_tensor

    def _compute_language_model_metrics(
        self,
        projected_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        model_inputs: BatchEncoding,
        targets: LensTargets,
        differentiable: bool,
    ) -> dict[str, Any]:
        projected_logits = self._upcast_for_scores(projected_logits)
        (
            aligned_projected_logits,
            aligned_reference_logits,
            target_ids,
            valid_mask,
            target_source,
        ) = self._prepare_language_model_metric_tensors(
            projected_logits,
            reference_logits,
            model_inputs,
            targets,
        )
        nb_evaluated_elements = int(valid_mask.sum().detach().cpu())
        if nb_evaluated_elements == 0:
            return {"target_source": target_source, "nb_evaluated_elements": 0}

        log_scores = F.log_softmax(aligned_projected_logits, dim=-1)
        scores = log_scores.exp()
        target_log_scores = torch.gather(log_scores, -1, target_ids.unsqueeze(-1)).squeeze(-1)
        target_scores = target_log_scores.exp()
        reference_scores = F.softmax(aligned_reference_logits, dim=-1)
        kl_values = F.kl_div(log_scores, reference_scores, reduction="none").sum(dim=-1).clamp_min(0.0)

        metrics: dict[str, Any] = {
            "target_source": target_source,
            "nb_evaluated_elements": nb_evaluated_elements,
            "mean_max_score": self._to_metric_value(
                self._masked_mean(scores.amax(dim=-1), valid_mask), differentiable
            ),
            "mean_target_score": self._to_metric_value(self._masked_mean(target_scores, valid_mask), differentiable),
            "target_cross_entropy": self._to_metric_value(
                -self._masked_mean(target_log_scores, valid_mask), differentiable
            ),
            "target_accuracy": self._to_metric_value(
                self._masked_mean((aligned_projected_logits.argmax(-1) == target_ids).float(), valid_mask),
                differentiable,
            ),
            "kl_divergence_to_model": self._to_metric_value(self._masked_mean(kl_values, valid_mask), differentiable),
            "model_top1_agreement": self._to_metric_value(
                self._masked_mean(
                    (aligned_projected_logits.argmax(-1) == aligned_reference_logits.argmax(-1)).float(),
                    valid_mask,
                ),
                differentiable,
            ),
        }
        cross_entropy = metrics["target_cross_entropy"]
        if self.language_model_mode == "causal":
            metrics["perplexity"] = (
                torch.exp(cross_entropy)
                if differentiable
                else float(torch.exp(torch.tensor(cross_entropy, dtype=torch.float64)))
            )
        return metrics

    def _compute_sequence_classification_metrics(
        self,
        projected_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        targets: LensTargets,
        differentiable: bool,
    ) -> dict[str, Any]:
        projected_logits = self._upcast_for_scores(projected_logits)
        reference_logits = reference_logits.detach().to(
            device=projected_logits.device,
            dtype=projected_logits.dtype,
        )
        if projected_logits.shape != reference_logits.shape:
            raise ValueError(
                "Lens logits must have the same shape as final model logits. "
                f"Got {tuple(projected_logits.shape)} and {tuple(reference_logits.shape)}."
            )

        target_tensor = self._prepare_sequence_classification_targets(
            targets,
            projected_logits.shape[0],
            projected_logits.device,
        )
        if target_tensor is not None and torch.any(
            (target_tensor < 0) | (target_tensor >= projected_logits.shape[-1])
        ):
            raise ValueError("Sequence-classification targets contain labels outside the model output range.")
        log_scores = F.log_softmax(projected_logits, dim=-1)
        scores = log_scores.exp()
        reference_scores = F.softmax(reference_logits, dim=-1)
        kl_divergence = F.kl_div(log_scores, reference_scores, reduction="batchmean").clamp_min(0.0)

        metrics: dict[str, Any] = {
            "target_source": "provided_targets" if target_tensor is not None else None,
            "nb_evaluated_elements": projected_logits.shape[0],
            "mean_max_score": self._to_metric_value(scores.amax(dim=-1).mean(), differentiable),
            "kl_divergence_to_model": self._to_metric_value(kl_divergence, differentiable),
            "model_top1_agreement": self._to_metric_value(
                (projected_logits.argmax(-1) == reference_logits.argmax(-1)).float().mean(),
                differentiable,
            ),
        }
        if target_tensor is None:
            return metrics

        target_log_scores = torch.gather(log_scores, -1, target_tensor.unsqueeze(-1)).squeeze(-1)
        metrics.update(
            {
                "mean_target_score": self._to_metric_value(target_log_scores.exp().mean(), differentiable),
                "target_cross_entropy": self._to_metric_value(-target_log_scores.mean(), differentiable),
                "target_accuracy": self._to_metric_value(
                    (projected_logits.argmax(-1) == target_tensor).float().mean(), differentiable
                ),
            }
        )
        return metrics

    def _compute_metrics(
        self,
        projected_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        model_inputs: BatchEncoding,
        targets: LensTargets,
        differentiable: bool,
    ) -> dict[str, Any]:
        if self.task == "language_model":
            return self._compute_language_model_metrics(
                projected_logits,
                reference_logits,
                model_inputs,
                targets,
                differentiable,
            )
        return self._compute_sequence_classification_metrics(
            projected_logits,
            reference_logits,
            targets,
            differentiable,
        )

    def _validate_targets(self, nb_samples: int, targets: LensTargets) -> None:
        if targets is None:
            return
        if self.task == "language_model":
            if not isinstance(targets, torch.Tensor):
                raise TypeError("Language-model targets must be a tensor.")
            if targets.ndim != 2 or targets.shape[0] != nb_samples:
                raise ValueError("Language-model targets must be a 2D tensor with one row per input.")
            return

        if isinstance(targets, bool):
            raise TypeError("Sequence-classification targets must be integer labels.")
        if isinstance(targets, Integral):
            if nb_samples != 1:
                raise ValueError("A scalar classification target can only be used with one input.")
            return
        if isinstance(targets, list):
            if len(targets) != nb_samples:
                raise ValueError("Sequence-classification targets must contain one label per input.")
            if any(isinstance(target, bool) or not isinstance(target, Integral) for target in targets):
                raise TypeError("Sequence-classification target lists must contain integers.")
            return
        if isinstance(targets, torch.Tensor):
            if targets.ndim == 0 or targets.shape[0] != nb_samples:
                raise ValueError("Sequence-classification targets must contain one label per input.")
            return
        raise TypeError("Sequence-classification targets must be an integer, list of integers, or tensor.")

    @staticmethod
    def _slice_targets(targets: LensTargets, start_index: int, end_index: int) -> LensTargets:
        if isinstance(targets, list | torch.Tensor):
            return targets[start_index:end_index]
        return targets

    @staticmethod
    def _merge_metric_batches(metric_batches: list[dict[str, Any]], differentiable: bool) -> dict[str, Any]:
        nonempty_batches = [batch for batch in metric_batches if batch["nb_evaluated_elements"] > 0]
        if not nonempty_batches:
            raise ValueError("No valid targets remain after applying token and padding masks.")

        target_source = nonempty_batches[0]["target_source"]
        metric_names = set(nonempty_batches[0]) - {"target_source", "nb_evaluated_elements", "perplexity"}
        total_count = sum(batch["nb_evaluated_elements"] for batch in nonempty_batches)
        merged: dict[str, Any] = {
            "target_source": target_source,
            "nb_evaluated_elements": total_count,
        }
        for batch in nonempty_batches:
            if batch["target_source"] != target_source:
                raise RuntimeError("Metric batches use inconsistent target sources.")
            batch_metric_names = set(batch) - {"target_source", "nb_evaluated_elements", "perplexity"}
            if batch_metric_names != metric_names:
                raise RuntimeError("Metric batches contain inconsistent metric fields.")

        for metric_name in sorted(metric_names):
            weighted_values = [batch[metric_name] * batch["nb_evaluated_elements"] for batch in nonempty_batches]
            merged[metric_name] = sum(weighted_values[1:], weighted_values[0]) / total_count

        if "target_cross_entropy" in merged and any("perplexity" in batch for batch in nonempty_batches):
            cross_entropy = merged["target_cross_entropy"]
            merged["perplexity"] = (
                torch.exp(cross_entropy)
                if differentiable
                else float(torch.exp(torch.tensor(cross_entropy, dtype=torch.float64)))
            )
        return merged

    def _validate_top_k(self, output_size: int, output_name: str) -> None:
        if self.top_k > output_size:
            raise ValueError(
                f"`top_k` cannot exceed the number of {output_name} entries: got {self.top_k} for {output_size}."
            )

    def _compute_top_outputs(
        self,
        logits: torch.Tensor,
        expected_ndim: int,
        output_name: str,
    ) -> LensTopKOutput:
        if logits.ndim != expected_ndim:
            raise ValueError(f"Expected {expected_ndim}D {output_name} logits, got shape {tuple(logits.shape)}.")
        self._validate_top_k(logits.shape[-1], output_name)
        stable_logits = self._upcast_for_scores(logits)
        top_logits, top_indices = torch.topk(stable_logits, self.top_k, dim=-1)
        top_scores = torch.exp(top_logits - torch.logsumexp(stable_logits, dim=-1, keepdim=True))
        return {
            "top_indices": top_indices.detach().cpu(),
            "top_scores": top_scores.detach().cpu(),
        }

    def _format_outputs(self, logits: torch.Tensor) -> LensTopKOutput:
        if self.task == "language_model":
            return self._compute_top_outputs(logits, 3, "vocabulary")
        return self._compute_top_outputs(logits, 2, "class")

    @staticmethod
    def _mask_language_model_padding(
        outputs: LensTopKOutput,
        model_inputs: BatchEncoding,
    ) -> LensTopKOutput:
        expected_shape = tuple(model_inputs["input_ids"].shape)
        if outputs["top_indices"].shape[:2] != expected_shape:
            raise ValueError("Language-model projections must preserve the input batch and sequence dimensions.")
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            return outputs

        padding_mask = ~attention_mask.detach().cpu().bool().unsqueeze(-1)
        return {
            "top_indices": outputs["top_indices"].masked_fill(padding_mask, 0),
            "top_scores": outputs["top_scores"].masked_fill(padding_mask, 0.0),
        }

    def _merge_output_batches(self, output_batches: list[LensTopKOutput]) -> LensTopKOutput:
        if self.task == "language_model":
            max_sequence_length = max(batch["top_indices"].shape[1] for batch in output_batches)
            padded_batches = []
            for batch in output_batches:
                padding_length = max_sequence_length - batch["top_indices"].shape[1]
                padded_batches.append(
                    {
                        "top_indices": F.pad(batch["top_indices"], (0, 0, 0, padding_length), value=0),
                        "top_scores": F.pad(batch["top_scores"], (0, 0, 0, padding_length), value=0.0),
                    }
                )
            output_batches = padded_batches

        return {
            "top_indices": torch.cat([batch["top_indices"] for batch in output_batches], dim=0),
            "top_scores": torch.cat([batch["top_scores"] for batch in output_batches], dim=0),
        }

    def _explain_inputs(self, inputs: LensInputs) -> LensResults:
        output_batches: list[LensTopKOutput] = []
        batch_size = self.model_with_split_points.batch_size
        self._validate_positive_integer(batch_size, "model_with_split_points.batch_size")
        with self._model_in_evaluation_mode(), torch.no_grad():
            for _, _, batch_inputs in self._iter_input_batches(inputs, batch_size):
                activation, _ = self._capture_projection_input(batch_inputs, include_reference_logits=False)
                batch_outputs = self._format_outputs(self._project_activation(activation))
                if self.task == "language_model":
                    batch_outputs = self._mask_language_model_padding(batch_outputs, batch_inputs)
                output_batches.append(batch_outputs)
        return {self.split_point: self._merge_output_batches(output_batches)}

    def explain(self, inputs: LensInputs) -> LensResults:
        """Project the configured split activation into the model output space.

        Args:
            inputs (LensInputs): Raw text or tensor-backed tokenized inputs.

        Returns:
            LensResults: Top-k indices and normalized scores keyed by the configured split point.
        """
        self._validate_bound_split_point()
        return self._explain_inputs(inputs)

    def metrics(
        self,
        inputs: LensInputs,
        targets: LensTargets = None,
        differentiable: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Compute decodability metrics at the configured split point.

        Causal language models use next-token targets; masked language models use token identities
        at the same position. Classification target metrics are included only when labels are given.

        Args:
            inputs (LensInputs): Raw text or tensor-backed tokenized inputs.
            targets (LensTargets): Optional token ids or class labels. Language-model labels require
                `BatchEncoding` inputs and a 2D tensor aligned with their token dimension; use `-100`
                at ignored positions.
            differentiable (bool): Preserve gradients through metric values when ``True``.

        Returns:
            dict[str, dict[str, Any]]: Metrics keyed by the configured split point. Every result contains
                `target_source`, `nb_evaluated_elements`, `mean_max_score`, `kl_divergence_to_model`, and
                `model_top1_agreement`. When targets are available it also contains `mean_target_score`,
                `target_cross_entropy`, and `target_accuracy`. `perplexity` is included only for causal
                next-token metrics. Softmax-derived values are scores because intermediate projections are
                not generally calibrated.

        Raises:
            TypeError: If targets or ``differentiable`` do not match the expected type.
            RuntimeError: If reference logits cannot be captured.
        """
        if not isinstance(differentiable, bool):
            raise TypeError("`differentiable` must be a boolean.")
        self._validate_bound_split_point()
        if self.task == "language_model" and targets is not None and not isinstance(inputs, BatchEncoding):
            raise TypeError("Language-model targets require a tensor-backed BatchEncoding input.")
        self._validate_targets(self._get_input_size(inputs), targets)
        batch_size = self.model_with_split_points.batch_size
        self._validate_positive_integer(batch_size, "model_with_split_points.batch_size")
        metric_batches = []

        gradient_context = torch.enable_grad() if differentiable else torch.no_grad()
        with self._model_in_evaluation_mode(), gradient_context:
            for start_index, end_index, batch_inputs in self._iter_input_batches(inputs, batch_size):
                activation, reference_logits = self._capture_projection_input(
                    batch_inputs,
                    include_reference_logits=True,
                )
                if reference_logits is None:  # pragma: no cover - guarded by capture
                    raise RuntimeError("Failed to capture final model logits.")
                batch_targets = self._slice_targets(targets, start_index, end_index)
                metric_batches.append(
                    self._compute_metrics(
                        self._project_activation(activation),
                        reference_logits,
                        batch_inputs,
                        batch_targets,
                        differentiable,
                    )
                )
        return {self.split_point: self._merge_metric_batches(metric_batches, differentiable)}

    def __call__(self, inputs: LensInputs) -> LensResults:
        """Alias for :meth:`explain`.

        Args:
            inputs (LensInputs): Raw text or tensor-backed tokenized inputs.

        Returns:
            LensResults: Top-k results at the configured split point.
        """
        return self.explain(inputs)
