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

"""
Shared utilities for lens methods.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints
from interpreto.typing import LabelNames, LensResults, LensTopKOutput

LensInputs = str | list[str] | BatchEncoding
LensTask = Literal["language_model", "sequence_classification"]
PoolingStrategy = Literal["cls", "mean", "last"]
LanguageModelMode = Literal["causal", "masked"]
LensTargets = int | list[int] | torch.Tensor | None


def _get_batch_encoding_size(inputs: BatchEncoding) -> int:
    for value in inputs.values():
        if isinstance(value, torch.Tensor):
            return value.shape[0]
    raise ValueError("The given BatchEncoding does not contain any tensor field.")


def _slice_batch_encoding(inputs: BatchEncoding, start_index: int, end_index: int) -> BatchEncoding:
    return BatchEncoding(
        {
            key: value[start_index:end_index] if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
    )


@dataclass(frozen=True)
class ProjectionSpec:
    """Projection path used by a lens method."""

    head_name: str
    pre_head_name: str | None = None
    pooling_strategy: PoolingStrategy | None = None


class BaseLens:
    """
    Shared implementation for Logit Lens style methods.

    The base class resolves the projection path from hidden states to model outputs,
    normalizes inputs, captures split activations on the wrapped model, and formats
    the corresponding intermediate predictions.
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        head_name: str | None = None,
        pre_head_name: str | None = None,
        pooling_strategy: PoolingStrategy | None = None,
        top_k: int = 5,
        device: torch.device | str | None = None,
    ) -> None:
        """
        Initialize a lens method.

        Args:
            model_with_split_points (ModelWithSplitPoints): Wrapped Hugging Face model used to
                extract split activations.
            head_name (str | None): Dot-separated path to the output head.
                If `None`, a short list of known paths is tried.
            pre_head_name (str | None): Optional dot-separated path to a module applied before
                the head. This is useful for architectures exposing a separated pooler or
                transformation block.
            pooling_strategy (Literal["cls", "mean", "last"] | None): Optional pooling applied
                before the projection for sequence classification when a simple head expects one
                vector per sample. Sequence-aware heads may consume 3D hidden states directly.
            top_k (int): Number of labels or tokens returned per prediction.
            device (torch.device | str | None): Device used by learned translators.
                Projection modules remain on the model device.

        Raises:
            TypeError: If the provided model is not a `ModelWithSplitPoints` or if `top_k` is not an integer.
            ValueError: If the tokenizer is missing, if `top_k` is not strictly positive,
                or if the projection path is invalid.
            NotImplementedError: If the wrapped model is a token classification model.
            RuntimeError: If the resolved projection contains meta tensors.
        """
        if not isinstance(model_with_split_points, ModelWithSplitPoints):
            raise TypeError(
                "The given model should be a ModelWithSplitPoints, "
                f"but {type(model_with_split_points)} was given."
            )

        if model_with_split_points.tokenizer is None:
            raise ValueError("The wrapped model should expose a tokenizer.")

        if isinstance(top_k, bool) or not isinstance(top_k, Integral):
            raise TypeError("`top_k` must be an integer.")
        if top_k < 1:
            raise ValueError("`top_k` must be a strictly positive integer.")

        self.model_with_split_points = model_with_split_points
        self.model = model_with_split_points._model
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = model_with_split_points.tokenizer
        self.task: LensTask = self._infer_task()
        self.language_model_mode: LanguageModelMode | None = (
            self._infer_language_model_mode() if self.task == "language_model" else None
        )
        self.top_k = top_k
        self.model_device = self._get_model_device()
        self.device = torch.device(device) if device is not None else self.model_device
        self.model_forward_keys = set(inspect.signature(self.model.forward).parameters)
        self.preferred_padding_side = self._get_preferred_padding_side()

        self.model.eval()
        projection = self._resolve_projection_spec(
            head_name=head_name,
            pre_head_name=pre_head_name,
            pooling_strategy=pooling_strategy,
        )
        self.head_name = projection.head_name
        self.pre_head_name = projection.pre_head_name
        self.pooling_strategy = projection.pooling_strategy
        self.model_head, self.model_pre_head = self._resolve_projection_modules()
        self._validate_projection()

    def _infer_task(self) -> LensTask:
        model_class_name = type(self.model).__name__

        if "ForTokenClassification" in model_class_name:
            raise NotImplementedError(
                "Token classification is not supported by the lens methods. "
                "Please use a language-model or sequence-classification checkpoint."
            )

        if "ForSequenceClassification" in model_class_name:
            return "sequence_classification"

        if "ForMaskedLM" in model_class_name or "ForCausalLM" in model_class_name:
            return "language_model"

        if model_class_name.endswith("LMHeadModel"):
            return "language_model"

        raise ValueError(
            "Unsupported model type for the lens methods. "
            "The supported families are `AutoModelForCausalLM`, `AutoModelForMaskedLM`, "
            "and `AutoModelForSequenceClassification`. "
            f"Received model class: {model_class_name}."
        )

    def _get_preferred_padding_side(self) -> str:
        model_class_name = type(self.model).__name__.lower()
        if self.task == "language_model" and (
            "causallm" in model_class_name or model_class_name.endswith("lmheadmodel")
        ):
            return "left"
        return "right"

    def _infer_language_model_mode(self) -> LanguageModelMode:
        model_class_name = type(self.model).__name__
        if "MaskedLM" in model_class_name:
            return "masked"
        return "causal"

    def _get_model_device(self) -> torch.device:
        for tensor in list(self.model.parameters()) + list(self.model.buffers()):
            if tensor.device.type != "meta":
                return tensor.device

        raise RuntimeError(
            "The wrapped model only exposes meta tensors. "
            "Please construct `ModelWithSplitPoints` from a fully loaded Hugging Face model "
            "before using the lens methods."
        )

    def _resolve_projection_spec(
        self,
        head_name: str | None,
        pre_head_name: str | None,
        pooling_strategy: PoolingStrategy | None,
    ) -> ProjectionSpec:
        if pooling_strategy not in {None, "cls", "mean", "last"}:
            raise ValueError(
                "`pooling_strategy` should be one of `None`, `'cls'`, `'mean'`, or `'last'`."
            )

        if head_name is not None:
            if self.task == "language_model" and pooling_strategy is not None:
                raise ValueError("`pooling_strategy` is only supported for sequence classification.")
            return ProjectionSpec(
                head_name=head_name,
                pre_head_name=pre_head_name,
                pooling_strategy=pooling_strategy,
            )

        if self.task == "language_model":
            return self._get_language_model_projection()

        return self._get_sequence_classification_projection()

    def _resolve_projection_modules(self) -> tuple[nn.Module, nn.Module | None]:
        model_head = self._get_module(self.head_name)
        self._raise_on_meta_module(model_head, self.head_name)

        model_pre_head: nn.Module | None = None
        if self.pre_head_name is not None:
            model_pre_head = self._get_module(self.pre_head_name)
            self._raise_on_meta_module(model_pre_head, self.pre_head_name)

        return model_head, model_pre_head

    def _get_language_model_projection(self) -> ProjectionSpec:
        normalized_candidates = [
            ProjectionSpec(head_name="lm_head", pre_head_name="transformer.ln_f"),
            ProjectionSpec(head_name="lm_head", pre_head_name="model.norm"),
            ProjectionSpec(head_name="lm_head", pre_head_name="model.decoder.final_layer_norm"),
            ProjectionSpec(head_name="lm_head", pre_head_name="decoder.final_layer_norm"),
            ProjectionSpec(head_name="embed_out", pre_head_name="gpt_neox.final_layer_norm"),
            ProjectionSpec(head_name="embed_out", pre_head_name="model.norm"),
        ]
        valid_normalized_candidates = [
            candidate
            for candidate in normalized_candidates
            if self._module_exists(candidate.head_name)
            and candidate.pre_head_name is not None
            and self._module_exists(candidate.pre_head_name)
        ]
        if len(valid_normalized_candidates) == 1:
            return valid_normalized_candidates[0]
        if len(valid_normalized_candidates) > 1:
            candidate_paths = ", ".join(
                [
                    f"pre_head_name={candidate.pre_head_name}, head_name={candidate.head_name}"
                    for candidate in valid_normalized_candidates
                ]
            )
            raise ValueError(
                "Ambiguous language-model projection path. "
                "Please provide `head_name` and `pre_head_name` explicitly. "
                f"Candidates: {candidate_paths}."
            )

        candidates = [
            ProjectionSpec(head_name="lm_head"),
            ProjectionSpec(head_name="embed_out"),
            ProjectionSpec(head_name="cls"),
            ProjectionSpec(head_name="generator_lm_head"),
        ]

        valid_candidates = [candidate for candidate in candidates if self._module_exists(candidate.head_name)]
        if not valid_candidates:
            checked_paths = ", ".join(candidate.head_name for candidate in candidates)
            raise ValueError(
                "Could not resolve a language-model projection head. "
                f"Checked: {checked_paths}. Please provide `head_name` explicitly."
            )

        if len(valid_candidates) > 1:
            candidate_paths = ", ".join(candidate.head_name for candidate in valid_candidates)
            raise ValueError(
                "Ambiguous language-model projection head. "
                f"Please provide `head_name` explicitly. Candidates: {candidate_paths}."
            )

        return valid_candidates[0]

    def _get_sequence_classification_projection(self) -> ProjectionSpec:
        candidates: list[ProjectionSpec] = []
        base_model_prefix = getattr(self.model, "base_model_prefix", None)

        if self._module_exists("classifier"):
            if base_model_prefix is not None and self._module_exists(f"{base_model_prefix}.pooler"):
                candidates.append(
                    ProjectionSpec(
                        head_name="classifier",
                        pre_head_name=f"{base_model_prefix}.pooler",
                        pooling_strategy=None,
                    )
                )
            else:
                candidates.append(ProjectionSpec(head_name="classifier", pooling_strategy=None))

        if not candidates and self._module_exists("score"):
            candidates.append(ProjectionSpec(head_name="score", pooling_strategy=None))

        if not candidates:
            checked_paths = ["classifier", "score"]
            if base_model_prefix is not None:
                checked_paths.append(f"{base_model_prefix}.pooler")
            raise ValueError(
                "Could not resolve a sequence-classification projection path. "
                f"Checked: {', '.join(checked_paths)}. Please provide `head_name` explicitly."
            )

        if len(candidates) > 1:
            candidate_paths = ", ".join(
                [
                    f"pre_head_name={candidate.pre_head_name}, head_name={candidate.head_name}"
                    for candidate in candidates
                ]
            )
            raise ValueError(
                "Ambiguous sequence-classification projection path. "
                "Please provide `head_name` and `pre_head_name` explicitly. "
                f"Candidates: {candidate_paths}."
            )

        return candidates[0]

    def _validate_projection(self) -> None:
        if self.task == "sequence_classification" and self.pooling_strategy is None:
            if self.model_pre_head is None and isinstance(self.model_head, nn.Linear):
                raise ValueError(self._build_missing_pooling_strategy_message())

    def _build_missing_pooling_strategy_message(self) -> str:
        projection_path = f"head_name={self.head_name}"
        return (
            "The resolved sequence-classification projection uses a simple vector head and expects "
            "one pooled vector per sample, "
            f"but `{projection_path}` does not pool internally. "
            "Please provide `pooling_strategy` explicitly with one of `'cls'`, `'mean'`, or `'last'`, "
            "or provide a compatible `pre_head_name`."
        )

    def _raise_on_meta_module(self, module: nn.Module, module_name: str) -> None:
        tensors = list(module.parameters(recurse=True)) + list(module.buffers(recurse=True))
        if any(tensor.device.type == "meta" for tensor in tensors):
            raise RuntimeError(
                "The lens projection contains meta tensors. "
                f"The module `{module_name}` is not materialized. "
                "Please build `ModelWithSplitPoints` from a fully loaded Hugging Face model "
                "instead of relying on an unresolved lazy-loading path."
            )

    def _module_exists(self, module_name: str) -> bool:
        try:
            module = self._get_module(module_name)
        except (AttributeError, IndexError, TypeError, ValueError):
            return False
        return module is not None

    def _get_module(self, module_name: str) -> nn.Module:
        current: Any = self.model
        for path_element in module_name.split("."):
            if path_element.isdigit():
                current = current[int(path_element)]
            else:
                current = getattr(current, path_element)

            if current is None:
                raise ValueError(f"Module `{module_name}` resolves to `None`.")

        if not isinstance(current, nn.Module):
            raise TypeError(f"Module `{module_name}` is not a torch.nn.Module.")

        return current

    def _ensure_padding_token(self) -> None:
        if self.tokenizer.pad_token is not None:
            return

        if self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return

        raise ValueError(
            "Raw text inputs require a tokenizer exposing a pad token or an eos token. "
            "Please configure the wrapped tokenizer before calling the lens methods."
        )

    def _prepare_inputs(self, inputs: LensInputs) -> BatchEncoding:
        if isinstance(inputs, BatchEncoding):
            if "input_ids" not in inputs or inputs["input_ids"].numel() == 0:
                raise ValueError("Empty BatchEncoding inputs are not supported.")
            return inputs

        if isinstance(inputs, str):
            texts = [inputs]
        elif isinstance(inputs, list):
            if not inputs:
                raise ValueError("Empty input lists are not supported.")
            if any(not isinstance(text, str) for text in inputs):
                raise TypeError("Input lists should only contain strings.")
            texts = inputs
        else:
            raise TypeError(
                "Unsupported input type. Expected `str`, `list[str]`, or `BatchEncoding`, "
                f"got {type(inputs)}."
            )

        self._ensure_padding_token()
        original_padding_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = self.preferred_padding_side
            return self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

    def _prepare_model_forward_inputs(self, inputs: BatchEncoding) -> dict[str, torch.Tensor]:
        prepared_inputs: dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if key in self.model_forward_keys and isinstance(value, torch.Tensor):
                prepared_inputs[key] = value.to(self.model_device)

        if "attention_mask" not in prepared_inputs and "input_ids" in prepared_inputs:
            prepared_inputs["attention_mask"] = torch.ones_like(prepared_inputs["input_ids"])

        return prepared_inputs

    def _normalize_split_points(self, split_points: str | list[str] | None) -> list[str]:
        if split_points is None:
            return list(self.model_with_split_points.split_points)

        requested_split_points = [split_points] if isinstance(split_points, str) else split_points
        invalid_split_points = [
            split_point
            for split_point in requested_split_points
            if split_point not in self.model_with_split_points.split_points
        ]
        if invalid_split_points:
            raise ValueError(
                "Unknown split points: "
                + ", ".join(invalid_split_points)
                + ". Available split points: "
                + ", ".join(self.model_with_split_points.split_points)
                + "."
            )
        return list(requested_split_points)

    def _get_attention_mask(self, model_inputs: BatchEncoding, hidden_states: torch.Tensor) -> torch.Tensor:
        if "attention_mask" in model_inputs:
            return model_inputs["attention_mask"].to(hidden_states.device)
        return torch.ones(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)

    def _get_special_tokens_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        if not self.tokenizer.all_special_ids:
            return torch.zeros_like(token_ids, dtype=torch.bool)

        special_token_ids = torch.tensor(self.tokenizer.all_special_ids, device=token_ids.device)
        return torch.isin(token_ids, special_token_ids)

    def _masked_mean(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_values = values.masked_select(valid_mask)
        if valid_values.numel() == 0:
            raise ValueError(
                "No valid targets were found for the selected inputs. "
                "Please check the provided labels or the tokenization settings."
            )
        return valid_values.mean()

    def _to_metric_value(self, value: torch.Tensor, differentiable: bool) -> torch.Tensor | float:
        if differentiable:
            return value
        return float(value.detach().cpu())

    def _pool_hidden_states(self, hidden_states: torch.Tensor, model_inputs: BatchEncoding) -> torch.Tensor:
        if self.pooling_strategy is None:
            return hidden_states

        if hidden_states.dim() != 3:
            raise ValueError(
                "Pooling strategies require 3D hidden states of shape `(batch_size, seq_len, hidden_size)`."
            )

        attention_mask = self._get_attention_mask(model_inputs, hidden_states).bool()
        if self.pooling_strategy == "cls":
            return hidden_states[:, 0, :]

        if self.pooling_strategy == "mean":
            weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            normalizer = weights.sum(dim=1).clamp_min(1.0)
            return (hidden_states * weights).sum(dim=1) / normalizer

        if self.pooling_strategy == "last":
            last_indices = attention_mask.long().sum(dim=1).clamp_min(1) - 1
            gather_index = last_indices.view(-1, 1, 1).expand(-1, 1, hidden_states.shape[-1])
            return hidden_states.gather(dim=1, index=gather_index).squeeze(1)

        raise ValueError(
            "`pooling_strategy` should be one of `None`, `'cls'`, `'mean'`, or `'last'`."
        )

    def _prepare_projection_inputs(
        self,
        hidden_states: torch.Tensor,
        model_inputs: BatchEncoding,
    ) -> torch.Tensor:
        if self.task != "sequence_classification" or self.pooling_strategy is None:
            return hidden_states
        return self._pool_hidden_states(hidden_states, model_inputs)

    def _capture_projection_inputs(
        self,
        model_inputs: BatchEncoding,
        split_points: list[str],
        differentiable: bool,
        include_reference_logits: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        forward_inputs = self._prepare_model_forward_inputs(model_inputs)
        activations: dict[str, torch.Tensor] = {}
        handles = []

        def build_hook(split_point: str):
            def hook(_module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor | tuple[torch.Tensor]) -> None:
                hidden_states = self.model_with_split_points._manage_output_tuple(output, split_point)
                activations[split_point] = self._prepare_projection_inputs(hidden_states, model_inputs)

            return hook

        for split_point in split_points:
            handles.append(self._get_module(split_point).register_forward_hook(build_hook(split_point)))

        context_manager = torch.enable_grad() if differentiable else torch.no_grad()
        try:
            with context_manager:
                outputs = self.model(**forward_inputs)
        finally:
            for handle in handles:
                handle.remove()

        missing_split_points = [split_point for split_point in split_points if split_point not in activations]
        if missing_split_points:
            raise RuntimeError(
                "Failed to capture activations for split points: " + ", ".join(missing_split_points) + "."
            )

        if not include_reference_logits:
            return activations, None

        logits = outputs.logits
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "The wrapped model should expose `logits` as a torch.Tensor, "
                f"got {type(logits)} instead."
            )

        return activations, logits.to(self.model_device)

    def _project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected_hidden_states = hidden_states.to(self.model_device)
        if self.model_pre_head is not None:
            projected_hidden_states = self.model_pre_head(projected_hidden_states)

        logits = self.model_head(projected_hidden_states)
        if isinstance(logits, tuple):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "The resolved projection head should return a torch.Tensor, "
                f"got {type(logits)} instead."
            )
        return logits

    def _transform_activations(self, split_point: str, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def _iter_projected_logits(
        self,
        projection_inputs: dict[str, torch.Tensor],
        split_points: list[str],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for split_point in split_points:
            transformed_hidden_states = self._transform_activations(split_point, projection_inputs[split_point])
            yield split_point, self._project_hidden_states(transformed_hidden_states)

    def _explain_prepared(
        self, model_inputs: BatchEncoding, split_points: str | list[str] | None = None
    ) -> LensResults:
        selected_split_points = self._normalize_split_points(split_points)
        projection_inputs, _ = self._capture_projection_inputs(
            model_inputs,
            selected_split_points,
            differentiable=False,
            include_reference_logits=False,
        )

        results: LensResults = {}
        with torch.no_grad():
            for split_point, logits in self._iter_projected_logits(
                projection_inputs,
                selected_split_points,
            ):
                results[split_point] = self._format_outputs(logits)

        return results

    def _prepare_language_model_metric_tensors(
        self,
        projected_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        model_inputs: BatchEncoding,
        targets: LensTargets,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        input_ids = model_inputs["input_ids"].to(projected_logits.device)
        attention_mask = self._get_attention_mask(model_inputs, projected_logits).bool()
        special_tokens_mask = self._get_special_tokens_mask(input_ids)

        if self.language_model_mode == "causal":
            aligned_projected_logits = projected_logits[:, :-1, :]
            aligned_reference_logits = reference_logits[:, :-1, :].detach()
            target_ids = input_ids[:, 1:]
            valid_mask = attention_mask[:, 1:] & ~special_tokens_mask[:, 1:]
            target_source = "next_token"
        else:
            aligned_projected_logits = projected_logits
            aligned_reference_logits = reference_logits.detach()
            target_ids = input_ids
            valid_mask = attention_mask & ~special_tokens_mask
            target_source = "token_identity"

        if targets is not None:
            if not isinstance(targets, torch.Tensor):
                raise TypeError(
                    "Language-model metrics only accept tensor targets. "
                    "When `targets` is omitted, token targets are derived from `inputs`."
                )

            target_tensor = targets.to(projected_logits.device).long()
            if self.language_model_mode == "causal" and target_tensor.shape == input_ids.shape:
                target_tensor = target_tensor[:, 1:]
            if target_tensor.shape != target_ids.shape:
                raise ValueError(
                    "Language-model targets should match the tokenized input shape. "
                    f"Expected {tuple(target_ids.shape)}, got {tuple(target_tensor.shape)}."
                )
            target_ids = target_tensor
            target_source = "provided_targets"

        valid_mask = valid_mask & target_ids.ne(-100)
        safe_target_ids = target_ids.masked_fill(~valid_mask, 0)
        return aligned_projected_logits, aligned_reference_logits, safe_target_ids, valid_mask, target_source

    def _prepare_sequence_classification_targets(
        self, targets: LensTargets, batch_size: int, device: torch.device
    ) -> torch.Tensor | None:
        if targets is None:
            return None

        if isinstance(targets, int):
            target_tensor = torch.tensor([targets], dtype=torch.long, device=device)
        elif isinstance(targets, list):
            target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        elif isinstance(targets, torch.Tensor):
            target_tensor = targets.to(device).long()
        else:
            raise TypeError(
                "Sequence-classification targets should be an integer, a list of integers, or a tensor."
            )

        if target_tensor.ndim == 2 and target_tensor.shape[-1] == 1:
            target_tensor = target_tensor.squeeze(-1)
        if target_tensor.ndim != 1:
            raise ValueError(
                "Sequence-classification targets should be a 1D tensor of shape `(batch_size,)`."
            )
        if target_tensor.shape[0] != batch_size:
            raise ValueError(
                "Sequence-classification targets should match the input batch size. "
                f"Expected {batch_size}, got {target_tensor.shape[0]}."
            )

        return target_tensor

    def _compute_language_model_metrics(
        self,
        projected_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        model_inputs: BatchEncoding,
        targets: LensTargets,
        differentiable: bool,
    ) -> dict[str, Any]:
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

        log_probabilities = F.log_softmax(aligned_projected_logits, dim=-1)
        probabilities = log_probabilities.exp()
        target_log_probabilities = torch.gather(
            log_probabilities,
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)
        target_probabilities = target_log_probabilities.exp()
        reference_probabilities = F.softmax(aligned_reference_logits, dim=-1)
        kl_values = F.kl_div(log_probabilities, reference_probabilities, reduction="none").sum(dim=-1)

        metrics: dict[str, Any] = {
            "target_source": target_source,
            "nb_evaluated_elements": int(valid_mask.sum().detach().cpu().item()),
            "mean_max_probability": self._to_metric_value(
                self._masked_mean(probabilities.amax(dim=-1), valid_mask),
                differentiable,
            ),
            "mean_target_probability": self._to_metric_value(
                self._masked_mean(target_probabilities, valid_mask),
                differentiable,
            ),
            "target_cross_entropy": self._to_metric_value(
                -self._masked_mean(target_log_probabilities, valid_mask),
                differentiable,
            ),
            "target_accuracy": self._to_metric_value(
                self._masked_mean((aligned_projected_logits.argmax(dim=-1) == target_ids).float(), valid_mask),
                differentiable,
            ),
            "kl_divergence_to_model": self._to_metric_value(
                self._masked_mean(kl_values, valid_mask),
                differentiable,
            ),
            "model_top1_agreement": self._to_metric_value(
                self._masked_mean(
                    (aligned_projected_logits.argmax(dim=-1) == aligned_reference_logits.argmax(dim=-1)).float(),
                    valid_mask,
                ),
                differentiable,
            ),
        }

        if self.language_model_mode == "causal":
            cross_entropy = metrics["target_cross_entropy"]
            if differentiable:
                metrics["perplexity"] = torch.exp(cross_entropy)
            else:
                metrics["perplexity"] = float(torch.exp(torch.tensor(cross_entropy)))

        return metrics

    def _compute_sequence_classification_metrics(
        self,
        projected_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        targets: LensTargets,
        differentiable: bool,
    ) -> dict[str, Any]:
        reference_logits = reference_logits.detach()
        target_tensor = self._prepare_sequence_classification_targets(
            targets,
            batch_size=projected_logits.shape[0],
            device=projected_logits.device,
        )
        log_probabilities = F.log_softmax(projected_logits, dim=-1)
        probabilities = log_probabilities.exp()
        reference_probabilities = F.softmax(reference_logits, dim=-1)

        metrics: dict[str, Any] = {
            "target_source": "provided_targets" if target_tensor is not None else None,
            "nb_evaluated_elements": projected_logits.shape[0],
            "mean_max_probability": self._to_metric_value(probabilities.amax(dim=-1).mean(), differentiable),
            "kl_divergence_to_model": self._to_metric_value(
                F.kl_div(log_probabilities, reference_probabilities, reduction="batchmean"),
                differentiable,
            ),
            "model_top1_agreement": self._to_metric_value(
                (projected_logits.argmax(dim=-1) == reference_logits.argmax(dim=-1)).float().mean(),
                differentiable,
            ),
        }

        if target_tensor is None:
            return metrics

        target_log_probabilities = torch.gather(
            log_probabilities,
            dim=-1,
            index=target_tensor.unsqueeze(-1),
        ).squeeze(-1)
        target_probabilities = target_log_probabilities.exp()
        metrics["mean_target_probability"] = self._to_metric_value(target_probabilities.mean(), differentiable)
        metrics["target_cross_entropy"] = self._to_metric_value(-target_log_probabilities.mean(), differentiable)
        metrics["target_accuracy"] = self._to_metric_value(
            (projected_logits.argmax(dim=-1) == target_tensor).float().mean(),
            differentiable,
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

    def _validate_top_k(self, output_size: int, output_name: str) -> None:
        if self.top_k > output_size:
            raise ValueError(
                f"`top_k` should not exceed the number of available {output_name}. "
                f"Received top_k={self.top_k} for {output_size} available {output_name}."
            )

    def _compute_top_outputs(
        self,
        logits: torch.Tensor,
        expected_ndim: int,
        output_name: str,
    ) -> LensTopKOutput:
        if logits.dim() != expected_ndim:
            raise ValueError(
                f"Lens projections should return a {expected_ndim}D tensor for {output_name} outputs, "
                f"got a tensor of shape {tuple(logits.shape)}."
            )

        self._validate_top_k(logits.shape[-1], output_name)
        top_logits, top_indices = torch.topk(logits, k=self.top_k, dim=-1)
        log_normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_scores = torch.exp(top_logits - log_normalizer)
        return {
            "top_indices": top_indices.detach().cpu(),
            "top_scores": top_scores.detach().cpu(),
        }

    def _format_outputs(self, logits: torch.Tensor) -> LensTopKOutput:
        if self.task == "language_model":
            return self._format_language_model_outputs(logits)
        return self._format_sequence_classification_outputs(logits)

    def _format_language_model_outputs(self, logits: torch.Tensor) -> LensTopKOutput:
        return self._compute_top_outputs(logits, expected_ndim=3, output_name="vocabulary entries")

    def _format_sequence_classification_outputs(self, logits: torch.Tensor) -> LensTopKOutput:
        if logits.dim() != 2:
            raise ValueError(
                "Sequence-classification lens projections should return 2D logits "
                "of shape `(batch_size, num_labels)`. "
                "If your projection returns sequence logits, please provide an explicit "
                "`pooling_strategy` or a compatible `pre_head_name`."
            )

        return self._compute_top_outputs(logits, expected_ndim=2, output_name="labels")

    def _iter_batches(self, inputs: LensInputs, batch_size: int) -> Iterator[LensInputs]:
        if batch_size < 1:
            raise ValueError("`batch_size` must be a strictly positive integer.")

        if isinstance(inputs, str):
            yield [inputs]
            return

        if isinstance(inputs, list):
            for start_index in range(0, len(inputs), batch_size):
                end_index = min(start_index + batch_size, len(inputs))
                yield inputs[start_index:end_index]
            return

        if isinstance(inputs, BatchEncoding):
            batch_encoding_size = _get_batch_encoding_size(inputs)
            for start_index in range(0, batch_encoding_size, batch_size):
                end_index = min(start_index + batch_size, batch_encoding_size)
                yield _slice_batch_encoding(inputs, start_index, end_index)
            return

        raise TypeError(
            "Unsupported input type. Expected `str`, `list[str]`, or `BatchEncoding`, "
            f"got {type(inputs)}."
        )

    def explain(self, inputs: LensInputs, split_points: str | list[str] | None = None) -> LensResults:
        """
        Compute lens predictions at the selected split points.

        Args:
            inputs (str | list[str] | BatchEncoding): Raw text or tokenized inputs.
            split_points (str | list[str] | None): Optional subset of split points.
                If `None`, all split points registered on `model_with_split_points` are used.

        Returns:
            dict[str, LensTopKOutput]: Mapping between split points and formatted predictions.
                Each split-point entry contains `top_indices` and `top_scores`.
                Decoding these indices to tokens or label names is handled by the visualization layer.
        """
        model_inputs = self._prepare_inputs(inputs)
        return self._explain_prepared(model_inputs, split_points=split_points)

    def metrics(
        self,
        inputs: LensInputs,
        targets: LensTargets = None,
        split_points: str | list[str] | None = None,
        differentiable: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Compute decodability metrics at the selected split points.

        For language models, hard-target metrics are computed on token identities derived from
        `inputs` when `targets` is omitted. Causal language models use next-token targets,
        while masked language models use token identities at the same position.

        For sequence classification, hard-target metrics require integer labels.
        When `targets` is omitted, the method still reports distributional metrics against the
        frozen final model distribution.

        Args:
            inputs (str | list[str] | BatchEncoding): Raw text or tokenized inputs.
            targets (int | list[int] | torch.Tensor | None): Optional hard targets.
                For sequence classification, provide one label per sample.
                For language models, tensor targets should match the tokenized input shape.
            split_points (str | list[str] | None): Optional subset of split points.
            differentiable (bool): If `True`, preserve gradients through the lens metric path.
                This is useful when reusing the scores as regularizers during training.

        Returns:
            dict[str, dict[str, Any]]: One metrics dictionary per split point.
                Each dictionary contains:
                - `mean_max_probability`
                - `kl_divergence_to_model`
                - `model_top1_agreement`
                and, when hard targets are available:
                - `mean_target_probability`
                - `target_cross_entropy`
                - `target_accuracy`
                - `perplexity` for causal language models
        """
        model_inputs = self._prepare_inputs(inputs)
        selected_split_points = self._normalize_split_points(split_points)
        projection_inputs, reference_logits = self._capture_projection_inputs(
            model_inputs,
            selected_split_points,
            differentiable=differentiable,
            include_reference_logits=True,
        )
        if reference_logits is None:
            raise RuntimeError("Failed to capture the reference logits required by the metrics path.")

        results: dict[str, dict[str, Any]] = {}
        context_manager = torch.enable_grad() if differentiable else torch.no_grad()
        with context_manager:
            for split_point, projected_logits in self._iter_projected_logits(
                projection_inputs,
                selected_split_points,
            ):
                results[split_point] = self._compute_metrics(
                    projected_logits,
                    reference_logits,
                    model_inputs,
                    targets,
                    differentiable,
                )

        return results

    def __call__(self, inputs: LensInputs, split_points: str | list[str] | None = None) -> LensResults:
        return self.explain(inputs, split_points=split_points)

    def lens(
        self,
        inputs: LensInputs,
        split_points: str | list[str] | None = None,
        label_names: LabelNames | None = None,
    ) -> LensResults:
        """
        Display a notebook-friendly summary of the lens predictions.

        Args:
            inputs (str | list[str] | BatchEncoding): Raw text or tokenized inputs.
            split_points (str | list[str] | None): Optional subset of split points.
            label_names (Mapping[int | str, str] | list[str] | tuple[str, ...] | None):
                Optional display names for sequence-classification labels.
                If `None`, raw label ids are shown.

        Returns:
            dict[str, LensTopKOutput]: Same output as `explain`.

        Examples:
            >>> results = lens.lens(
            ...     ["Interpreto is helpful", "Interpreto is practical"],
            ...     label_names={0: "negative", 1: "positive"},
            ... )
        """
        model_inputs = self._prepare_inputs(inputs)
        results = self._explain_prepared(model_inputs, split_points=split_points)
        from interpreto.visualizations import display_lens_results  # noqa: PLC0415

        display_lens_results(
            results,
            model_inputs,
            tokenizer=self.tokenizer,
            task=self.task,
            label_names=label_names,
        )
        return results
