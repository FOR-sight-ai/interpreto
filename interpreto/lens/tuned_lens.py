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

"""Tuned Lens implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers.tokenization_utils_base import BatchEncoding

from interpreto.concepts.splitters.model_with_split_points import ModelWithSplitPoints

from ._lens_base import BaseLens, LensInputs, PoolingStrategy

InitializationMode = Literal["default", "xavier", "logit_lens"]
_CHECKPOINT_FORMAT_VERSION = 1


class TunedLens(BaseLens):
    """Code: [:octicons-mark-github-24: `lens/tuned_lens.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/lens/tuned_lens.py)

    Learn an affine translator before applying the model prediction head.

    The translator follows the Tuned Lens method introduced by
    [Belrose et al., 2023](https://arxiv.org/abs/2303.08112). For a standard tuned lens,
    configure `ModelWithSplitPoints` at a transformer block output so the captured tensor is
    a residual-stream state. Other compatible activations can be projected, but do not have the
    same interpretation.

    Automatic projection is limited to model layouts whose final normalization and output head
    are represented by reusable modules. Explicit projection paths can be supplied for other
    layouts when those modules reproduce the model's actual output path.

    Args:
        model_with_split_points (ModelWithSplitPoints): Wrapped model containing the split point to translate.
        head_name (str | None): Optional path to the prediction head.
        pre_head_name (str | None): Optional path to a module applied before the prediction head.
        pooling_strategy (PoolingStrategy | None): Optional token pooling for a simple
            sequence-classification head.
        initialization_mode (InitializationMode): Translator initialization. ``"logit_lens"`` starts from the
            identity map, ``"xavier"`` uses Xavier uniform weights, and ``"default"`` keeps
            the PyTorch linear-layer initialization.
        top_k (int): Number of token or class scores returned per prediction.
        device (torch.device | str | None): Device used by the learned translator.

    Examples:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from interpreto import ModelWithSplitPoints, TunedLens
        >>> model_id = "hf-internal-testing/tiny-random-gpt2"
        >>> model = AutoModelForCausalLM.from_pretrained(model_id)
        >>> tokenizer = AutoTokenizer.from_pretrained(model_id)
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> wrapped_model = ModelWithSplitPoints(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     split_point="transformer.h.1",
        ... )
        >>> lens = TunedLens(wrapped_model, top_k=3)
        >>> _ = lens.fit(["Interpreto helps.", "Interpreto is practical."], epochs=1)
        >>> results = lens.explain("Interpreto is practical.")

    Raises:
        ValueError: If ``initialization_mode`` is invalid.

    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        head_name: str | None = None,
        pre_head_name: str | None = None,
        pooling_strategy: PoolingStrategy | None = None,
        initialization_mode: InitializationMode = "logit_lens",
        top_k: int = 5,
        device: torch.device | str | None = None,
    ) -> None:
        if initialization_mode not in {"default", "xavier", "logit_lens"}:
            raise ValueError("`initialization_mode` must be `'default'`, `'xavier'`, or `'logit_lens'`.")
        super().__init__(
            model_with_split_points=model_with_split_points,
            head_name=head_name,
            pre_head_name=pre_head_name,
            pooling_strategy=pooling_strategy,
            top_k=top_k,
            device=device,
        )

        self.initialization_mode = initialization_mode
        self.hidden_size = self._get_hidden_size()
        self.translator = nn.Linear(self.hidden_size, self.hidden_size, bias=True).to(self.device)
        self._initialize_translator()
        self.translator.eval()

    def _get_hidden_size(self) -> int:
        with self._model_in_evaluation_mode():
            latent_shape = self.model_with_split_points.get_latent_shape()
        if len(latent_shape) == 0:
            raise ValueError("The split activation has no hidden dimension.")
        hidden_size = latent_shape[-1]
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, Integral) or hidden_size < 1:
            raise ValueError(f"Invalid split hidden dimension: {hidden_size}.")
        return int(hidden_size)

    def _initialize_translator(self) -> None:
        if self.initialization_mode == "default":
            return
        if self.initialization_mode == "xavier":
            nn.init.xavier_uniform_(self.translator.weight)
        else:
            nn.init.eye_(self.translator.weight)
        nn.init.zeros_(self.translator.bias)

    def _transform_activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"The split activation width changed: expected {self.hidden_size}, got {hidden_states.shape[-1]}."
            )
        translator_input = hidden_states.to(
            device=self.translator.weight.device,
            dtype=self.translator.weight.dtype,
        )
        return self.translator(translator_input)

    def _prepare_loss_logits(
        self,
        projected_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if projected_logits.shape != target_logits.shape:
            raise ValueError(
                "Lens logits must have the same shape as final model logits. "
                f"Got {tuple(projected_logits.shape)} and {tuple(target_logits.shape)}."
            )
        projected_logits = self._upcast_for_scores(projected_logits)
        target_logits = target_logits.detach().to(
            device=projected_logits.device,
            dtype=projected_logits.dtype,
        )
        return projected_logits, target_logits

    def _language_model_loss(
        self,
        projected_logits: torch.Tensor,
        target_logits: torch.Tensor,
        model_inputs: BatchEncoding,
    ) -> tuple[torch.Tensor, int]:
        projected_logits, target_logits = self._prepare_loss_logits(projected_logits, target_logits)
        token_losses = (
            F.kl_div(
                F.log_softmax(projected_logits, dim=-1),
                F.softmax(target_logits, dim=-1),
                reduction="none",
            )
            .sum(dim=-1)
            .clamp_min(0.0)
        )
        valid_mask = self._get_attention_mask(model_inputs, projected_logits).bool()
        nb_evaluated_elements = int(valid_mask.sum().detach().cpu())
        if nb_evaluated_elements == 0:
            raise ValueError("Tuned Lens fitting requires at least one attended token.")
        return self._masked_mean(token_losses, valid_mask), nb_evaluated_elements

    def _sequence_classification_loss(
        self,
        projected_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        projected_logits, target_logits = self._prepare_loss_logits(projected_logits, target_logits)
        loss = F.kl_div(
            F.log_softmax(projected_logits, dim=-1),
            F.softmax(target_logits, dim=-1),
            reduction="batchmean",
        ).clamp_min(0.0)
        return loss, projected_logits.shape[0]

    def _compute_loss(
        self,
        projected_logits: torch.Tensor,
        target_logits: torch.Tensor,
        model_inputs: BatchEncoding,
    ) -> tuple[torch.Tensor, int]:
        if self.task == "language_model":
            return self._language_model_loss(projected_logits, target_logits, model_inputs)
        return self._sequence_classification_loss(projected_logits, target_logits)

    def _freeze_projection_parameters(self) -> list[tuple[nn.Parameter, bool]]:
        modules = [self.model_head]
        if self.model_pre_head is not None:
            modules.append(self.model_pre_head)

        parameter_states = []
        seen_parameter_ids: set[int] = set()
        for module in modules:
            for parameter in module.parameters():
                if id(parameter) in seen_parameter_ids:
                    continue
                seen_parameter_ids.add(id(parameter))
                parameter_states.append((parameter, parameter.requires_grad))
                parameter.requires_grad_(False)
        return parameter_states

    @staticmethod
    def _restore_projection_parameters(parameter_states: list[tuple[nn.Parameter, bool]]) -> None:
        for parameter, requires_grad in parameter_states:
            parameter.requires_grad_(requires_grad)

    @staticmethod
    def _validate_real(
        value: float,
        name: str,
        minimum: float,
        strict: bool,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"`{name}` must be a real number.")
        normalized = float(value)
        invalid_bound = normalized <= minimum if strict else normalized < minimum
        if not math.isfinite(normalized) or invalid_bound:
            qualifier = "greater than" if strict else "greater than or equal to"
            raise ValueError(f"`{name}` must be finite and {qualifier} {minimum}.")
        return normalized

    def fit(
        self,
        inputs: LensInputs,
        epochs: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Fit the translator to the final model distribution.

        Args:
            inputs (LensInputs): Raw text or tensor-backed tokenized training inputs.
            epochs (int): Number of passes over the inputs.
            learning_rate (float): AdamW learning rate.
            weight_decay (float): AdamW weight decay.
            batch_size (int | None): Optional fit batch size. The wrapped model batch size is used when omitted.

        Returns:
            dict[str, Any]: Loss history and fit metadata.

        Raises:
            RuntimeError: If final model logits cannot be captured.
        """
        self._validate_bound_split_point()
        self._validate_positive_integer(epochs, "epochs")
        learning_rate = self._validate_real(learning_rate, "learning_rate", 0.0, strict=True)
        weight_decay = self._validate_real(weight_decay, "weight_decay", 0.0, strict=False)
        if batch_size is None:
            fit_batch_size = self.model_with_split_points.batch_size
        else:
            self._validate_positive_integer(batch_size, "batch_size")
            fit_batch_size = int(batch_size)
        self._validate_positive_integer(fit_batch_size, "model_with_split_points.batch_size")
        self._get_input_size(inputs)

        optimizer = torch.optim.AdamW(
            self.translator.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        history: dict[str, Any] = {
            "loss": [],
            "split_point": self.split_point,
            "epochs": int(epochs),
        }
        parameter_states = self._freeze_projection_parameters()
        try:
            with self._model_in_evaluation_mode():
                self.translator.train()
                for _ in range(int(epochs)):
                    weighted_epoch_loss = 0.0
                    nb_evaluated_elements = 0
                    for _, _, batch_inputs in self._iter_input_batches(inputs, fit_batch_size):
                        with torch.no_grad():
                            activation, target_logits = self._capture_projection_input(
                                batch_inputs,
                                include_reference_logits=True,
                            )
                        if target_logits is None:  # pragma: no cover - guarded by capture
                            raise RuntimeError("Failed to capture final model logits during fitting.")

                        optimizer.zero_grad(set_to_none=True)
                        loss, batch_weight = self._compute_loss(
                            self._project_activation(activation),
                            target_logits,
                            batch_inputs,
                        )
                        loss.backward()
                        optimizer.step()
                        weighted_epoch_loss += float(loss.detach().cpu()) * batch_weight
                        nb_evaluated_elements += batch_weight

                    history["loss"].append(weighted_epoch_loss / nb_evaluated_elements)
        finally:
            optimizer.zero_grad(set_to_none=True)
            self.translator.eval()
            self._restore_projection_parameters(parameter_states)
        return history

    @staticmethod
    def _get_model_name_or_path(model: nn.Module) -> str | None:
        model_name_or_path = getattr(model.config, "_name_or_path", None)
        if model_name_or_path in {None, ""}:
            return None
        return str(model_name_or_path)

    def save(self, path: str | Path) -> None:
        """Save the translator and its declared model and projection contract.

        Args:
            path (str | Path): Checkpoint path.

        Returns:
            None: This method saves the checkpoint to disk.
        """
        self._validate_bound_split_point()
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "metadata": {
                "split_point": self.split_point,
                "head_name": self.head_name,
                "pre_head_name": self.pre_head_name,
                "pooling_strategy": self.pooling_strategy,
                "initialization_mode": self.initialization_mode,
                "top_k": self.top_k,
                "hidden_size": self.hidden_size,
                "task": self.task,
                "model_class": type(self.model).__name__,
                "model_name_or_path": self._get_model_name_or_path(self.model),
            },
            "translator": {name: tensor.detach().cpu() for name, tensor in self.translator.state_dict().items()},
        }
        torch.save(checkpoint, checkpoint_path)

    @staticmethod
    def _validate_checkpoint(checkpoint: Any) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        if not isinstance(checkpoint, Mapping):
            raise ValueError("A Tuned Lens checkpoint must be a mapping.")
        version = checkpoint.get("format_version")
        if isinstance(version, bool) or version != _CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported Tuned Lens checkpoint format: expected {_CHECKPOINT_FORMAT_VERSION}, got {version}."
            )
        metadata = checkpoint.get("metadata")
        translator = checkpoint.get("translator")
        if not isinstance(metadata, Mapping) or not isinstance(translator, Mapping):
            raise ValueError("A Tuned Lens checkpoint must contain `metadata` and `translator` mappings.")

        required_metadata = {
            "split_point",
            "head_name",
            "pre_head_name",
            "pooling_strategy",
            "initialization_mode",
            "top_k",
            "hidden_size",
            "task",
            "model_class",
            "model_name_or_path",
        }
        if not required_metadata.issubset(metadata):
            missing = sorted(required_metadata - set(metadata))
            raise ValueError(f"Tuned Lens checkpoint metadata is missing: {', '.join(missing)}.")
        if not isinstance(metadata["split_point"], str) or not isinstance(metadata["head_name"], str):
            raise ValueError("Checkpoint `split_point` and `head_name` values must be strings.")
        if metadata["pre_head_name"] is not None and not isinstance(metadata["pre_head_name"], str):
            raise ValueError("Checkpoint `pre_head_name` must be a string or `None`.")
        if metadata["pooling_strategy"] not in {None, "cls", "mean", "last"}:
            raise ValueError("Checkpoint `pooling_strategy` is invalid.")
        if metadata["initialization_mode"] not in {"default", "xavier", "logit_lens"}:
            raise ValueError("Checkpoint `initialization_mode` is invalid.")
        if isinstance(metadata["top_k"], bool) or not isinstance(metadata["top_k"], Integral) or metadata["top_k"] < 1:
            raise ValueError("Checkpoint `top_k` must be a strictly positive integer.")
        if (
            isinstance(metadata["hidden_size"], bool)
            or not isinstance(metadata["hidden_size"], Integral)
            or metadata["hidden_size"] < 1
        ):
            raise ValueError("Checkpoint `hidden_size` must be a strictly positive integer.")
        if metadata["task"] not in {"language_model", "sequence_classification"}:
            raise ValueError("Checkpoint `task` is invalid.")
        if not isinstance(metadata["model_class"], str):
            raise ValueError("Checkpoint `model_class` must be a string.")
        if metadata["model_name_or_path"] is not None and not isinstance(metadata["model_name_or_path"], str):
            raise ValueError("Checkpoint `model_name_or_path` must be a string or `None`.")
        if set(translator) != {"weight", "bias"} or not all(
            isinstance(tensor, torch.Tensor) for tensor in translator.values()
        ):
            raise ValueError("The checkpoint translator must contain tensor `weight` and `bias` entries.")
        return dict(metadata), dict(translator)

    @classmethod
    def from_checkpoint(
        cls,
        model_with_split_points: ModelWithSplitPoints,
        path: str | Path,
        device: torch.device | str | None = None,
    ) -> TunedLens:
        """Restore a tuned lens from a versioned tensor-only checkpoint.

        Args:
            model_with_split_points (ModelWithSplitPoints): Wrapped model used for restored inference.
            path (str | Path): Checkpoint path.
            device (torch.device | str | None): Device used by the restored translator.

        Returns:
            TunedLens: The restored tuned lens.

        Raises:
            ValueError: If the checkpoint schema or model contract is incompatible.
        """
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        metadata, translator_state = cls._validate_checkpoint(checkpoint)

        if metadata["split_point"] != model_with_split_points.split_point:
            raise ValueError(
                "The checkpoint split point does not match ModelWithSplitPoints: "
                f"{metadata['split_point']} != {model_with_split_points.split_point}."
            )
        if metadata["model_class"] != type(model_with_split_points._model).__name__:
            raise ValueError("The checkpoint was created for a different model class.")
        current_model_name_or_path = cls._get_model_name_or_path(model_with_split_points._model)
        if metadata["model_name_or_path"] != current_model_name_or_path:
            raise ValueError("The checkpoint was created for a different declared model identity.")

        lens = cls(
            model_with_split_points=model_with_split_points,
            head_name=metadata["head_name"],
            pre_head_name=metadata["pre_head_name"],
            pooling_strategy=metadata["pooling_strategy"],
            initialization_mode=metadata["initialization_mode"],
            top_k=metadata["top_k"],
            device=device,
        )
        if metadata["task"] != lens.task or metadata["hidden_size"] != lens.hidden_size:
            raise ValueError("The checkpoint task or hidden size is incompatible with the wrapped model.")
        try:
            lens.translator.load_state_dict(translator_state, strict=True)
        except RuntimeError as error:
            raise ValueError("The checkpoint translator state is incompatible with this split point.") from error
        lens.translator.eval()
        return lens
