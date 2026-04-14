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
Tuned Lens implementation built on top of `ModelWithSplitPoints`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers.tokenization_utils_base import BatchEncoding

from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints

from ._lens_base import BaseLens, LensInputs, PoolingStrategy

InitializationMode = Literal["default", "xavier", "logit_lens"]


class TunedLens(BaseLens):
    """Code: [:octicons-mark-github-24: `lens/tuned_lens.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/lens/tuned_lens.py)

    Learn one affine translator per split point before applying the model prediction head.

    `TunedLens` follows the method introduced by
    [Belrose et al., 2023](https://arxiv.org/abs/2303.08112).
    The original work focuses on autoregressive language models.
    Interpreto extends the same idea to sequence classification by aligning translated hidden states
    with the frozen classifier distribution.
    See the `ModelWithSplitPoints` documentation for split-point selection and activation extraction.
    Raw text inputs are tokenized internally by the lens methods with the wrapped tokenizer.

    For sequence classification, the projection contract matches `LogitLens`:
    - a model-specific pooler or transform may be resolved before a vector head
    - a sequence-aware classification head may consume the 3D hidden states directly
    - a bare vector head requires an explicit `pooling_strategy`

    Args:
        model_with_split_points (ModelWithSplitPoints): Wrapped model used to extract split activations.
        split_points (str | list[str] | None): Split points receiving a learned translator.
            If `None`, translators are created for all split points registered on `model_with_split_points`.
        head_name (str | None): Optional path to the prediction head.
        pre_head_name (str | None): Optional path to a module applied before the head.
        pooling_strategy (Literal["cls", "mean", "last"] | None): Optional pooling used for
            sequence classification when the classification head expects one vector per sample.
            Pooling is only applied when it is explicitly requested for bare vector heads.
        initialization_mode (Literal["default", "xavier", "logit_lens"]): Initialization applied
            to each translator.
            - `default` keeps the current `torch.nn.Linear` initialization.
            - `xavier` uses Xavier uniform initialization with zero bias.
            - `logit_lens` initializes each translator as the identity map, which starts tuning
              from the plain logit-lens projection.
        top_k (int): Number of labels or tokens returned per prediction.
        device (torch.device | str | None): Device used by the learned translators.

    Examples:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from interpreto import ModelWithSplitPoints, TunedLens
        >>> model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        >>> if tokenizer.pad_token is None:
        ...     tokenizer.pad_token = tokenizer.eos_token
        >>> model_with_split_points = ModelWithSplitPoints(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     split_points="transformer.h.1.mlp",
        ... )
        >>> lens = TunedLens(model_with_split_points, top_k=3, initialization_mode="logit_lens")
        >>> _ = lens.fit(["Interpreto helps.", "Interpreto is practical."], epochs=1)
        >>> explanations = lens.explain("Interpreto is practical.")

    Raises:
        ValueError: If the split points are invalid or if the checkpoint metadata is incompatible.
        NotImplementedError: If the wrapped model is a token classification model.
        RuntimeError: If the resolved projection contains meta tensors.
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        split_points: str | list[str] | None = None,
        head_name: str | None = None,
        pre_head_name: str | None = None,
        pooling_strategy: PoolingStrategy | None = None,
        initialization_mode: InitializationMode = "logit_lens",
        top_k: int = 5,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(
            model_with_split_points=model_with_split_points,
            head_name=head_name,
            pre_head_name=pre_head_name,
            pooling_strategy=pooling_strategy,
            top_k=top_k,
            device=device,
        )

        if initialization_mode not in {"default", "xavier", "logit_lens"}:
            raise ValueError(
                "`initialization_mode` should be one of `'default'`, `'xavier'`, or `'logit_lens'`."
            )

        self.translated_split_points = self._normalize_split_points(split_points)
        self.hidden_size = self._get_hidden_size()
        self.initialization_mode = initialization_mode
        self.translators = {
            split_point: self._build_translator().to(self.device)
            for split_point in self.translated_split_points
        }

    def _build_translator(self) -> nn.Linear:
        translator = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self._initialize_translator(translator)
        return translator

    def _initialize_translator(self, translator: nn.Linear) -> None:
        if self.initialization_mode == "default":
            return

        if self.initialization_mode == "xavier":
            nn.init.xavier_uniform_(translator.weight)
            if translator.bias is not None:
                nn.init.zeros_(translator.bias)
            return

        nn.init.eye_(translator.weight)
        if translator.bias is not None:
            nn.init.zeros_(translator.bias)

    def _get_hidden_size(self) -> int:
        for attribute_name in ["hidden_size", "n_embd", "d_model", "word_embed_proj_dim"]:
            hidden_size = getattr(self.model.config, attribute_name, None)
            if isinstance(hidden_size, int):
                return hidden_size

        modules = [self.model_head]
        if self.model_pre_head is not None:
            modules.append(self.model_pre_head)

        for module in modules:
            if hasattr(module, "in_features") and isinstance(module.in_features, int):
                return module.in_features

        raise ValueError(
            "Could not determine the hidden size required by the tuned lens translators. "
            "Please use a model exposing a standard transformer hidden size."
        )

    def _transform_activations(self, split_point: str, hidden_states: torch.Tensor) -> torch.Tensor:
        if split_point not in self.translators:
            raise ValueError(
                f"Split point `{split_point}` does not have a tuned translator. "
                f"Available split points: {', '.join(self.translators)}."
            )

        translator = self.translators[split_point]
        return translator(hidden_states.to(self.device)).to(self.model_device)

    def _iter_translator_parameters(self, split_points: list[str]) -> Iterator[nn.Parameter]:
        for split_point in split_points:
            yield from self.translators[split_point].parameters()

    def _language_model_loss(
        self,
        projected_logits: torch.Tensor,
        target_logits: torch.Tensor,
        model_inputs: BatchEncoding,
    ) -> torch.Tensor:
        target_probabilities = F.softmax(target_logits, dim=-1)
        log_probabilities = F.log_softmax(projected_logits, dim=-1)
        token_losses = F.kl_div(log_probabilities, target_probabilities, reduction="none").sum(dim=-1)
        attention_mask = self._get_attention_mask(model_inputs, projected_logits)
        valid_losses = token_losses.masked_select(attention_mask.bool())
        return valid_losses.mean()

    def _sequence_classification_loss(self, projected_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
        return F.kl_div(
            F.log_softmax(projected_logits, dim=-1),
            F.softmax(target_logits, dim=-1),
            reduction="batchmean",
        )

    def _compute_loss(
        self,
        projected_logits: torch.Tensor,
        target_logits: torch.Tensor,
        model_inputs: BatchEncoding,
    ) -> torch.Tensor:
        if self.task == "language_model":
            return self._language_model_loss(projected_logits, target_logits, model_inputs)
        return self._sequence_classification_loss(projected_logits, target_logits)

    def _freeze_projection_parameters(self) -> list[tuple[nn.Parameter, bool]]:
        projection_modules = [self.model_head]
        if self.model_pre_head is not None:
            projection_modules.append(self.model_pre_head)

        parameters_state: list[tuple[nn.Parameter, bool]] = []
        for module in projection_modules:
            for parameter in module.parameters():
                parameters_state.append((parameter, parameter.requires_grad))
                parameter.requires_grad_(False)
        return parameters_state

    def _restore_projection_parameters(self, parameters_state: list[tuple[nn.Parameter, bool]]) -> None:
        for parameter, requires_grad in parameters_state:
            parameter.requires_grad_(requires_grad)

    def fit(
        self,
        inputs: LensInputs,
        split_points: str | list[str] | None = None,
        epochs: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Fit translators on raw inputs.

        Args:
            inputs (str | list[str] | BatchEncoding): Training inputs.
            split_points (str | list[str] | None): Optional subset of translated split points.
            epochs (int): Number of training epochs.
            learning_rate (float): Optimizer learning rate.
            weight_decay (float): Optimizer weight decay.
            batch_size (int | None): Batch size used during tuning.
                If `None`, reuse `model_with_split_points.batch_size`.

        Returns:
            dict[str, Any]: Training history containing the mean loss per epoch and the fitted split points.
        """
        if epochs < 1:
            raise ValueError("`epochs` must be a strictly positive integer.")

        selected_split_points = self._normalize_split_points(split_points or self.translated_split_points)
        unknown_split_points = [split_point for split_point in selected_split_points if split_point not in self.translators]
        if unknown_split_points:
            raise ValueError(
                "The following split points do not have translators: " + ", ".join(unknown_split_points) + "."
            )

        fit_batch_size = batch_size or self.model_with_split_points.batch_size
        optimizer = torch.optim.AdamW(
            self._iter_translator_parameters(selected_split_points),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        history = {
            "loss": [],
            "split_points": list(selected_split_points),
            "epochs": epochs,
        }

        parameters_state = self._freeze_projection_parameters()
        try:
            for translator in self.translators.values():
                translator.train()

            for _ in range(epochs):
                epoch_loss = 0.0
                nb_batches = 0
                for batch_inputs in self._iter_batches(inputs, fit_batch_size):
                    model_inputs = self._prepare_inputs(batch_inputs)
                    projection_inputs, target_logits = self._capture_projection_inputs(
                        model_inputs,
                        selected_split_points,
                        differentiable=False,
                        include_reference_logits=True,
                    )
                    if target_logits is None:
                        raise RuntimeError("Failed to capture the reference logits required by the tuned lens fit path.")

                    optimizer.zero_grad()
                    total_loss: torch.Tensor | None = None
                    for _, projected_logits in self._iter_projected_logits(
                        projection_inputs,
                        selected_split_points,
                    ):
                        split_loss = self._compute_loss(projected_logits, target_logits, model_inputs)
                        total_loss = split_loss if total_loss is None else total_loss + split_loss

                    if total_loss is None:
                        raise RuntimeError("No split point was selected for tuning.")

                    total_loss = total_loss / len(selected_split_points)
                    total_loss.backward()
                    optimizer.step()

                    epoch_loss += float(total_loss.detach().cpu())
                    nb_batches += 1

                history["loss"].append(epoch_loss / max(nb_batches, 1))
        finally:
            self._restore_projection_parameters(parameters_state)
            for translator in self.translators.values():
                translator.eval()

        return history

    def save(self, path: str | Path) -> None:
        """
        Save the tuned translators and their configuration.

        Args:
            path (str | pathlib.Path): Checkpoint path.
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "metadata": {
                "split_points": list(self.translated_split_points),
                "head_name": self.head_name,
                "pre_head_name": self.pre_head_name,
                "pooling_strategy": self.pooling_strategy,
                "initialization_mode": self.initialization_mode,
                "top_k": self.top_k,
            },
            "translators": {
                split_point: translator.state_dict() for split_point, translator in self.translators.items()
            },
        }
        torch.save(checkpoint, checkpoint_path)

    @classmethod
    def from_checkpoint(
        cls,
        model_with_split_points: ModelWithSplitPoints,
        path: str | Path,
        device: torch.device | str | None = None,
    ) -> TunedLens:
        """
        Load a tuned lens from a checkpoint.

        Args:
            model_with_split_points (ModelWithSplitPoints): Wrapped model used for inference.
            path (str | pathlib.Path): Checkpoint path.
            device (torch.device | str | None): Device used by the learned translators.

        Returns:
            TunedLens: Restored tuned lens.
        """
        checkpoint = torch.load(Path(path), map_location="cpu")
        metadata = checkpoint["metadata"]
        lens = cls(
            model_with_split_points=model_with_split_points,
            split_points=metadata["split_points"],
            head_name=metadata["head_name"],
            pre_head_name=metadata["pre_head_name"],
            pooling_strategy=metadata["pooling_strategy"],
            initialization_mode=metadata.get("initialization_mode", "logit_lens"),
            top_k=metadata["top_k"],
            device=device,
        )

        missing_split_points = [
            split_point
            for split_point in lens.translated_split_points
            if split_point not in checkpoint["translators"]
        ]
        if missing_split_points:
            raise ValueError(
                "Missing translator states for split points: " + ", ".join(missing_split_points) + "."
            )

        for split_point in lens.translated_split_points:
            lens.translators[split_point].load_state_dict(checkpoint["translators"][split_point])
            lens.translators[split_point].eval()

        return lens
