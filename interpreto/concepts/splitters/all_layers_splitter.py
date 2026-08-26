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

"""Extraction of the residual stream at every transformer block boundary."""

from __future__ import annotations

from typing import Any

import torch
from nnsight.modeling.language import LanguageModel
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


class AllLayersSplitter(LanguageModel):
    """Extract the residual stream before and after every transformer block.

    The transformer blocks are inferred from the longest non-empty
    :class:`torch.nn.ModuleList` in the model. Activations are returned in model
    order: the input to the first block followed by the output of every block.
    A model with ``L`` transformer blocks therefore returns ``L + 1`` tensors
    of shape ``(1, sequence_length, model_width)``.

    This splitter is intended for methods that compare representations across
    model depths, such as Logit Lens and Tuned Lens. It does not implement the
    concept-specific ``BaseSplitter`` interface because there is no single split
    point or latent representation.

    Args:
        model_or_repo_id (str | PreTrainedModel): Hugging Face repository ID,
            local checkpoint path, or preloaded model.
        automodel (type[AutoModel]): Hugging Face AutoClass used when loading a
            model from a repository ID or local path.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast | None):
            Tokenizer associated with a preloaded model.
        config (PretrainedConfig | None): Optional model configuration passed
            to the model loader.
        device_map (torch.device | str | None): Device map passed to the model
            loader.
        **kwargs: Additional arguments passed to NNsight's ``LanguageModel``.

    Attributes:
        layer_split_points (list[str]): Dotted paths of the transformer blocks
            whose outputs are extracted, in model order.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from interpreto import AllLayersSplitter
        >>> splitter = AllLayersSplitter("gpt2")
        >>> activations = splitter.get_activations("Interpreto is useful.")
        >>> len(activations) == len(splitter.layer_split_points) + 1
        True
    """

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        *,
        automodel: type[AutoModel] = AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        config: PretrainedConfig | None = None,
        device_map: torch.device | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model and discover its transformer blocks."""
        super().__init__(
            model_or_repo_id,
            config=config,
            tokenizer=tokenizer,
            automodel=automodel,
            device_map=device_map,
            **kwargs,
        )

        module_lists = [
            (name, module)
            for name, module in self._model.named_modules()
            if name and isinstance(module, nn.ModuleList) and len(module) > 0
        ]
        if not module_lists:
            raise ValueError("Could not find a non-empty ModuleList containing the model's transformer blocks.")

        layer_name, layers = max(module_lists, key=lambda item: len(item[1]))
        self.layer_split_points = [f"{layer_name}.{index}" for index in range(len(layers))]

    @staticmethod
    def _hidden_state(value: Any, layer_name: str) -> torch.Tensor:
        """Extract a batched hidden state from a block input or output."""
        if isinstance(value, torch.Tensor) and value.ndim == 3:
            return value
        if isinstance(value, tuple):
            for candidate in value:
                if isinstance(candidate, torch.Tensor) and candidate.ndim == 3:
                    return candidate
        raise RuntimeError(f"Could not extract a 3D hidden state at `{layer_name}`.")

    def get_activations(self, inputs: str) -> list[torch.Tensor]:
        """Extract the residual stream for one text input.

        Args:
            inputs (str): Text passed to the wrapped model.

        Returns:
            list[torch.Tensor]: Input to the first transformer block followed by
                every transformer block output in ``layer_split_points`` order.
                Each tensor has shape ``(1, sequence_length, model_width)``.
        """
        with torch.no_grad():
            with self.trace(inputs) as tracer:
                cached = tracer.cache(
                    modules=self.layer_split_points,
                    include_inputs=True,
                    include_output=True,
                )

        # extract first activation as the first layer input
        first_layer = self.layer_split_points[0]
        activations = [self._hidden_state(cached[first_layer].input, f"{first_layer}.input")]

        # extract following activations as the layers outputs
        activations.extend(
            self._hidden_state(cached[layer_name].output, layer_name) for layer_name in self.layer_split_points
        )
        return activations

    def apply_head(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped model's prediction head to residual activations.

        The final transformer block is skipped and ``activations`` are used as
        its output. The wrapped model then executes its own downstream
        normalization, pooling, and prediction head. This avoids
        architecture-specific head names and preserves functional operations
        implemented in model ``forward`` methods.

        Args:
            activations (torch.Tensor): Residual activations with shape
                ``(1, sequence_length, model_width)``.

        Returns:
            torch.Tensor: Logits returned by the wrapped model, including the
                singleton batch dimension.
        """
        last_layer = self.get(self.layer_split_points[-1])
        with self.trace(inputs_embeds=activations):
            last_layer.skip(activations)
            logits = self.output.logits.save()
        return logits
