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
        **kwargs (Any): Additional arguments passed to NNsight's ``LanguageModel``.

    Raises:
        ValueError: If the model does not contain a non-empty module list.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from interpreto import AllLayersSplitter
        >>> splitter = AllLayersSplitter("gpt2")
        >>> activations = splitter.get_activations("Interpreto is useful.")
        >>> len(activations) == len(splitter.split_points) + 1
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
        super().__init__(
            model_or_repo_id,
            config=config,
            tokenizer=tokenizer,
            automodel=automodel,
            device_map=device_map,
            **kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        module_lists = [
            (name, module)
            for name, module in self._model.named_modules()
            if name and isinstance(module, nn.ModuleList) and len(module) > 0
        ]
        if not module_lists:
            raise ValueError("Could not find a non-empty ModuleList containing the model's transformer blocks.")

        layer_name, layers = max(module_lists, key=lambda item: len(item[1]))
        self.split_points = [f"model.{layer_name}.{index}" for index in range(len(layers))]
        self._block_output_arity: list[int | None] = [None] * len(self.split_points)

    @property
    def activation_names(self) -> list[str]:
        """Names of the residual states returned by :meth:`get_activations`."""
        return [f"{self.split_points[0]}.input", *self.split_points]

    def _prepare_input(
        self,
        *inputs: Any,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[tuple[Any, ...], dict[str, Any], int]:
        """Let NNsight create an invocation from activation inputs."""
        if inputs_embeds is None:
            return super()._prepare_input(*inputs, **kwargs)
        if inputs:
            raise ValueError("`inputs_embeds` cannot be combined with positional inputs.")
        return (), {"inputs_embeds": inputs_embeds, **kwargs}, inputs_embeds.shape[0]

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
                every transformer block output in ``split_points`` order.
                Each tensor has shape ``(1, sequence_length, model_width)``.
        """
        with torch.no_grad():
            with self.trace(inputs) as tracer:
                cached = tracer.cache(
                    modules=self.split_points,
                    include_inputs=True,
                    include_output=True,
                )

        # extract first activation as the first layer input
        first_layer = self.split_points[0]
        activations = [self._hidden_state(cached[first_layer].input, f"{first_layer}.input")]

        # extract following activations as the layers outputs
        outputs = [cached[layer_name].output for layer_name in self.split_points]
        # Some Transformers blocks wrap their hidden state in a tuple.
        self._block_output_arity = [len(output) if isinstance(output, tuple) else None for output in outputs]
        activations.extend(
            self._hidden_state(output, layer_name)
            for output, layer_name in zip(outputs, self.split_points, strict=True)
        )
        return activations

    def apply_head(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped model's prediction head to residual activations.

        The transformer blocks are skipped and ``activations`` are used as
        their output. The wrapped model then executes its own downstream
        normalization, pooling, and prediction head. This avoids
        architecture-specific head names and preserves functional operations
        implemented in model ``forward`` methods.

        Args:
            activations (torch.Tensor): Residual activations with shape
                ``(n, sequence_length, model_width)``. The leading dimension
                may represent several layer boundaries from the same input.

        Returns:
            torch.Tensor: Logits returned by the wrapped model for every
                activation in the leading dimension.
        """
        embedding_width = self._model.get_input_embeddings().weight.shape[-1]
        # The embedding code still validates its input even though every block is skipped.
        inputs_embeds = activations.new_zeros((*activations.shape[:-1], embedding_width))
        with self.trace(inputs_embeds=inputs_embeds):
            for split_point, output_arity in zip(self.split_points, self._block_output_arity, strict=True):
                replacement = activations if output_arity is None else (activations,) + (None,) * (output_arity - 1)
                self.get(split_point.removeprefix("model.")).skip(replacement)
            logits = self.output.logits.save()
        return logits
