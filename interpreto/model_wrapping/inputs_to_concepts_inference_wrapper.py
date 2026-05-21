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

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

from interpreto.model_wrapping.inference_wrapper import InferenceWrapper
from interpreto.typing import IncompatibilityError


class InputsToConceptsInferenceWrapper(InferenceWrapper):
    """Inference wrapper for input-to-concept attribution tasks.

    Handles model inference for the ``ModelForInputsToConcepts`` bridge model,
    extracting concept activations from raw inputs. This wrapper is used internally
    by ``InputsToConceptsAttributionsExplainer`` to score input perturbations.

    Note:
        Gradient-based attribution methods are not supported with this wrapper.
        Use perturbation-based methods (Lime, KernelShap, Occlusion, or Sobol) instead.
    """

    def __init__(
        self,
        model,
        gradients: bool = False,
        batch_size: int = 4,
        device: torch.device | None = None,
        **kwargs,
    ):
        """Initialize the inference wrapper.

        Args:
            model (ModelForInputsToConcepts): The bridge model mapping inputs to concepts.
            gradients (bool): Must be False. Gradient-based methods are incompatible.
            batch_size (int): Batch size for inference.
            device (torch.device | None): Device for inference.
            **kwargs: Additional keyword arguments forwarded to ``InferenceWrapper``.

        Raises:
            IncompatibilityError: If ``gradients=True`` is requested.
        """
        if gradients:
            raise IncompatibilityError(
                "Inputs to concepts models do not support gradient-based methods."
                + " Please use a perturbation-based method (Lime, KernelShap, Occlusion, or Sobol).",
            )
        super().__init__(model, gradients=gradients, batch_size=batch_size, device=device, **kwargs)

    def _extract_targets_from_logits(self, logits):
        raise NotImplementedError(
            "InputsToConceptsInferenceWrapper does not support computing targets from logits. "
            "Concept targets should be provided explicitly."
        )

    @property
    def padding_side(self):
        return "right"

    @jaxtyped(typechecker=beartype)
    def _target_logits(
        self, logits: Float[torch.Tensor, "b c"], targets: Int[torch.Tensor, "t"]
    ) -> Float[torch.Tensor, "b t"]:
        """
        For each sample, the targets specify which logits to extract.

        The target is common between each sample.
        """
        return logits[:, targets]
