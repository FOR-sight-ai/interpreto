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
Inference wrapper for image classification (e.g., ViT) models.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from transformers.modeling_utils import PreTrainedModel

from interpreto.model_wrapping.inference_wrapper import InferenceModes, InferenceWrapper
from interpreto.typing import TensorMapping


class ImageClassificationInferenceWrapper(InferenceWrapper):
    """
    Inference wrapper for image classification tasks (ViT-family models).

    Mirrors `ClassificationInferenceWrapper` with three modality-specific overrides:
    `__init__` drops the `pad_token_id` lookup (image configs have no pad token),
    `_prepare_inputs` stacks pixel tensors without padding, and `_compute_gradients`
    differentiates w.r.t. `pixel_values` and collapses the channel dimension.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        gradients: bool = False,
        input_x_gradient: bool = True,
        batch_size: int = 4,
        device: torch.device | None = None,
        mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
    ):
        # body mirrors InferenceWrapper.__init__ verbatim except for the dropped
        # `self.pad_token_id = model.config.pad_token_id` line (ViTConfig has no pad_token_id).
        # We can't call super() because the pad-token lookup happens inside it.
        self.model = model
        self.model.eval()
        self.gradients = gradients
        self.input_x_gradient = input_x_gradient

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not (getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)):
            self.model.to(device)  # type: ignore
        self.batch_size = batch_size

        assert callable(mode), "mode should be a callable function from `InferenceModes`"
        self.mode = mode

    @jaxtyped(typechecker=beartype)
    def _extract_targets_from_logits(self, logits: Float[torch.Tensor, "b c"]) -> Int[torch.Tensor, "b 1"]:
        """
        In classification, if no targets are specified, we explain the predicted class.
        The predicted class corresponds to the highest logits.
        Therefore, the target is the argmax of the output logits for each input sample.
        """
        return logits.argmax(dim=-1, keepdim=True)

    def _target_logits(
        self, logits: Float[torch.Tensor, "b c"], targets: Int[torch.Tensor, "t"]
    ) -> Float[torch.Tensor, "b t"]:
        """
        For each sample, the targets specify which logits to extract.

        The target is common between each sample.
        """
        return logits[:, targets]

    def _prepare_inputs(self, inputs: list[TensorMapping], for_gradients: bool = False) -> TensorMapping:
        """
        Stack a list of per-sample image tensor mappings into a single batched mapping.

        Unlike the text path, no padding is required: image processors produce
        fixed-shape tensors (3, H, W), so a plain `torch.stack` suffices.

        Args:
            inputs (list[TensorMapping]):
                Per-sample tensor mappings; each holds at minimum a `pixel_values`
                tensor of shape (3, H, W).
            for_gradients (bool, optional):
                Whether the inputs will be differentiated through.
                If True, `pixel_values` is detached and marked `requires_grad=True`.

        Returns:
            TensorMapping: Batched inputs on the model device.
        """
        pv = inputs[0]["pixel_values"]
        assert pv.ndim == 3 and pv.shape[0] == 3, (
            f"expected per-sample pixel_values of shape (3, H, W), got {tuple(pv.shape)}"
        )

        prepared = {
            key: torch.stack([elem[key] for elem in inputs]).to(self.device) for key in inputs[0].keys()
        }
        # key should always be pixel_values for ViT but we keep the more general form for later
        if for_gradients:
            prepared["pixel_values"] = prepared["pixel_values"].detach().requires_grad_(True)
        return prepared

    def _compute_gradients(
        self,
        inputs: TensorMapping,
        targeted_logits_chunk: Float[torch.Tensor, "c t"],
        chunk_slice: slice,
        last_chunk: bool,
    ) -> Float[torch.Tensor, "c t d l"]:
        """
        Compute the gradients of the targeted logits with respect to `pixel_values`.

        Done target by target to save memory. Spatial dimensions are flattened to
        `l = H*W`, but the 3 channels are kept as a `d` axis (signed, no abs):
        `(c, 3, H, W) -> (c, 3=d, H*W=l)`.

        Keeping channels signed and un-collapsed is what lets the cross-perturbation
        statistic (mean/var/squared-mean) in the aggregator be applied per channel —
        which VarGrad/SquareGrad need to compute the correct formula. The channel
        magnitude collapse (`abs().mean`) happens later, in the explainer, *after* the
        per-perturbation statistic. For image `d=3` is tiny, so unlike the text wrapper
        there is no memory reason to collapse early here.

        Args:
            inputs (TensorMapping):
                The batched inputs as returned by `_prepare_inputs(..., for_gradients=True)`.
                Must contain `pixel_values` with `requires_grad=True`.
            targeted_logits_chunk (torch.Tensor):
                The targeted logits associated to the input mappings, for the current chunk.
            chunk_slice (slice):
                The slice of the batch corresponding to the chunk.
            last_chunk (bool):
                Whether the chunk is the last one of the batch.
                If True, the autograd graph is not retained beyond the last target.

        Returns:
            gradients_chunk (torch.Tensor):
                Signed per-channel per-pixel gradients of shape `(c, t, 3, H*W)`.
        """
        c = chunk_slice.stop - chunk_slice.start
        #We don't need masking compared with the text logic
        pixel_values: Float[torch.Tensor, "b 3 h w"] = inputs["pixel_values"]  # type: ignore
        pixel_values_chunk: Float[torch.Tensor, f"{c} 3 h w"] = pixel_values[chunk_slice]

        t = targeted_logits_chunk.shape[1]
        gradients_list = []
        # iterate target-by-target — same pattern as text. For text this is mainly
        # a memory win (d=768/4096); for image (3 channels) the saving is small,
        # but we keep the structure for parity and so retain_graph timing matches.
        for k in range(t):
            last_target = k == (t - 1)
            # same autograd.grad call as text, just with pixel_values as the leaf
            # instead of inputs_embeds; retain_graph follows the text pattern
            target_wise_grads: Float[torch.Tensor, f"{c} 3 h w"] = torch.autograd.grad(
                outputs=targeted_logits_chunk[:, k].sum(),
                inputs=pixel_values,  # type: ignore
                retain_graph=not (last_target and last_chunk),
                create_graph=False,
            )[0][chunk_slice].detach()

            # input×gradient: same scheme as text, just multiplying against pixel_values_chunk
            # instead of inputs_embeds_chunk
            if self.input_x_gradient:
                target_wise_grads = target_wise_grads * pixel_values_chunk

            # Keep channels signed and un-collapsed: (c, 3, H, W) -> (c, 3=d, l=H*W).
            # The l ordering (row-major over H, W) matches the mask path and resize_to_image's
            # reshape(t, H, W). No abs / no channel mean here — those happen post-aggregator so the
            # per-perturbation statistic (mean/var/squared-mean) lands per channel (faithful VarGrad/
            # SquareGrad).
            target_wise_per_channel: Float[torch.Tensor, f"{c} d l"] = target_wise_grads.flatten(2, 3)
            gradients_list.append(target_wise_per_channel)

        # stack per-target results into (c, t, d, l)
        return torch.stack(gradients_list, dim=1)
