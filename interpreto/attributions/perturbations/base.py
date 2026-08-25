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
Base merged classes for perturbations used in attribution methods
"""
# TODO : remake all the docstrings of this file to fit with new method signatures

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from transformers.image_processing_utils import BaseImageProcessor

from interpreto.commons.granularity import (
    Granularity,
    GranularityCombinationStrategy,
    GranularityResizeStrategy,
    ImageGranularity,
)
from interpreto.typing import TensorMapping


class Perturbator(ABC):
    """
    Abstract Base class for perturbators.
    """

    # TODO: docstring — see todo/2026-07-30-to-do.md

    # Only what every perturbator has, whatever the modality and whatever the strategy.
    # Every other field is declared by the subclass that introduces it.
    __slots__ = (
        "processor",
        "granularity",
        "n_perturbations",
    )

    def __init__(
        self,
        *,
        processor: PreTrainedTokenizer | BaseImageProcessor | None = None,
        granularity: Granularity | None = None,
        n_perturbations: int = 1,
        **kwargs,
    ):
        # End of the cooperative chain: every class below has taken its own arguments out of
        # kwargs, so whatever is left was never claimed by anyone.
        if kwargs:
            raise ValueError(
                f"Unexpected arguments: {sorted(kwargs)}. Either too many arguments were given, "
                "or one of them is misspelled."
            )
        self.processor = processor
        self.granularity = granularity
        self.n_perturbations = n_perturbations

    @abstractmethod
    def perturb(self, inputs):
        pass

    def __call__(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        return self.perturb(model_inputs)


class TensorPerturbator(Perturbator):  # new class (just for typing and clarity)
    """
    Specific class for perturbators working on input embeddings
    All perturbators working on input embeddings only should inherit from this class

    By default, it only convert input IDs to embeddings using the model's input embedder.
    """

    __slots__ = ("inputs_embedder",)

    def __init__(self, *, inputs_embedder: torch.nn.Module | None = None, **kwargs):
        """
        Args:
            inputs_embedder: module converting input IDs to embeddings. Optional: the image
                side works directly on `pixel_values` and has no need for an embedder.
        """
        super().__init__(**kwargs)
        self.inputs_embedder = None if inputs_embedder is None else deepcopy(inputs_embedder).cpu()

    @abstractmethod
    def perturb_tensor(self):  # renaming of `perturb_embeds`
        pass


class TextTensorPerturbator(TensorPerturbator):
    """
    Text specific class that inherits from TensorPerturbator.
    """

    __slots__ = ()

    def __init__(self, *, inputs_embedder: torch.nn.Module, **kwargs):
        """
        Args:
            inputs_embedder: Model's module to convert input IDs to embeddings. Optional on
                `TensorPerturbator` but mandatory here: `perturb` cannot embed anything without it.
        """
        super().__init__(inputs_embedder=inputs_embedder, **kwargs)

    def perturb(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        # very input_ids are present
        if "input_ids" not in model_inputs:
            raise ValueError("model_inputs should contain 'input_ids'")

        inputs = deepcopy(model_inputs)

        # extract input ids
        input_ids: Float[torch.Tensor, "1 l"] = inputs["input_ids"].to(self.inputs_embedder.weight.device)  # type: ignore

        # convert input ids to embeddings
        inputs_embeds: Float[torch.Tensor, "1 l d"] = self.inputs_embedder(input_ids)

        # perturb embeddings
        perturbed_embeds: Float[torch.Tensor, "p l d"]
        mask: Float[torch.Tensor, "p l"] | None
        perturbed_embeds, mask = self.perturb_tensor(inputs_embeds)

        # repeat inputs elements to match perturbations
        p = perturbed_embeds.shape[0]
        for key, tensor in inputs.items():
            # repeat the first dimension to match perturbations
            inputs[key] = tensor.repeat((p, 1, 1)[: tensor.dim()])

        inputs["inputs_embeds"] = perturbed_embeds

        return inputs, mask

    def perturb_tensor(
        self, inputs_embeds: Float[torch.Tensor, "1 l d"]
    ) -> tuple[Float[torch.Tensor, "p l d"], Float[torch.Tensor, "p l"] | None]:
        """
        Perturb the input of the model, given as embeddings.
        This method should be implemented in subclasses to indeed perturb the embeddings.

        Args:
            inputs_embeds (torch.Tensor):
                Embeddings of the input tokens.
                Shape: (1, l, d)
        Returns:
            perturbed_embeds (torch.Tensor):
                Perturbed embeddings.
                Shape: (p, l, d)
            mask (torch.Tensor | None):
                Perturbation mask.
                Shape: (p, l)
        """
        return inputs_embeds, None


class MaskPerturbator(Perturbator):  # new class (just for typing and clarity)
    """
    Perturbator that hides granularity units by overwriting them with a baseline, rather than
    editing the input tensor directly.

    Owns the fields that only make sense for Mask based methods:
    the baseline written into masked units, the strategy used for the expansion, and
    the patch grid that defines the units on the image side.
    """

    __slots__ = (
        "replace_value",
        "granularity_combination_strategy",
        "patch_size",
    )

    def __init__(
        self,
        *,
        replace_value: int | float = 0.0,
        granularity_combination_strategy: GranularityCombinationStrategy = GranularityResizeStrategy.NEAREST,
        patch_size: int | None = None,
        **kwargs,
    ):
        """
        Args:
            replace_value: baseline written into masked units.
            granularity_combination_strategy: how a `(p, g)` mask is expanded to input resolution.
            patch_size: side length of a patch on the image side. Accepted here so the user can
                eventually choose it, but the image explainer currently overwrites it with
                `model.config.patch_size` at construction — see `base_merged.py:1103-1105`.
        """
        super().__init__(**kwargs)
        self.replace_value = replace_value
        self.granularity_combination_strategy = granularity_combination_strategy
        self.patch_size = patch_size

    @abstractmethod
    def get_mask(self):
        pass


class TextMaskPerturbator(MaskPerturbator):
    """
    Base class for perturbations consisting in applying masks on token (or groups of tokens)
    All perturbators working on input IDs by applying a mask should inherit from this class

    This class is combined with a method perturbator at runtime.
    """

    __slots__ = ()

    @jaxtyped(typechecker=beartype)
    @staticmethod
    def apply_mask(
        inputs: Int[torch.Tensor, "l 1"],
        mask: Float[torch.Tensor, "p g"],
        mask_value: torch.Tensor,
    ) -> Float[torch.Tensor, "p l"]:
        """
        Basic mask application method.

        If last dimension `d` is 1 (in case of tokens and not embeddings), this last dimension will be squeezed out
        and the returned tensor will have shape (num_sequences, n_perturbations, mask_dim).

        Args:
            inputs (torch.Tensor): inputs to mask
            mask (torch.Tensor): mask matrix to apply
            mask_value (torch.Tensor): tensor used as a mask (mask token, zero tensor, etc.)

        Returns:
            torch.Tensor: masked inputs
        """
        base: Float[torch.Tensor, "p l d"] = inputs.unsqueeze(-3) * (1 - mask).unsqueeze(
            -1
        )  # torch.einsum("ld,pl->pld", inputs, 1 - mask)
        masked: Float[torch.Tensor, "p l d"] = mask_value * mask.unsqueeze(
            -1
        )  # torch.einsum("pl,d->pld", mask, mask_value)
        return (base + masked).squeeze(-1)

    @jaxtyped(typechecker=beartype)
    @abstractmethod
    def get_mask(self, mask_dim: int, **kwargs) -> Float[torch.Tensor, "{self.n_perturbations} {mask_dim}"]:
        """
        Method returning a perturbation mask for a given set of inputs
        This method should be implemented in subclasses

        The created mask should be of size (n_perturbations, mask_dim)
        where mask_dim is the length of the sequence according to the granularity level (number of tokens, number of words, number of sentences...)

        Args:
            mask_dim (int): length of the sequence according to the granularity level
            kwargs: additional arguments if needed by the specific implementation of the mask

        Returns:
            torch.Tensor: mask to apply on the inputs, of shape (n_perturbations, mask_dim)
        """
        raise NotImplementedError()

    def perturb(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        """
        Method called to perturb the inputs of the model

        Args:
            model_inputs (MutableMapping): mapping given by the tokenizer

        Returns:
            tuple: model_inputs with perturbations and the specific granularity mask
        """
        model_inputs = model_inputs.copy()  # type: ignore

        if model_inputs["input_ids"].shape[0] != 1:
            raise ValueError(
                "Inputs are treated one by on one in the perturbator."
                + "But received a batch of inputs."
                + f"input_ids shape: {model_inputs['input_ids'].shape} - expected shape: (1, l)"
            )

        # compute association matrix between the granularity level and ALL_TOKENS
        association_matrix: Int[torch.Tensor, "g l"] = self.granularity.get_association_matrix(
            model_inputs,  # type: ignore
            self.processor,
        )[0].float()

        # compute granularity-wise perturbation mask based on the length of the sequence (granularity-wise)
        gran_mask: Float[torch.Tensor, "p g"] = self.get_mask(association_matrix.shape[0])

        # compute real perturbation mask
        real_mask: Float[torch.Tensor, "p l"] = gran_mask @ association_matrix

        model_inputs["input_ids"] = (
            self.apply_mask(
                inputs=model_inputs["input_ids"].T,
                mask=real_mask,
                mask_value=torch.Tensor([self.replace_value]),
            )
            .squeeze(-1)
            .to(torch.int)
        )

        # Repeat other keys in encoding for each perturbation
        for k in model_inputs.keys():
            if k != "input_ids":
                repeats = [1] * (model_inputs[k].dim())
                repeats[0] = model_inputs["input_ids"].shape[0]
                model_inputs[k] = model_inputs[k].repeat(*repeats)
        return model_inputs, gran_mask


class ImageTensorPerturbator(TensorPerturbator):
    """
    Image-side analog of `TextTensorPerturbator`.

    Operates directly on `pixel_values` of shape `(1, 3, H, W)`.

    Subclasses override `perturb_tensor` to produce a `(p, 3, H, W)` batch.

    Used by gradient-style methods (Saliency, SmoothGrad, IntegratedGradient).

    This class is combined with a method perturbator at runtime.
    """

    __slots__ = ()

    def perturb(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        if "pixel_values" not in model_inputs:
            raise ValueError("model_inputs should contain 'pixel_values'")

        inputs = deepcopy(model_inputs)
        pixel_values: Float[torch.Tensor, "1 3 H W"] = inputs["pixel_values"]

        perturbed_embeds: Float[torch.Tensor, "p 3 H W"]
        mask: Float[torch.Tensor, "p g"] | None
        perturbed_embeds, mask = self.perturb_tensor(pixel_values)

        inputs["pixel_values"] = perturbed_embeds
        return inputs, mask

    def perturb_tensor(
        self, pixel_values: Float[torch.Tensor, "1 3 H W"]
    ) -> tuple[Float[torch.Tensor, "p 3 H W"], Float[torch.Tensor, "p g"] | None]:
        """
        Default no-op: subclasses override to apply noise / interpolation / etc.

        Args:
            pixel_values: Shape (1, 3, H, W).
        Returns:
            perturbed_embeds: Shape (p, 3, H, W).
            mask: (p, g) or None.
        """
        return pixel_values, None


class ImageMaskPerturbator(MaskPerturbator):
    """
    Image-side analog of `TextMaskPerturbator`.

    Masking-based perturbator: each perturbation hides a subset of granularity
    units by overwriting their pixel positions with a
    constant `replace_value` baseline. This is the basis for perturbation
    methods (Occlusion, LIME, KernelShap, Sobol).

    perturb uses an interpolation strategy rather than an association matrix
    which was deemed more natural for images.

    This class is combined with a method perturbator at runtime.
    """

    __slots__ = ()

    @jaxtyped(typechecker=beartype)
    @abstractmethod
    def get_mask(self, mask_dim: int, **kwargs) -> Float[torch.Tensor, "{self.n_perturbations} {mask_dim}"]:
        """
        Return the granularity-wise perturbation mask, of shape `(n_perturbations, g)`.

        `mask_dim` is `g`, the number of granularity units. `1` marks a masked
        unit (replaced by the baseline), `0` marks a kept unit. Implemented by
        subclasses (random masking for LIME, single-unit masking for Occlusion, etc.).

        Args:
            mask_dim (int): number of granularity units `g`.
            kwargs: extra arguments for specific mask strategies.

        Returns:
            torch.Tensor: mask of shape `(n_perturbations, g)`.
        """
        raise NotImplementedError()

    def perturb(self, model_inputs: TensorMapping) -> tuple[TensorMapping, torch.Tensor | None]:
        if "pixel_values" not in model_inputs:
            raise ValueError("model_inputs should contain 'pixel_values'")

        inputs = deepcopy(model_inputs)
        pixel_values: Float[torch.Tensor, "1 3 H W"] = inputs["pixel_values"]

        if pixel_values.shape[0] != 1:
            raise ValueError(
                "Inputs are treated one by one in the perturbator, "
                f"but received pixel_values of shape {tuple(pixel_values.shape)} "
                "- expected shape (1, 3, H, W)."
            )

        _, c, h, w = pixel_values.shape
        l = h * w

        # PATCH is set directly by the explainer by reading model.config.
        if self.granularity == ImageGranularity.PATCH and self.patch_size is None:
            raise ValueError(
                "patch_size is None. It must be set "
                "from the model config. Normally the explainer does this at construction. "
                "If using the perturbator standalone, pass patch_size explicitly."
            )

        # granularity grid dims: PIXEL is per-pixel (identity resize), PATCH is the patch grid
        if self.granularity == ImageGranularity.PIXEL:
            gh, gw = h, w
        else:  # PATCH — patch_size guaranteed non-None by the guard above
            gh, gw = h // self.patch_size, w // self.patch_size

        # granularity-wise mask from the subclass: (p, g) with g = gh * gw
        gran_mask: Float[torch.Tensor, "p g"] = self.get_mask(gh * gw)
        p = gran_mask.shape[0]

        # expand to pixel space by resizing the (gh, gw) grid up to (h, w): (p, l)
        # NEAREST (the default) replicates each unit over its pixels and is bit-identical to the
        # old `gran_mask @ association_matrix`; output_size is computed here at the call site.
        grid: Float[torch.Tensor, "p gh gw"] = gran_mask.reshape(p, gh, gw)
        real_mask: Float[torch.Tensor, "p l"] = self.granularity_combination_strategy.resize(
            grid,
            output_size=(h, w),
        ).reshape(p, l)

        # apply the mask in flattened spatial space, broadcasting across channels
        flat: Float[torch.Tensor, "1 3 l"] = pixel_values.reshape(1, c, l)
        spatial_mask: Float[torch.Tensor, "p 1 l"] = real_mask.unsqueeze(1)
        perturbed_flat: Float[torch.Tensor, "p 3 l"] = flat * (1 - spatial_mask) + self.replace_value * spatial_mask
        perturbed_pixel_values: Float[torch.Tensor, "p 3 H W"] = perturbed_flat.reshape(-1, c, h, w)

        inputs["pixel_values"] = perturbed_pixel_values
        return inputs, gran_mask
