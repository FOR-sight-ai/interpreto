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
Image-side attribution base: `ImageAttributionOutput` dataclass and
`ImageClassificationAttributionExplainer` mirroring text equivalents in `base.py`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from PIL.Image import Image as PILImage
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.modeling_utils import PreTrainedModel

from interpreto.attributions.aggregations.base import Aggregator
from interpreto.attributions.base import AttributionExplainer
from interpreto.attributions.perturbations.base import Perturbator
from interpreto.attributions.perturbations.image_base import ImageMaskPerturbator, ImageTensorPerturbator
from interpreto.commons.generator_tools import split_iterator
from interpreto.commons.granularity import GranularityAggregationStrategy
from interpreto.commons.image_granularity import ImageGranularity
from interpreto.model_wrapping.image_classification_inference_wrapper import (
    ImageClassificationInferenceWrapper,
)
from interpreto.model_wrapping.inference_wrapper import InferenceModes
from interpreto.typing import ClassificationTarget, ModelInputs, SingleAttribution, TensorMapping


@dataclass(slots=True)
class ImageAttributionOutput:
    """
    Class to store the output of an image-attribution method.

    Mirrors `AttributionOutput` with two changes for the image modality:
    `granularity` is typed as `ImageGranularity` (default `PATCH`), and `elements`
    holds `(row, col)` integer tuples produced by `ImageGranularity.get_decomposition`
    instead of decoded strings.

    Attributes:
        attributions (SingleAttribution):
            Attribution score tensor(s) of shape `(t, l)` where `l = H*W` (PIXEL)
            or `l = num_patches` (PATCH). Stored FLAT — visualization reshapes to
            2D at render time via `model_inputs_to_explain["pixel_values"].shape[-2:]`.

        elements (list[tuple[int, int]] | torch.Tensor):
            Per-unit labels of shape matching `attributions`' last dim.
                - PIXEL: list of `(row, col)` pixel coordinates.
                - PATCH: list of `(patch_row, patch_col)` patch-grid coordinates.

        model_inputs_to_explain (TensorMapping):
            The output of `image_processor(image, return_tensors="pt")` (a `BatchFeature`,
            satisfies `TensorMapping`). Holds `pixel_values` of shape `(1, 3, H, W)`.

        raw_image (PIL.Image | np.ndarray | torch.Tensor | None):
            The user's pre-processing input, preserved so visualization can show an
            honest underlying image (the `pixel_values` above are post-normalization
            and not directly displayable). `None` when the user supplied an already-
            processed `BatchFeature` or when `preprocess=False` (input was already
            normalized — no raw form to recover).

        targets (torch.Tensor):
            The target class(es).

        classes (torch.Tensor | None):
            Optional tensor of class labels.

        granularity (ImageGranularity):
            The granularity level of the explanation. Defaults to `ImageGranularity.DEFAULT`
            (= `PATCH`).

        granularity_aggregation_strategy (GranularityAggregationStrategy):
            The aggregation method used to aggregate scores at the specified granularity.

        inference_mode (Callable[[torch.Tensor], torch.Tensor]):
            The mode used for inference (LOGITS, SOFTMAX, LOG_SOFTMAX). See `InferenceModes`.
    """

    attributions: SingleAttribution
    elements: list[tuple[int, int]] | torch.Tensor
    model_inputs_to_explain: TensorMapping
    targets: torch.Tensor
    classes: torch.Tensor | None = None
    granularity: ImageGranularity = ImageGranularity.DEFAULT
    granularity_aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN
    inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS
    raw_image: PILImage | np.ndarray | torch.Tensor | None = None


class ImageClassificationAttributionExplainer(AttributionExplainer):
    """
    Attribution explainer for image-classification models (ViT-family).

    Mirrors `ClassificationAttributionExplainer` with three modality-specific overrides:
    `__init__` swaps `tokenizer` for `image_processor` and drops the text-side
    `setup_token_ids` call (image configs have no pad/mask tokens),
    `process_model_inputs` accepts `BatchFeature` instead of `BatchEncoding`, and
    `explain` produces `ImageAttributionOutput` and passes `patch_size` / no tokenizer
    where the granularity API differs.
    """

    associated_inference_wrapper: ImageClassificationInferenceWrapper
    base_tensor_perturbator_class = ImageTensorPerturbator
    base_mask_perturbator_class = ImageMaskPerturbator

    def __init__(
        self,
        model: PreTrainedModel,
        image_processor: BaseImageProcessor,
        batch_size: int = 4,
        perturbator: Perturbator | None = None,
        aggregator: Aggregator | None = None,
        device: torch.device | None = None,
        granularity: ImageGranularity = ImageGranularity.DEFAULT,
        granularity_aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN,
        inference_mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        use_gradient: bool = False,
        input_x_gradient: bool = True,
        preprocess: bool = True,
    ) -> None:
        """
        Args mirror `AttributionExplainer.__init__` with `tokenizer` -> `image_processor`
        and `granularity` typed as `ImageGranularity`. See parent for shared semantics.

        Additional image-only arg:
            preprocess (bool): If True, `process_model_inputs` routes raw inputs
                (`PIL.Image`, `np.ndarray`, raw `torch.Tensor`) through `image_processor`.
                If False, inputs must already be `BatchFeature` or normalized
                `torch.Tensor` of shape `(1, 3, H, W)` (or `(3, H, W)`, auto-unsqueezed).
        """
        # does not call setup_token_id because there is no need for a PAD token

        self.image_processor = image_processor
        self.preprocess = preprocess

        self.inference_wrapper = self._associated_inference_wrapper(
            model,
            gradients=use_gradient,
            input_x_gradient=input_x_gradient,
            batch_size=batch_size,
            device=device,
            mode=inference_mode,
        )  # type: ignore
        self.perturbator = perturbator or ImageTensorPerturbator()
        self.aggregator = aggregator or Aggregator()
        self.granularity = granularity
        self.granularity_aggregation_strategy = granularity_aggregation_strategy
        # patch_size is sourced from the model config; required by ImageGranularity.PATCH
        self.patch_size = int(getattr(model.config, "patch_size", 16))
        # The explainer is the single source of truth for patch_size (it owns model.config).
        # A mask perturbator builds its (g, l) association matrix from patch_size, and the
        # explainer's (t, l) -> (t, g) aggregation interprets the result against the same g;
        # if the two disagree the mask<->score correspondence silently breaks. So we push the
        # authoritative value down, overriding the perturbator's placeholder default.
        # NOTE: this is the "version (a)" reconcile. If the isinstance wart or the
        # silently-overwritten default become a problem, switch to "version (b)" (perturbator
        # stops storing patch_size; explainer passes it into perturb() at call time).
        if isinstance(self.perturbator, ImageMaskPerturbator):
            self.perturbator.patch_size = self.patch_size

    def process_targets(
        self, targets: ClassificationTarget, expected_length: int | None = None
    ) -> list[Int[torch.Tensor, "t"]]:
        """
        Normalize classification targets into a list of 1D integer tensors.

        Args:
            targets (int | torch.Tensor | Iterable[int] | Iterable[torch.Tensor]):
                The classification target(s). Supported formats include:
                - int: Interpreted as a single target.
                - torch.Tensor:
                    * 1D tensors are treated as a sequence of individual targets.
                    * 2D tensors must have shape (n, t), where `n` is the number of targets.
                - Iterable[int]: Each integer is treated as a separate target.
                - Iterable[torch.Tensor]: Each tensor must be 1D and contain integers.

            expected_length (int | None, optional):
                If specified, validates that the number of targets matches this expected length.

        Returns:
            Iterable[torch.Tensor]
                A list of 1D integer tensors, one per input instance.

        Raises:
            ValueError
                - If the number of targets does not match `expected_length`.
            TypeError
                - If the type of `targets` is unsupported.
                - If tensor targets are not 1D or 2D.
                - If tensor values are not integers.
        """
        # integer
        if isinstance(targets, int):
            if expected_length is not None and expected_length != 1:
                raise ValueError(
                    "Mismatch between the inputs and targets length."
                    + f" Target is a single integer, but the length of the inputs is {expected_length}."
                )
            return [torch.tensor([targets])]

        # tensor
        if isinstance(targets, torch.Tensor):
            if targets.ndim == 1:
                # one dimensional tensors are treated as iterable of integer targets
                targets = targets.unsqueeze(-1)
            if expected_length is not None and expected_length != targets.shape[0]:
                raise ValueError(
                    "Mismatch between the inputs and targets length."
                    + f" Target tensor of {targets.shape[0]} elements, but the length of the inputs is {expected_length}."
                )
            if targets.ndim != 2:
                raise TypeError(
                    "Target tensor must be one-dimensional or two-dimensional."
                    + f" Target tensor has {targets.ndim} dimensions."
                )
            if torch.is_floating_point(targets):
                raise TypeError("Target tensor must be integers.")
            return list(targets.unbind(dim=0))

        # iterable
        if isinstance(targets, Iterable):
            if expected_length is not None and len(targets) != expected_length:  # type: ignore
                raise ValueError(
                    "Mismatch between the inputs and targets length."
                    + f" Target is an iterable of {len(targets)} elements, but the length of the inputs is {expected_length}."  # type: ignore
                )

            # iterable[int]
            if all(isinstance(t, int) for t in targets):
                return [torch.tensor([target]) for target in targets]

            # iterable[torch.Tensor]
            iterable_targets: list[torch.Tensor] = list(targets)  # type: ignore
            if all(isinstance(t, torch.Tensor) for t in iterable_targets):
                if any(target.ndim != 1 for target in iterable_targets):
                    raise TypeError("If the targets are iterable of tensors, the tensors must be one-dimensional.")
                if any(torch.is_floating_point(target) for target in iterable_targets):
                    raise TypeError("If the targets are iterable of tensors, they must be integers.")
                return iterable_targets

        raise TypeError(f"Target type {type(targets)} not supported.")

    def process_inputs_to_explain_and_targets(
        self,
        model_inputs: list[TensorMapping],
        targets: ClassificationTarget | None = None,
    ) -> tuple[list[TensorMapping], list[Int[torch.Tensor, "t"]]]:
        """
        Pre-processes model inputs and classification targets for explanation.

        This method ensures that:
        - If `targets` are not provided, they are computed by performing inference on `model_inputs` and selecting the predicted class using `argmax`.
        - The `targets` are then validated and converted using `self.process_targets`, ensuring the same length as `model_inputs`.

        Args:
        model_inputs : Iterable[TensorMapping]
            A batch of input mappings, typically containing tokenized inputs such as "input_ids", "attention_mask", etc.

        targets : int | torch.Tensor | Iterable[int] | Iterable[torch.Tensor] | None, optional
            Classification targets for each input. If None, targets are computed using model inference
            by selecting the index with the highest logit value for each input.

        Returns
        -------
        tuple[Iterable[TensorMapping], Iterable[torch.Tensor]]
            - model_inputs_to_explain: List of tokenized input mappings with required explanation metadata (e.g., special tokens mask).
            - sanitized_targets: List of 1D integer tensors, each corresponding to a target label for an input.

        Raises
        ------
        ValueError
            If the provided or inferred targets do not match the number of input instances, or if their format is invalid.
        """
        sanitized_targets: list[Int[torch.Tensor, "t"]]
        if targets is None:
            # compute targets from logits if not provided
            # inputs have already been split, so we only need to select the first element
            sanitized_targets = [t[0] for t in self.inference_wrapper(model_inputs)]
        else:
            # process targets and ensure they have the same length as inputs
            sanitized_targets = self.process_targets(targets, expected_length=len(model_inputs))

        return model_inputs, sanitized_targets

    def post_processing(self, contribution: Float[torch.Tensor, "t l"]):
        """
        stub implemented because post_processing is an abstact_method
        """
        pass

    def process_model_inputs(self, model_inputs: ModelInputs) -> list[TensorMapping]:
        """
        Normalize image inputs to a list of `BatchFeature` mappings, one per sample.

        Accepted input types depend on `self.preprocess`:
            - Always: `BatchFeature` (passed through after validation).
            - `preprocess=True`: also `PIL.Image`, `np.ndarray`, raw `torch.Tensor`
              → routed through `self.image_processor`.
            - `preprocess=False`: also `torch.Tensor` of shape `(1, 3, H, W)` or
              `(3, H, W)` (auto-unsqueezed) → wrapped as `BatchFeature` directly,
              assumed already normalized.
            - Iterables of any of the above are flattened recursively.

        Mirrors text `process_model_inputs` with `BatchFeature` for `BatchEncoding`
        and `pixel_values` for `input_ids`. The `str` branch is replaced by raw-image
        branches.
        """
        # check BatchFeature BEFORE Iterable — BatchFeature is a UserDict and would
        # otherwise match the Iterable branch and recurse into its keys.
        if isinstance(model_inputs, BatchFeature):
            return [self._validate_batch_feature(model_inputs)]

        # Raw-image types (PIL.Image is NOT Iterable on the class level; np.ndarray and
        # torch.Tensor ARE — so we must catch them before the Iterable branch).
        if isinstance(model_inputs, (PILImage, np.ndarray, torch.Tensor)):
            if self.preprocess:
                processed = self.image_processor(model_inputs, return_tensors="pt")
                return [self._validate_batch_feature(processed)]

            # preprocess=False: only Tensor is allowed; assume already normalized.
            if not isinstance(model_inputs, torch.Tensor):
                raise ValueError(
                    f"When preprocess=False, raw inputs must be torch.Tensor, got {type(model_inputs)}. "
                    "Either set preprocess=True, or pre-process via image_processor and pass BatchFeature."
                )
            tensor = model_inputs
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4 or tensor.shape[1] != 3 or tensor.shape[0] != 1:
                raise ValueError(
                    "When preprocess=False, expected pixel_values of shape (1, 3, H, W), "
                    f"got {tuple(tensor.shape)}."
                )
            return [BatchFeature(data={"pixel_values": tensor})]

        if isinstance(model_inputs, Iterable):
            return list(itertools.chain(*[self.process_model_inputs(item) for item in model_inputs]))

        raise ValueError(
            f"type {type(model_inputs)} not supported for method process_model_inputs in class {self.__class__.__name__}"
        )

    def _validate_batch_feature(self, bf: BatchFeature) -> BatchFeature:
        """Sanity-check that a BatchFeature has a single-sample (1, 3, H, W) pixel_values tensor."""
        if "pixel_values" not in bf.keys():
            raise ValueError(
                "The BatchFeature must contain the key 'pixel_values' to be processed by the "
                f"image attribution explainer. Got {list(bf.keys())}."
            )
        if not isinstance(bf["pixel_values"], torch.Tensor):
            raise ValueError(
                "The BatchFeature must hold a torch.Tensor for 'pixel_values'. "
                "Use `image_processor(image, return_tensors='pt')`. "
                f"Got {type(bf['pixel_values'])}."
            )
        if bf["pixel_values"].shape[0] != 1:
            raise ValueError(
                "The BatchFeature must contain a single sample to be processed. "
                "Pre-process images one by one, or pass an iterable. "
                f"Got {bf['pixel_values'].shape[0]} samples."
            )
        return bf

    def explain(
        self,
        model_inputs: ModelInputs,
        targets: ClassificationTarget | None = None,
    ) -> list[ImageAttributionOutput]:
        """
        Compute attributions for image-classification models.

        Mirrors `AttributionExplainer.explain` with image-side substitutions:
        - `granularity_score_aggregation` is called without `tokenizer` and with `patch_size`
          (text accepts `tokenizer`; image accepts `patch_size`).
        - `get_decomposition` uses `return_coordinates=True` (image equivalent of `return_text`).
        - The `attention_mask` post-processing block is dropped (no attention_mask for ViT).
        - The output is `ImageAttributionOutput` instead of `AttributionOutput`.
        - The generation `aggregate_targets` branch is dropped (image MVP is classification-only).
        """
        sanitized_model_inputs: list[TensorMapping] = self.process_model_inputs(model_inputs)

        model_inputs_to_explain: list[TensorMapping]
        sanitized_targets: list[Int[torch.Tensor, "t"]]
        model_inputs_to_explain, sanitized_targets = self.process_inputs_to_explain_and_targets(
            sanitized_model_inputs,
            targets,  # type: ignore
        )

        # Preserve each user-supplied raw image alongside its
        #  sanitized BatchFeature so
        # the per-sample ImageAttributionOutput can carry it for visualization. The
        # post-normalization pixel_values in model_inputs_to_explain are not directly
        # displayable. None for samples that came in as BatchFeature or under preprocess=False.
        raw_images: list[PILImage | np.ndarray | torch.Tensor | None]
        if isinstance(model_inputs, list):
            raw_images = [
                m if self.preprocess and isinstance(m, (PILImage, np.ndarray, torch.Tensor)) else None
                for m in model_inputs
            ]
        elif self.preprocess and isinstance(model_inputs, (PILImage, np.ndarray, torch.Tensor)):
            raw_images = [model_inputs]
        else:
            raw_images = [None] * len(model_inputs_to_explain)

        # Perturbations + scores: same flow as text. The default Perturbator() yields
        # the input unchanged with a None mask, which is what gradient methods need.
        pert_generator: Iterator[TensorMapping]
        mask_generator: Iterator[Int[torch.Tensor, "p l"] | None]
        pert_generator, mask_generator = split_iterator(
            self.perturbator.perturb(m) for m in model_inputs_to_explain
        )

        scores: Iterator[Float[torch.Tensor, "p t"]] = self.inference_wrapper(
            pert_generator, sanitized_targets
        )

        # Aggregate over perturbations: (p, t), (p, l) -> (t, l)
        contributions: Iterator[Float[torch.Tensor, "t l"]] = (
            self.aggregator(score.detach(), mask)
            for score, mask in zip(scores, mask_generator, strict=True)
        )

        # Aggregate over inputs for gradient-based methods: (t, l) -> (t, g).
        # Image granularity signature: patch_size instead of tokenizer; no aggregate_targets.
        granular_contributions: Iterator[Float[torch.Tensor, "t g"]] = (
            self.granularity.granularity_score_aggregation(
                contribution=contribution.cpu(),
                granularity_aggregation_strategy=self.granularity_aggregation_strategy,
                inputs=inputs,
                patch_size=self.patch_size,
                aggregate_inputs=self.inference_wrapper.gradients,
            )
            for contribution, inputs in zip(contributions, model_inputs_to_explain, strict=True)
        )

        # Coordinate labels per granularity unit (replaces text's decoded-token strings).
        # All samples share the same H, W after image_processor normalization, so the
        # decomposition is identical across samples — compute it once on the first sample
        # and share the reference. Text iterates per-sample because each sample can have
        # a different sequence length; for image that variation doesn't exist.
        # TODO: revisit — see project_vit_explainer_decomposition_refactor in auto-memory.
        # `get_decomposition` already replicates internally based on the input's batch dim
        # (`pixel_values.shape[0]`), but our per-sample BatchFeatures all have batch=1, so
        # the internal replication is a no-op and we redo it here. Cleaner long-term: either
        # strip the internal replication (return one decomposition) or concat samples into a
        # single batched BatchFeature upstream and let `get_decomposition` produce the full
        # `n_samples` list directly.
        shared_coords: list[tuple[int, int]] = self.granularity.get_decomposition(
            model_inputs_to_explain[0],
            patch_size=self.patch_size,
            return_coordinates=True,
        )[0]  # type: ignore
        granular_inputs_coords: list[list[tuple[int, int]]] = [shared_coords for _ in model_inputs_to_explain]

        results: list[ImageAttributionOutput] = []
        for contribution, model_input, elements, target, raw_image in zip(
            granular_contributions,
            model_inputs_to_explain,
            granular_inputs_coords,
            sanitized_targets,
            raw_images,
            strict=True,
        ):

            attribution_output = ImageAttributionOutput(
                attributions=contribution,
                elements=elements,
                model_inputs_to_explain=model_input,
                targets=target.cpu(),
                granularity=self.granularity,
                granularity_aggregation_strategy=self.granularity_aggregation_strategy,
                inference_mode=self.inference_wrapper.mode,
                raw_image=raw_image,
            )
            results.append(attribution_output)
        return results
