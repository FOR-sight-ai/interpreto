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


from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.image_processing_utils import BatchFeature

from interpreto.attributions.base import ImageClassificationAttributionExplainer
from interpreto.commons.granularity import ImageGranularity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_CLASSIFICATION_MODELS = [
    "hf-internal-testing/tiny-random-vit",
    "hf-internal-testing/tiny-random-BeitForImageClassification",
    "hf-internal-testing/tiny-random-ViTForImageClassification",
    "akahana/vit-base-cats-vs-dogs",
]


@pytest.mark.parametrize("model_name", IMAGE_CLASSIFICATION_MODELS)
@pytest.mark.parametrize(
    "use_gradient, granularity, clash",
    [
        (True, ImageGranularity.PIXEL, False),
        (True, ImageGranularity.PATCH, True),
        (False, ImageGranularity.PIXEL, True),
        (False, ImageGranularity.PATCH, False),
    ],
)
def test_init_explainer(model_name, use_gradient, granularity, clash):
    model = AutoModelForImageClassification.from_pretrained(model_name)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    if clash:
        with pytest.raises(ValueError):
            _ = ImageClassificationAttributionExplainer(
                model,
                image_processor,
                granularity,
                use_gradient=use_gradient,
                batch_size=2,
                device=DEVICE,
            )
    else:
        _ = ImageClassificationAttributionExplainer(
            model,
            image_processor,
            granularity,
            use_gradient=use_gradient,
            batch_size=2,
            device=DEVICE,
        )


_resolve_normalization_stats = ImageClassificationAttributionExplainer._resolve_normalization_stats


@pytest.mark.parametrize(
    "processor, expected_mean, expected_std",
    [
        # do_normalize=True: the processor's stats are used verbatim.
        (
            SimpleNamespace(do_normalize=True, image_mean=[0.1, 0.2, 0.3], image_std=[0.4, 0.5, 0.6]),
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ),
        # do_normalize=False: identity, NOT the processor's stats, which it keeps regardless.
        (
            SimpleNamespace(do_normalize=False, image_mean=[0.5] * 3, image_std=[0.5] * 3),
            [0.0],
            [1.0],
        ),
        # do_normalize absent entirely: getattr defaults to False, so also identity.
        (SimpleNamespace(), [0.0], [1.0]),
    ],
)
def test_resolve_normalization_stats_returns(processor, expected_mean, expected_std):
    # SimpleNamespace stands in for `self`: _resolve_normalization_stats only reads image_processor.
    mean, std = _resolve_normalization_stats(SimpleNamespace(image_processor=processor))

    # (-1, 1, 1) so a scalar broadcasts over C and a per-channel stat aligns with it.
    assert mean.shape == (len(expected_mean), 1, 1), (
        "_resolve_normalization_stats must return mean shaped (C, 1, 1) for broadcasting over (C, H, W)"
    )
    assert std.shape == (len(expected_std), 1, 1), (
        "_resolve_normalization_stats must return std shaped (C, 1, 1) for broadcasting over (C, H, W)"
    )
    assert mean.dtype is torch.float32, "_resolve_normalization_stats must return mean as float32"
    assert std.dtype is torch.float32, "_resolve_normalization_stats must return std as float32"
    assert torch.allclose(mean.flatten(), torch.tensor(expected_mean)), (
        "_resolve_normalization_stats must return the expected mean values"
    )
    assert torch.allclose(std.flatten(), torch.tensor(expected_std)), (
        "_resolve_normalization_stats must return the expected std values"
    )


# NOTE FOR REVIEWERS: test_inference_mode and test_process_targets below are copied
# from test_base.py (text modality) rather than written from scratch. This is
# intentional: ImageClassificationAttributionExplainer.process_targets is itself a
# copy of TextClassificationAttributionExplainer.process_targets (see base.py), so
# these tests are copies of copies, kept in sync with the text-side tests.
# I am also unsure about whether or not I should test private functions and have made the choice to do it.


@pytest.mark.parametrize("model_name", IMAGE_CLASSIFICATION_MODELS)
def test_process_targets(model_name):
    """
    Test the process_targets method for different input types.
    """
    model = AutoModelForImageClassification.from_pretrained(model_name)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    explainer = ImageClassificationAttributionExplainer(
        model, image_processor, ImageGranularity.PATCH, batch_size=2, device=DEVICE
    )

    # Single integer
    result = explainer.process_targets(3, expected_length=1)
    assert len(result) == 1, "a single int target must normalize to a list of exactly one tensor"  # type: ignore
    assert torch.equal(result[0], torch.tensor([3])), (  # type: ignore
        "a single int target of 3 must normalize to tensor([3])"
    )

    # Single integer with mismatch
    with pytest.raises(ValueError, match="Mismatch.*length of the inputs is 2"):
        explainer.process_targets(3, expected_length=2)

    # 1D tensor
    result = explainer.process_targets(torch.tensor([1, 2, 3]), expected_length=3)
    assert len(result) == 3, (  # type: ignore
        "a 1D tensor of n targets must normalize to a list of n tensors, one per target"
    )
    assert all(torch.equal(r.squeeze(), torch.tensor(v)) for r, v in zip(result, [1, 2, 3], strict=True)), (  # type: ignore
        "each tensor normalized from a 1D input must equal its corresponding original target value"
    )

    # 2D tensor
    tensor = torch.tensor([[1], [2], [3]])
    result = explainer.process_targets(tensor, expected_length=3)
    assert len(result) == 3, "a 2D tensor of shape (n, 1) must normalize to a list of n tensors"  # type: ignore
    assert all(r.shape == (1,) for r in result), "each tensor normalized from a 2D input must keep its original shape"

    # Tensor with floats
    tensor = torch.tensor([[1.0], [2.0]])
    with pytest.raises(TypeError, match="Target tensor must be integers."):
        explainer.process_targets(tensor)

    # Tensor with invalid ndim
    tensor = torch.tensor([[[1]]])
    with pytest.raises(TypeError, match="Target tensor must be one-dimensional or two-dimensional."):
        explainer.process_targets(tensor)

    # Iterable of ints
    result = explainer.process_targets([1, 2, 3], expected_length=3)
    assert len(result) == 3, "an iterable of n ints must normalize to a list of n tensors"  # type: ignore
    assert all(torch.equal(r, torch.tensor([v])) for r, v in zip(result, [1, 2, 3], strict=True)), (
        "each normalized tensor must keep the same value as its corresponding input tensor"
    )

    # Iterable of ints with mismatch
    with pytest.raises(ValueError, match="Mismatch.*length of the inputs is 2"):
        explainer.process_targets([1, 2, 3], expected_length=2)

    # Iterable of tensors
    tensors = [torch.tensor([1]), torch.tensor([2])]
    result = explainer.process_targets(tensors, expected_length=2)
    assert result == tensors, "an iterable of already-valid tensors must be passed through unchanged"

    # Iterable of float tensors
    tensors = [torch.tensor([1.0]), torch.tensor([2.0])]
    with pytest.raises(TypeError, match="must be integers"):
        explainer.process_targets(tensors)

    # Iterable of mixed-dim tensors
    tensors = [torch.tensor([[1]]), torch.tensor([2])]
    with pytest.raises(TypeError, match="must be one-dimensional"):
        explainer.process_targets(tensors)

    # Unsupported type
    with pytest.raises(TypeError, match="Target type .* not supported"):
        explainer.process_targets("invalid_input")  # type: ignore


@pytest.mark.parametrize("model_name", IMAGE_CLASSIFICATION_MODELS)
def test_validate_batch_feature(model_name):
    """
    Test the _validate_batch_feature method for the 3 failure cases and the valid case.
    """
    model = AutoModelForImageClassification.from_pretrained(model_name)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    explainer = ImageClassificationAttributionExplainer(
        model, image_processor, ImageGranularity.PATCH, batch_size=2, device=DEVICE
    )

    # Missing "pixel_values" key
    with pytest.raises(ValueError, match="must contain the key 'pixel_values'"):
        explainer._validate_batch_feature(BatchFeature(data={"input_ids": torch.zeros(1, 3, 4, 4)}))

    # "pixel_values" not a torch.Tensor
    with pytest.raises(ValueError, match="must hold a torch.Tensor for 'pixel_values'"):
        explainer._validate_batch_feature(BatchFeature(data={"pixel_values": [[1, 2], [3, 4]]}))

    # "pixel_values" with more than one sample
    with pytest.raises(ValueError, match="must hold a single-sample"):
        explainer._validate_batch_feature(BatchFeature(data={"pixel_values": torch.zeros(2, 3, 4, 4)}))

    with pytest.raises(ValueError, match="must be 4 dimensional"):
        explainer._validate_batch_feature(BatchFeature(data={"pixel_values": torch.zeros(3, 4, 4)}))


_validate = ImageClassificationAttributionExplainer._validate_batch_feature


@pytest.mark.parametrize("model_name", IMAGE_CLASSIFICATION_MODELS)
@pytest.mark.parametrize("patch_size", [2, 16])
@pytest.mark.parametrize("pixel_values", [torch.zeros(1, 3, 224, 224), torch.ones(1, 3, 16, 16)])
def test_validate_batch_feature_patch_size_returns(model_name, patch_size, pixel_values):
    explainer = SimpleNamespace(patch_size=patch_size)
    bf = BatchFeature(data={"pixel_values": pixel_values})
    _validate(explainer, bf)


@pytest.mark.parametrize("model_name", IMAGE_CLASSIFICATION_MODELS)
@pytest.mark.parametrize("patch_size", [7, 19])
@pytest.mark.parametrize("pixel_values", [torch.zeros(1, 3, 226, 226), torch.ones(1, 3, 16, 16)])
def test_validate_batch_feature_patch_size_raises(model_name, patch_size, pixel_values):
    explainer = SimpleNamespace(patch_size=patch_size)
    bf = BatchFeature(data={"pixel_values": pixel_values})
    with pytest.raises(ValueError):
        _validate(explainer, bf)


# NOTE FOR REVIEWERS: test_process_inputs_to_explain_and_targets below is copied
# from test_base.py (text modality) and adapted to BatchFeature/pixel_values inputs,
# for the same reason as above: process_inputs_to_explain_and_targets is itself a
# copy of the text-side method (see base.py).
@pytest.mark.parametrize("model_name", IMAGE_CLASSIFICATION_MODELS)
def test_process_inputs_to_explain_and_targets(model_name):
    model = AutoModelForImageClassification.from_pretrained(model_name)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    explainer = ImageClassificationAttributionExplainer(
        model, image_processor, ImageGranularity.PATCH, batch_size=2, device=DEVICE
    )
    explainer.inference_wrapper = lambda *args, **kwargs: [torch.tensor([[1]]), torch.tensor([[0]])]  # type: ignore

    # Model input example: two single-sample BatchFeatures with pixel_values
    model_inputs = [
        BatchFeature(data={"pixel_values": torch.zeros(1, 3, 4, 4)}),
        BatchFeature(data={"pixel_values": torch.ones(1, 3, 4, 4)}),
    ]

    # 1. Case with explicit targets (list)
    targets = [0, 1]
    processed_inputs, processed_targets = explainer.process_inputs_to_explain_and_targets(
        model_inputs, targets=targets
    )
    assert len(processed_inputs) == 2, (  # type: ignore
        "process_inputs_to_explain_and_targets must keep one processed input per model input"
    )
    assert len(processed_targets) == 2, (  # type: ignore
        "process_inputs_to_explain_and_targets must return one target per model input"
    )
    assert all(isinstance(t, torch.Tensor) for t in processed_targets), "each processed target must be a tensor"

    # 2. Case with explicit targets (tensor)
    targets_tensor = torch.tensor([1, 0])
    processed_inputs, processed_targets = explainer.process_inputs_to_explain_and_targets(
        model_inputs, targets=targets_tensor
    )
    assert len(processed_targets) == 2, (  # type: ignore
        "process_inputs_to_explain_and_targets must return one target per model input"
    )
    assert all(isinstance(t, torch.Tensor) for t in processed_targets), "each processed target must be a tensor"

    # 3. Case with no targets (should use logits + argmax)
    processed_inputs, processed_targets = explainer.process_inputs_to_explain_and_targets(model_inputs)
    processed_targets = list(processed_targets).copy()
    assert len(processed_targets) == 2, "process_inputs_to_explain_and_targets must return one target per model input"
    assert [t.item() for t in processed_targets] == [1, 0], (
        "with no targets given, process_inputs_to_explain_and_targets must fall back to per-input "
        "predictions from self.inference_wrapper (mocked here at the beginning of the test function "
        "to return 1 and 0)"
    )

    # 4. Mismatched targets
    with pytest.raises(ValueError, match="Mismatch.*length of the inputs"):
        explainer.process_inputs_to_explain_and_targets(model_inputs, targets=[1])  # Only one target for two inputs


def _assert_batch_feature_list(result, expected_length):
    assert isinstance(result, list), "process_model_inputs must return a list"
    assert len(result) == expected_length, "process_model_inputs must return one BatchFeature per input"
    assert all(isinstance(bf, BatchFeature) for bf in result), "each element of result must be a BatchFeature"
    assert "pixel_values" in result[0].keys(), "the returned result must carry a 'pixel_values' key"


@pytest.mark.parametrize("model_name", IMAGE_CLASSIFICATION_MODELS)
def test_process_model_inputs(model_name):
    """
    Test process_model_inputs: the ValueError case, and that valid inputs are
    normalized to a list of BatchFeature (with "pixel_values" as a key), one per
    sample, matching the input length for iterables.
    """
    model = AutoModelForImageClassification.from_pretrained(model_name)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    image = Image.new("RGB", (4, 4))

    # --- ValueError case: unsupported type (not BatchFeature, raw image, or Iterable) ---

    explainer = ImageClassificationAttributionExplainer(
        model,
        image_processor,
        ImageGranularity.PATCH,
        batch_size=2,
        device=DEVICE,
    )
    with pytest.raises(ValueError, match="not supported for method process_model_inputs"):
        explainer.process_model_inputs(42)  # type: ignore

    # --- Valid cases ---

    # BatchFeature passed directly
    bf = BatchFeature(data={"pixel_values": torch.zeros(1, 3, 16, 16)})
    result = explainer.process_model_inputs(bf)
    _assert_batch_feature_list(result, expected_length=1)

    # Raw PIL image: routed through image_processor
    result = explainer.process_model_inputs(image)
    _assert_batch_feature_list(result, expected_length=1)

    # Raw np.ndarray: routed through image_processor
    array = np.array(image)
    result = explainer.process_model_inputs(array)
    _assert_batch_feature_list(result, expected_length=1)

    # Iterable of raw PIL images: one BatchFeature per sample
    images = [image, image, image]
    result = explainer.process_model_inputs(images)
    _assert_batch_feature_list(result, expected_length=len(images))
