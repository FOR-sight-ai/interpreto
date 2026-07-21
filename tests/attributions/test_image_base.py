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

CLASSIFICATION_MODELS = [
    "hf-internal-testing/tiny-random-vit",
    "hf-internal-testing/tiny-random-BeitForImageClassification",
    "hf-internal-testing/tiny-random-ViTForImageClassification",
    "akahana/vit-base-cats-vs-dogs",
]


@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
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


def _stats_self(preprocess, image_processor=None):
    """A stand-in for `self`: _resolve_normalization_stats only reads these two attributes."""
    return SimpleNamespace(preprocess=preprocess, image_processor=image_processor)


_resolve = ImageClassificationAttributionExplainer._resolve_normalization_stats


@pytest.mark.parametrize(
    "preprocess, image_mean, image_std",
    [
        # preprocess=True owns the stats, so supplying any of them is the error.
        (True, 0.5, None),
        (True, None, 0.5),
        (True, 0.5, 0.5),
        # preprocess=False cannot guess them, so omitting any of them is the error.
        (False, None, None),
        (False, 0.5, None),
        (False, None, 0.5),
    ],
)
def test_resolve_normalization_stats_raises(preprocess, image_mean, image_std):
    processor = SimpleNamespace(do_normalize=True, image_mean=[0.5] * 3, image_std=[0.5] * 3)
    with pytest.raises(ValueError):
        _resolve(_stats_self(preprocess, processor), image_mean, image_std)


@pytest.mark.parametrize(
    "preprocess, processor, image_mean, image_std, expected_mean, expected_std",
    [
        # preprocess=True, do_normalize=True: the processor's stats are used verbatim.
        (
            True,
            SimpleNamespace(do_normalize=True, image_mean=[0.1, 0.2, 0.3], image_std=[0.4, 0.5, 0.6]),
            None,
            None,
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ),
        # do_normalize=False: identity, NOT the processor's stats, which it keeps regardless.
        (
            True,
            SimpleNamespace(do_normalize=False, image_mean=[0.5] * 3, image_std=[0.5] * 3),
            None,
            None,
            [0.0],
            [1.0],
        ),
        # do_normalize absent entirely: getattr defaults to False, so also identity.
        (True, SimpleNamespace(), None, None, [0.0], [1.0]),
        # preprocess=False: the caller's stats pass through, scalar and per-channel alike.
        (False, None, 0.5, 0.25, [0.5], [0.25]),
        (False, None, [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]),
    ],
)
def test_resolve_normalization_stats_returns(
    preprocess, processor, image_mean, image_std, expected_mean, expected_std
):
    mean, std = _resolve(_stats_self(preprocess, processor), image_mean, image_std)

    # (-1, 1, 1) so a scalar broadcasts over C and a per-channel stat aligns with it.
    assert mean.shape == (len(expected_mean), 1, 1)
    assert std.shape == (len(expected_std), 1, 1)
    assert mean.dtype is torch.float32
    assert std.dtype is torch.float32
    assert torch.allclose(mean.flatten(), torch.tensor(expected_mean))
    assert torch.allclose(std.flatten(), torch.tensor(expected_std))


# NOTE FOR REVIEWERS: test_inference_mode and test_process_targets below are copied
# from test_base.py (text modality) rather than written from scratch. This is
# intentional: ImageClassificationAttributionExplainer.process_targets is itself a
# copy of TextClassificationAttributionExplainer.process_targets (see base.py), so
# these tests are copies of copies, kept in sync with the text-side tests.
# I am also unsure about whether or not I should test private functions and have made the choice to do it.


@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
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
    assert len(result) == 1  # type: ignore
    assert torch.equal(result[0], torch.tensor([3]))  # type: ignore

    # Single integer with mismatch
    with pytest.raises(ValueError, match="Mismatch.*length of the inputs is 2"):
        explainer.process_targets(3, expected_length=2)

    # 1D tensor
    result = explainer.process_targets(torch.tensor([1, 2, 3]), expected_length=3)
    assert len(result) == 3  # type: ignore
    assert all(torch.equal(r.squeeze(), torch.tensor(v)) for r, v in zip(result, [1, 2, 3], strict=True))  # type: ignore

    # 2D tensor
    tensor = torch.tensor([[1], [2], [3]])
    result = explainer.process_targets(tensor, expected_length=3)
    assert len(result) == 3  # type: ignore
    assert all(r.shape == (1,) for r in result)

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
    assert len(result) == 3  # type: ignore
    assert all(torch.equal(r, torch.tensor([v])) for r, v in zip(result, [1, 2, 3], strict=True))

    # Iterable of ints with mismatch
    with pytest.raises(ValueError, match="Mismatch.*length of the inputs is 2"):
        explainer.process_targets([1, 2, 3], expected_length=2)

    # Iterable of tensors
    tensors = [torch.tensor([1]), torch.tensor([2])]
    result = explainer.process_targets(tensors, expected_length=2)
    assert result == tensors

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


@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
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


@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
@pytest.mark.parametrize("patch_size", [2, 16])
@pytest.mark.parametrize("pixel_values", [torch.zeros(1, 3, 224, 224), torch.ones(1, 3, 16, 16)])
def test_validate_batch_feature_patch_size_returns(model_name, patch_size, pixel_values):
    explainer = SimpleNamespace(patch_size=patch_size)
    bf = BatchFeature(data={"pixel_values": pixel_values})
    _validate(explainer, bf)


@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
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
@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
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

    # 1. Case with explicit targets (ints)
    targets = [0, 1]
    processed_inputs, processed_targets = explainer.process_inputs_to_explain_and_targets(
        model_inputs, targets=targets
    )
    assert len(processed_inputs) == 2  # type: ignore
    assert len(processed_targets) == 2  # type: ignore
    assert all(isinstance(t, torch.Tensor) for t in processed_targets)

    # 2. Case with explicit targets (tensor)
    targets_tensor = torch.tensor([1, 0])
    processed_inputs, processed_targets = explainer.process_inputs_to_explain_and_targets(
        model_inputs, targets=targets_tensor
    )
    assert len(processed_targets) == 2  # type: ignore
    assert all(isinstance(t, torch.Tensor) for t in processed_targets)

    # 3. Case with no targets (should use logits + argmax)
    processed_inputs, processed_targets = explainer.process_inputs_to_explain_and_targets(model_inputs)
    processed_targets = list(processed_targets).copy()
    assert len(processed_targets) == 2
    assert [t.item() for t in processed_targets] == [1, 0]

    # 4. Mismatched targets
    with pytest.raises(ValueError, match="Mismatch.*length of the inputs"):
        explainer.process_inputs_to_explain_and_targets(model_inputs, targets=[1])  # Only one target for two inputs


def _assert_batch_feature_list(result, expected_length):
    assert isinstance(result, list)
    assert len(result) == expected_length
    assert all(isinstance(bf, BatchFeature) for bf in result)
    assert "pixel_values" in result[0].keys()


@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
def test_process_model_inputs(model_name):
    """
    Test process_model_inputs: the 3 ValueError cases, and that valid inputs are
    normalized to a list of BatchFeature (with "pixel_values" as a key), one per
    sample, matching the input length for iterables.
    """
    model = AutoModelForImageClassification.from_pretrained(model_name)
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    image = Image.new("RGB", (4, 4))

    # --- ValueError cases ---

    # 1. Unsupported type (not BatchFeature, raw image, or Iterable)
    explainer = ImageClassificationAttributionExplainer(
        model,
        image_processor,
        ImageGranularity.PATCH,
        batch_size=2,
        device=DEVICE,
    )
    with pytest.raises(ValueError, match="not supported for method process_model_inputs"):
        explainer.process_model_inputs(42)  # type: ignore

    # 2. preprocess=False with a non-tensor raw input
    explainer_no_preprocess = ImageClassificationAttributionExplainer(
        model,
        image_processor,
        ImageGranularity.PATCH,
        batch_size=2,
        device=DEVICE,
        preprocess=False,
        image_mean=0.0,
        image_std=1.0,
    )
    with pytest.raises(ValueError, match="must be torch.Tensor"):
        explainer_no_preprocess.process_model_inputs(image)

    # 3. preprocess=False with a tensor of the wrong shape
    with pytest.raises(ValueError, match="expected pixel_values of shape"):
        explainer_no_preprocess.process_model_inputs(torch.zeros(2, 3, 4, 4))

    # --- Valid cases ---

    # BatchFeature passed directly
    bf = BatchFeature(data={"pixel_values": torch.zeros(1, 3, 16, 16)})
    result = explainer.process_model_inputs(bf)
    _assert_batch_feature_list(result, expected_length=1)

    # Raw PIL image, preprocess=True (default): routed through image_processor
    result = explainer.process_model_inputs(image)
    _assert_batch_feature_list(result, expected_length=1)

    # Raw np.ndarray, preprocess=True (default): routed through image_processor
    array = np.array(image)
    result = explainer.process_model_inputs(array)
    _assert_batch_feature_list(result, expected_length=1)

    # Raw tensor of shape (1, 3, H, W), preprocess=False: wrapped as-is
    result = explainer_no_preprocess.process_model_inputs(torch.zeros(1, 3, 4, 4))
    _assert_batch_feature_list(result, expected_length=1)

    # Iterable of raw PIL images, preprocess=True: one BatchFeature per sample
    images = [image, image, image]
    result = explainer.process_model_inputs(images)
    _assert_batch_feature_list(result, expected_length=len(images))
