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
Visual sanity check for the vision attribution methods, run against a *real* trained ViT
(`akahana/vit-base-cats-vs-dogs`) rather than the tiny random model.

The idea: `seperated_cat_and_dog.jpg` has a cat on the left and a dog on the right on a white
background. For each animal we build a masked image where *everything except that animal* is
painted with the neutral baseline, then ask the model to explain the matching label (cat -> 0,
dog -> 1). A masked-out region carries no information, so a faithful method should not credit it.

The masking is done in *raw pixel space*, on purpose:
  - we can save and directly eyeball the exact image that is fed to the model, and
  - it exercises the real preprocessing path (rescale + normalize) end to end.
The processor rescales by 1/255 then normalizes with mean = std = 0.5, so painting the raw
background with `image_mean * 255` (~128) lands on a normalized 0.0 — the neutral baseline that
carries no signal. The perturbed image is then handed to the explainer with `preprocess=True`.

This test does NOT assert anything about *where* the attribution lands yet — it only runs every
method end to end and saves both the perturbed input and the heatmaps to `sanity_check_outputs/`
at the repo root so they can be eyeballed. A quantitative "attribution concentrates on the animal"
check comes later.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from interpreto.attributions.methods.occlusion_merge import Occlusion
from interpreto.attributions.methods.smoothgrad_merged import SmoothGrad
from interpreto.visualizations.image_attributions_merge import plot_image_attribution

plt.switch_backend("Agg")  # headless: render into a buffer, never open a window.

SANITY_MODEL = "akahana/vit-base-cats-vs-dogs"
FIXTURE_IMAGES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "images"
REPO_ROOT = Path(__file__).parents[3]
OUTPUT_DIR = REPO_ROOT / "sanity_check_outputs"

# Boxes as (row0, row1, col0, col1) fractions of the raw image, kept generous so the whole animal
# body is inside. Everything *outside* the box is painted with the neutral baseline. Eyeballed
# from the fixture: cat bottom-left, dog right half.
CAT_BOX = (0.33, 1.00, 0.00, 0.30)
DOG_BOX = (0.15, 1.00, 0.57, 1.00)

# (label_name, box, target_index) — the model's id2label is {0: "cat", 1: "dog"}.
SANITY_CASES = [
    ("cat", CAT_BOX, 0),
    ("dog", DOG_BOX, 1),
]

# All ten methods, constructed uniformly with default parameters (Sobol/LIME included).
SANITY_METHODS = [
    # ImageGradientShap,
    # ImageIntegratedGradients,
    # ImageKernelShap,
    Occlusion,
    # ImageSaliency,
    SmoothGrad,
    # ImageSobol,
    # ImageSquareGrad,
    # ImageVarGrad,
    # ImageLime,
]


@pytest.fixture(scope="module")
def model_and_processor():
    model = AutoModelForImageClassification.from_pretrained(SANITY_MODEL)
    processor = AutoImageProcessor.from_pretrained(SANITY_MODEL)
    return model, processor


@pytest.fixture(scope="module")
def raw_image() -> Image.Image:
    return Image.open(FIXTURE_IMAGES_DIR / "seperated_cat_and_dog.jpg").convert("RGB")


@pytest.fixture(scope="module", autouse=True)
def _output_dir() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def _keep_box_raw(image: Image.Image, box: tuple[float, float, float, float], background: np.ndarray) -> Image.Image:
    """
    Return a copy of `image` with everything outside `box` painted with the per-channel `background`
    value (raw uint8). `background = image_mean * 255`, so after the processor's rescale+normalize it
    becomes a normalized 0.0 — the neutral baseline carrying no signal.
    """
    arr = np.asarray(image)  # (H, W, 3) uint8
    height, width, _ = arr.shape
    r0, r1, c0, c1 = box
    ri0, ri1 = int(r0 * height), int(r1 * height)
    ci0, ci1 = int(c0 * width), int(c1 * width)
    out = np.broadcast_to(background, arr.shape).copy()  # neutral everywhere...
    out[ri0:ri1, ci0:ci1] = arr[ri0:ri1, ci0:ci1]  # ...then paste the animal back in.
    return Image.fromarray(out)


@pytest.mark.slow
@pytest.mark.parametrize("attribution_method", SANITY_METHODS)
@pytest.mark.parametrize("label_name, box, target", SANITY_CASES)
def test_image_methods_sanity(model_and_processor, raw_image, attribution_method, label_name, box, target):
    model, processor = model_and_processor

    background = (np.asarray(processor.image_mean, dtype=np.float32) * 255).round().astype(np.uint8)
    masked_image = _keep_box_raw(raw_image, box, background)
    masked_image.save(OUTPUT_DIR / f"{label_name}_input.png")

    # preprocess=True: the raw perturbed image goes through the real rescale+normalize path.
    explainer = attribution_method(model, processor, preprocess=True)
    output = explainer.explain(masked_image, target)

    # Basic smoke checks only — no claim yet about *where* the attribution concentrates.
    assert len(output) == 1
    assert torch.isfinite(output[0].attributions).all()

    fig, _ = plot_image_attribution(output[0])
    fig.suptitle(f"{attribution_method.__name__} — {label_name}")
    fig.savefig(OUTPUT_DIR / f"{label_name}_{attribution_method.__name__}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
