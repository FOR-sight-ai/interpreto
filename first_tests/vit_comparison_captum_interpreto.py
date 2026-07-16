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

print("Beggining of the file ")
import functools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from interpreto.attributions import (
    ImageGradientShap,
    ImageIntegratedGradients,
    ImageKernelShap,
    ImageLime,
    ImageOcclusion,
    ImageSaliency,
    ImageSmoothGrad,
    ImageSobol,
    ImageSquareGrad,
    ImageVarGrad,
)
from interpreto.commons import ImageGranularity
from interpreto.visualizations.image_attributions import _draw_attribution_on_ax, _prepare_heatmap, _to_displayable

matplotlib.use("Agg")
from captum.attr import GradientShap, IntegratedGradients, KernelShap, Lime, NoiseTunnel, Occlusion, Saliency

print("End of imports")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForImageClassification.from_pretrained("akahana/vit-base-cats-vs-dogs").to(device)
processor = AutoImageProcessor.from_pretrained("akahana/vit-base-cats-vs-dogs")
print("End of models")

image = Image.open("/gpfs/users/debosschu/interpretability_libraries/interpreto/equal_cat_and_dog.jpg")
print(model.config)

os.makedirs("heatmap_interpreto_captum", exist_ok=True)


def model_forward(pixel_values):
    return model(pixel_values=pixel_values.to(device)).logits


def method_name(method_cls):
    target = getattr(method_cls, "func", method_cls)
    return target.__name__


def patch_feature_mask(pixel_values, patch_size=16):
    H, W = pixel_values.shape[-2:]
    mask = torch.zeros(1, 1, H, W, dtype=torch.long)
    patch_id = 0
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            mask[0, 0, i : i + patch_size, j : j + patch_size] = patch_id
            patch_id += 1
    return mask


def to_heatmap(attr_tensor):
    """Reduce captum (1, C, H, W) attribution tensor to (H, W) by abs-mean over channels."""
    return attr_tensor.squeeze(0).abs().mean(0).detach().numpy()


# Captum runners: pixel_values (1,3,H,W) with requires_grad, target_class -> (H,W) numpy
def run_captum_saliency(pv, target):
    return to_heatmap(Saliency(model_forward).attribute(pv, target=target))


def run_captum_gradient_shap(pv, target):
    baseline = torch.zeros_like(pv)
    return to_heatmap(GradientShap(model_forward).attribute(pv, baselines=baseline, target=target))


def run_captum_integrated_gradients(pv, target):
    baseline = torch.zeros_like(pv)
    return to_heatmap(IntegratedGradients(model_forward).attribute(pv, baselines=baseline, target=target, n_steps=50))


def run_captum_smoothgrad(pv, target):
    nt = NoiseTunnel(Saliency(model_forward))
    return to_heatmap(nt.attribute(pv, nt_type="smoothgrad", nt_samples=50, target=target))


def run_captum_squaregrad(pv, target):
    nt = NoiseTunnel(Saliency(model_forward))
    return to_heatmap(nt.attribute(pv, nt_type="smoothgrad_sq", nt_samples=50, target=target))


def run_captum_vargrad(pv, target):
    nt = NoiseTunnel(Saliency(model_forward))
    return to_heatmap(nt.attribute(pv, nt_type="vargrad", nt_samples=50, target=target))


def run_captum_kernel_shap(pv, target):
    mask = patch_feature_mask(pv)
    return to_heatmap(KernelShap(model_forward).attribute(pv, target=target, feature_mask=mask, n_samples=200))


def run_captum_lime(pv, target):
    mask = patch_feature_mask(pv)
    return to_heatmap(Lime(model_forward).attribute(pv, target=target, feature_mask=mask, n_samples=1000))


def run_captum_occlusion(pv, target):
    return to_heatmap(Occlusion(model_forward).attribute(pv, sliding_window_shapes=(3, 16, 16), target=target))


CAPTUM_MAP = {
    "ImageSaliency": run_captum_saliency,
    "ImageGradientShap": run_captum_gradient_shap,
    "ImageIntegratedGradients": run_captum_integrated_gradients,
    "ImageSmoothGrad": run_captum_smoothgrad,
    "ImageSquareGrad": run_captum_squaregrad,
    "ImageVarGrad": run_captum_vargrad,
    "ImageKernelShap": run_captum_kernel_shap,
    "ImageLime": run_captum_lime,
    "ImageOcclusion": run_captum_occlusion,
}

test_imagekernelshap = functools.partial(ImageKernelShap, n_perturbations=200)
test_imagesobol = functools.partial(ImageSobol, n_token_perturbations=32)
test_saliency = functools.partial(ImageSaliency, granularity=ImageGranularity.PIXEL, input_x_gradient=False)
test_gradient_shap = functools.partial(ImageGradientShap, granularity=ImageGranularity.PIXEL, input_x_gradient=False)
test_integrated_gradient = functools.partial(
    ImageIntegratedGradients, granularity=ImageGranularity.PIXEL, input_x_gradient=False
)
test_smoothgrad = functools.partial(
    ImageSmoothGrad, granularity=ImageGranularity.PIXEL, input_x_gradient=False, n_perturbations=50
)
test_imagegrad = functools.partial(
    ImageVarGrad, granularity=ImageGranularity.PIXEL, input_x_gradient=False, n_perturbations=50
)
test_squaregrad = functools.partial(
    ImageSquareGrad, granularity=ImageGranularity.PIXEL, input_x_gradient=False, n_perturbations=50
)
test_imagelime = functools.partial(ImageLime, n_perturbations=1000)

methods = [
    test_saliency,
    test_gradient_shap,
    test_integrated_gradient,
    test_smoothgrad,
    test_imagekernelshap,
    test_imagelime,
    test_imagesobol,
    ImageOcclusion,
    test_squaregrad,
    test_imagegrad,
]
methods = [test_saliency, test_imagelime, test_smoothgrad]
print("Beggining of the methods")

for method_cls in methods:
    name = method_name(method_cls)
    captum_fn = CAPTUM_MAP.get(name)

    print(f"Running {name}...")

    method = method_cls(model=model, image_processor=processor)
    output = method.explain(model_inputs=image, targets=[1])[0]
    target_class = int(output.targets[0].item())
    target_label = model.config.id2label[target_class]

    if captum_fn is None:
        print(f"  No captum equivalent for {name}, skipping.")
        continue

    pv = processor(images=image, return_tensors="pt").pixel_values.requires_grad_(True)
    captum_attr = captum_fn(pv, target_class)

    img_disp = _to_displayable(image)
    H_img, W_img = img_disp.shape[:2]
    interp_heatmap = _prepare_heatmap(output, target_idx=0, clip_percentile=0.1, absolute_value=False)
    captum_attr_clipped = np.clip(captum_attr, np.percentile(captum_attr, 0.1), np.percentile(captum_attr, 99.9))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{name} — {target_label}", fontsize=13)

    _draw_attribution_on_ax(
        axes[0],
        img_disp,
        interp_heatmap,
        cmap="vanimo",
        alpha=0.5,
        interpolation="nearest",
        colorbar=True,
        center_zero=None,
        grayscale_background=False,
        fig=fig,
    )
    axes[0].set_title("Interpreto")
    axes[0].axis("off")

    axes[1].imshow(img_disp)
    im = axes[1].imshow(
        captum_attr_clipped,
        extent=(0, W_img, H_img, 0),
        cmap="vanimo",
        alpha=0.5,
        interpolation="nearest",
        vmin=float(captum_attr_clipped.min()),
        vmax=float(captum_attr_clipped.max()),
    )
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("Captum")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(f"heatmap_interpreto_captum_v5/{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved heatmap_interpreto_captum_v5/{name}.png")

print("Done.")
