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
Probe models for concept-based interpretability.

This package provides lightweight torch-based probe models that map latent
activations to concept scores. All probes follow the
[ConceptModelProtocol][interpreto.typing.ConceptModelProtocol] interface (structurally) and
support `state_dict` serialization for reproducibility.

Probe families:
    - **Centroid probes** ([centroid][interpreto.concepts.probes.centroid]):
      assign concept scores based on distances to learned centroids.
    - **Linear probes** ([linear][interpreto.concepts.probes.linear]):
      learn a linear mapping from activations to concept scores.
    - **Sklearn probes** ([sklearn][interpreto.concepts.probes.sklearn]):
      wrap scikit-learn classifiers for single-concept experiments.

Supporting modules:
    - [normalizations][interpreto.concepts.probes.normalizations]: input normalization layers.
    - [bias_calibrators][interpreto.concepts.probes.bias_calibrators]: post-hoc bias calibration functions.
"""

from .base import ProbeExplainer
from .bias_calibrators import bce_bias, fpr_bias, lda_shared_var_bias, midpoint_bias, prevalence_bias
from .centroid import (
    CosineCentroidProbe,
    DiagonalMahalanobisCentroidProbe,
    DotProductCentroidProbe,
    SqL2CentroidProbe,
    SVDDCentroidProbe,
)
from .linear import (
    LinearRegressionProbe,
    LinearSVMProbe,
    LogisticRegressionProbe,
    MeansDiffProbe,
)
from .normalizations import Standardization, Whitening

__all__ = [
    "bce_bias",
    "fpr_bias",
    "lda_shared_var_bias",
    "midpoint_bias",
    "prevalence_bias",
    "CosineCentroidProbe",
    "DiagonalMahalanobisCentroidProbe",
    "DotProductCentroidProbe",
    "SqL2CentroidProbe",
    "SVDDCentroidProbe",
    "ProbeExplainer",
    "LinearRegressionProbe",
    "LinearSVMProbe",
    "LogisticRegressionProbe",
    "MeansDiffProbe",
    "Standardization",
    "Whitening",
]
