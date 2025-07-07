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

from interpreto.concepts.base import ConceptEncoderExplainer
from interpreto.typing import ConceptsActivations, LatentActivations


class Sparsity:
    r"""Code [:octicons-mark-github-24: `concepts/metrics/sparsity_metrics.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/metrics/sparsity_metrics.py)

    Evaluates the sparsity of the concepts activations.
    It takes in the `concept_explainer` and the `latent_activations`, compute the `concept_activations` and then compute the sparsity of the `concept_activations`.

    The sparsity is defined as:
    $$ \sum_{x}^{X} \sum_{i=1}^{cpt} \mathbb{1} ( | t(h(x))_i | > \epsilon ) $$
    TODO: make the formula work

    Attributes:
        concept_explainer (ConceptEncoderExplainer): The explainer used to compute concepts.
        epsilon (float): The threshold used to compute the sparsity.
    """

    def __init__(self, concept_explainer: ConceptEncoderExplainer, epsilon: float = 0.0):
        self.concept_explainer = concept_explainer
        self.epsilon = epsilon

    def compute(self, latent_activations: LatentActivations | dict[str, LatentActivations]) -> float:
        """Compute the metric.

        Args:
            latent_activations (LatentActivations | dict[str, LatentActivations]): The latent activations.

        Returns:
            float: The metric.
        """
        split_latent_activations: LatentActivations = self.concept_explainer._sanitize_activations(latent_activations)

        concepts_activations: ConceptsActivations = self.concept_explainer.encode_activations(split_latent_activations)

        return torch.mean(torch.abs(concepts_activations) > self.epsilon, dtype=torch.float32).item()


class SparsityRatio(Sparsity):
    r"""Code [:octicons-mark-github-24: `concepts/metrics/sparsity_metrics.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/metrics/sparsity_metrics.py)

    Evaluates the sparsity ratio of the concepts activations.
    It takes in the `concept_explainer` and the `latent_activations`, compute the `concept_activations` and then compute the sparsity ratio of the `concept_activations`.

    With $A$ latent activations obtained through $A = h(X)$, the sparsity ratio is defined as:
    $$ (1 / cpt) * \sum_{a}^{A} \sum_{i=1}^{cpt} \mathbb{1} ( | t(a)_i | > \epsilon ) $$
    TODO: make the formula work

    Attributes:
        concept_explainer (ConceptEncoderExplainer): The explainer used to compute concepts.
        epsilon (float): The threshold used to compute the sparsity.
    """

    def compute(self, latent_activations: LatentActivations | dict[str, LatentActivations]) -> float:
        """Compute the metric.

        Args:
            latent_activations (LatentActivations | dict[str, LatentActivations]): The latent activations.

        Returns:
            float: The metric.
        """
        sparsity = super().compute(latent_activations)
        return sparsity / self.concept_explainer.concept_model.nb_concepts


# TODO: add hoyer and co, see overcomplete
