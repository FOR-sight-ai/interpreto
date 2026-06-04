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
Implementation of the Cockatiel concept explainer
"""

from __future__ import annotations

from interpreto.concepts.methods.overcomplete import NMFConcepts
from interpreto.typing import ConceptsActivations, ModelInputs


class Cockatiel(NMFConcepts):
    """Code: [:octicons-mark-github-24: `concepts/methods/cockatiel.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/cockatiel.py)

    Implementation of the Cockatiel concept explainer by Jourdan et al. (2023)[^1].

    [^1]:
        Jourdan F., Picard A., Fel T., Risser A., Loubes JM., and Asher N. [COCKATIEL: COntinuous Concept ranKed ATtribution with Interpretable ELements for explaining neural net classifiers on NLP.](https://aclanthology.org/2023.findings-acl.317/)
        Findings of the Association for Computational Linguistics (ACL 2023), pp. 5120–5136, 2023.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
            It should have at least one split point on which `concept_model` can be fitted.
        split_point (str | None): The split point used to train the `concept_model`. Default: `None`, set only when
            the concept explainer is fitted.
        concept_model (overcomplete.optimization.SemiNMF): An [Overcomplete NMF](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/optimization/nmf.py) encoder-decoder.
        force_relu (bool): Whether to force the activations to be positive.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `activations_to_concepts` operation is differentiable.
        has_differentiable_concept_decoder (bool): Whether the `concepts_to_activations` operation is differentiable.
    """

    def input_concept_attribution(
        self,
        inputs: ModelInput,
        concept: int,
        **attribution_kwargs,
    ) -> list[float]:
        """
        Computes the attribution of each input to a given concept.

        Args:
            inputs (ModelInputs): The input data, which can be a string, a list of tokens/words/clauses/sentences, or a dataset.
            concept (int | list[int]): The concept index (or list of concepts indices) to analyze.

        Returns:
            A list of attribution scores for each input.
        """
        return super().input_concept_attribution(
            inputs, concept, "Occlusion", **attribution_kwargs
        )  # TODO: add occlusion class when it exists

    def concept_output_attribution(
        self, inputs: ModelInputs, concepts: ConceptsActivations, target: int, **attribution_kwargs
    ) -> list[float]:
        """Computes the attribution of each concept for the logit of a target output element.

        Args:
            inputs (ModelInputs): An input datapoint for the model.
            concepts (torch.Tensor): Concept activation tensor.
            target (int): The target class for which the concept output attribution should be computed.

        Returns:
            A list of attribution scores for each concept.
        """
        return super().concept_output_attribution(
            inputs, concepts, target, attribution_method="Sobol", **attribution_kwargs
        )  # TODO: add sobol class when it exists
