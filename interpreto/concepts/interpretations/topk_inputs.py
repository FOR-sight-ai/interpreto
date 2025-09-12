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
Base class for concept interpretation methods.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import torch

from interpreto import ModelWithSplitPoints
from interpreto.concepts.interpretations.base import (
    BaseConceptInterpretationMethod,
    verify_concepts_indices,
    verify_granular_inputs,
)
from interpreto.model_wrapping.model_with_split_points import ActivationGranularity
from interpreto.typing import ConceptModelProtocol, ConceptsActivations, LatentActivations


class TopKInputs(BaseConceptInterpretationMethod):
    """Code [:octicons-mark-github-24: `concepts/interpretations/topk_inputs.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/interpretations/topk_inputs.py)

    Implementation of the Top-K Inputs concept interpretation method also called MaxAct.
    It associate to each concept the inputs that activates it the most.
    It is the most natural way to interpret a concept, as it is the most natural way to explain a concept.
    Hence several papers used it without describing it.
    Nonetheless, we can reference Bricken et al. (2023) [^1] from Anthropic for their post on transformer-circuits.

    [^1]:
        Trenton Bricken*, Adly Templeton*, Joshua Batson*, Brian Chen*, Adam Jermyn*, Tom Conerly, Nicholas L Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, Chris Olah
        [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features)
        Transformer Circuits, 2023.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model with split points to use for the interpretation.
        split_point (str): The split point to use for the interpretation.
        concept_model (ConceptModelProtocol): The concept model to use for the interpretation.
        activation_granularity (ActivationGranularity): The granularity at which the interpretation is computed.
            Allowed values are `TOKEN`, `WORD`, `SENTENCE`, and `SAMPLE`.
            Ignored when use_vocab=True.
        k (int): The number of inputs to use for the interpretation.
        use_vocab (bool): If True, the interpretation will be computed from the vocabulary of the model.

    Examples:  # TODO: adapt example to added arguments
        >>> from datasets import load_dataset
        >>> from interpreto import ModelWithSplitPoints
        >>> from interpreto.concepts import NeuronsAsConcepts
        >>> from interpreto.concepts.interpretations import TopKInputs
        >>> # load and split the model
        >>> split = "bert.encoder.layer.1.output"
        >>> model_with_split_points = ModelWithSplitPoints(
        ...     "hf-internal-testing/tiny-random-bert",
        ...     split_points=[split],
        ...     automodel=AutoModelForMaskedLM,
        ...     batch_size=4,
        ... )
        >>> # NeuronsAsConcepts do not need to be fitted
        >>> concept_model = NeuronsAsConcepts(model_with_split_points=model_with_split_points, split_point=split)
        >>> # extracting concept interpretations
        >>> dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")["train"]["text"]
        >>> all_top_k_words = concept_model.interpret(
        ...     interpretation_method=TopKInputs,
        ...     activation_granularity=TopKInputs.activation_granularities.WORD,
        ...     k=2,
        ...     concepts_indices="all",
        ...     inputs=dataset,
        ...     latent_activations=activations,
        ... )
    """

    activation_granularities = ActivationGranularity

    def __init__(
        self,
        *,
        model_with_split_points: ModelWithSplitPoints,
        concept_model: ConceptModelProtocol,
        activation_granularity: ActivationGranularity = ActivationGranularity.WORD,
        split_point: str | None = None,
        concept_encoding_batch_size: int = 1024,
        k: int = 5,
        use_vocab: bool = False,
        device: torch.device | str | None = "cpu",
    ):
        super().__init__(
            model_with_split_points=model_with_split_points,
            concept_model=concept_model,
            split_point=split_point,
            activation_granularity=activation_granularity,
            concept_encoding_batch_size=concept_encoding_batch_size,
            device=device,
        )

        if activation_granularity not in (
            ActivationGranularity.TOKEN,
            ActivationGranularity.WORD,
            ActivationGranularity.SENTENCE,
            ActivationGranularity.SAMPLE,
        ):
            raise ValueError(
                f"The granularity {activation_granularity} is not supported. Supported `activation_granularities`: TOKEN, WORD, SENTENCE, and SAMPLE"
            )

        self.k = k
        self.use_vocab = use_vocab

    def interpret(
        self,
        concepts_indices: int | list[int],
        inputs: list[str] | None = None,
        latent_activations: LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
    ) -> Mapping[int, Any]:
        """
        Give the interpretation of the concepts dimensions in the latent space into a human-readable format.
        The interpretation is a mapping between the concepts indices and a list of inputs allowing to interpret them.
        The granularity of input examples is determined by the `activation_granularity` class attribute.

        The returned inputs are the most activating inputs for the concepts.

        If all activations are zero, the corresponding concept interpretation is set to `None`.

        Args:
            concepts_indices (int | list[int]): The indices of the concepts to interpret.
            inputs (list[str] | None): The inputs to use for the interpretation.
                Necessary if not `use_vocab`, as examples are extracted from the inputs.
            latent_activations (Float[torch.Tensor, "nl d"] | None): The latent activations matching the inputs. If not provided, it is computed from the inputs.
            concepts_activations (Float[torch.Tensor, "nl cpt"] | None): The concepts activations matching the inputs. If not provided, it is computed from the inputs or latent activations.

        Returns:
            Mapping[int, Any]: The interpretation of the concepts indices.

        """
        # compute the concepts activations from the provided source, can also create inputs from the vocabulary
        if self.use_vocab:
            sure_inputs, sure_concepts_activations = self.concepts_activations_from_vocab()
            granular_inputs = sure_inputs
        else:
            if inputs is None:
                raise ValueError("Inputs must be provided when `use_vocab` is False.")
            sure_inputs = inputs
            sure_concepts_activations = self.concepts_activations_from_source(
                inputs=inputs,
                latent_activations=latent_activations,
                concepts_activations=concepts_activations,
            )
            granular_inputs, _ = self.get_granular_inputs(sure_inputs)

        concepts_indices = verify_concepts_indices(
            concepts_activations=sure_concepts_activations, concepts_indices=concepts_indices
        )
        verify_granular_inputs(
            granular_inputs=granular_inputs,
            sure_concepts_activations=sure_concepts_activations,
            latent_activations=latent_activations,
            concepts_activations=concepts_activations,
        )

        return self._topk_inputs_from_concepts_activations(
            inputs=granular_inputs,
            concepts_activations=sure_concepts_activations,
            concepts_indices=concepts_indices,
        )

    def _topk_inputs_from_concepts_activations(
        self,
        inputs: list[str],  # (nl,)
        concepts_activations: ConceptsActivations,  # (nl, cpt)
        concepts_indices: list[int],  # TODO: sanitize this previously
    ) -> Mapping[int, Any]:
        # increase the number k to ensure that the top-k inputs are unique
        k = self.k * max(Counter(inputs).values())
        k = min(k, concepts_activations.shape[0])

        # Shape: (n*l, cpt_of_interest)
        concepts_activations = concepts_activations.T[concepts_indices].T

        # extract indices of the top-k input tokens for each specified concept
        topk_output = torch.topk(concepts_activations, k=k, dim=0)
        all_topk_activations = topk_output[0].T  # Shape: (cpt_of_interest, k)
        all_topk_indices = topk_output[1].T  # Shape: (cpt_of_interest, k)

        # create a dictionary with the interpretation
        interpretation_dict = {}
        # iterate over required concepts
        for cpt_idx, topk_activations, topk_indices in zip(
            concepts_indices, all_topk_activations, all_topk_indices, strict=True
        ):
            interpretation_dict[cpt_idx] = {}
            # iterate over k
            for activation, input_index in zip(topk_activations, topk_indices, strict=True):
                # ensure that the input is not already in the interpretation
                if len(interpretation_dict[cpt_idx]) >= self.k:
                    break
                if inputs[input_index] in interpretation_dict[cpt_idx]:
                    continue
                if activation == 0:
                    break
                # set the kth input for the concept
                interpretation_dict[cpt_idx][inputs[input_index]] = activation.item()

            # if no inputs were found for the concept, set it to None
            # TODO: see if we should remove the concept completely
            if len(interpretation_dict[cpt_idx]) == 0:
                interpretation_dict[cpt_idx] = None
        return interpretation_dict
