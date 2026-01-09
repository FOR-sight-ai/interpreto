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
from typing import Any, Literal

import torch

from interpreto.concepts.base import ConceptEncoderExplainer
from interpreto.concepts.interpretations.base import (
    BaseConceptInterpretationMethod,
)
from interpreto.model_wrapping.model_with_split_points import ActivationGranularity
from interpreto.typing import ConceptsActivations, LatentActivations


class TopKInputs(BaseConceptInterpretationMethod):
    """Code [:octicons-mark-github-24: `concepts/interpretations/topk_inputs.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/interpretations/topk_inputs.py)

    Implementation of the Top-K Inputs concept interpretation method also called MaxAct, or CMAW.
    It associate to each concept the inputs that activates it the most.
    It is the most natural way to interpret a concept, as it is the most natural way to explain a concept.
    Hence several papers used it without describing it.
    Nonetheless, we can reference Bricken et al. (2023) [^1] from Anthropic for their post on transformer-circuits.

    [^1]:
        Trenton Bricken*, Adly Templeton*, Joshua Batson*, Brian Chen*, Adam Jermyn*, Tom Conerly, Nicholas L Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, Chris Olah
        [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features)
        Transformer Circuits, 2023.

    Arguments:
        concept_explainer (ConceptEncoderExplainer):
            The concept explainer built on top of a `ModelWithSplitPoints`.

        activation_granularity (ActivationGranularity):
            The granularity at which the interpretation is computed.
            Allowed values are `CLS_TOKEN`, `TOKEN`, `WORD`, `SENTENCE`, and `SAMPLE`.
            Ignored when `use_vocab=True`.

        concept_encoding_batch_size (int):
            The batch size to use for the concept encoding.

        k (int):
            The number of inputs to use for the interpretation.

        use_vocab (bool):
            If True, the interpretation will be computed from the vocabulary of the model.

        use_unique_words (bool):
            If True, the interpretation will be computed from the unique words of the inputs.
            Incompatible with `use_vocab=True`.
            Default unique words selects all different word from the input.
            It can be tuned through the `unique_words_kwargs` argument.

        unique_words_kwargs (dict):
            The kwargs to pass to the `extract_unique_words` function.
            See [`extract_unique_words`][interpreto.concepts.interpretations.extract_unique_words] for more details.
            Possible arguments are `count_min_threshold`, `lemmatize`, `words_to_ignore`.

        concept_model_device (torch.device | str | None):
            The device to use for the concept model forward pass.
            If None, does not change the device.

    Examples:
        **Minimal example**, finding the topk tokens activating a neuron:
        >>> from transformers import AutoModelForCausalLM
        >>>
        >>> from interpreto import ModelWithSplitPoints
        >>> from interpreto.concepts import NeuronsAsConcepts, TopKInputs
        >>>
        >>> # load and split the the GPT2 model
        >>> mwsp = ModelWithSplitPoints(
        ...     "gpt2",
        ...     split_points=[11],           # split at the 12th layer
        ...     automodel=AutoModelForCausalLM,
        ...     device_map="auto",
        ...     batch_size=2048,
        ... )
        >>>
        >>> # Use `NeuronsAsConcepts` to use the concept-based pipeline with neurons
        >>> concept_explainer = NeuronsAsConcepts(mwsp)
        >>>
        >>> method = TopKInputs(
        ...     concept_explainer=concept_explainer,
        ...     use_vocab=True,             # use the vocabulary of the model and test all tokens (50257 with GPT2)
        ...     k=10,                       # get the top 10 tokens for each neuron
        ... )
        >>>
        >>> topk_tokens = method.interpret(
        ...     concepts_indices="all",     # interpret the three first neurons of the 7th layer
        ... )
        >>>
        >>> print(list(topk_tokens[1].keys()))
        ['hostages', 'choke', 'infring', 'herpes', 'nuns', 'phylogen', 'watched', 'alitarian', 'tattoos', 'fisher']
        >>> # Results are not interpretable, due to superposition and such.
        >>> # This is why we use dictionary to find concept direction!

        **Classification example**, we should fit concepts on the [CLS] token activations,
        then use `TopKInputs` with `use_unique_words=True` and `activation_granularity=CSL_TOKEN`:
        >>> from datasets import load_dataset
        >>> from transformers import AutoModelForSequenceClassification
        >>>
        >>> from interpreto import ModelWithSplitPoints
        >>> from interpreto.concepts import ICAConcepts, TopKInputs
        >>>
        >>> CLS_TOKEN = ModelWithSplitPoints.activation_granularities.CLS_TOKEN
        >>>
        >>> # load and split an IMDB classification model
        >>> mwsp = ModelWithSplitPoints(
        ...     "textattack/bert-base-uncased-imdb",
        ...     split_points=[11],              # split at the last layer
        ...     automodel=AutoModelForSequenceClassification,
        ...     device_map="cuda",
        ...     batch_size=64,
        ... )
        >>>
        >>> # load the IMDB dataset and compute a dataset of [CLS] token activations
        >>> imdb = load_dataset("stanfordnlp/imdb", split="train")["text"][:1000]
        >>> activations = mwsp.get_activations(imdb, activation_granularity=CLS_TOKEN)
        >>>
        >>> # Load an fit a concept-based explainer
        >>> concept_explainer = ICAConcepts(mwsp, nb_concepts=20)
        >>> concept_explainer.fit(activations)
        >>>
        >>> method = TopKInputs(
        ...     concept_explainer=concept_explainer,
        ...     activation_granularity=CLS_TOKEN,
        ...     k=5,                            # get the top 10 tokens for each concept
        ...     use_unique_words=True,          # necessary to get topk words on the [CLS] token
        ...     unique_words_kwargs={
        ...         "count_min_threshold": 5,   # only consider words that appear at least 5 times in the dataset
        ...         "lemmatize": True,          # group words by their lemma (e.g., "bad" and "badly" are grouped together)
        ...     }
        ... )
        >>>
        >>> topk_words = method.interpret(
        ...     inputs=imdb,
        ...     concepts_indices="all",     # interpret the three first neurons of the 7th layer
        ... )
        >>>
        >>> print(list(topk_words[1].keys()))
        ['bad', 'bad.', 'hackneyed', 'clichéd', 'cannibal']

        **Generation example**, use either `TOKEN` or `WORD` granularity for activations.
        `WORD` allows to select the topk words for each concept without recomputing the activations.
        >>> from datasets import load_dataset
        >>> from transformers import AutoModelForCausalLM
        >>>
        >>> from interpreto import ModelWithSplitPoints
        >>> from interpreto.concepts import ICAConcepts, TopKInputs
        >>>
        >>> WORD = ModelWithSplitPoints.activation_granularities.WORD
        >>>
        >>> # load and split the the GPT2 model
        >>> mwsp = ModelWithSplitPoints(
        ...     "Qwen/Qwen3-0.6B",
        ...     split_points=[9],              # split at the 10th layer
        ...     automodel=AutoModelForCausalLM,
        ...     device_map="auto",
        ...     batch_size=16,
        ... )
        >>>
        >>> # load the IMDB dataset and compute a dataset of words activations
        >>> imdb = load_dataset("stanfordnlp/imdb", split="train")["text"][:1000]
        >>> activations = mwsp.get_activations(imdb, activation_granularity=WORD)
        >>>
        >>> # Load an fit a concept-based explainer
        >>> concept_explainer = ICAConcepts(mwsp, nb_concepts=10)
        >>> concept_explainer.fit(activations)
        >>>
        >>> method = TopKInputs(
        ...     concept_explainer=concept_explainer,
        ...     activation_granularity=WORD,    # we want the topk words for each concept
        ...     k=10,                           # get the top 10 words for each concept
        ...     device="cuda",
        ... )
        >>>
        >>> topk_tokens = method.interpret(
        ...     concepts_indices="all",     # interpret the three first neurons of the 7th layer
        ...     inputs=imdb,
        ...     latent_activations=activations, # use previously computed activations (same granularity)
        ... )

    """

    activation_granularities = ActivationGranularity

    def __init__(
        self,
        *,
        concept_explainer: ConceptEncoderExplainer,
        activation_granularity: ActivationGranularity = ActivationGranularity.WORD,
        concept_encoding_batch_size: int = 1024,
        k: int = 5,
        use_vocab: bool = False,
        use_unique_words: bool = False,
        unique_words_kwargs: dict = {},
        concept_model_device: torch.device | str | None = None,
    ):
        super().__init__(
            concept_explainer=concept_explainer,
            activation_granularity=activation_granularity,
            concept_encoding_batch_size=concept_encoding_batch_size,
            use_vocab=use_vocab,
            use_unique_words=use_unique_words,
            unique_words_kwargs=unique_words_kwargs,
            concept_model_device=concept_model_device,
        )

        self.k = k

    def interpret(
        self,
        concepts_indices: int | list[int] | Literal["all"],
        inputs: list[str] | None = None,
        latent_activations: dict[str, torch.Tensor] | LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
    ) -> Mapping[int, Any]:
        """
        Give the interpretation of the concepts dimensions in the latent space into a human-readable format.
        The interpretation is a mapping between the concepts indices and a list of inputs allowing to interpret them.
        The granularity of input examples is determined by the `activation_granularity` class attribute.

        The returned inputs are the most activating inputs for the concepts.

        If all activations are zero, the corresponding concept interpretation is set to `None`.

        Args:
            concepts_indices (int | list[int] | Literal["all"]):
                The indices of the concepts to interpret. If "all", all concepts are interpreted.

            inputs (list[str] | None):
                The inputs to use for the interpretation.
                Necessary if not `use_vocab`,as examples are extracted from the inputs.

            latent_activations (dict[str, torch.Tensor] | Float[torch.Tensor, "nl d"] | None):
                The latent activations matching the inputs. If not provided,
                it is computed from the inputs.

            concepts_activations (Float[torch.Tensor, "nl cpt"] | None):
                The concepts activations matching the inputs. If not provided,
                it is computed from the inputs or latent activations.

        Returns:
            Mapping[int, Any]: The interpretation of the concepts indices.

        """
        sure_concepts_indices, granular_inputs, sure_concepts_activations, _ = (
            self.get_granular_inputs_and_concept_activations(
                concepts_indices=concepts_indices,
                inputs=inputs,
                latent_activations=latent_activations,
                concepts_activations=concepts_activations,
            )
        )
        sure_concepts_indices: list[int]
        granular_inputs: list[str]
        sure_concepts_activations: torch.Tensor

        return self._topk_inputs_from_concepts_activations(
            inputs=granular_inputs,
            concepts_activations=sure_concepts_activations,
            concepts_indices=sure_concepts_indices,
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
