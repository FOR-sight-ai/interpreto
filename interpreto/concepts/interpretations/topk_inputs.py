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

from collections.abc import Mapping
from typing import Any, Literal

import torch

from interpreto.concepts.base import ConceptEncoderExplainer
from interpreto.concepts.interpretations.base import (
    BaseConceptInterpretationMethod,
)
from interpreto.concepts.splitters.text_tokens_splitter import TokenPooling
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
            The fitted concept explainer used to encode activations.

        token_pooling (TokenPooling):
            Optional pooling applied to token activations before interpretation.

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
            The kwargs to pass to the `extract_ngrams` function.
            See [`extract_ngrams`][interpreto.concepts.interpretations.extract_ngrams] for more details.
            Possible arguments are `count_min_threshold`, `lemmatize`, `words_to_ignore`.

    Examples:
        **Vocabulary examples** rank every token in the model vocabulary:
        >>> from interpreto import TextTokensSplitter
        >>> from interpreto.concepts import NeuronsAsConcepts, TopKInputs
        >>>
        >>> splitter = TextTokensSplitter(
        ...     "gpt2",
        ...     split_point=11,
        ...     device_map="auto",
        ... )
        >>> method = TopKInputs(
        ...     concept_explainer=NeuronsAsConcepts(splitter),
        ...     use_vocab=True,
        ...     k=5,
        ... )
        >>> topk_tokens = method.interpret(concepts_indices=[0, 1])

        **Classification examples** use independently encoded words or n-grams
        because the splitter exposes one pooled representation per input:
        >>> from interpreto import SplitterForClassification
        >>> from interpreto.concepts import NeuronsAsConcepts, TopKInputs
        >>>
        >>> classifier = SplitterForClassification(
        ...     "textattack/bert-base-uncased-imdb",
        ...     device_map="auto",
        ... )
        >>> reviews = ["A remarkably good film", "A predictable story"]
        >>> method = TopKInputs(
        ...     concept_explainer=NeuronsAsConcepts(classifier),
        ...     k=5,
        ...     use_unique_words=3,            # include up to 3-grams in the interpretation
        ...     unique_words_kwargs={
        ...         "count_min_threshold": 5,   # only consider words that appear at least 5 times in the dataset
        ...         "lemmatize": True,          # group words by their lemma (e.g., "bad" and "badly" are grouped together)
        ...     }
        ... )
        >>> topk_words = method.interpret(
            inputs=reviews,
            concepts_indices="all",             # interpret all neurons at the [CLS] token position
        )

        **Precomputed activations** can be passed directly,
        ensure the interpretation token_pooling matches the activations one (if used):
        >>> from interpreto import TextTokensSplitter
        >>> from interpreto.concepts import NeuronsAsConcepts, TopKInputs
        >>>
        >>> splitter = TextTokensSplitter(
        ...     "gpt2",
        ...     split_point=11,
        ...     device_map="auto",
        ... )
        >>> texts = ["Interpreto explains models", "Concepts reveal patterns"]
        >>> pooled_activations, _ = splitter.get_activations(texts, token_pooling="mean")
        >>> method = TopKInputs(
        ...     concept_explainer=NeuronsAsConcepts(splitter),
        ...     token_pooling="mean",
        ...     k=2,
        ... )
        >>> topk_texts = method.interpret(
        ...     inputs=texts,
        ...     latent_activations=pooled_activations,
        ...     concepts_indices=[0, 1],
        ... )

    """

    def __init__(
        self,
        *,
        concept_explainer: ConceptEncoderExplainer,
        token_pooling: TokenPooling = None,
        concept_encoding_batch_size: int = 1024,
        k: int = 5,
        use_vocab: bool = False,
        use_unique_words: bool | int = 0,
        unique_words_kwargs: dict = {},
    ):
        super().__init__(
            concept_explainer=concept_explainer,
            token_pooling=token_pooling,
            concept_encoding_batch_size=concept_encoding_batch_size,
            use_vocab=use_vocab,
            use_unique_words=use_unique_words,
            unique_words_kwargs=unique_words_kwargs,
        )

        self.k = k

    def interpret(
        self,
        concepts_indices: int | list[int] | Literal["all"] = "all",
        inputs: list[str] | None = None,
        latent_activations: LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
    ) -> Mapping[int, Any]:
        """
        Give the interpretation of the concepts dimensions in the latent space into a human-readable format.
        The interpretation is a mapping between the concepts indices and a list of inputs allowing to interpret them.
        Input examples are whole samples for pooled representations and individual tokens otherwise.

        The returned inputs are the most activating inputs for the concepts.

        If all activations are zero, the corresponding concept interpretation is set to `None`.

        Args:
            concepts_indices (int | list[int] | Literal["all"]):
                The indices of the concepts to interpret. If "all", all concepts are interpreted.

            inputs (list[str] | None):
                The inputs to use for the interpretation.
                Necessary if not `use_vocab`,as examples are extracted from the inputs.

            latent_activations (Float[torch.Tensor, "nl d"] | None):
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
        inputs: list[str],
        concepts_activations: torch.Tensor,
        concepts_indices: list[int],
    ):
        device = concepts_activations.device

        # ------------------------------------------------
        # we first reduce inputs which are the same by max
        input_to_id = {}
        unique_inputs = []
        inverse_ids_list = []

        # associate inputs ids to unique inputs
        for x in inputs:
            if input_to_id.get(x) is None:
                input_to_id[x] = len(input_to_id)
            inverse_ids_list.append(input_to_id[x])

        inverse_ids = torch.tensor(inverse_ids_list, device=device, dtype=torch.long)

        # acts:     (n, n_concepts)
        acts = concepts_activations[:, concepts_indices]
        unique_inputs = list(input_to_id.keys())
        n_unique = len(unique_inputs)
        n_concepts = acts.shape[1]

        # init reduced activations
        reduced = torch.full(
            (n_unique, n_concepts),
            -torch.inf,
            device=device,
            dtype=acts.dtype,
        )

        # reduce activations by max for each unique input
        # we go from (n, n_concepts) to (n_unique, n_concepts)
        reduced.scatter_reduce_(
            dim=0,
            index=inverse_ids[:, None].expand(-1, n_concepts),
            src=acts,
            reduce="amax",
            include_self=True,
        )

        # ----------------------------------------------------------------------
        # Then we compute topk (among max value for each input) for each concept
        # top_values:     (k, n_concepts)
        # top_unique_ids: (k, n_concepts)
        top_values, top_unique_ids = torch.topk(reduced, k=self.k, dim=0)

        # convert top indices to top inputs
        # iterate over required concepts
        interpretation_dict = {}
        for cpt_idx, values, ids in zip(concepts_indices, top_values.T, top_unique_ids.T, strict=True):
            out = {}
            # iterate over k
            for activation, unique_id in zip(values, ids, strict=True):
                if (
                    activation == 0
                ):  # TODO: see if we should remove negative values, or maybe optionally return them as top-bottom
                    break
                out[unique_inputs[int(unique_id)]] = activation.item()

            interpretation_dict[cpt_idx] = out if out else None

        return interpretation_dict
