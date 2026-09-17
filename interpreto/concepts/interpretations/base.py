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

import warnings
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any, Literal

import nltk
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from interpreto.concepts.base import ConceptEncoderExplainer
from interpreto.concepts.splitters.splitter_for_classification import (
    SplitterForClassification,
)
from interpreto.concepts.splitters.text_tokens_splitter import TextTokensSplitter, TokenPooling
from interpreto.typing import ConceptsActivations, LatentActivations


@lru_cache(maxsize=1)
def _ensure_nltk_resources(lemmatize: bool) -> None:
    """
    Ensure NLTK resources are downloaded.

    Only used in `extract_ngrams`.

    The `lru_cache` ensures the download are only called once.
    """
    needed = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }

    if lemmatize:
        needed["wordnet"] = "corpora/wordnet.zip"

    for package, resource_path in needed.items():
        # Even if already present, nltk still reaches internet which can crash if no internet connection
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(
                package,
                quiet=True,
                raise_on_error=True,
            )


@jaxtyped(typechecker=beartype)
def extract_ngrams(
    inputs: Iterable[str],
    n: int = 1,
    count_min_threshold: int = 1,
    return_counts: bool = False,
    lemmatize: bool = False,
    words_to_ignore: list[str] | None = None,
) -> list[str] | Counter[str]:
    """
    Extract n-grams (from 1-gram up to n-gram of words) from a list of texts.

    If n=3, it extracts 1-grams, 2-grams, and 3-grams.

    Args:
        inputs (Iterable[str]):
            The texts to extract n-grams from.

        n (int):
            The maximum n-gram size. All sizes from 1 to n are extracted.

        count_min_threshold (int, optional):
            The minimum total number of occurrences of an n-gram in the whole `inputs`.

        return_counts (bool, optional):
            Whether to return the counts of each n-gram.
            Defaults to False.

        lemmatize (bool, optional):
            Whether to lemmatize words before counting.

        words_to_ignore (list[str] | None, optional):
            A list of words to ignore (applied to individual tokens before forming n-grams).

    Returns:
        list[str] | Counter[str]:
            The list of unique n-grams or the counts of each n-gram.
    """
    _ensure_nltk_resources(lemmatize=lemmatize)

    if lemmatize:
        lemmatizer = WordNetLemmatizer()

    tuple_ngram_counts: Counter[tuple[str]] = Counter()

    for text in inputs:
        tokens = word_tokenize(text)

        # preprocess tokens
        processed = []
        for word in tokens:
            if lemmatize:
                word = lemmatizer.lemmatize(word.lower())  # noqa: PLW2901  # type: ignore  (ignore possibly unbound)
            if words_to_ignore is not None and word in words_to_ignore:
                continue
            processed.append(word)
            tuple_ngram_counts[(word,)] += 1  # unigram tuple

        for size in range(2, n + 1):  # skips size 1 as covered over
            for i in range(len(processed) - size + 1):
                tuple_ngram_counts[tuple(processed[i : i + size])] += 1  # >1-gram tuples

    str_ngram_counts: Counter[str] = Counter(
        {
            " ".join(key): count  # convert ngram tuples to strings
            for key, count in tuple_ngram_counts.items()
            if count >= count_min_threshold  # filter too rare n-grams
        }
    )

    if return_counts:
        return str_ngram_counts

    return list(str_ngram_counts.keys())


def verify_concepts_indices(
    concepts_activations: ConceptsActivations,
    concepts_indices: int | list[int],
) -> list[int]:
    # take subset of concepts as specified by the user
    if isinstance(concepts_indices, int):
        concepts_indices = [concepts_indices]

    if not isinstance(concepts_indices, list) or not all(isinstance(c, int) for c in concepts_indices):
        raise ValueError(f"`concepts_indices` should be 'all', an int, or a list of int. Received {concepts_indices}.")

    if max(concepts_indices) >= concepts_activations.shape[1] or min(concepts_indices) < 0:
        raise ValueError(
            f"At least one concept index out of bounds. `max(concepts_indices)`: {max(concepts_indices)} >= {concepts_activations.shape[1]}."
        )

    return concepts_indices


def verify_granular_inputs(
    granular_inputs: list[str],
    sure_concepts_activations: ConceptsActivations,
    mode: str,
):
    """Validate that granular inputs and concept activation rows agree."""
    if len(granular_inputs) != sure_concepts_activations.shape[0]:
        raise ValueError(
            f"The number of granular inputs ({len(granular_inputs)}) does not match the number of "
            f"concept activation rows ({sure_concepts_activations.shape[0]}) for {mode}. "
            "Precomputed activations must match the selected interpretation mode: use token-level "
            "activations for token-level text splitters, and one pooled activation per input when "
            "`token_pooling` is set."
        )


class BaseConceptInterpretationMethod(ABC):
    """Code: [:octicons-mark-github-24: `concepts/interpretations/base.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/interpretations/base.py)

    Abstract class defining an interface for concept interpretation.
    Its goal is to make the dimensions of the concept space interpretable by humans.

    Attributes:
        concept_explainer (ConceptEncoderExplainer):
            The concept explainer used to compute the concept activations.

        token_pooling (TokenPooling):
            Optional pooling applied to token activations. With pooling,
            interpretation examples are whole inputs rather than tokens.

        concept_encoding_batch_size (int):
            The batch size to use for the concept encoding.

        use_vocab (bool):
            Whether to use the vocabulary to extract granular inputs.
            If False, granular inputs are extracted from the inputs.

        use_unique_words (bool):
            If True, the interpretation will be computed from the unique words of the inputs.
            Incompatible with `use_vocab=True`.
            Default unique words selects all different word from the input.
            It can be tuned through the `unique_words_kwargs` argument.

        unique_words_kwargs (dict):
            The kwargs to pass to the `extract_ngrams` function.
            see `interpreto.concepts.interpretations.topk_inputs.extract_ngrams` for more details.
            Possible arguments are `count_min_threshold`, `lemmatize`, `words_to_ignore`.

    """

    def __init__(
        self,
        concept_explainer: ConceptEncoderExplainer,
        token_pooling: TokenPooling = None,
        concept_encoding_batch_size: int = 1024,
        use_vocab: bool = False,
        use_unique_words: bool | int = 0,
        unique_words_kwargs: dict = {},
    ):
        if use_unique_words and use_vocab:
            raise ValueError("Cannot use both `use_unique_words` and `use_vocab`. Please use only one of them.")

        self.concept_explainer: ConceptEncoderExplainer = concept_explainer
        self.token_pooling: TokenPooling = token_pooling
        self.concept_encoding_batch_size: int = concept_encoding_batch_size
        self.use_vocab: bool = use_vocab
        self.use_unique_words: int = int(use_unique_words)
        self.unique_words_kwargs: dict = unique_words_kwargs

    def _is_pooled_mode(self) -> bool:
        """Whether the splitter exposes one representation per input."""
        return isinstance(self.concept_explainer.splitter, SplitterForClassification) or self.token_pooling is not None

    def _mode_description(self) -> str:
        """Describe the representation mode for alignment errors."""
        if isinstance(self.concept_explainer.splitter, SplitterForClassification):
            return "classification (one pooled representation per input)"
        if self.token_pooling is not None:
            return f"pooled text representations (token_pooling={self.token_pooling!r})"
        return "token-level text representations"

    @abstractmethod
    def interpret(
        self,
        concepts_indices: int | list[int],
        inputs: list[str] | None = None,
        latent_activations: LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
    ) -> Mapping[int, Any]:
        """
        Interpret the concepts dimensions in the latent space into a human-readable format.
        The interpretation is a mapping between the concepts indices and an object allowing to interpret them.
        It can be a label, a description, examples, etc.

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
            Mapping[int, Any]:
                The interpretation of each of the specified concepts.
        """
        raise NotImplementedError

    def concepts_activations_from_source(
        self,
        *,
        inputs: list[str] | None = None,
        latent_activations: Float[torch.Tensor, "nl d"] | None = None,
        concepts_activations: Float[torch.Tensor, "nl cpt"] | None = None,
    ) -> Float[torch.Tensor, "nl cpt"]:
        """
        Computes the concepts activations from the given samples.
        Samples can be provided as raw text (`inputs`), latent activations (`latent_activations`),
        or directly concept activations (`concepts_activations`).

        Args:
            inputs (list[str] | None): The indices of the concepts to interpret.
            latent_activations (Float[torch.Tensor, "nl d"] | None): The latent activations
            concepts_activations (Float[torch.Tensor, "nl cpt"] | None): The concepts activations

        Returns:
            Float[torch.Tensor, "nl cpt"] :
        """

        if concepts_activations is not None:
            return concepts_activations

        if latent_activations is not None:
            # batch over latent activations for concept encoding
            concepts_activations_list = []
            with torch.no_grad():
                for batch_idx in range(0, latent_activations.shape[0], self.concept_encoding_batch_size):
                    # concept model forward pass
                    batch_concepts_activations = self.concept_explainer.activations_to_concepts(
                        latent_activations[batch_idx : batch_idx + self.concept_encoding_batch_size]
                    ).cpu()
                    concepts_activations_list.append(batch_concepts_activations)
            concepts_activations = torch.cat(concepts_activations_list, dim=0)
            return concepts_activations

        if inputs is not None:
            extraction_kwargs: dict[str, Any] = {}
            if isinstance(self.concept_explainer.splitter, TextTokensSplitter):
                extraction_kwargs["token_pooling"] = self.token_pooling
            latent_activations, _ = self.concept_explainer.splitter.get_activations(
                inputs,
                **extraction_kwargs,
            )
            return self.concepts_activations_from_source(latent_activations=latent_activations, inputs=inputs)

        raise ValueError(
            "No source provided. Please provide either `inputs`, `latent_activations`, or `concepts_activations`."
        )

    @jaxtyped(typechecker=beartype)
    def concepts_activations_from_vocab(
        self,
    ) -> tuple[list[str], Float[torch.Tensor, "nl cpt"]]:
        """
        Computes the concepts activations for each token of the vocabulary

        Each vocabulary ID is passed as one input. Classification splitters add
        their tokenizer's model-specific formatting; token splitters process
        the ID directly.

        Returns:
            tuple[list[str], Float[torch.Tensor, "nl cpt"]]:
                - The list of tokens in the vocabulary
                - The concept activations for each token
        """
        # extract and sort the vocabulary
        vocab_dict: dict[str, int] = self.concept_explainer.splitter.tokenizer.get_vocab()
        inputs, vocab_ids = zip(*vocab_dict.items(), strict=True)  # type: ignore
        inputs = list(inputs)

        splitter = self.concept_explainer.splitter
        if isinstance(splitter, TextTokensSplitter):
            model_inputs = torch.tensor(vocab_ids).unsqueeze(1)
        else:
            model_inputs = vocab_ids
        latent_activations, _ = splitter.get_activations(model_inputs)

        # compute the vocabulary's concepts activations
        with torch.no_grad():
            concepts_activations = self.concept_explainer.activations_to_concepts(latent_activations)
        return inputs, concepts_activations

    @jaxtyped(typechecker=beartype)
    def get_granular_inputs(
        self,
        inputs: list[str],  # (n)
    ) -> tuple[list[str], list[int]]:
        """Return display units matching the splitter's representation rows.

        Args:
            inputs (list[str]): n text samples

        Returns:
            granular_inputs (list[str]):
                The granular inputs extracted from the inputs, flattened.
                [Example1_Tok1, Example1_Tok2, ... Example2_Tok1, Example2_Tok2, ...]

            granular_sample_ids (list[int]):
                The sample id for each granular input.
                [0, 0, ... 1, 1, ...]
        """
        if self._is_pooled_mode():
            return inputs, list(range(len(inputs)))

        splitter = self.concept_explainer.splitter
        if not isinstance(splitter, TextTokensSplitter):
            raise TypeError("Token-level interpretation requires a TextTokensSplitter.")

        tokenized, tokens_mask = splitter._tokenize_and_get_mask(inputs, include_special_tokens=False)
        input_ids = tokenized["input_ids"]
        granular_inputs: list[str] = []
        granular_sample_ids: list[int] = []
        for sample_index, (ids_row, mask_row) in enumerate(zip(input_ids, tokens_mask, strict=True)):  # type: ignore
            tokens = splitter.tokenizer.convert_ids_to_tokens(ids_row[mask_row].tolist())
            granular_inputs.extend(tokens)
            granular_sample_ids.extend([sample_index] * len(tokens))
        return granular_inputs, granular_sample_ids

    def get_granular_inputs_and_concept_activations(
        self,
        concepts_indices: int | list[int] | Literal["all"],
        inputs: list[str] | None = None,
        latent_activations: LatentActivations | None = None,
        concepts_activations: ConceptsActivations | None = None,
    ) -> tuple[list[int], list[str], Float[torch.Tensor, "nl cpt"], list[int]]:
        """
        Compute the granular inputs and concept activations for the specified concepts.

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
            sure_concepts_indices (list[int]):
                The indices of the concepts to interpret.

            granular_inputs (list[str]):
                The granular inputs for the specified concepts.

            sure_concepts_activations (Float[torch.Tensor, "nl cpt"]):
                The concepts activations matching the granular inputs.

            granular_sample_ids (list[int]):
                The input sample index for each granular input.

        """
        if concepts_indices == "all":
            concepts_indices = list(range(self.concept_explainer.concept_model.nb_concepts))

        # compute the concepts activations from the provided source, can also create inputs from the vocabulary
        if self.use_vocab:
            # --------------------------------------------------------------------------------------
            # Case 1: use_vocab=True
            granular_inputs: list[str]
            sure_concepts_activations: Float[torch.Tensor, "nl cpt"]
            granular_inputs, sure_concepts_activations = self.concepts_activations_from_vocab()

            granular_sample_ids: list[int] = list(range(len(granular_inputs)))
        else:
            if inputs is None:
                raise ValueError("Inputs must be provided when `use_vocab` is False.")

            if self.use_unique_words >= 1:
                # ----------------------------------------------------------------------------------
                # Case 2: use_unique_words >= 1
                # first list unique words/ngrams from the inputs and compute the activations from them
                if not self._is_pooled_mode():
                    raise ValueError(
                        "`use_unique_words` requires pooled representations. "
                        "Use SplitterForClassification or set `token_pooling` on TextTokensSplitter."
                    )
                granular_inputs: list[str] = extract_ngrams(
                    inputs=inputs,
                    n=self.use_unique_words,
                    return_counts=False,
                    **self.unique_words_kwargs,
                )  # type: ignore  (sure list[str] with return_counts=False)
                if latent_activations is not None and concepts_activations is not None:
                    warnings.warn(
                        "`latent_activations` or `concepts_activations` were provided, "
                        "but `use_unique_words` is True. "
                        "Therefore, the inputs and activations will likely mismatch. "
                        "Either do not provide `latent_activations` and `concepts_activations`, "
                        "or use `interpreto.concepts.interpretation.extract_ngrams` yourself, "
                        "and set `use_unique_words` to False.",
                        stacklevel=2,
                    )
                sure_concepts_activations = self.concepts_activations_from_source(
                    inputs=granular_inputs,
                    latent_activations=latent_activations,
                    concepts_activations=concepts_activations,
                )

                granular_sample_ids: list[int] = list(range(len(granular_inputs)))
            else:
                # ----------------------------------------------------------------------------------
                # Case 3: Default, use_vocab=False and use_unique_words=False
                sure_concepts_activations = self.concepts_activations_from_source(
                    inputs=inputs,
                    latent_activations=latent_activations,
                    concepts_activations=concepts_activations,
                )
                granular_inputs: list[str]
                granular_sample_ids: list[int]
                granular_inputs, granular_sample_ids = self.get_granular_inputs(inputs)

        sure_concepts_indices = verify_concepts_indices(
            concepts_activations=sure_concepts_activations,
            concepts_indices=concepts_indices,
        )
        verify_granular_inputs(
            granular_inputs=granular_inputs,
            sure_concepts_activations=sure_concepts_activations,
            mode=self._mode_description(),
        )

        return (
            sure_concepts_indices,
            granular_inputs,
            sure_concepts_activations,
            granular_sample_ids,
        )
