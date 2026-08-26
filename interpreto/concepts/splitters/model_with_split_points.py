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

import gc
import warnings
from collections.abc import Callable, Iterable
from enum import Enum
from math import ceil
from typing import Any

import nnsight
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from tqdm import tqdm
from transformers import (
    AutoModel,
    BatchEncoding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    T5ForConditionalGeneration,
)

from interpreto.commons.granularity import Granularity, GranularityAggregationStrategy
from interpreto.concepts.splitters.base_splitter import BaseSplitter, InitializationError  # noqa: F401
from interpreto.typing import ConceptsActivations, LatentActivations


class ActivationGranularity(Enum):
    """
    Activation selection strategies for `ModelWithSplitPoints.get_activations()`.

    - ``ALL_TOKENS``:
        the raw activations are flattened ``(n x l, d)``.
        Hence, each token activation is now considered as a separate element.
        This includes special tokens such as [CLS], [SEP], [EOS], [PAD], etc.

    - ``CLS_TOKEN``:
        for each sample, only the first token (e.g. ``[CLS]``) activation is returned ``(n, d)``.
        This will raise an error if the model is not `ForSequenceClassification`.

    - ``SAMPLE``:
        special tokens are removed and the remaining ones are aggregated on the whole sample ``(n, d)``.

    - ``SENTENCE``:
        special tokens are removed and the remaining ones are aggregate by sentences.
        Then the activations are flattened.
        ``(n x g, d)`` where `g` is the number of sentences in the input.
        The split is defined by `interpreto.commons.granularity.Granularity.SENTENCE`.

    - ``TOKEN``:
        the raw activations are flattened, but the special tokens are removed.
        ``(n x g, d)`` where `g` is the number of non-special tokens in the input.
        This is the default granularity.

    - ``WORD``:
        the special tokens are removed and the remaining ones are aggregate by words.
        Then the activations are flattened.
        ``(n x g, d)`` where `g` is the number of words in the input.
        The split is defined by `interpreto.commons.granularity.Granularity.WORD`.
    """

    ALL_TOKENS = Granularity.ALL_TOKENS
    CLS_TOKEN = "cls_token"
    SAMPLE = "sample"
    SENTENCE = Granularity.SENTENCE
    TOKEN = Granularity.TOKEN
    WORD = Granularity.WORD


AG = ActivationGranularity


class ModelWithSplitPoints(BaseSplitter):
    """Code: [:octicons-mark-github-24: `concepts/splitters/model_with_split_points.py`](https://github.com/FOR-sight-ai/interpreto/blob/main/interpreto/concepts/splitters/model_with_split_points.py)

    The `ModelWithSplitPoints` is a wrapper around your HuggingFace model.
    Its goal is to allow you to split your model at specified locations and extract activations.

    It is one of the key component of the Concept-Based Explainers framework in Interpreto.
    Indeed, any Interpreto concept explainer is built around a `ModelWithSplitPoints` object.
    Because, splitting the model is the first step of the concept-based explanation process.

    It is based on the `LanguageModel` class from NNsight and inherits its functionalities.
    In a sense, the LanguageModel class is a wrapper around the HuggingFace model.
    The `ModelWithSplitPoints` class is a wrapper around the LanguageModel class.

    We often shorten the `ModelWithSplitPoints` class as `MWSP` and instances as `mwsp`.

    Arguments:
        model_or_repo_id (str | transformers.PreTrainedModel): One of:

            * A `str` corresponding to the ID of the model that should be loaded from the HF Hub.
            * A `str` corresponding to the local path of a folder containing a compatible checkpoint.
            * A preloaded `transformers.PreTrainedModel` object.
            If a string is provided, a automodel should also be provided.

        split_point (str | int): The split location inside the model.
            Either one of the following:

            * A `str` corresponding to the path of a split point inside the model.
            * An `int` corresponding to the n-th layer.

            Example: `split_point='cls.predictions.transform.LayerNorm'` correspond to a split
            after the LayerNorm layer in the MLM head (assuming a `BertForMaskedLM` model in input).

        split_points (str | int | list[str] | list[int], deprecated): Backward-compatible alias for
            `split_point`. If a list/tuple is provided, only the first element is used.

        automodel (type[AutoModel]): Huggingface [AutoClass](https://huggingface.co/docs/transformers/en/model_doc/auto#natural-language-processing)
            corresponding to the desired type of model (e.g. `AutoModelForSequenceClassification`).

            :warning: `automodel` **must be defined** if `model_or_repo_id` is `str`, since the the model class
                cannot be known otherwise.

        config (PretrainedConfig): Custom configuration for the loaded model.
            If not specified, it will be instantiated with the default configuration for the model.

        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast | None): Custom tokenizer for the loaded model.
            If not specified, it will be instantiated with the default tokenizer for the model.

            :warning: If `model_or_repo_id` is a `transformers.PreTrainedModel` object, then `tokenizer` **must be defined**.

        batch_size (int): Batch size for the model.

        device_map (torch.device | str | None): Device map for the model. Directly passed to the model.

    Attributes:
        activation_granularities (ActivationGranularity):
            Enumeration of the available granularities for the `get_activations` method.

        aggregation_strategies (GranularityAggregationStrategy):
            Enumeration of the available aggregation strategies for the `get_activations` method.

        automodel (type[AutoModel]): The [AutoClass](https://huggingface.co/docs/transformers/en/model_doc/auto#natural-language-processing)
            corresponding to the loaded model type.

        batch_size (int): Batch size for the model.

        output_tuple_index (int | None): If the output at the split point is a tuple, this is the index of the hidden state.
            If `None`, an element with 3 dimensions is searched for.
            If not found, an error is raised.
            If several elements are found, an error is raised.

        repo_id (str): Either the model id in the HF Hub, or the path from which the model was loaded.

        tokenizer (PreTrainedTokenizer): Tokenizer for the loaded model, either given by the user or loaded from the repo_id.

        _model (transformers.PreTrainedModel): Huggingface transformers model wrapped by NNSight.

    Examples:
        Minimal example with gpt2:
        >>> from transformers import AutoModelForCausalLM
        >>> from interpreto import ModelWithSplitPoints
        >>> model_with_split_points = ModelWithSplitPoints(
        ...     "gpt2",
        ...     split_point=10,  # split at the 10th layer
        ...     automodel=AutoModelForCausalLM,
        ...     device_map="auto",
        ... )
        >>> activations, _ = model_with_split_points.get_activations(
        ...     inputs="interpreto is magic",
        ...     activation_granularity=ModelWithSplitPoints.activation_granularities.TOKEN,  # highly recommended for generation
        ... )

        Load the model from its repository id, split it at the first layer,
        and get the raw activations for the first layer.
        >>> from datasets import load_dataset
        >>> from interpreto import ModelWithSplitPoints
        >>> # load and split the model
        >>> model_with_split_points = ModelWithSplitPoints(
        ...     "bert-base-uncased",
        ...     split_point="bert.encoder.layer.1.output",
        ...     automodel=AutoModelForSequenceClassification,
        ...     batch_size=64,
        ...     device_map="cuda" if torch.cuda.is_available() else "cpu",
        ... )
        >>> # get activations
        >>> dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")["train"]["text"]
        >>> activations, _ = model_with_split_points.get_activations(
        ...     dataset,
        ...     activation_granularity=ModelWithSplitPoints.activation_granularities.CLS_TOKEN,  # highly recommended for classification
        ... )

        Load the model then pass it the `ModelWithSplitPoint`, split it at the first layer,
        get the word activations for the tenth layer, skip special tokens, and aggregate tokens activations by mean into words.
        >>> from transformers import AutoModelCausalLM, AutoTokenizer
        >>> from datasets import load_dataset
        >>> from interpreto import ModelWithSplitPoints as MWSP
        >>> # load the model
        >>> model = AutoModelCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        >>> # wrap and split the model at the 10th layer
        >>> model_with_split_points = MWSP(
        ...     model,
        ...     tokenizer=tokenizer,
        ...     split_point=10,  # split at the 10th layer
        ...     batch_size=16,
        ...     device_map="auto",
        ... )
        >>> # get activations at the word granularity
        >>> dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")["train"]["text"]
        >>> activations, _ = model_with_split_points.get_activations(
        ...     dataset,
        ...     activation_granularity=MWSP.activation_granularities.WORD,
        ...     aggregation_strategy=MWSP.aggregation_strategies.MEAN,  # average tokens activations by words
        ... )
    """

    # attributes to easily allow users to access the ENUMs
    activation_granularities = ActivationGranularity
    aggregation_strategies = GranularityAggregationStrategy

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        split_point: str | int | list[str] | list[int] | tuple[str, ...] | tuple[int, ...] | None = None,
        *args: tuple[Any],
        split_points: str | int | list[str] | list[int] | tuple[str, ...] | tuple[int, ...] | None = None,
        automodel: type[AutoModel] | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        config: PretrainedConfig | None = None,
        batch_size: int = 1,
        device_map: torch.device | str | None = None,
        **kwargs,
    ) -> None:
        # For parameters list, see class docstring. It was moved to change the order in the documentation.
        """Initialize a ModelWithSplitPoints object.

        Most of the work is forwarded to the `BaseSplitter` class initialization.
        Which is in turn a wrapper around the `nnsight.LanguageModel` class.

        Raises:
            InitializationError (ValueError): If the model cannot be loaded, because of a missing `tokenizer` or `automodel`.
            ValueError: If the `device_map` is set to 'auto' and the model is not a generation model.
            TypeError: If the `model_or_repo_id` is not a `str` or a `transformers.PreTrainedModel`.
        """
        # Handle deprecated `split_points` parameter
        if split_point is not None and split_points is not None:
            raise ValueError("Specify only one of `split_point` or deprecated `split_points`.")
        if split_points is not None:
            split_point = self._deprecated_split_points_to_split_point(split_points)
        elif isinstance(split_point, list | tuple):
            split_point = self._deprecated_split_points_to_split_point(split_point)
        if split_point is None:
            raise TypeError("Missing required argument `split_point`.")

        # Delegate to BaseSplitter (handles validation, loading, split point, device, tokenizer)
        super().__init__(
            model_or_repo_id,
            split_point,
            *args,
            automodel=automodel,
            tokenizer=tokenizer,
            config=config,
            batch_size=batch_size,
            device_map=device_map,
            **kwargs,
        )

    @staticmethod
    def _deprecated_split_points_to_split_point(
        split_points: str | int | list[str] | list[int] | tuple[str, ...] | tuple[int, ...],
    ) -> str | int:
        """Convert the deprecated `split_points` API to the singular `split_point` API."""
        if isinstance(split_points, list | tuple):
            if len(split_points) == 0:
                raise ValueError("At least one split point must be provided.")
            warnings.warn(
                "Multiple split points are deprecated. Only a single split point is supported. "
                f"Using the first element: '{split_points[0]}'. "
                "`split_points` will be removed in version 0.6.0. "
                "Please update your code to pass `split_point` instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return split_points[0]

        warnings.warn(
            "`split_points` is deprecated and will be removed in version 0.6.0. "
            "Please update your code to pass `split_point` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return split_points

    @property
    def split_points(self) -> list[str]:
        """Deprecated alias returning the split point as a single-element list."""
        warnings.warn(
            "`split_points` is deprecated and will be removed in version 0.6.0. Please use `split_point` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [self._split_point]

    @split_points.setter
    def split_points(
        self,
        split_points: str | int | list[str] | list[int] | tuple[str, ...] | tuple[int, ...],
    ) -> None:
        """Deprecated alias for setting `split_point`."""
        self.split_point = self._deprecated_split_points_to_split_point(split_points)

    def _get_granularity_indices(
        self,
        inputs: BatchEncoding | torch.Tensor,
        activation_granularity: ActivationGranularity,
    ) -> list[list[list[int]]]:
        """Get the indices of the granularity level, might be None.

        The indices correspond to how Granularity work in general in Interpreto.
        Called by the `get_activations` and `_get_concept_output_gradients` methods.
        They are used to select the activations through the `_apply_selection_strategy` method.
        But also to put back the activations through the `_reintegrate_selected_activations` method.

        Args:
            inputs (BatchEncoding | torch.Tensor): Inputs to the model forward pass before or after tokenization.
                In the case of a `torch.Tensor`, we assume a batch dimension and token ids.
            activation_granularity (ActivationGranularity): Selection strategy for activations.
                See `get_activations` for more details.

        Returns:
            list[list[list[int]]]: The indices of the granularity level.
                One sublist for each sample,
                for each sample: one subsublist for each granularity element,
                for each granularity element: list of indices of tokens composing the granularity element.
        """

        # Apply selection rule
        match activation_granularity:
            case AG.CLS_TOKEN:
                # get either the tensor or the input_ids tensor
                inputs_tensor: torch.Tensor = inputs if isinstance(inputs, torch.Tensor) else inputs["input_ids"]  # type: ignore
                n = inputs_tensor.shape[0]

                if inputs_tensor[0, 0] != self.tokenizer.cls_token_id:
                    raise ValueError(
                        "The first token of the input tensor is not the CLS token. "
                        "Please provide a tensor with the CLS token as the first token. "
                        "This may happen if you asking for a ``CLS_TOKEN`` granularity while not doing classification."
                    )

                # select the first token of each sample
                return [[[0]]] * n

            case AG.ALL_TOKENS:
                # get either the tensor or the input_ids tensor shape
                inputs_tensor: Float[torch.Tensor, "n l"] = (
                    inputs if isinstance(inputs, torch.Tensor) else inputs["input_ids"]
                )  # type: ignore  (weird type from huggingface `BatchEncoding`["input_ids"])
                n, l = inputs_tensor.shape

                # select all tokens of each sample
                return [[[i] for i in list(range(l))]] * n

            case AG.TOKEN | AG.WORD | AG.SENTENCE | AG.SAMPLE:
                if not isinstance(inputs, BatchEncoding):
                    raise ValueError(
                        "Cannot get indices without a tokenizer if granularity is TOKEN or SAMPLE. "
                        + "Please provide a tokenizer or set granularity to ALL_TOKENS."
                        + f"Got: {type(inputs)}"
                    )

                # for SAMPLE granularity, we select tokens activations before aggregating them
                if activation_granularity == AG.SAMPLE:
                    activation_granularity = AG.TOKEN

                # extract indices of activations to keep from inputs
                return activation_granularity.value.get_indices(
                    inputs=inputs,
                    tokenizer=self.tokenizer,
                )

            case _:
                raise ValueError(f"Invalid activation selection strategy: {activation_granularity}")

    @jaxtyped(typechecker=beartype)
    def _apply_selection_strategy(
        self,
        activations: Float[torch.Tensor, "n l d"],
        granularity_indices: list[list[list[int]]],
        activation_granularity: ActivationGranularity,
        aggregation_strategy: GranularityAggregationStrategy | None,
    ) -> list[Float[torch.Tensor, "g d"]]:
        """Apply selection strategy to activations.

        In theory, we could use the same code for most granularities thanks to the `granularity_indices` argument.
        However, we do special cases to go faster for some granularities.

        The way activations indices are treated is far from trivial. Here is an example:
        This indices are the same we defined in `Granularity`, lets take an example with the `WORD` granularity.

        >>> example:list[str] = [
        ...     "A BC DEF",
        ...     "abc de f"
        ... ]
        >>> indices = Granularity.WORD.get_indices(example, tokenizer)
        >>> indices
        [
             [ [0], [1, 2], [3, 4, 5] ],
             [ [0, 1, 2], [3, 4], [5] ],
        ]

        Here, the word `"abc"` belongs to the second sample;
        therefore, we need to look at the second element of the `indices` list.
        `"abc"` is the first word, thus the first granular element of this second sample.
        Therefore, `[0, 1, 2]`,
        which tells us that the word `"abc"` is formed with the first three tokens of the second sample.

        If we had to use this information to obtain the activations of the word `"abc"`,
        we would look at the activations of shape (n, l, d).
        Then extract the elements of interest by `activations[1, [0, 1, 2], :]`,
        the second sample, first three tokens, all of the model dimensions.
        The final step would be the aggregation over the token dimension.

        By applying this operation to all words, we would obtain six activation vectors, as we have six words.
        Words are kept by sample, here there are two samples, sol the list has two elements.
        Each element is a tensor of shape (g, d), where g is the number of granular elements in one input.
        In our case g is 3 for both samples, so the list has two elements of shape (3, d).

        Args:
            activations (InterventionProxy): Activations to apply selection strategy to.
            activation_granularity (ActivationGranularity): Selection strategy to apply. see :meth:`ModelWithSplitPoints.get_activations`.
            aggregation_strategy (GranularityAggregationStrategy | None): Aggregation strategy to apply. see :meth:`ModelWithSplitPoints.get_activations`.
            granularity_indices (list[list[list[int]]]): Indices of the granularity level, might be None.

        Returns:
            activation_list (list[torch.tensor]):
                List of activations, one element for each sample. (len(activation_list) == n)
                Each element of the list is a tensor of shape (g, d),
                where g depends on the granularity strategy and the length of the input.
        """
        if granularity_indices is None:
            if activation_granularity in [AG.TOKEN, AG.SAMPLE, AG.WORD, AG.SENTENCE]:
                raise ValueError(
                    "This should never happen as we apply `_get_granularity_indices` prior. "
                    "granularity_indices cannot be None when activation_granularity is TOKEN, SAMPLE, WORD or SENTENCE."
                )

        # Apply selection rule
        match activation_granularity:
            case AG.CLS_TOKEN:
                # select the first token of each sample
                return list(activations[:, 0, :].unsqueeze(1))

            case AG.ALL_TOKENS:
                # select all tokens of each sample
                return list(activations)

            case AG.TOKEN | AG.SAMPLE:
                if aggregation_strategy is None and activation_granularity == AG.SAMPLE:
                    raise ValueError("aggregation_strategy cannot be None when activation_granularity is SAMPLE.")

                # select activations based on indices
                activation_list: list[Float[torch.Tensor, "g d"]] = []

                # iterate over samples
                for i, indices in enumerate(granularity_indices):  # type: ignore
                    # flatten indices to a one dimensional tensor for faster indexing
                    indices_tensor = torch.tensor(indices).squeeze(1)
                    selected_activations = activations[i, indices_tensor]

                    # aggregate activations for SAMPLE strategy
                    if activation_granularity == AG.SAMPLE:
                        selected_activations = aggregation_strategy.aggregate(  # type: ignore
                            selected_activations,
                            dim=-2,
                        )

                    # add to the selected activations list
                    activation_list.append(selected_activations)

                return activation_list

            case AG.WORD | AG.SENTENCE:
                if aggregation_strategy is None:
                    raise ValueError(
                        "aggregation_strategy cannot be None when activation_granularity is WORD or SENTENCE."
                    )

                # select activations based on indices
                activation_list: list[Float[torch.Tensor, "g d"]] = []

                # iterate over samples
                for i, indices in enumerate(granularity_indices):  # type: ignore
                    sample_activations_list: list[Float[torch.Tensor, "1 d"]] = []
                    # iterate over activations
                    for index in indices:
                        # select activation for the current granularity element
                        granular_activations = activations[i, index]

                        # aggregate token activations over the granularity element
                        aggregated_activations = aggregation_strategy.aggregate(granular_activations, dim=-2)

                        sample_activations_list.append(aggregated_activations)

                    # cat activations for the current sample
                    sample_activations: Float[torch.Tensor, "g d"] = torch.cat(sample_activations_list, dim=0)
                    activation_list.append(sample_activations)

                return activation_list

            case _:
                raise ValueError(f"Invalid activation selection strategy: {activation_granularity}")

    @jaxtyped(typechecker=beartype)
    def _reintegrate_selected_activations(
        self,
        initial_activations: Float[torch.Tensor, "n l d"],
        new_activations: Float[torch.Tensor, "n l d"] | Float[torch.Tensor, "ng d"],
        activation_granularity: ActivationGranularity,
        aggregation_strategy: GranularityAggregationStrategy | None,
        granularity_indices: list[list[list[int]]],
    ) -> Float[torch.Tensor, "n l d"]:
        """
        Reintegrates the selected activations into the initial activations.

        It is the opposite of `_apply_selection_strategy`.

        It is not possible to reconstruct the latent activations from the granular activations alone.
        For example, the `TOKEN` granularity removes the special tokens, so the reconstructed activations
        cannot be the same as the initial activations.

        Therefore this function is used to reintegrate the reconstructed activations back into the initial activations.
        When activations were aggregated, they are unfolded (often copied) to match back the number of tokens.

        Args:
            initial_activations (Float[torch.Tensor, "n l d"]): The initial activations tensor.
            new_activations (Float[torch.Tensor, "n l d"] | Float[torch.Tensor, "ng d"]): The new activations tensor.
            granularity_indices (list[list[list[int]]]): The indices of the granularity level.
            activation_granularity (ActivationGranularity): The granularity level.
            aggregation_strategy (GranularityAggregationStrategy | None): The aggregation strategy to use.

        Returns:
            Float[torch.Tensor, "n l d"]: The reintegrated activations tensor.
        """
        new_activations = new_activations.to(device=initial_activations.device, dtype=initial_activations.dtype)

        match activation_granularity:
            case AG.CLS_TOKEN:
                # reintegrate the reconstructed CLS token activations into the initial activations
                initial_activations = initial_activations.clone()
                initial_activations[:, 0, :] = new_activations
                return initial_activations

            case AG.ALL_TOKENS:
                # reshape the reconstructed activations to match the initial activations shape
                return new_activations.view(initial_activations.shape)

            case AG.TOKEN:
                # iterate over samples
                current_index = 0
                for i, indices in enumerate(granularity_indices):
                    # flatten indices to a one dimensional tensor for faster indexing
                    indices_tensor = torch.tensor(indices).squeeze(1)

                    # reintegrate the reconstructed activations of non-special tokens into the initial activations
                    initial_activations[i, indices_tensor] = new_activations[
                        current_index : current_index + len(indices)
                    ]
                    current_index += len(indices)

                return initial_activations

            case AG.WORD | AG.SENTENCE:
                if aggregation_strategy is None:
                    raise ValueError(
                        "aggregation_strategy cannot be None when activation_granularity is WORD or SENTENCE."
                    )

                # iterate over samples
                current_index = 0
                for i, indices in enumerate(granularity_indices):
                    indices: list[list[int]]
                    # iterate over activations
                    for index in indices:
                        index: list[int]  # list of token indices for a given granularity element (word/sentence)
                        # extract the activations for the current word/sentence
                        aggregated_activations = new_activations[current_index : current_index + 1]

                        # repeat the activations to match the length of the word/sentence
                        unfolded_activations = aggregation_strategy.unfold(aggregated_activations, len(index))
                        torch_index = torch.tensor(index).to(initial_activations.device)

                        # reintegrate the repeated granular activations into the initial activations
                        initial_activations[i, torch_index] = unfolded_activations.to(initial_activations.device)
                        current_index += 1
                return initial_activations

            case AG.SAMPLE:
                raise ValueError(
                    "Activations aggregated at the sample level cannot be reintegrated. "
                    "Please choose another granularity level, such as ALL_TOKENS, TOKEN, WORD, or SENTENCE."
                )

            case _:
                raise ValueError(f"Invalid activation selection strategy: {activation_granularity}")

    def inputs_to_activations(
        self,
        inputs: list[str],
        *,
        activation_granularity: ActivationGranularity,
        aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN,
        flatten_activations: bool = True,
        forward_kwargs: dict[str, Any] = {},
    ) -> list[LatentActivations] | LatentActivations:
        """Extract activations from raw inputs.

        Args:
            inputs (list[str]): Raw text inputs.
            activation_granularity (ActivationGranularity): Selection strategy for activations.
            aggregation_strategy (GranularityAggregationStrategy): Aggregation strategy to use for activations.
            flatten_activations (bool): Whether to flatten the activations into a single tensor of shape (n*g, d).
            forward_kwargs (dict[str, Any]): Additional keyword arguments passed to the model forward pass.

        Returns:
            list[LatentActivations] | LatentActivations: Sample-wise list of activations or a single flattened tensor.
        """
        # tokenize text inputs for granularity selection
        # include "offsets_mapping" for sentence selection strategy
        tokenized = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )

        # get granularity indices
        granularity_indices: list[list[list[int]]] = self._get_granularity_indices(tokenized, activation_granularity)

        # extract offset mapping not supported by forward but was necessary for sentence selection strategy
        tokenized.pop("offset_mapping", None)

        sp_module = self.get(self._split_point)
        output_name = "nns_output" if hasattr(sp_module, "nns_output") else "output"

        # forward till the split point
        with self.trace(tokenized, **forward_kwargs) as tracer:
            outputs = getattr(sp_module, output_name).save()
            tracer.stop()

        # manage the output tuple and extract the (n, l, d) activations from it
        full_activations: Float[torch.Tensor, "n l d"] = self._manage_output_tuple(outputs, self._split_point)

        # select relevant activations with respect to the granularity strategy
        # potentially aggregate activations over the granularity elements
        # this merges the `n` and `g` dimensions with `g` a subset of `n`
        # shape (n, l, d) only for `ALL` granularity, thus raw activations
        granular_activations: list[Float[torch.Tensor, "g d"]] = self._apply_selection_strategy(
            activations=full_activations,
            granularity_indices=granularity_indices,
            activation_granularity=activation_granularity,
            aggregation_strategy=aggregation_strategy,
        )
        granular_activations = [
            act.detach().to(device="cpu", dtype=torch.float32, copy=True) for act in granular_activations
        ]

        if flatten_activations:
            return torch.cat(granular_activations, dim=0)

        return granular_activations

    def get_activations(  # noqa: PLR0912  # ignore too many branches  # too many special cases
        self,
        inputs: list[str] | torch.Tensor | BatchEncoding,
        *,
        activation_granularity: ActivationGranularity,
        aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN,
        pad_side: str | None = None,
        tqdm_bar: bool = False,
        include_predicted_classes: bool = False,
        flatten_activations: bool = True,
        forward_kwargs: dict[str, Any] = {},
    ) -> tuple[LatentActivations, torch.Tensor | None] | tuple[list[LatentActivations], list[torch.Tensor] | None]:
        """

        Get intermediate activations for the model split point on the given `inputs`.

        Optionally include the model predictions in the returned tuple.

        Args:
            inputs list[str] | torch.Tensor | BatchEncoding:
                Inputs to the model forward pass before or after tokenization.
                In the case of a `torch.Tensor`, we assume a batch dimension and token ids.

            activation_granularity (ActivationGranularity):
                Selection strategy for activations.
                In the model, activations have the shape `(n, l, d)`, where `d` is the model dimension.
                This parameters specifies which elements of these tensors are selected.
                If the granularity is larger then tokens, i.e. words and sentences, the activations are aggregated.
                The parameter `aggregation_strategy` specifies how the activations are aggregated.

                **It is highly recommended to use `CLS_TOKEN` for classification tasks and `TOKEN` for other tasks.**

                Available options are:

                - ``ModelWithSplitPoints.activation_granularities.ALL_TOKENS``:
                    the raw activations are flattened ``(n x l, d)``.
                    Hence, each token activation is now considered as a separate element.
                    This includes special tokens such as [CLS], [SEP], [EOS], [PAD], etc.

                - ``ModelWithSplitPoints.activation_granularities.CLS_TOKEN``:
                    for each sample, only the first token (e.g. ``[CLS]``) activation is returned ``(n, d)``.
                    This will raise an error if the model is not `ForSequenceClassification`.

                - ``ModelWithSplitPoints.activation_granularities.SAMPLE``:
                    special tokens are removed and the remaining ones are aggregated on the whole sample ``(n, d)``.

                - ``ModelWithSplitPoints.activation_granularities.SENTENCE``:
                    special tokens are removed and the remaining ones are aggregate by sentences.
                    Then the activations are flattened.
                    ``(n x g, d)`` where `g` is the number of sentences in the input.
                    The split is defined by `interpreto.commons.granularity.Granularity.SENTENCE`.

                - ``ModelWithSplitPoints.activation_granularities.TOKEN``:
                    the raw activations are flattened, but the special tokens are removed.
                    ``(n x g, d)`` where `g` is the number of non-special tokens in the input.
                    This is the default granularity.

                - ``ModelWithSplitPoints.activation_granularities.WORD``:
                    the special tokens are removed and the remaining ones are aggregate by words.
                    Then the activations are flattened.
                    ``(n x g, d)`` where `g` is the number of words in the input.
                    The split is defined by `interpreto.commons.granularity.Granularity.WORD`.

            aggregation_strategy (GranularityAggregationStrategy):
                Strategy to aggregate token activations into larger inputs granularities.
                Applied for `WORD`, `SENTENCE` and `SAMPLE` activation strategies.
                Token activations of shape  n * (l, d) are aggregated on the sequence length dimension.
                The concatenated into (ng, d) tensors.

                Existing strategies are:

                - ``ModelWithSplitPoints.aggregation_strategies.SUM``:
                    Tokens activations are summed along the sequence length dimension.

                - ``ModelWithSplitPoints.aggregation_strategies.MEAN``:
                    Tokens activations are averaged along the sequence length dimension.

                - ``ModelWithSplitPoints.aggregation_strategies.MAX``:
                    The maximum of the token activations along the sequence length dimension is selected.

                - ``ModelWithSplitPoints.aggregation_strategies.SIGNED_MAX``:
                    The maximum of the absolute value of the activations multiplied by its initial sign.
                    signed_max([[-1, 0, 1, 2], [-3, 1, -2, 0]]) = [-3, 1, -2, 2]

            pad_side (str | None):
                'left' or 'right' — side on which to apply padding along dim=1 only for ALL strategy.
                Forced right for classification models and left for causal LMs.

            tqdm_bar (bool):
                Whether to display a progress bar.

            include_predicted_classes (bool):
                Whether to include the predicted classes in the output tuple.
                Only applicable for classification models.

            flatten_activations (bool):
                Whether to flatten the activations tensors.

                - If True, the activations will be flattened from (n, l, d) to (n x l, d).
                    It allows storing the activations for the split point in a single tensor.

                - If False, a list of sample-wise activations will be returned.

            forward_kwargs (dict):
                Additional keyword arguments passed to the model forward pass.

        Returns:
            activations (LatentActivations | [list[LatentActivations]:
                The extracted activations either in a sample-wise list are flattened.
            predictions (torch.Tensor | list[torch.Tensor] | None):
                The predicted classes, if requested.
        """
        # set default pad side value and catch unsupported cases
        if self._model.__class__.__name__.endswith("ForSequenceClassification"):
            pad_side = "right"
        else:
            if self._model.__class__.__name__.endswith("ForCausalLM"):
                pad_side = "left"
            else:
                pad_side = pad_side or "left"
            if include_predicted_classes:
                raise ValueError(
                    "`include_predicted_classes` is only supported for classification models. "
                    f"Provided model is a {self._model.__class__.__name__}."
                )
        self.tokenizer.padding_side = pad_side

        # add padding token to vocabulary if not present (model and tokenizer)
        if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))  # type: ignore  # weird huggingface typing

        # batch inputs
        if isinstance(inputs, BatchEncoding):
            batch_generator = []
            # manage key by key batching for BatchEncoding
            for i in range(0, len(inputs), self.batch_size):
                end_idx = min(i + self.batch_size, len(inputs))
                batch_generator.append({key: value[i:end_idx] for key, value in inputs.items()})
        elif isinstance(inputs, list | torch.Tensor):
            # create a generator for iterable of inputs and tensors
            batch_generator = (
                inputs[i : min(i + self.batch_size, len(inputs))] for i in range(0, len(inputs), self.batch_size)
            )
        else:
            raise TypeError(
                f"Invalid inputs type: {type(inputs)}. Expected: list[str] | torch.Tensor | BatchEncoding."
            )

        # wrap generator in tqdm for progress bar
        tqdm_wrapped_batch_generator = tqdm(
            batch_generator,
            desc="Computing activations",
            unit="batch",
            total=ceil(len(inputs) / self.batch_size),
            disable=not tqdm_bar,
        )

        # initialize activation and prediction storage
        activations: list[LatentActivations] = []
        predictions: list[torch.Tensor] = []

        sp_module = self.get(self._split_point)

        # iterate over batch of inputs
        with torch.no_grad():
            # several call of the same model should be grouped in an nnsight session
            for batch_inputs in tqdm_wrapped_batch_generator:
                # ------------------------------------------------------------------------------
                # prepare inputs and compute granular indices
                if isinstance(batch_inputs, list):
                    # tokenize text inputs for granularity selection
                    # include "offsets_mapping" for sentence selection strategy
                    tokenized_inputs = self.tokenizer(
                        batch_inputs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        return_offsets_mapping=True,
                    )

                    # special case for T5 in a generation setting
                    if isinstance(self.args[0], T5ForConditionalGeneration):
                        # TODO: find a way for this not to be necessary
                        tokenized_inputs["decoder_input_ids"] = tokenized_inputs["input_ids"]
                else:
                    # the input was already tokenized
                    tokenized_inputs = batch_inputs

                # get granularity indices
                granularity_indices: list[list[list[int]]] = self._get_granularity_indices(
                    tokenized_inputs, activation_granularity
                )

                # extract offset mapping not supported by forward but was necessary for sentence selection strategy
                if isinstance(tokenized_inputs, (BatchEncoding, dict)):  # noqa: UP038
                    tokenized_inputs.pop("offset_mapping", None)

                # ------------------------------------------------------------------------------
                # model forward pass with nnsight to extract activations and predictions

                # all model calls use trace with nnsight
                # call model forward pass and save split point outputs
                with self.trace(tokenized_inputs, **forward_kwargs) as tracer:
                    output_name = "nns_output" if hasattr(sp_module, "nns_output") else "output"
                    batch_outputs = getattr(sp_module, output_name).save()

                    # for classification optionally compute and save the predictions
                    if include_predicted_classes:
                        batch_predictions: Float[torch.Tensor, "n"] = (
                            self.output.logits.argmax(dim=-1).cpu().save()  # type: ignore  (under specification from NNsight)
                        )
                    else:
                        tracer.stop()

                # free memory after each batch, necessary with nnsight, overwise, memory piles up
                torch.cuda.empty_cache()

                # ------------------------------------------------------------------------------
                # apply granularity selection and aggregation of activations and predictions
                # manage the output tuple and extract the (n, l, d) activations from it
                batch_sp_activations: Float[torch.Tensor, "n l d"] = self._manage_output_tuple(
                    batch_outputs, self._split_point
                )

                # select relevant activations with respect to the granularity strategy
                # potentially aggregate activations over the granularity elements
                # this merges the `n` and `g` dimensions with `g` a subset of `n`
                # shape (n, l, d) only for `ALL` granularity, thus raw activations
                granular_activations: list[Float[torch.Tensor, "g d"]] = self._apply_selection_strategy(
                    activations=batch_sp_activations,
                    granularity_indices=granularity_indices,
                    activation_granularity=activation_granularity,
                    aggregation_strategy=aggregation_strategy,
                )

                activations.extend(
                    act.detach().to(device="cpu", dtype=torch.float32, copy=True) for act in granular_activations
                )

                if include_predicted_classes:
                    if not flatten_activations:
                        predictions.extend(
                            list(batch_predictions)  # type: ignore  (ignore possibly unbound)
                        )
                    else:
                        # adapt predictions to match the granularity indices
                        repeats: Float[torch.Tensor, "ng"] = torch.tensor(
                            [len(indices) for indices in granularity_indices]
                        )

                        # predictions have a shape (n,), which we convert to (ng,)
                        # by repeating each predicted class as many times as the number of granularity elements in a sample
                        repeated_predictions = torch.repeat_interleave(
                            batch_predictions,  # type: ignore  (ignore possibly unbound)
                            repeats,
                            dim=0,
                        )
                        predictions.append(repeated_predictions)

        # ------------------------------------------------------------------------------------------
        # concat activation batches and validate that activations have the expected type
        if flatten_activations:
            # two dimensional tensor (n*g, d)
            flattened_activations = torch.cat(activations, dim=0)

            if include_predicted_classes:
                return flattened_activations, torch.cat(predictions, dim=0)
            return flattened_activations, None

        # validate that activations have the expected type
        if not all(isinstance(act, torch.Tensor) for act in activations):
            raise RuntimeError("Invalid output. Expected a list of torch.Tensor activations.")

        if include_predicted_classes:
            return activations, predictions
        return activations, None

    @jaxtyped(typechecker=beartype)
    def _get_concept_output_gradients(  # noqa: PLR0912  # ignore too many branches
        self,
        inputs: list[str] | torch.Tensor | BatchEncoding,
        activations_to_concepts: Callable[[LatentActivations], ConceptsActivations],
        concepts_to_activations: Callable[[ConceptsActivations], LatentActivations],
        targets: list[int] | None = None,
        activation_granularity: ActivationGranularity = AG.TOKEN,
        aggregation_strategy: GranularityAggregationStrategy | None = GranularityAggregationStrategy.MEAN,
        concepts_x_gradients: bool = True,
        tqdm_bar: bool = False,
        batch_size: int | None = None,
        forward_kwargs: dict[str, Any] = {},
    ) -> list[Float[torch.Tensor, "t g c"]]:
        """Get intermediate activations for all model split points

        :warning: This method should not be called directly. The concept explainer should be used instead.

        Args:
            inputs list[str] | torch.Tensor | BatchEncoding:
                Inputs to the model forward pass before or after tokenization.
                In the case of a `torch.Tensor`, we assume a batch dimension and token ids.

            activation_granularity (ActivationGranularity):
                Selection strategy for activations. Options are:

                - ``ModelWithSplitPoints.activation_granularities.ALL_TOKENS``:
                    the raw activations are flattened ``(n x l, d)``.
                    Hence, each token activation is now considered as a separate element.
                    This includes special tokens such as [CLS], [SEP], [EOS], [PAD], etc.

                - ``ModelWithSplitPoints.activation_granularities.CLS_TOKEN``:
                    for each sample, only the first token (e.g. ``[CLS]``) activation is returned ``(n, d)``.
                    This will raise an error if the model is not `ForSequenceClassification`.

                - ``ModelWithSplitPoints.activation_granularities.SENTENCE``:
                    special tokens are removed and the remaining ones are aggregate by sentences.
                    Then the activations are flattened.
                    ``(n x g, d)`` where `g` is the number of sentences in the input.
                    The split is defined by `interpreto.commons.granularity.Granularity.SENTENCE`.

                - ``ModelWithSplitPoints.activation_granularities.TOKEN``:
                    the raw activations are flattened, but the special tokens are removed.
                    ``(n x g, d)`` where `g` is the number of non-special tokens in the input.
                    This is the default granularity.

                - ``ModelWithSplitPoints.activation_granularities.WORD``:
                    the special tokens are removed and the remaining ones are aggregate by words.
                    Then the activations are flattened.
                    ``(n x g, d)`` where `g` is the number of words in the input.
                    The split is defined by `interpreto.commons.granularity.Granularity.WORD`.

            aggregation_strategy:
                Strategy to aggregate token activations into larger inputs granularities.
                Applied for `WORD` and `SENTENCE` activation strategies.
                Token activations of shape  n * (l, d) are aggregated on the sequence length dimension.
                The concatenated into (ng, d) tensors.
                Existing strategies are:

                - ``ModelWithSplitPoints.aggregation_strategies.SUM``:
                    Tokens activations are summed along the sequence length dimension.

                - ``ModelWithSplitPoints.aggregation_strategies.MEAN``:
                    Tokens activations are averaged along the sequence length dimension.

                - ``ModelWithSplitPoints.aggregation_strategies.MAX``:
                    The maximum of the token activations along the sequence length dimension is selected.

                - ``ModelWithSplitPoints.aggregation_strategies.SIGNED_MAX``:
                    The maximum of the absolute value of the activations multiplied by its initial sign.
                    signed_max([[-1, 0, 1, 2], [-3, 1, -2, 0]]) = [-3, 1, -2, 2]

            tqdm_bar (bool):
                Whether to display a progress bar.

            forward_kwargs (dict):
                Additional keyword arguments passed to the model forward pass.

        Returns:
            gradients (list[torch.Tensor]): The gradients of the model output with respect to the concept activations.
            List length: correspond to the number of inputs.
                Tensor shape: (t, g, c) with t the target dimension, g the number of granularity elements in one input, and c the number of
                concepts.
        """
        # sanity check
        if activation_granularity is AG.SAMPLE:
            raise ValueError(
                "The activation granularity cannot be SAMPLE to compute the concept output gradients. "
                "Please choose another granularity strategy among: ALL_TOKENS, CLS_TOKEN, TOKEN, WORD, SENTENCE. "
            )

        # add padding token to vocabulary if not present (model and tokenizer)
        if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self._model.resize_token_embeddings(len(self.tokenizer))  # type: ignore  (weird huggingface typing)

        # the `targets` parameter need to be loaded in self for nnsight to allow its access inside the trace context
        self.targets = targets

        # batch inputs
        grad_batch_size = batch_size or self.batch_size
        if isinstance(inputs, BatchEncoding):
            batch_generator = []
            # manage key by key batching for BatchEncoding
            for i in range(0, len(inputs), grad_batch_size):
                end_idx = min(i + grad_batch_size, len(inputs))
                batch_generator.append({key: value[i:end_idx] for key, value in inputs.items()})
        else:  # sequence of inputs or tensors
            # create a generator for iterable of inputs and tensors
            batch_generator = (
                inputs[i : min(i + grad_batch_size, len(inputs))] for i in range(0, len(inputs), grad_batch_size)
            )

        # wrap generator in tqdm for progress bar
        tqdm_wrapped_batch_generator = tqdm(
            batch_generator,
            desc="Computing gradients",
            unit="batches",
            total=ceil(len(inputs) / grad_batch_size),
            disable=not tqdm_bar,
        )

        gradients_list: list[Float[torch.Tensor, "ng c"]] = []
        # iterate over batch of inputs
        for batch_inputs in tqdm_wrapped_batch_generator:
            # --------------------------------------------------------------------------------------
            # prepare inputs and compute granular indices
            # tokenize text inputs
            if isinstance(batch_inputs, list):
                if activation_granularity == AG.CLS_TOKEN:
                    self.tokenizer.padding_side = "right"
                tokenized_inputs = self.tokenizer(
                    batch_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    return_offsets_mapping=True,
                )
                if isinstance(self.args[0], T5ForConditionalGeneration):
                    # TODO: find a way for this not to be necessary
                    tokenized_inputs["decoder_input_ids"] = tokenized_inputs["input_ids"]
            else:
                tokenized_inputs = batch_inputs

            granularity_indices: list[list[list[int]]] = self._get_granularity_indices(  # type: ignore  (cannot be None with given activation granularity)
                tokenized_inputs, activation_granularity
            )

            # extract offset mapping not supported by forward but necessary for word/sentence selection strategy
            if isinstance(tokenized_inputs, (BatchEncoding, dict)):  # noqa: UP038
                tokenized_inputs.pop("offset_mapping", None)

            # TODO: test if we can use `with model.edit():` from nnsight
            # in theory, it would be much faster

            # --------------------------------------------------------------------------------------
            # model forward pass with nnsight to compute concepts activations and predictions
            # then backward from the predictions to the concepts activations (gradients)

            # all model calls use trace with nnsight
            with self.trace(tokenized_inputs, **forward_kwargs):
                curr_module = self.get(self._split_point)
                # Handle case in which module has .output attribute, and .nns_output gets overridden instead
                module_out_name = "nns_output" if hasattr(curr_module, "nns_output") else "output"

                # get activations
                layer_outputs = getattr(curr_module, module_out_name)
                raw_activations: Float[torch.Tensor, "n l d"] = self._manage_output_tuple(
                    layer_outputs, self._split_point
                )
                n, l, d = raw_activations.shape  # number of samples, sequence length, and model dimension
                ng = sum([len(indices) for indices in granularity_indices])  # number of granularity elements

                # apply selection strategy
                selected_activations: list[Float[torch.Tensor, "g {d}"]]
                selected_activations = self._apply_selection_strategy(
                    activations=raw_activations,  # use the last batch of activations
                    granularity_indices=granularity_indices,
                    activation_granularity=activation_granularity,
                    aggregation_strategy=aggregation_strategy,
                )
                # concatenate the selected activations into a single tensor
                flattened_activations: Float[torch.Tensor, ng, d] = torch.cat(selected_activations, dim=0)

                # encode activations into concepts
                concept_activations: Float[torch.Tensor, "{ng} c"] = activations_to_concepts(
                    flattened_activations.to(dtype=torch.float32)
                )
                del selected_activations, flattened_activations
                c = concept_activations.shape[-1]

                # decode concepts back into activations
                decoded_activations: Float[torch.Tensor, ng, d] = concepts_to_activations(concept_activations)

                # reintegrate decoded activations into the original activations
                reconstructed_activations: Float[torch.Tensor, n, l, d] = self._reintegrate_selected_activations(
                    initial_activations=raw_activations,
                    new_activations=decoded_activations,
                    granularity_indices=granularity_indices,
                    activation_granularity=activation_granularity,
                    aggregation_strategy=aggregation_strategy,
                )
                del decoded_activations, raw_activations

                # reintegrate the reconstructed activations into the original layer outputs
                if isinstance(layer_outputs, tuple):
                    layer_outputs = list(layer_outputs)
                    layer_outputs[self.output_tuple_index] = reconstructed_activations  # type: ignore
                else:
                    layer_outputs = reconstructed_activations

                # assign the new outputs to the module output
                if hasattr(curr_module, "nns_output"):
                    curr_module.nns_output = layer_outputs  # type: ignore  (under specification from NNsight)
                else:
                    curr_module.output = layer_outputs  # type: ignore  (under specification from NNsight)

                # ----------------------------------------------------------------------------------
                # Manipulate logits and targets to prepare gradients computation
                # get logits
                logits: Float[torch.Tensor, "{n} t_all"]  # number of samples and number of possible targets
                all_logits = self.output.logits

                if len(all_logits.shape) == 3:  # generation (n, l, v)
                    # in the case of a generation model, take the maximum logits over the vocabulary dimension
                    logits, _ = all_logits.max(dim=-1)  # (n, l)
                else:  # classification (n, nb_classes)
                    logits = all_logits

                # sum over samples to batch gradients calls (it has no impact on the final gradients)
                logits: Float[torch.Tensor, "t_all"] = logits.sum(dim=0)

                # compute gradients for each target
                if self.targets is None:
                    current_targets: Iterable[int] = range(logits.shape[0])
                else:
                    current_targets: Iterable[int] = self.targets

                t = len(current_targets)  # number of targets

                # TODO: find a way to compute gradients for all targets simultaneously

                # ----------------------------------------------------------------------------------
                # compute gradients for each target separately
                targets_gradients_list: list[Float[torch.Tensor, ng, c]] = []
                for t in current_targets:
                    # sum over samples but compute the gradients for each target separately
                    with logits[t].backward(retain_graph=True):  # type: ignore
                        # compute the gradient of the concept activations
                        concept_activations_grad: Float[torch.Tensor, ng, c] = concept_activations.grad.clone()  # type: ignore

                        # clean gradient for following operations
                        concept_activations.grad.zero_()  # type: ignore

                        # for gradient x concepts, multiply by concepts
                        if concepts_x_gradients:
                            concept_activations_grad *= concept_activations
                    targets_gradients_list.append(concept_activations_grad)

                targets_gradients: Float[torch.Tensor, t, ng, d] = (
                    torch.stack(targets_gradients_list, dim=0).detach().cpu().save()  # type: ignore  (nnsight under specification)
                )
                del (
                    targets_gradients_list,
                    concept_activations,
                    concept_activations_grad,  # type: ignore (possibly unbound grad),
                    logits,
                    all_logits,
                )

                # split gradients for each input sentence from (t, ng, d) to n * (t, g, d)
                start = 0
                for indices_list in granularity_indices:
                    end = start + len(indices_list)
                    gradients_list.append(targets_gradients[:, start:end, :])
                    start = end

                gc.collect()

            # free memory after each batch, necessary with nnsight, overwise, memory piles up
            torch.cuda.empty_cache()

        return gradients_list

    def get_latent_shape(self) -> torch.Size:
        """Get the shape of the latent activations at the split point.

        Use the `scan` operation from NNsight to get the shape of the activations.
        It basically builds the computation graph, but it is much quicker than a forward.

        Returns:
            torch.Size: Shape of the activations for the split point.
        """
        shape = None
        with self.scan("scan"):
            curr_module = self.get(self._split_point)
            module_out_name = "nns_output" if hasattr(curr_module, "nns_output") else "output"
            module = getattr(curr_module, module_out_name)
            if isinstance(module, tuple):
                for candidate in module:
                    if candidate.dim() == 3:
                        module = candidate
                        break
            shape = nnsight.save(module.shape)  # type: ignore  (under specification from NNsight)
        return shape
