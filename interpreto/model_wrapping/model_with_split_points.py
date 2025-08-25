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
from collections.abc import Callable, Iterable
from enum import Enum
from math import ceil
from typing import Any

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from nnsight.modeling.language import LanguageModel
from tqdm import tqdm
from transformers import AutoModel, T5ForConditionalGeneration
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from interpreto.commons.granularity import Granularity, GranularityAggregationStrategy
from interpreto.model_wrapping.splitting_utils import get_layer_by_idx, sort_paths, validate_path, walk_modules
from interpreto.typing import ConceptsActivations, LatentActivations


class InitializationError(ValueError):
    """Raised to signal a problem with model initialization."""


class ActivationGranularity(Enum):
    """Activation selection strategies for `ModelWithSplitPoints.get_activations()`."""

    ALL = "all"
    CLS_TOKEN = "cls_token"
    ALL_TOKENS = Granularity.ALL_TOKENS
    TOKEN = Granularity.TOKEN
    WORD = Granularity.WORD
    SENTENCE = Granularity.SENTENCE
    SAMPLE = "sample"


AG = ActivationGranularity


class ModelWithSplitPoints(LanguageModel):
    """Code: [:octicons-mark-github-24: model_wrapping/model_with_split_points.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/model_wrapping/model_with_split_points.py)

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

        split_points (str | Sequence[str] | int | Sequence[int]): One or more to split locations inside the model.
            Either one of the following:

            * A `str` corresponding to the path of a split point inside the model.
            * An `int` corresponding to the n-th layer.
            * A `Sequence[str]` or `Sequence[int]` corresponding to multiple split points.

            Example: `split_points='cls.predictions.transform.LayerNorm'` correspond to a split
            after the LayerNorm layer in the MLM head (assuming a `BertForMaskedLM` model in input).

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

        output_tuple_index (int | None): If the output at the split point is a tuple, this is the index of the hidden state.
            If `None`, an element with 3 dimensions is searched for.
            If not found, an error is raised.
            If several elements are found, an error is raised.

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

        split_points (list[str]): Getter/setters for model paths corresponding to split points inside the loaded model.
            Automatically handle validation, sorting and resolving int paths to strings.

        tokenizer (PreTrainedTokenizer): Tokenizer for the loaded model, either given by the user or loaded from the repo_id.

        _model (transformers.PreTrainedModel): Huggingface transformers model wrapped by NNSight.

    Examples:
        Minimal example with gpt2:
        >>> from transformers import AutoModelForCausalLM
        >>> from interpreto import ModelWithSplitPoints
        >>> model_with_split_points = ModelWithSplitPoints(
        ...     "gpt2",
        ...     split_points=10,  # split at the 10th layer
        ...     automodel=AutoModelForCausalLM,
        ...     device_map="auto",
        ... )
        >>> activations_dict = model_with_split_points.get_activations(
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
        ...     split_points="bert.encoder.layer.1.output",
        ...     automodel=AutoModelForSequenceClassification,
        ...     batch_size=64,
        ...     device_map="cuda" if torch.cuda.is_available() else "cpu",
        ... )
        >>> # get activations
        >>> dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")["train"]["text"]
        >>> activations_dict = model_with_split_points.get_activations(
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
        ...     split_points=10,  # split at the 10th layer
        ...     batch_size=16,
        ...     device_map="auto",
        ... )
        >>> # get activations at the word granularity
        >>> dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")["train"]["text"]
        >>> activations = model_with_split_points.get_activations(
        ...     dataset,
        ...     activation_granularity=MWSP.activation_granularities.WORD,
        ...     aggregation_strategy=MWSP.aggregation_strategies.MEAN,  # average tokens activations by words
        ... )
    """

    _example_input = "hello"  # placeholder input for the nnsight `scan` method
    # attributes to easily allow users to access the ENUMs
    activation_granularities = ActivationGranularity
    aggregation_strategies = GranularityAggregationStrategy

    def __init__(
        self,
        model_or_repo_id: str | PreTrainedModel,
        split_points: str | int | list[str] | list[int] | tuple[str] | tuple[int],
        *args: tuple[Any],
        automodel: type[AutoModel] | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        config: PretrainedConfig | None = None,
        batch_size: int = 1,
        device_map: torch.device | str | None = None,
        output_tuple_index: int | None = None,
        **kwargs,
    ) -> None:
        # For parameters list, see class docstring. It was moved to change the order in the documentation.
        """Initialize a ModelWithSplitPoints object.

        Most of the work is forwarded to the `LanguageModel` class initialization from NNsight.

        Raises:
            InitializationError (ValueError): If the model cannot be loaded, because of a missing `tokenizer` or `automodel`.
            ValueError: If the `device_map` is set to 'auto' and the model is not a generation model.
            TypeError: If the `model_or_repo_id` is not a `str` or a `transformers.PreTrainedModel`.
        """
        if isinstance(model_or_repo_id, PreTrainedModel):
            if tokenizer is None:
                raise InitializationError(
                    "Tokenizer is not set. When providing a model instance, the tokenizer must be set."
                )
        elif isinstance(model_or_repo_id, str):  # Repository ID
            if automodel is None:
                raise InitializationError(
                    "Model autoclass not found.\n"
                    "The model class can be omitted if a pre-loaded model is passed to `model_or_repo_id` "
                    "param.\nIf an HF Hub ID is used, the corresponding autoclass must be specified in `automodel`.\n"
                    "Example: ModelWithSplitPoints('bert-base-uncased', automodel=AutoModelForMaskedLM, ...)"
                )
        else:
            raise TypeError(
                f"Invalid model_or_repo_id type: {type(model_or_repo_id)}. "
                "Expected `str` or `transformers.PreTrainedModel`."
            )

        # Handles model loading through nnsight.LanguageModel._load
        super().__init__(
            model_or_repo_id,
            *args,
            config=config,
            tokenizer=tokenizer,  # type: ignore (under specification from NNsight)
            automodel=automodel,  # type: ignore (under specification from NNsight)
            device_map=device_map,
            **kwargs,
        )

        # set split points
        self._model_paths = list(walk_modules(self._model))
        self.split_points = split_points  # this uses the setter which handles validation
        self._model: PreTrainedModel  # specify type of `_model` attribute from NNsight
        if self.repo_id is None:
            self.repo_id = self._model.config.name_or_path
        self.batch_size = batch_size

        if not isinstance(model_or_repo_id, str):
            if device_map is not None:
                if device_map == "auto":
                    raise ValueError(
                        "'auto' device_map is only supported when loading a generation model from a repository id. "
                        "Please specify a device_map, e.g. 'cuda' or 'cpu'."
                    )
                self.to(device_map)  # type: ignore  (under specification from NNsight)

        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set. When providing a model instance, the tokenizer must be set.")
        self.output_tuple_index = output_tuple_index

    @property
    def split_points(self) -> list[str]:
        return self._split_points

    @split_points.setter
    def split_points(self, split_points: str | int | list[str] | list[int] | tuple[str] | tuple[int]) -> None:
        """Split points are automatically validated and sorted upon setting"""
        # sanitize split points to a list of strings and ints
        pre_conversion_split_points = split_points if isinstance(split_points, list | tuple) else [split_points]

        # convert layer idx to full path
        post_conversion_split_points: list[str] = []
        for split in pre_conversion_split_points:
            # Handle conversion of layer idx to full path
            if isinstance(split, int):
                str_split = get_layer_by_idx(split, model_paths=self._model_paths)
            else:
                str_split = split
            post_conversion_split_points.append(str_split)

            # Validate whether the split exists in the model
            validate_path(self._model, str_split)

        # Sort split points to match execution order
        self._split_points: list[str] = sort_paths(post_conversion_split_points, model_paths=self._model_paths)

    @staticmethod
    def _pad_and_concat(
        tensor_list: list[Float[torch.Tensor, "n_i l_i d"]], pad_side: str, pad_value: float
    ) -> Float[torch.Tensor, "sum(n_i) max_l d"]:
        """
        Concatenates a list of 3D tensors along dim=0 after padding their second dimension to the same length.

        Args:
            tensor_list (List[Tensor]): List of tensors with shape (n_i, l_i, d)
            pad_side (str): 'left' or 'right' — side on which to apply padding along dim=1
            pad_value (float): Value to use for padding

        Returns:
            Tensor: Tensor of shape (sum(n_i), max_l, d)
        """
        if pad_side not in ("left", "right"):
            raise ValueError("pad_side must be either 'left' or 'right'")

        max_l = max(t.shape[1] for t in tensor_list)
        padded = []

        for t in tensor_list:
            n, l, d = t.shape
            pad_len = max_l - l

            if pad_len == 0:
                padded_tensor = t
            else:
                if pad_side == "right":
                    pad = (0, 0, 0, pad_len)  # pad dim=1 on the right
                else:  # pad_side == 'left'
                    pad = (0, 0, pad_len, 0)  # pad dim=1 on the left
                padded_tensor = F.pad(t, pad, value=pad_value)

            padded.append(padded_tensor)

        return torch.cat(padded, dim=0)

    def _get_granularity_indices(
        self,
        inputs: BatchEncoding | torch.Tensor,
        activation_granularity: ActivationGranularity,
    ) -> list[list[list[int]]] | None:
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
            list[list[list[int]]] | None: The indices of the granularity level.
                One sublist for each sample,
                for each sample: one subsublist for each granularity element,
                for each granularity element: list of indices of tokens composing the granularity element.
        """

        # Apply selection rule
        match activation_granularity:
            case AG.ALL:
                return None

            case AG.CLS_TOKEN:
                # get either the tensor or the input_ids tensor
                inputs_tensor: torch.Tensor = inputs if isinstance(inputs, torch.Tensor) else inputs["input_ids"]  # type: ignore
                n = inputs_tensor.shape[0]

                if inputs_tensor[0, 0] != self.tokenizer.cls_token_id:
                    raise ValueError(
                        "The first token of the input tensor is not the CLS token. "
                        "Please provide a tensor with the CLS token as the first token."
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
                        "Cannot get indices without a tokenizer if granularity is TOKEN or SAMPLE."
                        + "Please provide a tokenizer or set granularity to ALL_TOKENS."
                        + f"Got: {type(inputs)}"
                    )

                # for SAMPLE granularity, we select tokens activations before aggregating them
                if activation_granularity == AG.SAMPLE:
                    activation_granularity = AG.TOKEN

                # extract indices of activations to keep from inputs
                return Granularity.get_indices(
                    inputs=inputs,
                    granularity=activation_granularity.value,  # type: ignore
                    tokenizer=self.tokenizer,
                )

            case _:
                raise ValueError(f"Invalid activation selection strategy: {activation_granularity}")

    @jaxtyped(typechecker=beartype)
    def _apply_selection_strategy(
        self,
        activations: Float[torch.Tensor, "n l d"],
        granularity_indices: list[list[list[int]]] | None,
        activation_granularity: ActivationGranularity,
        aggregation_strategy: GranularityAggregationStrategy | None,
    ) -> Float[torch.Tensor, "n l d"] | Float[torch.Tensor, "ng d"]:
        """Apply selection strategy to activations.

        In theory, we could use the same code for most granularities thanks to the `granularity_indices` argument.
        However, we do special cases to go faster for some granularities.

        The way activations indices are treated is far from trivial. Here is an example:
        This indices are the same we defined in `Granularity`, lets take an example with the `WORD` granularity.

        >>> example:list[str] = [
        ...     "A BC DEF",
        ...     "abc de f"
        ... ]
        >>> indices = Granularity.get_indices(example, Granularity.WORD, tokenizer)
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
        Then extract the elements of interest by `activations[1, [0, 1, 2],:]`,
        the second sample, first three tokens, all of the model dimensions.
        The final step would be the aggregation over the token dimension.

        By applying this operation to all words, we would obtain six activation vectors, as we have six words.
        These are stacked to build the granular activations of shape `(ng, d)`
        where `ng` is the total number of granular elements over the n provided samples.

        Args:
            activations (InterventionProxy): Activations to apply selection strategy to.
            activation_granularity (ActivationGranularity): Selection strategy to apply. see :meth:`ModelWithSplitPoints.get_activations`.
            aggregation_strategy (GranularityAggregationStrategy | None): Aggregation strategy to apply. see :meth:`ModelWithSplitPoints.get_activations`.
            granularity_indices (list[list[list[int]]] | None): Indices of the granularity level, might be None.

        Returns:
            torch.Tensor: The aggregated activations. (n, l, d) or (ng, d)
        """
        if granularity_indices is None:
            if activation_granularity in [AG.TOKEN, AG.SAMPLE, AG.WORD, AG.SENTENCE]:
                raise ValueError(
                    "This should never happen as we apply `_get_granularity_indices` prior. "
                    "granularity_indices cannot be None when activation_granularity is TOKEN, SAMPLE, WORD or SENTENCE."
                )

        # Apply selection rule
        match activation_granularity:
            case AG.ALL:
                # raw activations
                return activations

            case AG.CLS_TOKEN:
                # select the first token of each sample
                return activations[:, 0, :]

            case AG.ALL_TOKENS:
                # select all tokens of each sample
                return activations.flatten(0, 1)

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
                        selected_activations = GranularityAggregationStrategy.aggregate(
                            selected_activations,
                            strategy=aggregation_strategy,  # type: ignore
                            dim=-2,
                        )

                    # add to the selected activations list
                    activation_list.append(selected_activations)

                # concat all activations
                flatten_activations: Float[torch.Tensor, "ng d"] = torch.concat(activation_list, dim=0)
                return flatten_activations

            case AG.WORD | AG.SENTENCE:
                if aggregation_strategy is None:
                    raise ValueError(
                        "aggregation_strategy cannot be None when activation_granularity is WORD or SENTENCE."
                    )

                # select activations based on indices
                activation_list: list[Float[torch.Tensor, "g d"]] = []

                # iterate over samples
                for i, indices in enumerate(granularity_indices):  # type: ignore
                    # iterate over activations
                    for index in indices:
                        # select activation for the current granularity element
                        granular_activations = activations[i, index]

                        # aggregate token activations over the granularity element
                        aggregated_activations = GranularityAggregationStrategy.aggregate(
                            granular_activations, strategy=aggregation_strategy, dim=-2
                        )

                        activation_list.append(aggregated_activations)

                # concat all activations
                flatten_activations: Float[torch.Tensor, "ng d"] = torch.concat(activation_list, dim=0)
                return flatten_activations

            case _:
                raise ValueError(f"Invalid activation selection strategy: {activation_granularity}")

    @jaxtyped(typechecker=beartype)
    def _reintegrate_selected_activations(
        self,
        initial_activations: Float[torch.Tensor, "n l d"],
        new_activations: Float[torch.Tensor, "n l d"] | Float[torch.Tensor, "ng d"],
        activation_granularity: ActivationGranularity,
        aggregation_strategy: GranularityAggregationStrategy | None,
        granularity_indices: list[list[list[int]]] | None = None,
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
        match activation_granularity:
            case AG.ALL:
                # raw activations
                return new_activations

            case AG.CLS_TOKEN:
                # reintegrate the reconstructed CLS token activations into the initial activations
                initial_activations[:, 0, :] = new_activations
                return initial_activations

            case AG.ALL_TOKENS:
                # reshape the reconstructed activations to match the initial activations shape
                return new_activations.view(initial_activations.shape)

            case AG.TOKEN:
                if granularity_indices is None:
                    raise ValueError("granularity_indices cannot be None when activation_granularity is TOKEN.")

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
                if granularity_indices is None:
                    raise ValueError(
                        "granularity_indices cannot be None when activation_granularity is WORD or SENTENCE."
                    )
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
                        unfolded_activations = GranularityAggregationStrategy.unfold(
                            aggregated_activations, aggregation_strategy, len(index)
                        )
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

    def _manage_output_tuple(self, activations: torch.Tensor | tuple[torch.Tensor], split_point: str) -> torch.Tensor:
        """
        Handles the case in which the model has a tuple of outputs,
        and we need to know which element is the hidden state.

        The hypothesis is that the hidden state has three dimensions (n, l, d).
        Therefore, in the case of a tuple of tensors,
        this function returns the tensor with three dimensions.

        Args:
            activations (torch.Tensor | tuple[torch.Tensor]): The activations to manage.
            split_point (str): The split point for interpretable error messages.

        Returns:
            torch.Tensor: The managed activations.
            int | None: The index of the hidden state in the tuple.

        Raises:
            TypeError: If the activations are not a `torch.tensor` or a valid tuple.
            RuntimeError: If the activations are a tuple, but we were not able to determine which element is the hidden state.
        """
        if isinstance(activations, torch.Tensor):
            if activations.dim() != 3:
                raise ValueError(
                    f"Invalid activations for split point '{split_point}'. "
                    f"Expected a 3D tensor of shape (n, l, d), "
                    f"got a tensor of shape {activations.shape}. "
                    "It is recommended to look for another split point."
                )
            return activations

        if not isinstance(activations, tuple):
            raise TypeError(
                f"Failed to manipulate activations for split point '{split_point}'. "
                f"Wrong type of activations. Expected torch.Tensor or tuple[torch.Tensor], got {type(activations)}: {activations}"
            )

        if self.output_tuple_index is not None:
            return activations[self.output_tuple_index]

        for i, candidate in enumerate(activations):
            if candidate.dim() == 3:
                self.output_tuple_index: int | None = i
                return candidate

        raise RuntimeError(
            f"Failed to manipulate activations for split point '{split_point}'. "
            "Activations are tuples, and no tensor with three dimensions was found. "
            f"Found tensors of shape: {(t.shape for t in activations)}. "
            "It is recommended to look for another split point."
        )

    def get_activations(  # noqa: PLR0912  # ignore too many branches  # too many special cases
        self,
        inputs: list[str] | torch.Tensor | BatchEncoding,
        activation_granularity: ActivationGranularity,
        aggregation_strategy: GranularityAggregationStrategy = GranularityAggregationStrategy.MEAN,
        pad_side: str | None = None,
        tqdm_bar: bool = False,
        include_predicted_classes: bool = False,
        model_forward_kwargs: dict[str, Any] = {},
    ) -> dict[str, LatentActivations]:
        """

        Get intermediate activations for all model split points on the given `inputs`.

        Also include the model predictions in the returned activations dictionary.

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

                - ``ModelWithSplitPoints.activation_granularities.ALL``:
                    the activations are returned as is ``(n, l, d)``.
                    They are padded manually so that each batch of activations can be concatenated.

                - ``ModelWithSplitPoints.activation_granularities.CLS_TOKEN``:
                    only the first token (e.g. ``[CLS]``) activation is returned ``(n, d)``.

                - ``ModelWithSplitPoints.activation_granularities.ALL_TOKENS``:
                    every token activation is treated as a separate element ``(n x l, d)``.
                    This includes special tokens such as [CLS], [SEP], [EOS], [PAD], etc.

                - ``ModelWithSplitPoints.activation_granularities.TOKEN``: remove special tokens.

                - ``ModelWithSplitPoints.activation_granularities.WORD``:
                    aggregate by words following the split defined by
                    `interpreto.commons.granularity.Granularity.WORD`.

                - ``ModelWithSplitPoints.activation_granularities.SENTENCE``:
                    aggregate by sentences following the split defined by
                    :class:`~interpreto.commons.granularity.Granularity.SENTENCE`.
                    Requires `spacy` to be installed.

                - ``ModelWithSplitPoints.activation_granularities.SAMPLE``:
                    activations are aggregated on the whole sample.

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

            tqdm_bar (bool):
                Whether to display a progress bar.

            include_predicted_classes (bool):
                Whether to include the predicted classes in the output dictionary.
                Only applicable for classification models.

            model_forward_kwargs (dict):
                Additional keyword arguments passed to the model forward pass.

        Returns:
            (dict[str, LatentActivations]) Dictionary having one key, value pair for each split point defined for the model. Keys correspond to split
                names in `self.split_points`, while values correspond to the extracted activations for the split point
                for the given `inputs`.
        """
        # set default pad side value and catch unsupported cases
        if self._model.__class__.__name__.endswith("Classification"):
            pad_side = pad_side or "right"
        else:
            pad_side = pad_side or "left"
            if include_predicted_classes:
                raise ValueError(
                    "`include_predicted_classes` is only supported for classification models. "
                    f"Provided model is a {self._model.__class__.__name__}."
                )

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
        else:  # sequence of inputs or tensors
            # create a generator for iterable of inputs and tensors
            batch_generator = (
                inputs[i : min(i + self.batch_size, len(inputs))] for i in range(0, len(inputs), self.batch_size)
            )

        # wrap generator in tqdm for progress bar
        tqdm_wrapped_batch_generator = tqdm(
            batch_generator,
            desc="Computing activations",
            unit="batch",
            total=ceil(len(inputs) / self.batch_size),
            disable=not tqdm_bar,
        )

        # initialize activations dictionary
        activations: dict = {}
        for split_point in self.split_points + ["predictions"]:
            activations[split_point] = []

        # iterate over batch of inputs
        with torch.no_grad():
            # several call of the same model should be grouped in an nnsight session
            with self.session():
                for batch_inputs in tqdm_wrapped_batch_generator:
                    # ------------------------------------------------------------------------------
                    # prepare inputs and compute granular indices
                    if isinstance(batch_inputs, list):
                        # tokenize text inputs for granularity selection
                        # include "offsets_mapping" for sentence selection strategy
                        if activation_granularity == AG.CLS_TOKEN:
                            self.tokenizer.padding_side = "right"
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
                    granularity_indices = self._get_granularity_indices(tokenized_inputs, activation_granularity)

                    # extract offset mapping not supported by forward but was necessary for sentence selection strategy
                    if isinstance(tokenized_inputs, (BatchEncoding, dict)):  # noqa: UP038
                        tokenized_inputs.pop("offset_mapping", None)

                    # ------------------------------------------------------------------------------
                    # model forward pass with nnsight to extract activations and predictions

                    # all model calls use trace with nnsight
                    # call model forward pass and save split point outputs
                    with self.trace(tokenized_inputs, **model_forward_kwargs) as tracer:
                        # nnsight quick way to obtain the activations for all split points
                        batch_activations = tracer.cache(modules=[self.get(sp) for sp in self.split_points])  # type: ignore  (under specification from NNsight)

                        # for classification optionally compute and save the predictions
                        if include_predicted_classes:
                            batch_predictions: Float[torch.Tensor, "n"] = (
                                self.output.logits.argmax(dim=-1).cpu().save()  # type: ignore  (under specification from NNsight)
                            )

                    # free memory after each batch, necessary with nnsight, overwise, memory piles up
                    torch.cuda.empty_cache()

                    # ------------------------------------------------------------------------------
                    # apply granularity selection and aggregation of activations and predictions
                    for sp in self.split_points:
                        # extracting the activations for the current split point
                        sp_module = batch_activations["model." + sp]
                        output_name = "nns_output" if hasattr(sp_module, "nns_output") else "output"
                        batch_outputs = getattr(sp_module, output_name)

                        # manage the output tuple and extract the (n, l, d) activations from it
                        batch_sp_activations: Float[torch.Tensor, "n l d"] = self._manage_output_tuple(
                            batch_outputs, sp
                        )

                        # select relevant activations with respect to the granularity strategy
                        # potentially aggregate activations over the granularity elements
                        # this merges the `n` and `g` dimensions with `g` a subset of `n`
                        # shape (n, l, d) only for `ALL` granularity, thus raw activations
                        granular_activations: Float[torch.Tensor, "ng d"] | Float[torch.Tensor, "n l d"] = (
                            self._apply_selection_strategy(
                                activations=batch_sp_activations,
                                granularity_indices=granularity_indices,
                                activation_granularity=activation_granularity,
                                aggregation_strategy=aggregation_strategy,
                            )
                        )

                        activations[sp].append(granular_activations)

                    if include_predicted_classes:
                        # for granularities outside of `ALL`
                        if granularity_indices is not None:
                            # adapt predictions to match the granularity indices
                            repeats: Float[torch.Tensor, "ng"] = torch.tensor(
                                [len(indices) for indices in granularity_indices]
                            )

                            # predictions have a shape (n,), which we convert to (ng,)
                            # by repeating each predicted class as many times as the number of granularity elements in a sample
                            repeated_predictions = torch.repeat_interleave(batch_predictions, repeats, dim=0)  # type: ignore  (ignore possibly unbound)
                            activations["predictions"].append(repeated_predictions)

        # ------------------------------------------------------------------------------------------
        # concat activation batches and validate that activations have the expected type
        for split_point in self.split_points:
            if activation_granularity == AG.ALL:
                # three dimensional tensor (n, l, d)
                activations[split_point] = ModelWithSplitPoints._pad_and_concat(
                    activations[split_point], pad_side, 0.0
                )
            else:
                # two dimensional tensor (n*g, d)
                activations[split_point] = torch.cat(activations[split_point], dim=0)

        if include_predicted_classes:
            activations["predictions"] = torch.cat(activations["predictions"], dim=0)
        else:
            activations.pop("predictions", None)

        # validate that activations have the expected type
        for layer, act in activations.items():
            if not isinstance(act, torch.Tensor):
                raise RuntimeError(
                    f"Invalid output for layer '{layer}'. Expected torch.Tensor activation, got {type(act)}: {act}"
                )
        return activations  # type: ignore

    @jaxtyped(typechecker=beartype)
    def _get_concept_output_gradients(  # noqa: PLR0912  # ignore too many branches
        self,
        inputs: list[str] | torch.Tensor | BatchEncoding,
        encode_activations: Callable[[LatentActivations], ConceptsActivations],
        decode_concepts: Callable[[ConceptsActivations], LatentActivations],
        targets: list[int] | None = None,
        split_point: str | None = None,
        activation_granularity: ActivationGranularity = AG.TOKEN,
        aggregation_strategy: GranularityAggregationStrategy | None = GranularityAggregationStrategy.MEAN,
        concepts_x_gradients: bool = False,
        tqdm_bar: bool = False,
        batch_size: int | None = None,
        model_forward_kwargs: dict[str, Any] = {},
    ) -> list[Float[torch.Tensor, "t g c"]]:
        """Get intermediate activations for all model split points

        :warning: This method should not be called directly. The concept explainer should be used instead.

        Args:
            inputs list[str] | torch.Tensor | BatchEncoding:
                Inputs to the model forward pass before or after tokenization.
                In the case of a `torch.Tensor`, we assume a batch dimension and token ids.

            activation_granularity (ActivationGranularity):
                Selection strategy for activations. Options are:

                - ``ModelWithSplitPoints.activation_granularities.CLS_TOKEN``:
                    only the first token (e.g. ``[CLS]``) activation is returned ``(n, d)``.

                - ``ModelWithSplitPoints.activation_granularities.ALL_TOKENS``:
                    every token activation is treated as a separate element ``(n x l, d)``.
                    This includes special tokens such as [CLS], [SEP], [EOS], [PAD], etc.

                - ``ModelWithSplitPoints.activation_granularities.TOKEN``: remove special tokens.

                - ``ModelWithSplitPoints.activation_granularities.WORD``:
                    aggregate by words following the split defined by
                    :class:`~interpreto.commons.granularity.Granularity.WORD`.

                - ``ModelWithSplitPoints.activation_granularities.SENTENCE``:
                    aggregate by sentences following the split defined by
                    :class:`~interpreto.commons.granularity.Granularity.SENTENCE`.
                    Requires `spacy` to be installed.

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

            model_forward_kwargs (dict):
                Additional keyword arguments passed to the model forward pass.

        Returns:
            gradients (list[torch.Tensor]): The gradients of the model output with respect to the concept activations.
            List length: correspond to the number of inputs.
                Tensor shape: (t, g, c) with t the target dimension, g the number of granularity elements in one input, and c the number of
                concepts.
        """
        # sanity check
        if activation_granularity in [AG.ALL, AG.SAMPLE]:
            raise ValueError(
                "The activation granularity cannot be ALL or SAMPLE to compute the concept output gradients. "
                "Please choose another granularity strategy among: ALL_TOKENS, CLS_TOKEN, TOKEN, WORD, SENTENCE. "
            )

        # add padding token to vocabulary if not present (model and tokenizer)
        if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self._model.resize_token_embeddings(len(self.tokenizer))  # type: ignore  (weird huggingface typing)

        # the `targets` parameter need to be loaded in self for nnsight to allow its access inside the trace context
        self.targets = targets

        # manage the split point
        if split_point is not None:
            local_split_point: str = split_point
        elif not self.split_points:
            raise ValueError(
                "The activations cannot correspond to `model_with_split_points` model. "
                "The `model_with_split_points` model do not have `split_point` defined. "
            )
        elif len(self.split_points) > 1:
            raise ValueError("Cannot determine the split point with multiple `model_with_split_points` split points. ")
        else:
            local_split_point: str = self.split_points[0]

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
        with self.session():
            # iterate over batch of inputs
            for batch_inputs in tqdm_wrapped_batch_generator:
                # --------------------------------------------------------------------------------------
                # prepare inputs and compute granular indices
                # tokenize text inputs
                if isinstance(batch_inputs, list):
                    if activation_granularity == AG.CLS_TOKEN:
                        self.tokenizer.padding_side = "right"
                    tokenized_inputs = self.tokenizer(
                        batch_inputs, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True
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
                with self.trace(tokenized_inputs, **model_forward_kwargs):
                    curr_module = self.get(local_split_point)
                    # Handle case in which module has .output attribute, and .nns_output gets overridden instead
                    module_out_name = "nns_output" if hasattr(curr_module, "nns_output") else "output"

                    # get activations
                    layer_outputs = getattr(curr_module, module_out_name)
                    raw_activations: Float[torch.Tensor, "n l d"] = self._manage_output_tuple(
                        layer_outputs, local_split_point
                    )
                    n, l, d = raw_activations.shape  # number of samples, sequence length, and model dimension
                    ng = sum([len(indices) for indices in granularity_indices])  # number of granularity elements

                    # apply selection strategy
                    selected_activations: Float[torch.Tensor, ng, d]
                    selected_activations = self._apply_selection_strategy(
                        activations=raw_activations,  # use the last batch of activations
                        granularity_indices=granularity_indices,
                        activation_granularity=activation_granularity,
                        aggregation_strategy=aggregation_strategy,
                    )

                    # encode activations into concepts
                    concept_activations: Float[torch.Tensor, "{ng} c"] = encode_activations(selected_activations)
                    del selected_activations
                    c = concept_activations.shape[-1]

                    # decode concepts back into activations
                    decoded_activations: Float[torch.Tensor, ng, d] = decode_concepts(concept_activations)

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
                            concept_activations_grad: Float[torch.Tensor, ng, c] = concept_activations.grad  # type: ignore

                            # for gradient x concepts, multiply by concepts
                            if concepts_x_gradients:
                                concept_activations_grad *= concept_activations
                        targets_gradients_list.append(concept_activations_grad)

                    targets_gradients: Float[torch.Tensor, t, ng, d] = (
                        torch.stack(targets_gradients_list, dim=0).detach().cpu().save()  # type: ignore  (nnsight under specification)
                    )
                    del targets_gradients_list, concept_activations, concept_activations_grad, logits, all_logits

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

    def get_split_activations(
        self, activations: dict[str, LatentActivations], split_point: str | None = None
    ) -> LatentActivations:
        """
        Extract activations for the specified split point.
        If no split point is specified, it works if and only if the `model_with_split_points` has only one split point.
        Verify that the given activations are valid for the `model_with_split_points` and `split_point`.
        Cases in which the activations are not valid include:

        * Activations are not a valid dictionary.
        * Specified split point does not exist in the activations.

        Args:
            activations (dict[str, LatentActivations]): A dictionary with model paths as keys and the corresponding
                tensors as values.
            split_point (str | None): The split point to extract activations from.
                If None, the `split_point` of the explainer is used.

        Returns:
            (LatentActivations): The activations for the explainer split point.

        Examples:
            >>> from interpreto import ModelWithSplitPoints as MWSP
            >>> model = ModelWithSplitPoints("bert-base-uncased", split_points=4,
            >>>                              automodel=AutoModelForSequenceClassification)
            >>> activations_dict: dict[str, LatentActivations] = model.get_activations(
            ...     "interpreto is magic",
            ... )
            >>> activations: LatentActivations = model.get_split_activations(activations_dict)
            >>> activations.shape
            torch.Size([1, 12, 768])

        Raises:
            ValueError: If not split point is specified and the `model_with_split_points` has more than one split point.
            TypeError: If the activations are not a valid dictionary.
            ValueError: If the specified split point is not found in the activations.
        """
        if split_point is not None:
            local_split_point: str = split_point
        elif not self.split_points:
            raise ValueError(
                "The activations cannot correspond to `model_with_split_points` model. "
                "The `model_with_split_points` model do not have `split_point` defined. "
            )
        elif len(self.split_points) > 1:
            raise ValueError("Cannot determine the split point with multiple `model_with_split_points` split points. ")
        else:
            local_split_point: str = self.split_points[0]

        if not isinstance(activations, dict) or not all(isinstance(act, torch.Tensor) for act in activations.values()):
            raise TypeError(
                "Invalid activations for the concept explainer. "
                "Activations should be a dictionary of model paths and torch.Tensor activations. "
                f"Got: '{type(activations)}'"
            )
        activations_split_points: list[str] = list(activations.keys())  # type: ignore
        if local_split_point not in activations_split_points:
            raise ValueError(
                f"Fitted split point '{local_split_point}' not found in activations.\n"
                f"Available split_points: {', '.join(activations_split_points)}."
            )

        return activations[local_split_point]  # type: ignore

    def get_latent_shape(
        self,
        inputs: str | list[str] | BatchEncoding | None = None,
    ) -> dict[str, torch.Size]:
        """Get the shape of the latent activations at the specified split point.

        Use the `scan` operation from NNsight to get the shape of the activations.
        It basically builds the computation graph, but it it much quicker than a forward.

        Args:
            inputs (str | list[str] | BatchEncoding | None): Inputs to the model forward pass before or after tokenization.
                In the case of a `torch.Tensor`, we assume a batch dimension and token ids.

        Returns:
            dict[str, torch.Size]: Dictionary with the shape of the activations for each split point.
        """
        sizes = {}
        with self.scan(self._example_input if inputs is None else inputs):
            for split_point in self.split_points:
                curr_module = self.get(split_point)
                module_out_name = "nns_output" if hasattr(curr_module, "nns_output") else "output"
                module = getattr(curr_module, module_out_name)
                if isinstance(module, tuple):
                    for candidate in module:
                        if candidate.dim() == 3:
                            module = candidate
                            break
                sizes[split_point] = module.shape  # type: ignore  (under specification from NNsight)
        return sizes
