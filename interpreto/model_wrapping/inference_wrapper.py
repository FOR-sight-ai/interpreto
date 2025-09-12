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
Basic inference wrapper for explaining models.

This module provides a base class for inference wrappers that can be used to
perform inference on various models. The InferenceWrapper class is designed to
handle device management, embedding inputs, and batching of inputs for efficient
processing. The class is designed to be subclassed for specific model types and tasks.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, MutableMapping
from enum import Enum
from functools import singledispatchmethod
from typing import Any, overload

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel

from interpreto.typing import IncompatibilityError, ModelInputs, TensorMapping

# TODO: make jaxtyping in the whole file!!


class InferenceModes(Enum):
    """
    Enum class for inference modes.

    Attributes:
        LOGITS: Return the logits.
        SOFTMAX: Return the softmax of the logits.
        LOG_SOFTMAX: Return the log softmax of the logits.
    """

    LOGITS = staticmethod(lambda logits: logits)
    SOFTMAX = staticmethod(lambda logits: F.softmax(logits, dim=-1))
    LOG_SOFTMAX = staticmethod(lambda logits: F.log_softmax(logits, dim=-1))


# TODO : move that somewhere else
# TODO: simplify this function, this is overly complex for what we need,
# the dim and padding dims are always 0 an 1
def concat_and_pad(
    *tensors: torch.Tensor | None,
    pad_left: bool,
    dim: int = 0,
    pad_value: int = 0,
    pad_dims: Iterable[int] | None = None,
) -> torch.Tensor:
    """
    Concatenate and pad tensors to the maximum length of each dimension.

    Args:
        *tensors (torch.Tensor | None): tensors to concatenate (can be of different shapes but must have the same number of dimensions). Can be None.
        pad_left (bool): if True, padding is done on the left side of the tensor, otherwise on the right side.
        dim (int, optional): Dimension along which the tensors will be concatenated. Defaults to 0.
        pad_value (int, optional): Value used to pad the tensors. Defaults to 0.
        pad_dims (Iterable[int] | None, optional): Dimensions to pad. Defaults to None.

    Returns:
        torch.Tensor: result of the concatenation

    Raises:
        ValueError: if the tensors have different number of dimensions.
        TypeError: If the `tensors` argument is not a valid sequence of tensors or if
            `pad_dims` contains invalid dimensions.
        ValueError: If the concatenation dimension is in the padding dimensions.
        ValueError: If the tensors have different shapes along the dimensions not in the padding dimensions.

    Example:
        >>> t1 = torch.randn(2, 3, 4)
        >>> t2 = torch.randn(3, 2, 5)
        >>> t3 = torch.randn(1, 6, 1)
        >>> result = concat_and_pad(t1, t2, t3, pad_left=True, dim=0, pad_value=-1, pad_dims=[1, 2])
        >>> print(result.shape)
        torch.Size([6, 6, 5])  # After padding and concatenation along the first dimension
    """
    _tensors = [a for a in tensors if a is not None and a.numel()]
    if not _tensors:
        raise ValueError("No tensors provided for concatenation.")
    if any(t.dim() != _tensors[0].dim() for t in _tensors[1:]):
        raise ValueError("All tensors must have the same number of dimensions.")

    tensors_dim = _tensors[0].dim()
    pad_dims = pad_dims or []
    if dim is not None and dim in pad_dims:
        raise ValueError(f"`pad_dims`: {pad_dims} should not contain the dimension to pad, `dim`: {dim}")
    for t_dim in range(tensors_dim):
        if t_dim not in pad_dims and t_dim != dim:
            if any(t.shape[t_dim] != _tensors[0].shape[t_dim] for t in _tensors):
                raise ValueError(
                    f"All tensors must have the same shape along the dimensions not in the padding dimensions {pad_dims}, but got {[t.shape for t in _tensors]}"
                )
    max_length_per_dim = [max(t.shape[d] for t in _tensors) for d in pad_dims]

    padded_tensors: list[torch.Tensor] = []
    for t in _tensors:
        pad = [0, 0] * tensors_dim
        for pad_dim, pad_length in zip(pad_dims, max_length_per_dim, strict=True):
            # update padding indication to pad the right dimension
            pad_index = -2 * (pad_dim % tensors_dim) - 1 - pad_left
            pad[pad_index] = pad_length - t.shape[pad_dim]
        # pad the tensor
        padded_tensors.append(torch.nn.functional.pad(t, pad, value=pad_value))
    # return the concatenation of all tensors
    return torch.cat(padded_tensors, dim=dim)


class InferenceWrapper(ABC):
    """
    Base class for inference wrapper objects.
    This class is designed to wrap a model and provide a consistent interface for
    performing inference on the model's inputs. It handles device management,
    embedding inputs, and batching of inputs for efficient processing.
    The class is designed to be subclassed for specific model types and tasks.

    Attributes:
        model (PreTrainedModel): The model to be wrapped.
        batch_size (int): The maximum batch size for processing inputs.
        device (torch.device | None): The device on which the model is loaded.
    """

    # static attribute to indicate whether to pad on the left or right side
    # this is a class attribute and should be set in subclasses
    PAD_LEFT = True

    def __init__(
        self,
        model: PreTrainedModel,
        batch_size: int = 4,
        device: torch.device | None = None,
        mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
    ):
        self.model = model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)  # type: ignore
        self.batch_size = batch_size

        assert callable(mode), "mode should be a callable function from `InferenceModes`"
        self.mode = mode

        # Pad token id should be set by the explainer
        self.pad_token_id = None

    @property
    def device(self) -> torch.device:
        """
        Returns:
            torch.device: The device on which the model is loaded.
        """
        return self.model.device

    @device.setter
    def device(self, device: torch.device):
        """
        Sets the device on which the model is loaded.

        Args:
            device (torch.device): wanted device (e.g., "cpu" or "cuda").
        """
        self.model.to(device)  # type: ignore

    def to(self, device: torch.device):
        """
        Move the model to the specified device.

        Args:
            device (torch.device): The device to which the model should be moved.
        """
        self.device = device

    def cpu(self):
        """
        Move the model to the CPU.
        """
        self.device = torch.device("cpu")

    def cuda(self):
        """
        Move the model to the GPU.
        """
        self.device = torch.device("cuda")

    def embed(self, model_inputs: TensorMapping) -> TensorMapping:
        """
        Embed the inputs using the model's input embeddings.

        Args:
            model_inputs (TensorMapping): input mapping containing either "input_ids" or "inputs_embeds".

        Raises:
            ValueError: If neither "input_ids" nor "inputs_embeds" are present in the input mapping.

        Returns:
            TensorMapping: The input mapping with "inputs_embeds" added.
        """
        # If input embeds are already present, return the unmodified model inputs
        if "inputs_embeds" in model_inputs:
            return model_inputs
        # If input ids are present, get the embeddings and add them to the model inputs
        if "input_ids" in model_inputs:
            base_shape = model_inputs["input_ids"].shape
            input_ids = model_inputs["input_ids"].flatten(0, -2).to(self.device)
            flatten_embeds = self.model.get_input_embeddings()(input_ids)
            model_inputs["inputs_embeds"] = flatten_embeds.view(*base_shape, flatten_embeds.shape[-1])
            return model_inputs
        # If neither input ids nor input embeds are present, raise an error
        raise ValueError("model_inputs should contain either 'input_ids' or 'inputs_embeds'")

    def call_model(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> BaseModelOutput:
        """
        Perform a call to the wrapped model with the given input embeddings and attention mask.

        Args:
            input_ids (torch.Tensor | None): input ids to be passed to the model.
            inputs_embeds (torch.Tensor | None): input embeddings to be passed to the model.
            attention_mask (torch.Tensor | None): attention mask to be passed to the model.

        Returns:
            ModelOutput: The output of the model.

        Note:
            If the batch size of the input embeddings exceeds the wrapper's batch size, a warning is issued.
        """
        if inputs_embeds is None and input_ids is None:
            raise ValueError("Either inputs_embeds or input_ids must be provided.")

        # Check that batch size of inputs_embeds is not greater than the wrapper's batch size
        if input_ids is not None and input_ids.shape[0] > self.batch_size:
            raise ValueError(
                f"Batch size of {input_ids.shape[0]} is greater than the wrapper's batch size of {self.batch_size}. "
                f"Consider adjust the batch size or the wrapper of split your data.",
            )
        if inputs_embeds is not None and inputs_embeds.shape[0] > self.batch_size:
            raise ValueError(
                f"Batch size of {inputs_embeds.shape[0]} is greater than the wrapper's batch size of {self.batch_size}. "
                f"Consider adjust the batch size or the wrapper of split your data.",
            )

        # send input to device
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Call wrapped model
        if inputs_embeds is not None:
            try:
                return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            except NotImplementedError as e:
                raise IncompatibilityError from e

        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    @overload
    def get_logits(self, model_inputs: TensorMapping) -> torch.Tensor: ...

    @overload
    def get_logits(self, model_inputs: Iterable[TensorMapping]) -> Generator[torch.Tensor, None, str]: ...

    @singledispatchmethod
    def get_logits(self, model_inputs: ModelInputs) -> torch.Tensor | Generator[torch.Tensor, None, str]:
        """
        Get the logits from the model for the given inputs.

        This method propose two different treatments of the inputs:
        If the input is a mapping, it will be processed as a single input and given directly to the model.
        The method will return the logits of the model as a torch.Tensor.

        If the input is an iterable of mappings, it will be processed as a batch of inputs.
        The method will yield the logits of the model for each input as a torch.Tensor.

        Args:
            model_inputs (Any): input mappings to be passed to the model or iterable of input mappings.

        Raises:
            NotImplementedError: If the input type is not supported.

        Returns:
            torch.Tensor | Generator[torch.Tensor, None, None]: logits associated to the input mappings.

        Example:
            Single input given as a mapping
                >>> model_inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}
                >>> logits = wrapper.get_logits(model_inputs)
                >>> print(logits.shape)

            Sequence of inputs given as an iterable of mappings (generator, list, etc.)
                >>> model_inputs = [{"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])},
                ...                 {"input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]])}]
                >>> logits = wrapper.get_logits(model_inputs)
                >>> for logit in logits:
                ...     print(logit.shape)

        """
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_logits in class {self.__class__.__name__}"
        )

    @get_logits.register(MutableMapping)  # type: ignore
    def _get_logits_from_mapping(self, model_inputs: TensorMapping) -> torch.Tensor:
        """
        Get the logits from the model for the given inputs.
        registered for MutableMapping type.

        Args:
            model_inputs (TensorMapping): input mapping containing either "input_ids" or "inputs_embeds".

        Returns:
            torch.Tensor: logits associated to the input mapping.
        """
        # if embeddings has not been calculated yet, embed the inputs
        # model_inputs = self.embed(model_inputs)  # TODO: add back if needed for gradient-based methods
        inputs_key = "input_ids" if "input_ids" in model_inputs.keys() else "inputs_embeds"
        # depending on the number of dimensions of the input
        match model_inputs[inputs_key].dim():
            case 2:  # (sequence_length, embedding_size)
                return self.call_model(
                    **{inputs_key: model_inputs[inputs_key], "attention_mask": model_inputs["attention_mask"]}
                ).logits  # type: ignore
            case 3:  # (batch_size, sequence_length, embedding_size)
                # If a batch dimension is given, split the inputs into chunks of batch_size
                inputs_chunks = model_inputs[inputs_key].split(self.batch_size)
                mask_chunks = model_inputs["attention_mask"].split(self.batch_size)

                # call the model on each chunk and concatenate the results
                return torch.cat(
                    [
                        self.call_model(**{inputs_key: inputs_chunk, "attention_mask": mask_chunk}).logits  # type: ignore
                        for inputs_chunk, mask_chunk in zip(inputs_chunks, mask_chunks, strict=False)
                    ],
                )
            case _:  # (..., sequence_length, embedding_size) e.g. (batch_size, n_perturbations, sequence_length, embedding_size)
                # flatten the first dimension to a single batch dimension
                # then call the model on the flattened inputs and reshape the result to the original batch structure
                flat_model_inputs = {
                    inputs_key: model_inputs[inputs_key].flatten(0, -3),
                    "attention_mask": model_inputs["attention_mask"].flatten(0, -2),
                }
                prediction = self._get_logits_from_mapping(flat_model_inputs)
                return prediction.view(*model_inputs[inputs_key].shape[:-2], -1)

    # @get_logits.register(Iterable)  # type: ignore
    # def _get_logits_from_iterable(self, model_inputs: Iterable[TensorMapping]) -> Generator[torch.Tensor, None, None]:
    #     """
    #     Get the logits from the model for the given inputs.
    #     registered for Iterable type.
    #     Args:
    #         model_inputs (Iterable[TensorMapping]): Iterable of input mappings containing either "input_ids" or "inputs_embeds".
    #     Yields:
    #         torch.Tensor: logits associated to the input mappings.
    #     """
    #     for model_input in model_inputs:
    #         yield self._get_logits_from_mapping(model_input)

    @get_logits.register(Iterable)  # type: ignore
    def _get_logits_from_iterable(self, model_inputs: Iterable[TensorMapping]) -> Generator[torch.Tensor, None, str]:
        """
        Yield the logits associated to the model inputs.
        For each group of model inputs, of the input iterable the logits are yielded.
        Here the goal is to call the model the least number of times possible.
        The constraint is that each model call should be done with at most batch_size inputs.

        `model_inputs` is an iterable of model inputs, each model inputs can have a different number of samples.
        Let's take the example of three model inputs: n1 = 3, n2 = 8, and n3 = 4, with a batch size of 5.
        The batch will be as follows:
        [[1, 1, 1, 2, 2], [2, 2, 2, 2, 2], [2, 3, 3, 3, 3]]

        To do so the algorithm is the following:
            while true:
                if there is enough data in the output buffer:
                    yield the output buffer
                    if last item:
                        break
                    continue
                if there is a batch available:
                    call the model
                    concatenate the results to the output buffer
                if there is enough data in the input buffer:
                    make a batch
                if there is not enough data in the input buffer:
                    try to get the next item from the input stream
                    add it to the input buffer

        Args:
            model_inputs (Iterable[TensorMapping]): Iterable of input mappings containing either "input_ids" or "inputs_embeds".

        Yields:
            torch.Tensor: logits associated to the input mappings.
        """
        # create an iterator from the input iterable
        model_inputs = iter(model_inputs)

        # If no pad token id has been given
        if self.pad_token_id is None:
            # raise ValueError(
            #     "Asking to pad but the tokenizer does not have a padding token. Please select a token to use as pad_token (tokenizer.pad_token = tokenizer.eos_token e.g.) or add a new pad token via tokenizer.add_special_tokens({'pad_token': '[PAD]'})"
            # )
            raise ValueError(
                "Padding token is not set in the inference wrapper. Please assign it explicitly by setting: inference_wrapper.pad_token_id = tokenizer.pad_token_id"
            )

        batch: torch.Tensor | None = None
        batch_mask: torch.Tensor | None = None

        inputs_key: str = "input_ids"  # default key for inputs, will be updated if inputs_embeds are used
        n_tokens: list[int] = []
        result_indexes: list[int] = []
        input_buffer: list[torch.Tensor] = []
        mask_buffer: list[torch.Tensor] = []
        result_buffer: list[torch.Tensor] = []

        last_item = False

        # Generation loop
        while True:
            # if there is no element left in the input stream and the result buffer is empty, break the loop
            if last_item and not result_indexes:
                break
            # check if the output buffer contains enough data to correspond to the next element
            if result_indexes and len(result_buffer) >= result_indexes[0]:
                # pop the first index from the result indexes
                index = result_indexes.pop(0)
                n_token = n_tokens.pop(0)

                # in the case of generation, input tokens might have been padded differently depending on the batch
                # we need to remove the padding from the result buffer
                if len(result_buffer[0].shape) == 2:
                    # TODO: verify we do not destroy everything
                    assert self.PAD_LEFT, "results shapes suggest generation but the padding is not set to left"
                    for i in range(index):
                        result_buffer[i] = result_buffer[i][-n_token:]

                # yield the associated logits
                yield torch.stack(result_buffer[:index])
                # remove the yielded logits from the result buffer
                result_buffer = result_buffer[index:]
                continue
            # check if the batch of inputs is large enough to be processed (or if the last item is reached)
            if batch is not None and (last_item or len(batch) == self.batch_size):
                # Call the model
                logits = self.call_model(**{inputs_key: batch, "attention_mask": batch_mask}).logits  # type: ignore

                # Concatenate the results to the output buffer
                result_buffer += [*logits.detach()]  # TODO: see if I need to pass to cpu here

                ##################### FIXME #####################
                # The .detach().clone() if used to avoid memory issues provoked by the bad usage of the result_buffer
                # This will block the gradient calculation on the yielded logits
                # Gradient calculation currently only call _get_logits_from_mapping register for the jacobian calculation
                # This code works but should be improved in the future
                ###############################################
                # result_buffer = concat_and_pad(result_buffer, logits, pad_left=self.PAD_LEFT).detach().clone()

                # update batch and mask
                batch = batch_mask = None
                continue
            # check if the input buffer contains enough data to fill the batch
            if len(input_buffer) >= self.batch_size or last_item:
                # calculate the missing length of the batch
                missing_length = self.batch_size - len(batch if batch is not None else ())
                # fill the batch with the missing data
                batch = concat_and_pad(
                    batch,
                    *input_buffer[:missing_length],
                    pad_left=self.PAD_LEFT,
                    dim=0,
                    pad_value=self.pad_token_id,
                    pad_dims=(1,),
                )
                batch_mask = concat_and_pad(
                    batch_mask,
                    *mask_buffer[:missing_length],
                    pad_left=self.PAD_LEFT,
                    dim=0,
                    pad_value=0,
                    pad_dims=(1,),
                )
                # remove the used data from the input buffer
                input_buffer = input_buffer[missing_length:]
                mask_buffer = mask_buffer[missing_length:]
                continue
            # If there is not enough data in the input buffer, get the next item from the input stream
            try:
                # Get next item input and mask
                next_item = next(model_inputs)
                inputs_key = "input_ids" if "input_ids" in next_item.keys() else "inputs_embeds"
                # next_item = self.embed(next(model_inputs))  # TODO: remove embed if not absolutely necessary

                # update buffers and lists
                n_tokens.append(next_item[inputs_key].shape[1])
                result_indexes.append(next_item[inputs_key].shape[0])
                input_buffer += [elem.unsqueeze(0) for elem in next_item[inputs_key]]
                mask_buffer += [elem.unsqueeze(0) for elem in next_item["attention_mask"]]
            # If the input stream is empty
            except StopIteration:
                if last_item:
                    # This should never happen
                    warnings.warn(
                        "Tried to get the next item from the input stream but it is empty a second time. This should never happen.",
                        stacklevel=2,
                    )
                last_item = True
        # Check that all the buffers are empty
        if any(len(element) for element in [result_buffer, input_buffer, mask_buffer, result_indexes, n_tokens]):
            warnings.warn(
                "Some data were not well fetched in inference wrapper,"
                + " please check your code if you made custom method or notify it to the developers."
                + " Remaining data in buffers:\n"
                + f"\tinput_buffer: {len(input_buffer)}\n"
                + f"\tmask_buffer: {len(mask_buffer)}\n"
                + f"\tresult_indexes: {result_indexes}\n"
                + f"\tresult_buffer: {len(result_buffer)}\n"
                + f"\tn_tokens: {len(n_tokens)}\n"
                + f"\tlast_item: {last_item}\n",
                stacklevel=2,
            )
            return "Some data were not well fetched in inference wrapper"
        return "All data were well fetched in inference wrapper"

    def _reshape_inputs(self, tensor: torch.Tensor, non_batch_dims: int = 2) -> torch.Tensor:
        """
        reshape inputs to have a single batch dimension.
        """
        # TODO : see if there is a better way to do this
        assert tensor.dim() >= non_batch_dims, "The given tensor have less dimensions than non_batch_dims parameter"
        if tensor.dim() == non_batch_dims:
            return tensor.unsqueeze(0)
        assert tensor.shape[0] == 1, (
            "When passing a sequence or a generator of inputs to the inference wrapper, please consider giving sequence of perturbations of single elements instead of batches (shape should be (1, n_perturbations, ...))"
        )
        if tensor.dim() == non_batch_dims + 1:
            return tensor
        return self._reshape_inputs(tensor[0], non_batch_dims=non_batch_dims)

    @singledispatchmethod
    @abstractmethod
    def get_targeted_logits(
        self, model_inputs: Any, targets: torch.Tensor
    ) -> torch.Tensor | Generator[torch.Tensor, None, None]:
        raise NotImplementedError(
            f"get_targeted_logits not implemented for {self.__class__.__name__}. Implement this method is necessary to use gradient-based methods."
        )

    @overload
    def get_gradients(
        self, model_inputs: TensorMapping, targets: torch.Tensor, input_x_gradient: bool = False
    ) -> torch.Tensor: ...

    @overload
    def get_gradients(
        self, model_inputs: Iterable[TensorMapping], targets: Iterable[torch.Tensor], input_x_gradient: bool = False
    ) -> Iterable[torch.Tensor]: ...

    # @allow_nested_iterables_of(MutableMapping)
    @singledispatchmethod
    def get_gradients(
        self, model_inputs: ModelInputs, targets: torch.Tensor, input_x_gradient: bool = False
    ) -> torch.Tensor | Generator[torch.Tensor, None, None]:
        """
        Get the gradients of the logits associated to a given target with respect to the inputs.

        Args:
            model_inputs (Any): input mappings to be passed to the model or iterable of input mappings.
            targets (torch.Tensor): target tensor to be used to get the logits.
            targets shape should be either (t) or (n, t) where n is the batch size and t is the number of targets for which we want the logits.

        Raises:
            NotImplementedError: If the input type is not supported.

        Returns:
            torch.Tensor|Generator[torch.Tensor, None, None]: gradients of the logits.

        Example:
            Single input given as a mapping
                >>> model_inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}
                >>> targets = torch.tensor([1, 2])
                >>> gradients = wrapper.get_gradients(model_inputs, targets)
                >>> print(gradients)
            Sequence of inputs given as an iterable of mappings (generator, list, etc.)
                >>> model_inputs = [{"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])},
                ...                 {"input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]])}]
                >>> targets = torch.tensor([[1, 2], [3, 4]])
                >>> gradients = wrapper.get_gradients(model_inputs, targets)
                >>> for grad in gradients:
                ...     print(grad)
        """
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_gradients in class {self.__class__.__name__}"
        )

    @get_gradients.register(MutableMapping)  # type: ignore
    def _get_gradients_from_mapping(
        self, model_inputs: TensorMapping, targets: torch.Tensor, input_x_gradient: bool = False
    ) -> torch.Tensor:  # TODO: add jaxtyping
        model_inputs = self.embed(model_inputs)
        inputs_embeds = model_inputs["inputs_embeds"]

        def get_score(inputs_embeds: torch.Tensor):
            return self.get_targeted_logits(
                {"inputs_embeds": inputs_embeds, "attention_mask": model_inputs["attention_mask"]}, targets
            )

        # Compute gradient of the selected logits:
        grad_matrix = torch.autograd.functional.jacobian(get_score, inputs_embeds)  # (n, lt, n, l, d)

        grad_matrix = grad_matrix[
            torch.arange(grad_matrix.shape[0]), :, torch.arange(grad_matrix.shape[0])
        ]  # (n, lt, l, d)
        if input_x_gradient:
            grad_matrix = grad_matrix * inputs_embeds.unsqueeze(1)

        grad_matrix = grad_matrix.abs().mean(
            dim=-1
        )  # (n, lt, l)  # average over the embedding dimension  # TODO: discuss if th average should be forced or an argument
        return grad_matrix

    @get_gradients.register(Iterable)  # type: ignore
    def _get_gradients_from_iterable(
        self, model_inputs: Iterable[TensorMapping], targets: Iterable[torch.Tensor], input_x_gradient: bool = False
    ) -> Iterable[torch.Tensor]:
        for model_input, target in zip(model_inputs, targets, strict=True):
            # check that the model input and target have the same batch size
            result = self._get_gradients_from_mapping(model_input, target, input_x_gradient=input_x_gradient)
            yield result

        # yield from (
        #     self.get_gradients(model_input, target) for model_input, target in zip(model_inputs, targets, strict=True)
        # )
