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

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_utils import PreTrainedModel

from interpreto.typing import IncompatibilityError, TensorMapping


class InferenceModes(Enum):
    """
    Enum class for inference modes.

    Attributes:
        LOGITS: Return the logits.
        SOFTMAX: Return the softmax of the logits.
        LOG_SOFTMAX: Return the log softmax of the logits.
    """

    LOGITS = auto()
    SOFTMAX = auto()
    LOG_SOFTMAX = auto()

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        match self:
            case InferenceModes.LOGITS:
                return logits
            case InferenceModes.SOFTMAX:
                return F.softmax(logits, dim=-1)
            case InferenceModes.LOG_SOFTMAX:
                return F.log_softmax(logits, dim=-1)


@dataclass
class ChunkMetadata:
    group_id: int
    batch_start: int
    batch_end: int
    last_group_chunk: bool

    sequence_start: int = 0
    sequence_end: int | None = None

    @property
    def sequence_slice(self) -> slice:
        return slice(self.sequence_start, self.sequence_end)


class Batch:
    """
    Class to represent a batch of inputs and targets for the model.
    Its goal is to link the batch sent to the model and the corresponding initial inputs.

    It is iteratively filled with chunks via `add_chunk()`.
    A chunk is part of group of perturbed elements.
    These perturb elements correspond to one initial input.
    So thanks to the metadata, once the batch is treated,
    `_call_batch()` group back outputs corresponding to the same initial input and yield them together.

    Attributes:
        inputs (list[TensorMapping]):
            The list of inputs to send together in a batch.
            They are flattened and distinguished between each others thanks to chunk metadata.
        targets (list[torch.Tensor | None]):
            List of targets, there is one target for each chunk.
        chunks (list[ChunkMetadata]):
            A list of metadata elements, chunks correspond to perturbed inputs
            from the same initial input.
    """

    def __init__(self):
        self.inputs: list[TensorMapping] = []
        self.targets: list[torch.Tensor | None] = []
        self.chunks: list[ChunkMetadata] = []

    def __bool__(self):
        return bool(self.inputs)

    def __len__(self):
        return len(self.inputs)

    def add_chunk(
        self,
        chunk_inputs: list[TensorMapping],
        chunk_targets: torch.Tensor | None,
        group_id: int,
        last_group_chunk: bool = False,
        *,
        sequence_key: str | None = None,
        padding_side: Literal["left", "right"] = "right",
    ) -> None:
        """Add a chunk and record its original sequence extent.

        Samples in a chunk must share the same input geometry.

        `sequence_key` identifies the input field whose sequence padding must
        be removed from gradients. None means no sequence trimming.

        The recorded slice removes padding introduced during batching; it
        does not identify padding already present in the original inputs.
        """
        sequence_start = 0
        sequence_end = None

        if sequence_key is not None:
            sequence_length = chunk_inputs[0][sequence_key].shape[0]

            if padding_side == "left" and sequence_length > 0:
                # Negative indexing remains valid whatever the final batch length.
                sequence_start = -sequence_length
            else:
                # Also represents an empty sequence correctly: slice(0, 0).
                sequence_end = sequence_length

        self.chunks.append(
            ChunkMetadata(
                group_id=group_id,
                batch_start=len(self.inputs),
                batch_end=len(self.inputs) + len(chunk_inputs),
                last_group_chunk=last_group_chunk,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
            )
        )

        self.inputs.extend(chunk_inputs)
        self.targets.append(chunk_targets)


def _split_group(  # TODO: check attributions.base there is also a splitting there
    grouped_elements: TensorMapping | tuple[TensorMapping, torch.Tensor],
) -> tuple[list[TensorMapping], torch.Tensor | None]:
    """
    Split a group of perturbed samples into a list of individual samples.

    Args:
        grouped_elements (dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]):
            A group of perturbed samples, either inputs and targets or just inputs.
            inputs is a dictionary of torch tensor.

    Returns:
        split_inputs (list[dict[str, torch.Tensor]):
            The inputs converted from a dict to a list of dict.
        split_targets (list[torch.Tensor] | None):
            Either split targets or None depending on the presence of targets in the `grouped_elements`.
    """
    # grouped elements might be a tuple if targets are included
    if isinstance(grouped_elements, tuple):
        grouped_inputs, group_targets = grouped_elements
    else:
        grouped_inputs = grouped_elements
        group_targets = None

    # extract keys and very that all fields have the same number of samples
    keys = list(grouped_inputs.keys())
    n_samples = grouped_inputs[keys[0]].shape[0]

    for key in keys[1:]:
        if grouped_inputs[key].shape[0] != n_samples:
            # TODO: rather than raising an error, we could extend each tensor to the same length
            # This is currently duplicated in several perturbators
            keys_samples = {k: grouped_inputs[k].shape[0] for k in keys}
            raise ValueError(f"All fields must have the same number of samples, but received {keys_samples}.")

    # split inputs dict into list of input dict
    split_inputs: list[TensorMapping] = [{key: grouped_inputs[key][i] for key in keys} for i in range(n_samples)]

    if group_targets is not None:
        return split_inputs, group_targets

    return split_inputs, None


class InferenceWrapper(ABC):
    """
    Base class for inference wrapper objects.
    This class is designed to wrap a model and provide a consistent interface for
    performing inference on the model's inputs. It handles device management,
    embedding inputs, and batching of inputs for efficient processing.
    The class is designed to be subclassed for specific model types and tasks.

    Attributes:
        model (PreTrainedModel):
            The model to be wrapped.
        gradients (bool):
            Wether to compute the gradients of the `targeted_logits` with respect to the inputs.
            Requires `targets` and the `inputs_embeds` element in `model_inputs`.
        input_x_gradient (bool, optional):
            If True and ``gradients`` is set, multiplies the input embeddings
            with their gradients before reducing them. Defaults to ``True``.
        batch_size (int):
            The maximum batch size for processing inputs.
        device (torch.device | None):
            The device on which the model is loaded.
        padding_side (str):
            The padding side to use for batching inputs.
            It should be either "right" or "left".
            This is used only for text. Image require padding only when patchified, and in this case, most use right padding.
    """

    padding_side: Literal["left", "right"]
    grad_key: str = "inputs_embeds"

    def __init__(
        self,
        model: PreTrainedModel,
        gradients: bool = False,
        input_x_gradient: bool = True,
        batch_size: int = 4,
        device: torch.device | None = None,
        mode: Callable[[torch.Tensor], torch.Tensor] = InferenceModes.LOGITS,
        padding_side: Literal["left", "right"] | None = None,
    ):
        self.model = model
        self.model.eval()
        self.gradients = gradients
        self.input_x_gradient = input_x_gradient
        self.pad_token_id = model.config.pad_token_id

        self.to(device=device)
        self.batch_size = batch_size

        assert callable(mode), "mode should be a callable function from `InferenceModes`"
        self.mode = mode
        if padding_side is not None:
            self.padding_side = padding_side

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

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Move the model to the specified device.

        Args:
            device: The device to which the model should be moved.
            dtype: The dtype to which the model should be moved.
        """
        # guard against sharded models
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map and len(set(map(str, device_map.values()))) > 1:
            if device is not None or dtype is not None:
                raise ValueError(
                    "Cannot move or cast a sharded model. Please use `device=None` and manage the device manually."
                )
            return

        current = self.model.device
        device = torch.device(device) if isinstance(device, str) else (device or current)

        if device.type == "cuda" and device.index is None and current.type == "cuda":
            device = current

        if device.type == "meta":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device == current and (dtype is None or dtype == self.model.dtype):
            return

        self.model.to(device=device, dtype=dtype)

    @property
    def dtype(self):
        return self.model.dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype):
        self.model.to(dtype=dtype)  # type: ignore

    @abstractmethod
    def _extract_targets_from_logits(self, logits: torch.Tensor, chunk: ChunkMetadata):
        """
        Extract the targets from a tensor of logits.
        """
        raise NotImplementedError(f"_extract_targets_from_logits not implemented for {self.__class__.__name__}.")

    @abstractmethod
    def _target_logits(
        self,
        logits: Float[torch.Tensor, "c ..."],
        targets: Int[torch.Tensor, "t"],
        chunk: ChunkMetadata,
    ) -> Float[torch.Tensor, "c t"]:
        """
        Get the targeted logits from the model output logits and the given targets.

        Logits are expected to come from a single initial input,
        therefore, the target is the same for each one of them

        Args:
            logits (torch.Tensor):
                The output logits from the model.
                They all correspond to the same initial input.
                They are a chunk as defined by `_call_batch`
            targets (torch.Tensor):
                The target tensor to be used to get the targeted logits.
                The target is the same for all logits of the chunk.

        Returns:
            targeted_logits (torch.Tensor):
                The targeted logits associated to the input mappings.
        """
        raise NotImplementedError(f"_target_logits not implemented for {self.__class__.__name__}.")

    def _compute_gradients(
        self,
        inputs: TensorMapping,
        targeted_logits_chunk: Float[torch.Tensor, "c t"],
        chunk: ChunkMetadata,
        last_chunk: bool,
    ) -> Float[torch.Tensor, "c t l"]:
        """
        Compute the gradients of the targeted logits with respect to the inputs.

        It is done target by target to save memory.
        This way, we can directly aggregate on the model width dimension after the gradients computation.

        Args:
            inputs_embeds (TensorMapping):
                The inputs embeddings and attention mask.
                They all correspond to the same initial input.
                Which means that they all have the same attention mask.
            targeted_logits_chunk (torch.Tensor):
                The targeted logits associated to the input mappings.
            chunk (ChunkMetadata):
                The metadata of the chunk, including its slice in the batch.
            last_chunk (bool):
                Whether the chunk is the last one of the batch.
                If True, the graph are not retained for the next iteration.

        Returns:
            gradients_chunk (torch.Tensor):
                The gradients associated with the targeted logits for the chunk.
        """
        c = chunk.batch_end - chunk.batch_start
        chunk_batch_slice = slice(chunk.batch_start, chunk.batch_end)
        # outputs are gradients
        inputs_full: Float[torch.Tensor, "b ..."] = inputs[self.grad_key]
        inputs_chunks: Float[torch.Tensor, f"{c} ..."] = inputs_full[chunk_batch_slice, chunk.sequence_slice].detach()

        # iterate on targets for memory efficiency
        t = targeted_logits_chunk.shape[1]
        gradients_list = []
        for k in range(t):
            last_target = k == (t - 1)
            # we compute gradients one target at a time to save memory and avoid jacobian computations
            target_wise_grads: Float[torch.Tensor, f"{c} ..."] = torch.autograd.grad(
                # sum the targeted logits to get a scalar output for the backward pass, but this does not change the gradients as the targeted logits are independent between each others
                outputs=targeted_logits_chunk[:, k].sum(),
                inputs=inputs_full,  # type: ignore
                # we retain the graph for all but the last iteration to avoid unnecessary recomputation of the forward pass
                retain_graph=not (last_target and last_chunk),
                # we do not create a new graph as we do not need higher order gradients
                create_graph=False,
            )[0][chunk_batch_slice, chunk.sequence_slice].detach()

            if self.input_x_gradient:
                # we multiply the input embeddings with their gradients before reducing them
                target_wise_grads: Float[torch.Tensor, f"{c} ..."] = target_wise_grads * inputs_chunks.detach()

            if target_wise_grads.ndim == 4:
                # spatial images: (chunk, channels, h, w) -> (chunk, h*w, channels) renamed (c, l, d)
                target_wise_grads: Float[torch.Tensor, f"{c} l d"] = target_wise_grads.flatten(2).transpose(1, 2)

            if target_wise_grads.ndim == 3:  # for images and text, this is always true
                # language sequence, patchified images, and flattened spatial images: (c, l, d) -> (c, l)
                # we average the gradients over the model width dimension of input_embeds to save memory
                # TODO: this aggregation is arbitrary and hardcoded, it should be discussed
                target_wise_grads: Float[torch.Tensor, f"{c} l"] = target_wise_grads.mean(dim=-1)

            gradients_list.append(target_wise_grads)

        # stack target-wise gradients
        # TODO: check when everything should be moved to CPU
        return torch.stack(gradients_list, dim=1)

    def _prepare_inputs(self, inputs: list[TensorMapping], for_gradients: bool = False) -> TensorMapping:
        """
        Prepare the inputs for the model:
        - pad
        - stack
        - put on device
        - set requires_grad to True for inputs_embeds if gradients are required

        Args:
            inputs (list[TensorMapping]):
                The inputs to be padded.
            for_gradients (bool, optional):
                Whether the inputs are for gradients or not.
                If True, the inputs are already embedded, the "inputs_embeds" key is present.
                Also, we do not need to include the "input_ids" key.

        Returns:
            TensorMapping: The padded inputs.
        """
        prepared = {}
        image_sequence_keys = {"pixel_attention_mask"}
        text_sequence_keys = {
            "input_ids",
            "inputs_embeds",
            "attention_mask",
            "token_type_ids",
            "mm_token_type_ids",
        }

        for key in inputs[0]:
            if key == "offset_mapping":
                continue

            tensors = [sample[key] for sample in inputs]

            if (key == "pixel_values" and tensors[0].ndim == 2) or key in image_sequence_keys:
                # Patchified images: (P, D) -> (B, P_max, D)
                batched = pad_sequence(
                    tensors,
                    batch_first=True,
                    padding_value=0,
                    padding_side="right",
                )

            elif key in text_sequence_keys:
                # Text and multimodal token sequences: (L,) -> (B, L_max)
                batched = pad_sequence(
                    tensors,
                    batch_first=True,
                    padding_value=self.pad_token_id if key == "input_ids" else 0,
                    padding_side=self.padding_side,
                )

            else:
                # Spatial images (C, H, W), image_grid_thw (3,), etc.
                batched = torch.stack(tensors)

            prepared[key] = batched.to(self.device)

        if for_gradients:
            if self.grad_key == "inputs_embeds":
                # remove the input_ids key as we cannot provide it together with inputs_embeds
                prepared.pop("input_ids", None)

            # ensure inputs_embeds are present and set requires_grad to True
            if self.grad_key not in prepared:
                raise ValueError(
                    "The `model_inputs` should have been embedded when requesting gradients. "
                    "The 'inputs_embeds' key is missing. "
                    "This should never happen, please raise an issue in GitHub."
                )
            prepared[self.grad_key] = prepared[self.grad_key].detach().requires_grad_(True)  # type: ignore
        return prepared

    def _model_forward(self, **inputs) -> torch.Tensor:
        """
        Forward pass through the model.

        Useless by default, but opens an entry-point for easy modification via inheritance.

        Args:
            **inputs: The inputs to be passed to the model.

        Returns:
            torch.Tensor: The output of the model.
        """
        try:
            # try model forward
            return self.mode(self.model(**inputs).logits)
        except NotImplementedError as e:
            key_shapes = {k: v.shape for k, v in inputs.items()}
            raise IncompatibilityError(
                f"The model {self.model.__class__.__name__} is not compatible with the provided inputs. "
                f"Input keys and shapes: {key_shapes}. "
                f"This can happen for T5 when applying gradient-based methods, they do the forward from the `inputs_embeds`. "
            ) from e

    @jaxtyped(typechecker=beartype)
    def _call_batch(
        self, batch: Batch, outputs_buffer: MutableMapping[int, list[torch.Tensor]]
    ) -> Iterator[torch.Tensor]:
        """
        Run one model batch, dispatch logits to their original groups,
        and return the groups that became complete during this flush.

        Args:
            batch (Batch):
                The batch to be processed.
                It includes metadata allowing to group back outputs distributed across batches
            outputs_buffer (MutableMapping[int, list[torch.Tensor]]):
                The buffer to store the outputs.
                A dictionary of output chunks, grouped by group id.
                Once we reach the last chunk of a group, we yield the concatenated outputs.

        Returns:
            Iterator[torch.Tensor]:
                The outputs of the model. The `__call__` yields from it.
        """
        # pad and concat individual inputs into a single BatchEncoding
        # also, put them on device
        inputs = self._prepare_inputs(batch.inputs, for_gradients=self.gradients and batch.targets[0] is not None)

        # torch.no_grad activated only if gradients not necessary
        with torch.no_grad() if not self.gradients else nullcontext():
            # call model forward
            logits: Float[torch.Tensor, "b ..."] = self._model_forward(**inputs)  # type: ignore

            # apply output processing chunk by chunk
            # because generation outputs have different size depending on the initial source sample
            for chunk_id, (chunk, chunk_targets) in enumerate(zip(batch.chunks, batch.targets, strict=True)):
                chunk_slice = slice(chunk.batch_start, chunk.batch_end)
                # extract subset of logits corresponding to the chunk
                logits_chunk: Float[torch.Tensor, "c ..."] = logits[chunk_slice]

                if chunk_targets is not None:
                    # select logits through targets indexing
                    targeted_logits_chunk: Float[torch.Tensor, "c t"] = self._target_logits(
                        logits_chunk,
                        chunk_targets.to(self.device),
                        chunk,
                    )

                    if self.gradients:
                        # outputs are gradients
                        outputs_chunk = self._compute_gradients(
                            inputs=inputs,  # type: ignore
                            targeted_logits_chunk=targeted_logits_chunk,
                            chunk=chunk,
                            last_chunk=(chunk_id == (len(batch.chunks) - 1)),
                        )
                    else:
                        # outputs are targeted logits
                        outputs_chunk = targeted_logits_chunk
                else:
                    # outputs are targets
                    # (c, t) compute targets from logits
                    outputs_chunk = self._extract_targets_from_logits(logits_chunk, chunk).cpu()

                outputs_buffer[chunk.group_id].append(outputs_chunk)

                # once we reach the last chunk of a group, yield the concatenated outputs
                if chunk.last_group_chunk:
                    yield torch.cat(outputs_buffer.pop(chunk.group_id), dim=0)
                del logits_chunk
        del batch

    def __call__(
        self,
        model_inputs: Iterable[TensorMapping],
        targets: Iterable[Int[torch.Tensor, "t"]] | None = None,
    ) -> Iterator[torch.Tensor]:
        """
        Most computationally heavy function of the attributions module,
        it manages the batching of the inputs and the calls to the model.

        As most elements in the module, it is a generator.
        It can be a generator of several types of outputs:
            - targets as argmax of logits if targets are not provided
            - targeted logits if targets are provided and self.gradients is False
            - gradients if targets are provided and self.gradients is True

        In all case, the operations are done on a batch.

        This method takes in `model_inputs`, an iterable over groups of perturbed samples.
        Each group can have a different number of elements. Which might not correspond to the `batch_size`.
        To optimize model batch calling, groups might be split between several batches and vice-versa.

        Let's take the example of three model inputs: n1 = 3, n2 = 8, and n3 = 4, with a batch size of 5.
        The batches will be as follows: [[1, 1, 1, 2, 2], [2, 2, 2, 2, 2], [2, 3, 3, 3, 3]]

        The algorithm is the following:
            for group in model_inputs:
                while group not empty:
                    extract chunk from group to fill the batch
                    if batch is full:
                        call the model
                        dispatch outputs to their original groups
                        yield groups that are complete

        Args:
            model_inputs (Iterable[TensorMapping]):
                The inputs grouped by initial samples, could contain `input_ids` or `inputs_embeds`

            targets (Iterable[torch.Tensor] | None):
                Indices used to select the logits to return `targeted_logits` or for gradients computations.
                If missing, they are computed and return as the output.

        Returns:
            An iterator yielding one tensor per input in `model_inputs` iterator.

            The meaning of each yielded tensor depends on the arguments:

            - If `targets is None`, yields the predicted target indices for the group. Shape (t,).
            - If `targets is not None` and `self.gradients` is `False`, yields the
            targeted logits for the group. Shape (p, t).
            - If `targets is not None` and `self.gradients` is `True`, yields the
            gradients associated with the targeted logits for the group. Shape (p, t, l).

            Each yielded tensor contains the outputs reconstructed for a full original
            group of perturbed samples, even if that group was split across several
            model batches.
        """

        outputs_buffer: MutableMapping[int, list[torch.Tensor]] = defaultdict(list)
        current_batch = Batch()

        # include targets if available
        group_iterator = zip(model_inputs, targets, strict=True) if targets else model_inputs

        # iterate on group of perturbed samples (one group per initial input)
        for group_id, grouped_elements in enumerate(group_iterator):
            # each tensor in the group dict correspond to several perturbations
            # we split this in a list of dict where each correspond to one perturbation
            split_inputs, group_targets = _split_group(grouped_elements)
            n = len(split_inputs)

            # While possible construct and send batches from the group
            start = 0
            while start < n:
                # extract a chunk of the group to add to the current batch
                take = min(
                    self.batch_size - len(current_batch),  # either a subset of the group to fill the batch
                    n - start,  # or the full group
                )

                # elements for slicing, it keeps relevant element indices
                sample = split_inputs[start]
                sequence_key, padding_side = None, self.padding_side
                if "inputs_embeds" in sample:
                    sequence_key = "inputs_embeds"
                elif "pixel_values" in sample and sample["pixel_values"].ndim == 2:
                    sequence_key, padding_side = "pixel_values", "right"
                elif "input_ids" in sample:
                    sequence_key = "input_ids"

                # add chunk to the batch
                current_batch.add_chunk(
                    chunk_inputs=split_inputs[start : start + take],
                    chunk_targets=group_targets,
                    group_id=group_id,
                    last_group_chunk=(start + take == n),
                    sequence_key=sequence_key,
                    padding_side=padding_side,
                )

                # count number of elements from the group already chunked into the batches
                start += take

                # if the batch is full, call the model and yield completed outputs
                if len(current_batch) == self.batch_size:
                    yield from self._call_batch(current_batch, outputs_buffer)

                    current_batch = Batch()

        # final call with potentially incomplete batch
        if current_batch:
            yield from self._call_batch(current_batch, outputs_buffer)
