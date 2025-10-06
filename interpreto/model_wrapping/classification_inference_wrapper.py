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
Base class for classification inference wrappers.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, MutableMapping
from functools import singledispatchmethod
from typing import Any

import torch

from interpreto.model_wrapping.inference_wrapper import InferenceWrapper
from interpreto.typing import TensorMapping


class ClassificationInferenceWrapper(InferenceWrapper):
    """
    Basic inference wrapper for classification tasks.
    """

    # Padding is done on the right for classification tasks
    PAD_LEFT = False

    @staticmethod
    def process_target(
        target: torch.Tensor, batch_dims: tuple[int] | torch.Size
    ) -> torch.Tensor:  # TODO: find another name, this one is used in base attribution explainer
        """
        Process the target tensor to match the shape of the logits tensor.

        Args:
            target (torch.Tensor): target tensor
            batch_dims (tuple[int] | torch.Size): batch dimension of the logits tensor

        Raises:
            ValueError: if the target tensor has more than 2 dimensions

        Returns:
            torch.Tensor: processed target tensor
        """
        n = 1
        view_index = [1 for _ in batch_dims]
        if target.dim() == 2:
            n = target.shape[0]
            assert n in (1, batch_dims[0]), (
                f"target batch size {n} should be either 1 or logits batch size ({batch_dims[0]})"
            )
            view_index[0] = n
        target = target.view(*view_index, -1)
        return target.expand(*batch_dims, -1)

    @singledispatchmethod
    def get_targets(self, model_inputs: Any) -> torch.Tensor | Generator[torch.Tensor, None, None]:
        """
        Get the predicted target from the model inputs.

        This method propose two different treatments of the inputs:
        If the input is a mapping, it will be processed as a single input and given directly to the model.
        The method will return the predicted target as a torch.Tensor.

        If the input is an iterable of mappings, it will be processed as a batch of inputs.
        The method will yield the targets of the model for each input as a torch.Tensor.

        Args:
            model_inputs (Any): input mappings to be passed to the model or iterable of input mappings.

        Raises:
            NotImplementedError: If the input type is not supported.

        Returns:
            torch.Tensor | Generator[torch.Tensor, None, None]: logits associated to the input mappings.

        Example:
            Single input given as a mapping
                >>> model_inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}
                >>> target = wrapper.get_targets(model_inputs)

            Sequence of inputs given as an iterable of mappings (generator, list, etc.)
                >>> model_inputs = [{"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])},
                ...                 {"input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]])}]
                >>> targets = wrapper.get_targets(model_inputs)

        """
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_targets in class {self.__class__.__name__}"
        )

    @get_targets.register(MutableMapping)  # TODO: remove single dispatch, `_get_targets_from_mapping` never called
    def _get_targets_from_mapping(self, model_inputs: TensorMapping) -> torch.Tensor:
        """
        Get the target from the model for the given inputs.
        registered for MutableMapping type.

        Args:
            model_inputs (TensorMapping): input mapping containing either "input_ids" or "inputs_embeds".

        Returns:
            torch.Tensor: target predicted by the model for the given input mapping.
        """
        return self._get_logits_from_mapping(model_inputs).argmax(dim=-1)

    @get_targets.register(Iterable)
    def _get_targets_from_iterable(self, model_inputs: Iterable[TensorMapping]) -> Generator[torch.Tensor, None, None]:
        """
        Get the targets from the model for the given inputs.
        registered for Iterable type.

        Args:
            model_inputs (Iterable[TensorMapping]): _description_

        Yields:
            torch.Tensor: target predicted by the model for the given input mapping.
        """
        yield from (prediction.argmax(dim=-1) for prediction in self._get_logits_from_iterable(model_inputs))

    @singledispatchmethod  # TODO: evaluate necessity of single dispacth, always `Iterable` in the explainers and `MutableMapping` for gradients
    def get_targeted_logits(
        self,
        model_inputs: Any,
        targets: torch.Tensor | Iterable[torch.Tensor],
    ) -> torch.Tensor | Generator[torch.Tensor, None, None]:
        """
        Get the logits associated to a collection of targets.

        Args:
            model_inputs (Any): input mappings to be passed to the model or iterable of input mappings.
            targets (torch.Tensor): target tensor to be used to get the logits.
            targets shape should be either (t) or (n, t) where n is the batch size and t is the number of targets for which we want the logits.

        Raises:
            NotImplementedError: If the input type is not supported.

        Returns:
            torch.Tensor|Generator[torch.Tensor, None, None]: logits selected for the given targets.

        Example:
            Single input given as a mapping
                >>> model_inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}
                >>> targets = torch.tensor([1, 2])
                >>> target_logits = wrapper.get_targeted_logits(model_inputs, targets)
                >>> print(target_logits)

            Sequence of inputs given as an iterable of mappings (generator, list, etc.)
                >>> model_inputs = [{"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])},
                ...                 {"input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]])}]
                >>> targets = torch.tensor([[1, 2], [3, 4]])
                >>> target_logits = wrapper.get_targeted_logits(model_inputs, targets)
                >>> for logits in target_logits:
                ...     print(logits)
        """
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_target_logits in class {self.__class__.__name__}"
        )

    @get_targeted_logits.register(MutableMapping)
    def _get_targeted_logits_from_mapping(
        self,
        model_inputs: TensorMapping,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the logits associated to a collection of targets.
        registered for MutableMapping type.

        Args:
            model_inputs (TensorMapping): input mappings to be passed to the model
            targets (torch.Tensor): target tensor to be used to get the logits.
            targets shape should be either (t) or (n, t) where n is the batch size and t is the number of targets for which we want the logits.

        Returns:
            torch.Tensor: logits given by the model for the given targets.
        """
        logits = self._get_logits_from_mapping(model_inputs)

        # Apply post-processing mode
        logits = self.mode(logits)
        targets = self.process_target(targets, logits.shape[:-1])
        return logits.gather(-1, targets)

    @get_targeted_logits.register(Iterable)
    def _get_targeted_logits_from_iterable(
        self,
        model_inputs: Iterable[TensorMapping],
        targets: Iterable[torch.Tensor],
    ) -> Generator[torch.Tensor, None, None]:
        """
        Get the logits associated to a collection of targets.
        registered for Iterable type.

        Args:
            model_inputs (Iterable[TensorMapping]): iterable of input mappings to be passed to the model
            targets (torch.Tensor): target tensor to be used to get the logits.
            targets shape should be either (t) or (n, t) where n is the batch size and t is the number of targets for which we want the logits.

        Yields:
            torch.Tensor: logits given by the model for the given targets.
        """
        predictions = self._get_logits_from_iterable(model_inputs)
        for logits, target in zip(predictions, targets, strict=True):
            logits_mode = self.mode(logits)
            yield logits_mode.gather(-1, self.process_target(target, logits_mode.shape[:-1]))
        # for index, logits in enumerate(predictions):
        #    yield logits.gather(-1, self.process_target(targets[multiple_index and index], logits.shape[:-1]))
