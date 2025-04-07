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

from interpreto.commons.generator_tools import enumerate_generator
from interpreto.commons.model_wrapping.inference_wrapper import InferenceWrapper


class ClassificationInferenceWrapper(InferenceWrapper):
    """
        Basic inference wrapper for classification tasks.
    """

    # Padding is done on the right for classification tasks
    PAD_LEFT = False

    def _process_target(self, target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Process the target tensor to match the shape of the logits tensor.

        Args:
            target (torch.Tensor): target tensor
            logits (torch.Tensor): logits tensor

        Raises:
            ValueError: if the target tensor has more than 2 dimensions

        Returns:
            torch.Tensor: processed target tensor
        """
        # TODO : find another way to do that
        match target.dim():
            case 0:
                # if target is a scalar, we add a (t) dimension and reprocess it
                return self._process_target(target.unsqueeze(0), logits)
            case 1:  # (t)
                # if target tensor count only one dimension, we repeat it along the batch dimension
                index_shape = list(logits.shape)
                index_shape[-1] = target.shape[0]
                return target.expand(index_shape)
            case 2:  # (n, t)
                if target.shape[0] == 1:
                    # if target tensor batch dimension is one, we repeat it
                    return target.expand(logits.shape[0], -1)
                return target
        raise ValueError(f"Target tensor should have 0, 1 or 2 dimensions, but got {target.dim()} dimensions")

    @singledispatchmethod
    def get_targets(self, model_inputs:Any)-> torch.Tensor|Generator[torch.Tensor, None, None]:
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
                >>> print(target)

            Sequence of inputs given as an iterable of mappings (generator, list, etc.)
                >>> model_inputs = [{"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])},
                ...                 {"input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]])}]
                >>> targets = wrapper.get_targets(model_inputs)
                >>> for target in targets:
                ...     print(target)

        """
        raise NotImplementedError(
            f"type {type(model_inputs)} not supported for method get_targets in class {self.__class__.__name__}"
        )

    @get_targets.register(MutableMapping)
    def _get_targets_from_mapping(self, model_inputs: MutableMapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the target from the model for the given inputs.
        registered for MutableMapping type.

        Args:
            model_inputs (MutableMapping[str, torch.Tensor]): input mapping containing either "input_ids" or "inputs_embeds".

        Returns:
            torch.Tensor: target predicted by the model for the given input mapping.
        """
        return self._get_logits_from_mapping(model_inputs).argmax(dim=-1)

    @get_targets.register(Iterable)
    def _get_targets_from_iterable(self, model_inputs: Iterable[MutableMapping[str, torch.Tensor]]) -> Generator[torch.Tensor, None, None]:
        """
        Get the targets from the model for the given inputs.
        registered for Iterable type.

        Args:
            model_inputs (Iterable[MutableMapping[str, torch.Tensor]]): _description_

        Yields:
            torch.Tensor: target predicted by the model for the given input mapping.
        """
        yield from (prediction.argmax(dim=-1) for prediction in self._get_logits_from_iterable(iter(model_inputs)))

    @singledispatchmethod
    def get_targeted_logits(self, model_inputs:Any, targets: torch.Tensor)-> torch.Tensor|Generator[torch.Tensor, None, None]:
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
    def _get_targeted_logits_from_mapping(self, model_inputs: MutableMapping[str, torch.Tensor], targets: torch.Tensor):
        """
        Get the logits associated to a collection of targets.
        registered for MutableMapping type.

        Args:
            model_inputs (MutableMapping[str, torch.Tensor]): input mappings to be passed to the model
            targets (torch.Tensor): target tensor to be used to get the logits.
            targets shape should be either (t) or (n, t) where n is the batch size and t is the number of targets for which we want the logits.

        Returns:
            torch.Tensor: logits given by the model for the given targets.
        """
        logits = self._get_logits_from_mapping(model_inputs)
        targets = self._process_target(targets, logits)
        return logits.gather(-1, targets)

    @get_targeted_logits.register(Iterable)
    def _get_targeted_logits_from_iterable(self, model_inputs: Iterable[MutableMapping[str, torch.Tensor]], targets: torch.Tensor)->Generator[torch.Tensor, None, None]:
        """
        Get the logits associated to a collection of targets.
        registered for Iterable type.

        Args:
            model_inputs (Iterable[MutableMapping[str, torch.Tensor]]): iterable of input mappings to be passed to the model
            targets (torch.Tensor): target tensor to be used to get the logits.
            targets shape should be either (t) or (n, t) where n is the batch size and t is the number of targets for which we want the logits.

        Yields:
            torch.Tensor: logits given by the model for the given targets.
        """
        predictions = self._get_logits_from_iterable(iter(model_inputs))
        # TODO : refaire ça proprement
        if targets.dim() in (0, 1):
            targets = targets.view(1, -1)
        single_index = int(targets.shape[0] > 1)
        for index, logits in enumerate_generator(predictions):
            yield logits.gather(-1, targets[single_index and index].unsqueeze(0).expand(logits.shape[0], -1))

    @singledispatchmethod
    def get_gradients(self, model_inputs:Any, targets: torch.Tensor):
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

    @get_gradients.register(MutableMapping)
    def _get_gradients_from_mapping(self, model_inputs: MutableMapping[str, torch.Tensor], targets: torch.Tensor)->torch.Tensor:
        """
        Get the gradients of the logits associated to a given target with respect to the inputs.
        registered for MutableMapping type.
        This method uses torch.autograd.functional.jacobian to compute the gradients.

        Args:
            model_inputs (Iterable[MutableMapping[str, torch.Tensor]]): input mapping to be passed to the model
            targets (torch.Tensor): target tensor to be used to get the logits.

        Returns:
            torch.Tensor: gradient of the output of the model with respect to the given inputs
        """
        # calculate embeddings if needed
        model_inputs = self.embed(model_inputs)

        # function to differentiate
        def f(embed):
            return self._get_targeted_logits_from_mapping(embed, targets)

        # return the jacobian of the function for a given input
        return (
            torch.autograd.functional.jacobian(f, model_inputs["inputs_embeds"], create_graph=True, strict=False)
            .sum(dim=1)
            .abs()
            .mean(axis=-1)
        )

    @get_gradients.register(Iterable)
    def _get_gradients_from_iterable(self, model_inputs: Iterable[MutableMapping[str, torch.Tensor]], targets: torch.Tensor)->Generator[torch.Tensor, None, None]:
        """
        Get the gradients of the logits associated to a given target with respect to the inputs.
        registered for Iterable type.

        Args:
            model_inputs (Iterable[MutableMapping[str, torch.Tensor]]): iterable input mappings to be passed to the model
            targets (torch.Tensor): target tensor to be used to get the logits.

        Yields:
            torch.Tensor: gradients of the logits associated to the given target with respect to the inputs.
        """
        # TODO : see if we can do that in a more efficient way
        yield from (
            self.get_gradients(model_input, target)
            for model_input, target in zip(model_inputs, targets, strict=True)
        )
