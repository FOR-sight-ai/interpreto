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


import torch

from interpreto.attributions.base import TextGenerationExplainer


def create_targets_test(tokenizer):
    """
    Create a set of targets for testing the process_targets method.
    """
    # str:
    target1a = "I like kitten"
    target1b = "Interpreto is magic"

    # TensorMapping:
    target2a = tokenizer(target1a, return_tensors="pt", return_offsets_mapping=True)
    target2b = tokenizer([target1b], return_tensors="pt")

    # TensorMapping with multiple elements:
    target2c = tokenizer([target1a, target1b], return_tensors="pt", padding=True, return_offsets_mapping=True)

    # torch.Tensor:
    target3a = target2a["input_ids"]
    target3b = target2b["input_ids"]

    # list of str:
    target41 = [target1a, target1b]

    # list of TensorMapping:
    target42 = [target2a, target2b]
    target42c = [target2a, target2b, target2c]

    # list of torch.Tensor:
    target43 = [target3a, target3b]

    return [
        target1a,
        target2a,
        target3a,  # 3 first targets have 1 element
        target2c,
        target41,
        target42,
        target43,  # 4 next targets hace 2 elements
        target42c,  # the last targets has 4 elements
    ]


def test_process_targets(gpt2_model, gpt2_tokenizer):
    """
    Test the process_targets method for different input types.
    """
    explainer = TextGenerationExplainer(gpt2_model, gpt2_tokenizer, batch_size=2)
    list_targets = create_targets_test(gpt2_tokenizer)

    for target in list_targets:
        results = explainer.process_targets(target)
        assert isinstance(results, list), "The output of the process_targets must be a list"
        assert all(isinstance(result, torch.Tensor) for result in results), (
            "The elements of the list must be of type torch.Tensor."
        )
        assert all(result.dim() == 1 for result in results), "The elements of the list must have 1 dimension."
