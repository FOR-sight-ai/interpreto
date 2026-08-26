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

from interpreto import AllLayersSplitter


def test_extracts_every_bert_and_gpt2_layer(bert_model, bert_tokenizer, gpt2_model, gpt2_tokenizer):
    """BERT and GPT-2 expose the input stream and every block output in order."""
    for model, tokenizer, layer_path in (
        (bert_model, bert_tokenizer, "bert.encoder.layer"),
        (gpt2_model, gpt2_tokenizer, "transformer.h"),
    ):
        splitter = AllLayersSplitter(model, tokenizer=tokenizer)
        activations = splitter.get_activations("Interpreto is useful.")

        assert splitter.layer_split_points == [
            f"{layer_path}.{index}" for index in range(len(model.get_submodule(layer_path)))
        ]
        assert len(activations) == len(splitter.layer_split_points) + 1
        assert all(activation.ndim == 2 and activation.shape == activations[0].shape for activation in activations)
