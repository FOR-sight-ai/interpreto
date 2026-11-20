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

"""Session fixtures reserved for NNsight-backed concept tests.

NNsight instruments models in place. Override the root text model fixtures with
independent instances so instrumentation cannot leak into attribution tests,
which deep-copy embedding layers onto CPU. Tokenizers are independent too,
because splitters may change their padding configuration.
"""

import torch
from pytest import fixture
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from interpreto import SplitterForClassification, TextTokensSplitter
from interpreto.typing import LatentActivations

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@fixture(scope="session")
def bert_model():
    return AutoModelForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-bert").to(DEVICE)


@fixture(scope="session")
def bert_tokenizer():
    return AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")


@fixture(scope="session")
def gpt2_model():
    return AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(DEVICE)


@fixture(scope="session")
def gpt2_tokenizer():
    return AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")


@fixture(scope="session")
def splitted_encoder_ml(bert_model, bert_tokenizer) -> SplitterForClassification:
    """Reuse the concept-only BERT instance, isolated from attribution models."""
    return SplitterForClassification(
        bert_model,
        tokenizer=bert_tokenizer,
        batch_size=4,
        device_map=DEVICE,
    )


@fixture(scope="session")
def text_tokens_splitter(gpt2_model, gpt2_tokenizer) -> TextTokensSplitter:
    """Expose token representations using the concept-only GPT-2 instance."""
    return TextTokensSplitter(
        gpt2_model,
        tokenizer=gpt2_tokenizer,
        split_point=1,
        task="text-generation",
        batch_size=8,
        device_map=DEVICE,
    )


@fixture(scope="session")
def activations(splitted_encoder_ml: SplitterForClassification, sentences: list[str]) -> LatentActivations:
    """Return one classification representation per input sample."""
    latent_activations, _ = splitted_encoder_ml.get_activations(sentences)
    return latent_activations
