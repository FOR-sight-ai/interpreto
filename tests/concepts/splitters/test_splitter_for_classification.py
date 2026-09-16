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

import pytest
import torch

from interpreto import SplitterForClassification as SSC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_IDS = [
    "hf-internal-testing/tiny-random-bert",
    "hf-internal-testing/tiny-random-gpt2",
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "hf-internal-testing/tiny-random-roberta",
    "hf-internal-testing/tiny-random-t5",
]


@pytest.fixture(scope="module")
def split_seq_cls():
    return SSC(
        "hf-internal-testing/tiny-random-bert",
        batch_size=2,
        device_map=DEVICE,
    )


def test_loading_possibilities(bert_model, bert_tokenizer):
    """
    Test loading model with and without split points
    """
    # wrong module name
    with pytest.raises(ValueError):
        SSC(bert_model, split_point="wrong.module.name", tokenizer=bert_tokenizer)

    # tokenizer inferred from the model repository
    splitter = SSC(bert_model)
    assert splitter.tokenizer is not None

    # correct module name
    splitter = SSC(bert_model, split_point="classifier", tokenizer=bert_tokenizer)
    assert splitter.split_point == "classifier", (
        f"split_point mismatch: got {splitter.split_point}, expected 'classifier'"
    )

    # no module name
    splitter = SSC("hf-internal-testing/tiny-random-bert")
    assert splitter.split_point == "classifier", (
        f"split_point mismatch: got {splitter.split_point}, expected 'classifier'"
    )


def test_get_latent_shape(split_seq_cls: SSC):
    """The latent shape contains one classification representation."""
    shape = split_seq_cls.get_latent_shape()
    expected_shape = (1, split_seq_cls.config.hidden_size)
    assert shape == expected_shape, f"Latent shape mismatch: got {shape}, expected {expected_shape}"


@pytest.mark.parametrize(
    "repo_id",
    [
        "hf-internal-testing/tiny-random-bert",
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "hf-internal-testing/tiny-random-t5",
    ],
)
def test_standalone_token_ids_use_tokenizer_formatting(repo_id):
    """Standalone IDs use each tokenizer's special-token and padding policy."""
    splitter = SSC(repo_id, batch_size=2, device_map=DEVICE)
    token_ids = [
        token_id
        for token, token_id in splitter.tokenizer.get_vocab().items()
        if token_id not in splitter.tokenizer.all_special_ids
        and splitter.tokenizer.encode(token, add_special_tokens=False) == [token_id]
    ][:3]

    prepared = splitter._prepare_batch(token_ids, {})
    template, token_position = splitter._get_standalone_token_template()
    assert prepared["input_ids"][:, token_position].tolist() == token_ids
    for name, value in template.items():
        if name != "input_ids":
            assert torch.equal(prepared[name], value.repeat(len(token_ids), 1))

    direct_activations = splitter.inputs_to_activations(token_ids)
    activations, _ = splitter.get_activations(token_ids)
    tensor_activations, _ = splitter.get_activations(prepared["input_ids"])
    assert direct_activations.shape == (len(token_ids), splitter.config.hidden_size)
    assert activations.shape == tensor_activations.shape == (len(token_ids), splitter.config.hidden_size)


@pytest.mark.parametrize("repo_id", REPO_IDS)
def test_get_activation_and_gradient(repo_id, sentences):
    """
    Test that the `get_activations` and `_get_concept_output_gradients` methods return the expected shapes.
    """
    # --------------------------------------------------------
    # Setup the model with split points, tokenizer, and tokens
    splitter = SSC(
        repo_id,
        batch_size=2,
        device_map=DEVICE,
    )
    # -----------------------------------------------------------
    # Define expected shapes for the different granularity levels
    batch = len(sentences)
    hidden = splitter.config.hidden_size

    # ----------------------------------------------
    # Define a concept encoder/decoder weight matrix
    # We want W@W.T to be approximately the identity matrix
    nb_concepts = 2 * hidden
    initial = torch.randn(nb_concepts, hidden)
    decoder_weights = torch.linalg.qr(initial)[0].to(DEVICE)
    encoder_weights = decoder_weights.T

    expected_activations_shape = (batch, hidden)
    expected_gradients_shape = (1, nb_concepts)

    # ---------------
    # Get activations
    # activations
    activations, predictions = splitter.get_activations(sentences)
    assert activations is not None, "get_activations returned None"
    assert activations is not None, "Activations are None"
    assert activations.shape == expected_activations_shape, (  # type: ignore
        f"Activations shape mismatch: got {tuple(activations.shape)}, "  # type: ignore
        f"expected {expected_activations_shape}"
    )
    assert activations.dtype == torch.float32

    # predictions
    assert predictions is not None, "Predictions are None"
    assert predictions.shape[0] == expected_activations_shape[0], (  # type: ignore
        f"Predictions batch mismatch: got {predictions.shape[0]}, "  # type: ignore
        f"expected {expected_activations_shape[0]}"
    )

    assert splitter.get_latent_shape() == torch.Size([1, hidden])

    expected_logits = []
    with torch.no_grad():
        for i in range(0, len(sentences), splitter.batch_size):
            with splitter.trace(sentences[i : i + splitter.batch_size]):
                batch_logits = splitter.output.logits.save()
            expected_logits.append(batch_logits)
    expected_logits = torch.cat(expected_logits)

    replayed_logits = splitter.activations_to_outputs(activations)
    assert replayed_logits.shape == expected_logits.shape
    assert torch.allclose(replayed_logits.cpu(), expected_logits.cpu(), atol=1e-5)
    assert torch.equal(replayed_logits.argmax(dim=-1).cpu(), predictions.cpu())

    # -------------
    # Get gradients
    grads_list = splitter._get_concept_output_gradients(
        sentences,
        activations_to_concepts=lambda x: x @ encoder_weights,
        concepts_to_activations=lambda x: x @ decoder_weights,
        targets=None,
    )
    assert grads_list is not None, "_get_concept_output_gradients returned None"
    assert len(grads_list) == len(sentences), (
        f"Gradients list length mismatch: got {len(grads_list)}, expected {len(sentences)}"
    )  # there should be as many gradients as inputs
    for grads in grads_list:
        assert grads is not None, "A gradients tensor is None"
        # we expect the shape of the gradients to be (t, 1, c)
        # with t the number of targets, 1, and c the number of concepts
        assert grads.shape[1:] == expected_gradients_shape, (
            f"Gradient shape mismatch: got {grads.shape}, expected {expected_gradients_shape}"
        )  # number of granularity elements


def test_batching(split_seq_cls: SSC, huge_text: list[str]):
    split_seq_cls.get_activations(huge_text)


if __name__ == "__main__":
    test_get_activation_and_gradient(
        "hf-internal-testing/tiny-random-roberta",
        sentences=["Hello world!", "Can you hear me?", "Yes, who is it?", "It's me!"],
    )
