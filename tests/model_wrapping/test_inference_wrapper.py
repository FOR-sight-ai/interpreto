from __future__ import annotations

import math
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from interpreto.model_wrapping.classification_inference_wrapper import ClassificationInferenceWrapper
from interpreto.model_wrapping.generation_inference_wrapper import GenerationInferenceWrapper
from interpreto.model_wrapping.inference_wrapper import Batch

TASK_INFERENCE_WRAPPERS = {
    "classification": ClassificationInferenceWrapper,
    "generation": GenerationInferenceWrapper,
}

TARGET_MAX = 3


class DummyBatchEncoding(dict[str, torch.Tensor]):
    """Minimal batch container supporting the `.to()` call used by the wrappers."""

    def to(self, device: torch.device) -> DummyBatchEncoding:
        for key, value in self.items():
            self[key] = value.to(device)
        return self


class CountingIdentityModel(nn.Module):
    """Return deterministic logits from the given source tensors and count forward calls."""

    def __init__(self, task: str, batch_size=None):
        super().__init__()
        self.task = task
        self.forward_calls = 0
        self.register_buffer("_anchor", torch.zeros(1))
        self.batch_size = batch_size
        self.config = SimpleNamespace(pad_token_id=0)

    @property
    def device(self) -> torch.device:
        return self._anchor.device  # type: ignore

    @property
    def dtype(self) -> torch.dtype:
        return self._anchor.dtype  # type: ignore

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> SimpleNamespace:
        self.forward_calls += 1

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")

            # (b, seq_len, vocab_size)
            inputs_embeds = F.one_hot(input_ids, num_classes=TARGET_MAX).to(torch.float32)

        if self.batch_size is not None:
            assert inputs_embeds.shape[0] == self.batch_size, "Placeholder model called with wrong batch size"

        if self.task == "classification":
            # (b, n_classes)
            # we take the sum over the sequence length dimension
            # so from token-ids, it basically is a value count
            # [0, 2, 1, 1] -> [1, 2, 1]
            logits = inputs_embeds.sum(dim=1)

        elif self.task == "generation":
            # (b, seq_len, vocab_size)
            logits = torch.roll(inputs_embeds, shifts=-1, dims=1)

        else:
            raise NotImplementedError("Unknown task")

        return SimpleNamespace(logits=logits)


def split_group(group: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    n_samples = next(iter(group.values())).shape[0]
    return [{key: value[sample_idx] for key, value in group.items()} for sample_idx in range(n_samples)]


@pytest.mark.parametrize(
    "task, source, behavior",
    [
        ("classification", "input_ids", "targets"),
        ("classification", "input_ids", "targeted_logits"),
        ("classification", "inputs_embeds", "gradients"),
        # Constructing targets from logits is not supported for generation.
        ("generation", "input_ids", "targeted_logits"),
        ("generation", "inputs_embeds", "gradients"),
    ],
)
def test_call_batch_behaviors(task: str, source: str, behavior: str):
    """
    Test the behavior of the `_call_batch` method.

    This method manages the following different behaviors:
        - extracting targets when no targets are provided,
        - selecting targeted logits,
        - computing gradients of those targeted logits.
    """
    # `_call_batch` owns the behavioral split between:
    #   - extracting targets when no targets are provided,
    #   - selecting targeted logits,
    #   - computing gradients of those targeted logits.
    # Non-gradient branches should therefore use `input_ids`, which is the real wrapper path.
    targets = None if behavior == "targets" else torch.tensor([2, 0], dtype=torch.long)
    model = CountingIdentityModel(task)
    wrapper = TASK_INFERENCE_WRAPPERS[task](
        model=model,
        gradients=(behavior == "gradients"),
        input_x_gradient=True,
        device=torch.device("cpu"),
    )

    # set predefined inputs
    input_ids = torch.tensor([1, 0, 2, 1], dtype=torch.long)
    # [0, 2, 1, 1] -> [[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    inputs_embeds = F.one_hot(input_ids, num_classes=TARGET_MAX).to(torch.float32)
    attention_mask = torch.ones_like(input_ids)
    group = {"input_ids": input_ids, "inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
    # targets correspond to the two last token ids
    targets = torch.tensor([2, 1], dtype=torch.long) if behavior != "targets" else None

    # construct batch as expected by inference wrappers
    batch = Batch()
    batch.add_chunk([group], targets, group_id=0, last_group_chunk=True)

    # call the wrapper on a single batch
    outputs = list(wrapper._call_batch(batch, defaultdict(list)))

    assert model.forward_calls == 1, "Model was called too many times"
    assert len(outputs) == 1, (
        f"Inference wrapper for {task} returned more than one output for one input: {len(outputs)}"
    )

    if behavior == "targets":
        # we know its classification
        # we expect classification logits argmax
        assert torch.allclose(outputs[0], torch.tensor([1], dtype=torch.long)), "Classification targets are not argmax"
    elif behavior == "targeted_logits":
        if task == "classification":
            # extract logits (value count) with targets indices
            # targets[0] = 2, 2 appears once, so expected[0] = 1
            # targets[1] = 1, 1 appears twice, so expected[1] = 2
            assert torch.allclose(outputs[0], torch.tensor([1, 2], dtype=torch.float32)), (
                "Classification targeted logits are not correct"
            )
        elif task == "generation":
            # extract logits (one hot) with targets indices
            # the targets correspond to the shifted input_ids, so the targeted one hot are ones
            assert torch.allclose(outputs[0], torch.tensor([1, 1], dtype=torch.float32)), (
                "Generation targeted logits are not correct"
            )
        else:
            raise ValueError(f"Unknown task {task}")
    elif behavior == "gradients":
        # (2, 4)
        assert outputs[0].shape == (1, len(targets), len(input_ids)), (  # type: ignore
            f"Gradients have wrong shape for task {task}"
        )
        if task == "classification":
            expected = [[0, 0, 1, 0], [1, 0, 0, 1]]  # first is position of the 2 and second the position of the ones
            assert torch.allclose(outputs[0], torch.tensor(expected, dtype=torch.float32) * (1 / TARGET_MAX)), (
                "Classification gradients are not correct"
            )
        elif task == "generation":
            # generated token corresponds to the previous token, and the targets are the two lasts
            expected = [[0, 0, 1, 0], [0, 0, 0, 1]]
            assert torch.allclose(outputs[0], torch.tensor(expected, dtype=torch.float32) * (1 / TARGET_MAX)), (
                "Generation gradients are not correct"
            )
        else:
            raise ValueError(f"Unknown task {task}")
    else:
        raise ValueError(f"Unknown behavior {behavior}")


@pytest.mark.parametrize("task", TASK_INFERENCE_WRAPPERS.keys())
def test_batching_management(task: str):
    """
    Test the input elements are correctly distributed across batches.

    Which means, that the model is called the right number of times.

    Following inference wrapper `__call__` docstring, we make 3 groups of inputs of sizes 3, 8, and 4.
    They should be distributed as follows:
        [[1, 1, 1, 2, 2], [2, 2, 2, 2, 2], [2, 3, 3, 3, 3]]
    """
    # parameters
    batch_size = 5
    nb_samples_groups = [3, 8, 4]
    nb_calls = math.ceil(sum(nb_samples_groups) / batch_size)

    # counter model and wrapper
    model = CountingIdentityModel(task, batch_size=batch_size)
    wrapper = TASK_INFERENCE_WRAPPERS[task](
        model=model,
        batch_size=batch_size,
        device=torch.device("cpu"),
    )

    # define inputs of requested sizes
    groups = [{"input_ids": torch.eye(n, dtype=torch.long)} for n in nb_samples_groups]
    torch.manual_seed(0)
    targets = [torch.randint(0, TARGET_MAX, (math.ceil(n / 2),), dtype=torch.long) for n in nb_samples_groups]

    outputs = list(wrapper(groups, targets))

    assert model.forward_calls == nb_calls, "Model was called too many times"
    assert len(outputs) == len(groups)

    for out, n in zip(outputs, nb_samples_groups, strict=True):
        assert out.shape == (n, math.ceil(n / 2)), "Output shape is not correct"


BAB_CONFIGS = [
    BitsAndBytesConfig(load_in_8bit=True),
    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_type=torch.float16),
    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_type=torch.float16),
    BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0),
    BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head", "output"]),
]

QUANTIZED_MODELS = [
    "hf-internal-testing/tiny-random-gpt2",
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes quantization requires CUDA")
@pytest.mark.parametrize("model_name", QUANTIZED_MODELS)
@pytest.mark.parametrize("bab_config", BAB_CONFIGS)
def test_inference_wrapper_with_quantized_models(model_name, bab_config):
    pytest.importorskip("bitsandbytes")

    nb_inputs = 3
    nb_outputs = 2

    try:
        quantized_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bab_config)
    except RuntimeError as error:
        if "CUDA SETUP ERROR" in str(error) or "automatic conversion of the weights" in str(error):
            pytest.skip(f"bitsandbytes runtime is not available in this environment: {error}")
        raise

    wrapped_model = GenerationInferenceWrapper(quantized_model)

    input_ids = torch.arange(0, nb_inputs + nb_outputs, dtype=torch.long).view((1, -1))
    attention_mask = torch.ones_like(input_ids)
    input_ids_dict = [{"input_ids": input_ids, "attention_mask": attention_mask}]
    input_embeddings = quantized_model.get_input_embeddings()(input_ids.to(device=quantized_model.device)).to(
        dtype=torch.float32
    )
    input_embeddings_dict = [{"inputs_embeds": input_embeddings, "attention_mask": attention_mask}]
    targets = [torch.zeros((nb_outputs,), dtype=torch.long)]

    outputs_from_ids = next(wrapped_model(input_ids_dict, targets))
    outputs_from_embeds = next(wrapped_model(input_embeddings_dict, targets))

    assert outputs_from_ids is not None
    assert outputs_from_embeds is not None
    assert isinstance(outputs_from_ids, torch.Tensor)
    assert isinstance(outputs_from_embeds, torch.Tensor)
    assert outputs_from_ids.shape == (1, nb_outputs)
    assert outputs_from_embeds.shape == (1, nb_outputs)
    assert torch.allclose(outputs_from_ids, outputs_from_embeds, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    # test_call_batch_behaviors("classification", "input_ids", "targets")
    # test_call_batch_behaviors("classification", "input_ids", "targeted_logits")
    # test_call_batch_behaviors("classification", "inputs_embeds", "gradients")
    # test_call_batch_behaviors("generation", "input_ids", "targeted_logits")
    # test_call_batch_behaviors("generation", "inputs_embeds", "gradients")
    # test_batching_management("classification")
    # test_batching_management("generation")

    if torch.cuda.is_available():
        import itertools

        for model_name, bab_config in itertools.product(QUANTIZED_MODELS, BAB_CONFIGS):
            test_inference_wrapper_with_quantized_models(model_name, bab_config)
