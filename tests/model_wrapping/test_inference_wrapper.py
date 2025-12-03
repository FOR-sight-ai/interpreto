import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from interpreto.model_wrapping.generation_inference_wrapper import GenerationInferenceWrapper

# Define quantization configurations
BAB_CONFIGS = [
    BitsAndBytesConfig(load_in_8bit=True),
    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_type=torch.float16),
    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_type=torch.float16),
    BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0),
    BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head", "output"]),
]

# Define models to test
MODELS = [
    "hf-internal-testing/tiny-random-gpt2",
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
]


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("bab_config", BAB_CONFIGS)
def test_inference_wrapper_with_quantized_models(model_name, bab_config):
    """
    Test the GenerationInferenceWrapper with quantized HuggingFace models.
    """
    # Load the quantized model
    quantized_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bab_config)

    # Wrap the model
    wrapped_model = GenerationInferenceWrapper(quantized_model)

    # Prepare dummy input
    input_ids = torch.tensor([[1, 2, 3, 4]])
    input_embeddings = quantized_model.get_input_embeddings()(input_ids.to(device=quantized_model.device)).to(
        dtype=torch.float32
    )

    # Perform inference
    outputs_from_ids = wrapped_model.get_logits({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})
    outputs_from_embeds = wrapped_model.get_logits(
        {"inputs_embeds": input_embeddings, "attention_mask": torch.ones_like(input_ids)}
    )

    # Assertions
    assert outputs_from_ids is not None
    assert outputs_from_embeds is not None
    assert isinstance(outputs_from_ids, torch.Tensor)
    assert isinstance(outputs_from_embeds, torch.Tensor)
    assert outputs_from_ids.shape[0] == input_ids.shape[0] == outputs_from_embeds.shape[0]  # Ensure batch size matches
