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

import random
import string
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutput

from interpreto import (
    GradientShap,
    IntegratedGradients,
    KernelShap,
    Lime,
    Occlusion,
    Saliency,
    SmoothGrad,
    Sobol,
    SquareGrad,
    VarGrad,
)
from interpreto.commons.granularity import TextGranularity

MODEL_ID = "bhadresh-savani/distilbert-base-uncased-emotion"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0


ATTRIBUTION_METHODS = [
    GradientShap,
    IntegratedGradients,
    KernelShap,
    Lime,
    Occlusion,
    Saliency,
    SmoothGrad,
    Sobol,
    SquareGrad,
    VarGrad,
]

# ==================================================================================================
# Utils for the classification sanity check
EXAMPLES = [
    ("I have been depressed for two years...", "sadness", "depressed"),
    ("I am so happy for you!", "joy", "happy"),
    ("I love you so much.", "love", "love"),
    ("I am outraged by your behavior.", "anger", "outraged"),
    ("You scared me.", "fear", "scared"),
    ("I am surprised by the news.", "surprise", "surprised"),
]


@pytest.fixture(scope="module")
def emotion_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model.eval()
    label_to_id = {label: label_id for label_id, label in model.config.id2label.items()}
    return model, tokenizer, label_to_id


# ==================================================================================================
# Classification sanity checks


@pytest.mark.slow
@pytest.mark.parametrize(
    "method_class",
    ATTRIBUTION_METHODS,
    ids=[method_class.__name__ for method_class in ATTRIBUTION_METHODS],
)
def test_classification_attribution_methods_detect_emotion_word(emotion_model, method_class):
    """
    For the six emotion of the model, we constructed a sentence with a clear word determining the emotion.
    We check that the most important word is indeed the one that determines the emotion.
    """
    set_seed()
    model, tokenizer, _ = emotion_model

    texts = [text for text, _, _ in EXAMPLES]

    explainer = method_class(model, tokenizer=tokenizer)

    set_seed()
    attribution_outputs = explainer.explain(texts)

    for attribution_output, (text, expected_label, expected_word) in zip(attribution_outputs, EXAMPLES, strict=True):
        predicted_label = model.config.id2label[int(attribution_output.targets.item())]
        assert predicted_label == expected_label, (
            f"{MODEL_ID} predicted {predicted_label!r} instead of {expected_label!r} for input {text!r}."
        )

        scores = attribution_output.attributions.squeeze(0)
        scored_words = list(zip(attribution_output.elements, scores.tolist(), strict=True))
        top_word, _ = max(scored_words, key=lambda item: item[1])

        assert top_word == expected_word, (
            f"{method_class.__name__} expected top word {expected_word!r} but got {top_word!r} "
            f"for target {expected_label!r} on input {text!r}. "
            f"Scores: {[(word, round(score, 6)) for word, score in scored_words]}"
        )


# ==================================================================================================
# Utils for the generation sanity check
REPEAT_SEQUENCE_ALPHABET = string.ascii_uppercase
COPY_SHIFT = 2
SEPARATOR_TOKEN = " "


class LetterTokenizer(PreTrainedTokenizer):
    """Minimal character tokenizer for this test: 26 capitals plus a separator/mask space token."""

    vocab_files_names: dict[str, str] = {}

    def __init__(self):
        self.tokens = [SEPARATOR_TOKEN, *string.ascii_uppercase]
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        super().__init__(pad_token=SEPARATOR_TOKEN, mask_token=SEPARATOR_TOKEN)
        self.model_max_length = 2 * COPY_SHIFT + 8

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def all_special_ids(self) -> list[int]:
        # Keep the separator token available to the granularity decomposition.
        return []

    def get_vocab(self) -> dict[str, int]:
        return self.token_to_id.copy()

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        tokens = list(text)
        if any(token not in self.token_to_id for token in tokens):
            unsupported = {token for token in tokens if token not in self.token_to_id}
            raise ValueError(f"Unsupported tokens: {sorted(unsupported)!r}.")
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.token_to_id[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_token[index]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        return token_ids_0 if token_ids_1 is None else [*token_ids_0, *token_ids_1]

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None):
        return ()

    def _decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:  # type: ignore
        return "".join(self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens))

    def __call__(  # type: ignore
        self,
        text: str | list[str],
        return_tensors: str = "pt",
        return_offsets_mapping: bool = False,
        truncation: bool = True,
        padding: bool | str = False,
    ) -> BatchEncoding:
        # We keep a tiny custom __call__ because the slow tokenizer base class does not provide offsets mappings.
        texts = [text] if isinstance(text, str) else text
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        offset_mappings: list[list[tuple[int, int]]] = []

        max_length = max(len(sample) for sample in texts) if texts else 0
        for sample in texts:
            sample_ids = []
            sample_offsets = []
            for index, character in enumerate(sample):
                if character not in self.token_to_id:
                    raise ValueError(f"Unsupported character {character!r}.")
                sample_ids.append(self.token_to_id[character])
                sample_offsets.append((index, index + 1))
            pad_length = max_length - len(sample_ids) if padding else 0
            input_ids.append(sample_ids + [self.pad_token_id] * pad_length)
            attention_masks.append([1] * len(sample_ids) + [0] * pad_length)
            offset_mappings.append(sample_offsets + [(0, 0)] * pad_length)

        data = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }
        if return_offsets_mapping:
            data["offset_mapping"] = torch.tensor(offset_mappings, dtype=torch.long)
        return BatchEncoding(data=data, tensor_type=return_tensors)


class ShiftCopyForCausalLM(nn.Module):
    """Linear toy CausalLM: shift embeddings by 10 positions, then decode with the embedding transpose."""

    def __init__(self, vocab_size: int, shift: int = COPY_SHIFT):
        super().__init__()
        self.shift = shift
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        # Identity embeddings make the transpose an exact inverse on our toy vocabulary.
        self.embedding.weight.data.copy_(torch.eye(vocab_size))
        self.embedding.weight.requires_grad_(False)
        self.config = SimpleNamespace(max_position_embeddings=2 * shift + 8, pad_token_id=0)
        self.generation_config = SimpleNamespace(pad_token_id=0)

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embedding.weight.dtype

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def resize_token_embeddings(self, new_num_tokens: int):
        return self.embedding

    def _shift_embeddings(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs_embeds.shape[1]
        # Position i reads only from i - shift. The prompt therefore needs `shift + 1` tokens:
        # 10 reference letters plus the trailing separator token.
        shift_matrix = torch.zeros(
            (sequence_length, sequence_length), device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        if sequence_length > self.shift:
            indices = torch.arange(self.shift, sequence_length, device=inputs_embeds.device)
            shift_matrix[indices, indices - self.shift] = 1.0
        return torch.einsum("ij,bjd->bid", shift_matrix, inputs_embeds)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            inputs_embeds = self.embedding(input_ids)
        shifted_embeds = self._shift_embeddings(inputs_embeds)  # type: ignore
        logits = torch.einsum("bld,vd->blv", shifted_embeds, self.embedding.weight)
        return CausalLMOutput(logits=logits)  # type: ignore

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = COPY_SHIFT,
        **kwargs,
    ) -> torch.Tensor:
        generated_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(input_ids=generated_ids, attention_mask=attention_mask).logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
        return generated_ids


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_repeat_example(sequence: list[str]) -> tuple[str, str]:
    prompt = "".join(sequence)
    # The trailing separator aligns the last prompt position with the first copied token under the 10-step shift.
    return prompt, SEPARATOR_TOKEN + prompt


@pytest.fixture(scope="module")
def shift_copy_model():
    tokenizer = LetterTokenizer()
    model = ShiftCopyForCausalLM(vocab_size=len(tokenizer)).to(DEVICE)
    model.eval()
    return model, tokenizer


# ==================================================================================================
# Generation sanity checks


@pytest.mark.slow
@pytest.mark.parametrize(
    "method_class",
    ATTRIBUTION_METHODS,
    ids=[method_class.__name__ for method_class in ATTRIBUTION_METHODS],
)
def test_generation_attribution_methods_detect_copied_token(shift_copy_model, method_class):
    """
    We hand make a model which repeats the previous tokens with a specified shift.
    As we generate sequence of the shift length, then it basically repeats the inputs.

    We hand make this, so we only use sequence of letters, each letter is a token.

    So our goal is to see if attribution methods detect that the most important inputs tokens are the one copied.
    """
    set_seed()
    # sample a sequence of the shift length (i.e. "abcde")
    sequence = random.sample(REPEAT_SEQUENCE_ALPHABET, COPY_SHIFT)

    # construct the prompt and the target (i.e. in: "abcde", out: " abcdef")
    prompt, target = build_repeat_example(sequence)

    model, tokenizer = shift_copy_model
    prompt_token_count = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]

    explainer = method_class(
        model, tokenizer=tokenizer, granularity=TextGranularity.TOKEN, batch_size=1, device=DEVICE
    )

    set_seed()
    [attribution_output] = explainer.explain(prompt, target)

    generated_tokens = [tokenizer.decode([token_id]) for token_id in attribution_output.targets.tolist()]
    assert generated_tokens == [SEPARATOR_TOKEN] + sequence, (
        f"Shift copy model generated {generated_tokens!r} instead of {sequence!r} for prompt {prompt!r}."
    )

    for row_index, expected_token in enumerate(sequence):
        scores = attribution_output.attributions[row_index, :prompt_token_count]
        valid_scores = scores.masked_fill(scores.isnan(), float("-inf"))

        # Top-2 is enough here: some methods also rank the separator or adjacent copied context highly.
        top_k = min(2, valid_scores.numel())
        top_indices = torch.topk(valid_scores, k=top_k).indices.tolist()
        top_tokens = [attribution_output.elements[index].strip() for index in top_indices]

        assert expected_token in top_tokens, (
            f"{method_class.__name__} expected input token {expected_token!r} in top-{top_k} but got {top_tokens!r} "
            f"for generated token {generated_tokens[row_index]!r} at position {row_index} on prompt {prompt!r}. "
            f"Prompt scores: {[(token.strip(), None if torch.isnan(score) else round(score.item(), 6)) for token, score in zip(attribution_output.elements[:prompt_token_count], scores, strict=True)]}"
        )


if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model.eval()
    test_classification_attribution_methods_detect_emotion_word((model, tokenizer, None), Saliency)
    # tokenizer = LetterTokenizer()
    # model = ShiftCopyForCausalLM(vocab_size=len(tokenizer)).to(DEVICE)
    # model.eval()
    # test_generation_attribution_methods_detect_copied_token((model, tokenizer), Sobol)
