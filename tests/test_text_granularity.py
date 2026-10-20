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

from __future__ import annotations

import re
import sys

import pytest
import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

# local import – the module is supplied alongside this test file
from interpreto import Granularity
from interpreto.commons.granularity import GranularityAggregationStrategy

# -------
# Helpers


def _build_expected_matrix(indices: list[list[int]], seq_len: int) -> torch.Tensor:
    """Utility building a bool matrix from *indices* (g × lp)."""
    mat = torch.zeros((len(indices), seq_len), dtype=torch.bool)
    for g, token_pos in enumerate(indices):
        mat[g, token_pos] = True
    return mat


# --------
# Fixtures


@pytest.fixture(scope="session")
def real_bert_tokenizer():
    """Load the BERT‑base uncased tokenizer once for the whole session."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def simple_text():
    """A single‑sentence, single‑paragraph short text."""
    return "word longword verylongword"


@pytest.fixture(scope="module")
def complex_text():
    """5 sentences and 7 part‑sentences, with various punctuation and spacing."""
    return "Although it was raining, we went for a walk. Mr. Smith was with us... We took umbrellas?\nThis is url.com and I love, this !?! Ahah!"


# ----------------------------------------
# ALL_TOKENS / TOKEN / WORD granularities


def test_low_level_granularities_indices(simple_text, real_bert_tokenizer):
    """Checks *get_indices* for ALL_TOKENS / TOKEN / WORD on a tiny text."""

    tokens = real_bert_tokenizer(simple_text, return_tensors="pt", return_offsets_mapping=True)
    seq_len = tokens["input_ids"].shape[1]

    # Human‑readable token list
    print("\nBERT tokens:", real_bert_tokenizer.tokenize(simple_text))

    # ALL_TOKENS
    all_tokens_indices = Granularity.ALL_TOKENS.get_indices(tokens, real_bert_tokenizer)[0]
    assert all_tokens_indices == [[i] for i in range(seq_len)]

    # TOKEN
    token_indices = Granularity.TOKEN.get_indices(tokens, real_bert_tokenizer)[0]
    special = set(real_bert_tokenizer.all_special_ids)
    expected_token_pos = [i for i, tid in enumerate(tokens["input_ids"][0]) if int(tid) not in special]
    assert [idx[0] for idx in token_indices] == expected_token_pos

    # WORD
    word_indices = Granularity.WORD.get_indices(tokens, real_bert_tokenizer)[0]
    # We know there are exactly 3 human words in *simple_text*: "word longword verylongword".
    assert len(word_indices) == 3
    # First word should decode to "word"
    first_word_ids = [int(tokens["input_ids"][0][i]) for i in word_indices[0]]
    assert real_bert_tokenizer.decode(first_word_ids) == "word"


def test_low_level_granularities_matrices_and_decomposition(simple_text, real_bert_tokenizer):
    """Cross‑validate indices ⇄ matrices ⇄ decompositions for low‑level granularities."""

    tokens = real_bert_tokenizer(simple_text, return_tensors="pt", return_offsets_mapping=True)
    seq_len = tokens["input_ids"].shape[1]

    for gran in (Granularity.ALL_TOKENS, Granularity.TOKEN, Granularity.WORD):
        indices = gran.get_indices(tokens, real_bert_tokenizer)[0]

        # Association matrix
        assoc = gran.get_association_matrix(tokens, real_bert_tokenizer)[0]
        expected_mat = _build_expected_matrix(indices, seq_len)
        assert torch.equal(assoc, expected_mat)

        # Decomposition (ids)
        decomp_ids = gran.get_decomposition(tokens, real_bert_tokenizer)[0]
        # Compare raw ids – order + content must match indices
        assert decomp_ids == [[int(tokens["input_ids"][0][i]) for i in grp] for grp in indices]

        # Decomposition (text)
        decomp_text: list[str] = gran.get_decomposition(tokens, real_bert_tokenizer, return_text=True)[0]  # type: ignore
        # Join all segments and strip spaces; must equal original (without specials)
        match gran:
            case Granularity.ALL_TOKENS:
                # We remove the special tokens: decomp_text[1:-1]
                joined = " ".join(seg.strip() for seg in decomp_text[1:-1]).replace(" ##", "")
                assert joined == simple_text
            case Granularity.TOKEN:
                joined = " ".join(seg.strip() for seg in decomp_text).replace(" ##", "")
                assert joined == simple_text
            case Granularity.WORD:
                joined = " ".join(seg.strip() for seg in decomp_text)
                assert joined == simple_text


def test_starts_word():
    assert not Granularity._starts_word("##foo"), "'##foo' cannot be the start of a word"
    assert not Granularity._starts_word("@@bar"), "'@@bar' cannot be the start of a word"
    assert Granularity._starts_word("__init__"), "'__init__' should be considered the start of a word"
    assert Granularity._starts_word("ĠHello"), "'ĠHello' should be considered the start of a word"
    assert Granularity._starts_word(" World"), "' World' should be considered the start of a word"
    assert not Granularity._starts_word("token"), "'token' cannot be the start of a word"


def test_granularity_score_aggregation_alltokens_manual_ids(simple_text, real_bert_tokenizer):
    """Test score aggregation for ALL_TOKENS"""

    tokenizer = real_bert_tokenizer
    tokens = tokenizer(simple_text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = tokens["input_ids"]
    seq_len = input_ids.shape[1]

    # Fake scores = ascending from 0 to seq_len-1
    fake_scores = torch.arange(seq_len).float().unsqueeze(0)

    # ALL_TOKENS → passthrough
    agg_all_tokens = Granularity.ALL_TOKENS.granularity_score_aggregation(
        contribution=fake_scores,
        inputs=tokens,
        tokenizer=tokenizer,
        aggregate_inputs=True,
    )
    assert torch.equal(agg_all_tokens, fake_scores)


def test_granularity_score_aggregation_token_manual_ids(real_bert_tokenizer):
    """
    Test TOKEN-level aggregation using manually constructed input_ids.

    The first token is a special token (e.g., [CLS]), and the remaining tokens are regular.
    The method should exclude the special token and return the scores for the regular tokens only.
    """

    tokenizer = real_bert_tokenizer
    special_ids = set(tokenizer.all_special_ids)

    # Ensure the tokenizer has special tokens
    assert special_ids is not None, "Tokenizer should have special tokens defined."

    # Select one special token ID (e.g., [CLS] or [SEP])
    special_token_id = list(special_ids)[0]

    # Select 4 token IDs that are not in the set of special tokens
    non_special_ids = [i for i in range(100, 1000) if i not in special_ids][:4]
    assert len(non_special_ids) >= 4, "Not enough non-special token IDs available."

    # Construct input_ids tensor: [special_token, regular_token1, ..., regular_token4]
    # Shape: (1, 5)
    input_ids = torch.tensor([[special_token_id] + non_special_ids])
    tokens = BatchEncoding({"input_ids": input_ids})

    # Create dummy scores: 2 scores per token:
    fake_scores = torch.tensor([[10.0, 20.0, 31.1, 41.2, 55.2], [12.0, 21.0, 30.0, 40.0, 50.0]])  # shape (2, 5)

    # Apply TOKEN-level aggregation (no actual reduction since each token is treated individually)
    aggregated = Granularity.TOKEN.granularity_score_aggregation(
        contribution=fake_scores,
        inputs=tokens,
        tokenizer=tokenizer,
        aggregate_inputs=True,
    )

    # Expect scores of regular tokens only (i.e., skip the first one)
    expected = fake_scores[:, 1:]

    # Assert the result matches the expected filtered scores
    assert torch.allclose(aggregated, expected, atol=1e-5)


def _manual_aggregate(
    scores: torch.Tensor,  # (t, lp)
    indices: list[list[int]],  # groups of token positions
    strategy: GranularityAggregationStrategy,
) -> torch.Tensor:  # (t, g)
    """Manually reproduce the selected aggregation strategy (for word and sentence) over scores."""
    chunks: list[torch.Tensor] = []

    for group in indices:
        chunk = scores[:, group]  # (t, |group|)

        if chunk.shape[1] == 1:  # only one token: no aggregation
            agg = chunk
        elif strategy is GranularityAggregationStrategy.MEAN:
            agg = chunk.mean(dim=1, keepdim=True)
        elif strategy is GranularityAggregationStrategy.MAX:
            agg = chunk.max(dim=1, keepdim=True).values
        elif strategy is GranularityAggregationStrategy.MIN:
            agg = chunk.min(dim=1, keepdim=True).values
        elif strategy is GranularityAggregationStrategy.SUM:
            agg = chunk.sum(dim=1, keepdim=True)
        elif strategy is GranularityAggregationStrategy.SIGNED_MAX:
            idx = chunk.abs().argmax(dim=1).unsqueeze(1)
            agg = chunk.gather(1, idx)
        elif strategy is GranularityAggregationStrategy.FIRST:
            agg = chunk.narrow(1, 0, 1)
        elif strategy is GranularityAggregationStrategy.LAST:
            agg = chunk.narrow(1, chunk.shape[1] - 1, 1)
        else:  # should never happen
            raise ValueError(f"Unknown strategy: {strategy}")

        chunks.append(agg)

    return torch.cat(chunks, dim=1)  # (t, g)


STRATS = [
    GranularityAggregationStrategy.MEAN,
    GranularityAggregationStrategy.MAX,
    GranularityAggregationStrategy.MIN,
    GranularityAggregationStrategy.SUM,
    GranularityAggregationStrategy.SIGNED_MAX,
    GranularityAggregationStrategy.FIRST,
    GranularityAggregationStrategy.LAST,
]


@pytest.mark.parametrize("strategy", STRATS)
def test_aggregate_and_unfold(strategy):
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)

    agg = strategy.aggregate(tensor, dim=0)

    assert agg.shape == (1, 3), (
        "Wrong activation aggregation output shape. ",
        f"Expected shape (1, 3), got {agg.shape}",
    )

    unfolded = strategy.unfold(agg, 2)

    assert unfolded.shape == (2, 3), (
        "Wrong activation unfolding output shape. ",
        f"Expected shape (2, 3), got {unfolded.shape}",
    )


@pytest.mark.parametrize("strategy", STRATS)
def test_word_aggregation_matches_manual(simple_text, real_bert_tokenizer, strategy):
    """
    Checks that word-level aggregation performed by the library
    matches our manual aggregation implementation.
    """

    tok = real_bert_tokenizer(simple_text, return_tensors="pt", return_offsets_mapping=True)
    seq_len = tok["input_ids"].shape[1]

    # two scores per token:
    scores = torch.randn(2, seq_len)

    # Group token indices by word
    indices = Granularity.WORD.get_indices(tok, real_bert_tokenizer)[0]

    expected = _manual_aggregate(scores, indices, strategy)

    obtained = Granularity.WORD.granularity_score_aggregation(
        contribution=scores,
        granularity_aggregation_strategy=strategy,
        inputs=tok,
        tokenizer=real_bert_tokenizer,
        aggregate_inputs=True,
    )

    assert torch.allclose(obtained, expected, atol=1e-6)


# ----------------------------------------------------------
# SENTENCE granularities


def test_sentence_granularities_indices(complex_text, real_bert_tokenizer):
    """Basic sanity checks on the hierarchy of sentence granularities."""

    tokens = real_bert_tokenizer(complex_text, return_tensors="pt", return_offsets_mapping=True)

    sent_idx = Granularity.SENTENCE.get_indices(tokens, real_bert_tokenizer)[0]

    assert len(sent_idx) == 5, f"Expected 5 sentences, got {len(sent_idx)}"


def test_sentence_granularities_indices_part_sentence(complex_text, real_bert_tokenizer):
    """Basic sanity checks on the hierarchy of sentence granularities."""

    tokens = real_bert_tokenizer(complex_text, return_tensors="pt", return_offsets_mapping=True)

    part_sent_idx = Granularity.PART_SENTENCE.get_indices(tokens, real_bert_tokenizer)[0]

    assert len(part_sent_idx) == 7, f"Expected 7 part-sentences, got {len(part_sent_idx)}"


def test_sentence_part_sentence_granularity_with_different_tokenizers():
    def assert_segment_matches(repo_id: str, segment: str, ref: str):
        seg = segment.lower()

        # Allow an optional leading newline for the "This is the end..." sentence
        if ref == "this is the end of the test!!!":
            seg = seg.lstrip("\n")
            assert ref in seg, f"Expected '{ref}' in segment '{segment}' for tokenizer {repo_id}"

        # Allow "url.com" OR "url. com" (optional space after the dot) for the URL sentence
        elif ref == "this url.com is?":
            assert re.search(r"this url\. ?com is\?", seg), (
                f"Expected '{ref}' (or 'this url. com is?') in segment '{segment}' for tokenizer {repo_id}"
            )

        # Default strict containment check for other sentences
        else:
            assert ref in seg, f"Expected '{ref}' in segment '{segment}' for tokenizer {repo_id}"

    for repo_id in [
        "hf-internal-testing/tiny-random-bert",
        "hf-internal-testing/tiny-random-gpt2",
        "hf-internal-testing/tiny-random-roberta",
        "hf-internal-testing/tiny-random-t5",
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
    ]:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        text = "This is : a test. Mr. is this a test?\nThis is the end of the test!!! Or, is it?! This url.com is?"

        tokens = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)

        # --- SENTENCE granularity ---
        sent_idx = Granularity.SENTENCE.get_indices(tokens, tokenizer)[0]
        sent_text = Granularity.SENTENCE.get_decomposition(tokens, tokenizer, return_text=True)[0]

        assert len(sent_idx) == 5, (
            f"[SENTENCE] Expected 5 sentences, got {len(sent_idx)} for tokenizer {repo_id}",
            f"{sent_text}",
        )

        expected_sent = [
            "this is : a test.",
            "mr. is this a test?",
            "this is the end of the test!!!",
            "or, is it?!",
            "this url.com is?",
        ]

        for segment, ref in zip(sent_text, expected_sent, strict=True):
            assert_segment_matches(repo_id, segment, ref)

        # --- PART_SENTENCE granularity (also splits on ',' and ':') ---
        part_idx = Granularity.PART_SENTENCE.get_indices(tokens, tokenizer)[0]
        part_text = Granularity.PART_SENTENCE.get_decomposition(tokens, tokenizer, return_text=True)[0]

        # We expect extra splits due to ',' and ':' :
        assert len(part_idx) == 7, (
            f"[PART_SENTENCE] Expected 7 segments, got {len(part_idx)} for tokenizer {repo_id}",
            f"{part_text}",
        )

        expected_part = [
            "this is :",
            "a test.",
            "mr. is this a test?",
            "this is the end of the test!!!",
            "or,",
            "is it?!",
            "this url.com is?",
        ]

        for segment, ref in zip(part_text, expected_part, strict=True):
            seg = segment.lower()

            # Allow an optional leading newline on the first chunk after the line break
            if ref == "this is the end of the test!!!":
                seg = seg.lstrip("\n")
                assert ref in seg, f"Expected '{ref}' in segment '{segment}' for tokenizer {repo_id}"
            else:
                assert_segment_matches(repo_id, segment, ref)


def test_sentence_granularities_matrices_and_decomposition(complex_text, real_bert_tokenizer):
    """Full round‑trip checks for SENTENCE and PART-SENTENCE granularities."""

    tokens = real_bert_tokenizer(complex_text, return_tensors="pt", return_offsets_mapping=True)
    seq_len = tokens["input_ids"].shape[1]

    for gran in (Granularity.SENTENCE, Granularity.PART_SENTENCE):
        indices = gran.get_indices(tokens, real_bert_tokenizer)[0]

        # Association matrix
        assoc = gran.get_association_matrix(tokens, real_bert_tokenizer)[0]
        expected_mat = _build_expected_matrix(indices, seq_len)
        assert torch.equal(assoc, expected_mat)

        # Decomposition (ids)
        decomp_ids = gran.get_decomposition(tokens, real_bert_tokenizer)[0]
        assert decomp_ids == [[int(tokens["input_ids"][0][i]) for i in grp] for grp in indices]

        # Decomposition (text)
        decomp_text: list[str] = gran.get_decomposition(tokens, real_bert_tokenizer, return_text=True)[0]  # type: ignore
        # Silent print for manual inspection when running directly
        # (f"\n[{gran}] decomposition:\n", decomp_text)

        # Each segment must be non‑empty & appear verbatim in the original text
        raw_text = real_bert_tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
        for segment in decomp_text:
            assert segment.strip() in raw_text

        # Join without spaces – coverage check (no char lost)
        joined = " ".join(seg.strip() for seg in decomp_text)
        assert joined == raw_text


@pytest.mark.parametrize("strategy", STRATS)
def test_sentence_aggregation_matches_manual(complex_text, real_bert_tokenizer, strategy):
    """
    Same as the WORD-level test, but for SENTENCE and PART-SENTENCE-level granularity.
    """

    tok = real_bert_tokenizer(complex_text, return_tensors="pt", return_offsets_mapping=True)
    seq_len = tok["input_ids"].shape[1]
    scores = torch.randn(3, seq_len)  # three scores per token for variation

    for gran in (Granularity.SENTENCE, Granularity.PART_SENTENCE):
        indices = gran.get_indices(tok, real_bert_tokenizer)[0]

        expected = _manual_aggregate(scores, indices, strategy)

        obtained = gran.granularity_score_aggregation(
            contribution=scores,
            granularity_aggregation_strategy=strategy,
            inputs=tok,
            tokenizer=real_bert_tokenizer,
            aggregate_inputs=True,
        )
        assert torch.allclose(obtained, expected, atol=1e-6)


# -----------------------------------------------------------------
# Manual run
# -----------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover – convenience only
    # Run tests when executed directly (no pytest needed)
    print("Running granularity tests…\n")
    sys.exit(pytest.main([__file__]))
