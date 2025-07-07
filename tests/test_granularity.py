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

import sys

import pytest
import torch
from transformers import AutoTokenizer

# local import – the module is supplied alongside this test file
from interpreto import Granularity
from interpreto.commons.granularity import _HAS_SPACY

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
    """Multi‑clause, multi‑sentence, multi‑paragraph text for spaCy levels."""
    return (
        "Although it was raining, we went for a walk. "  # 1st sentence (2 clauses)
        "We took umbrellas.\n\n"  # 2nd sentence, same paragraph
        "It was fun."  # New paragraph
    )


# ----------------------------------------
# ALL_TOKENS / TOKEN / WORD granularities


def test_low_level_granularities_indices(simple_text, real_bert_tokenizer):
    """Checks *get_indices* for ALL_TOKENS / TOKEN / WORD on a tiny text."""

    tokens = real_bert_tokenizer(simple_text, return_tensors="pt", return_offsets_mapping=True)
    seq_len = tokens["input_ids"].shape[1]

    # Human‑readable token list
    print("\nBERT tokens:", real_bert_tokenizer.tokenize(simple_text))

    # ALL_TOKENS
    all_tokens_indices = Granularity.get_indices(tokens, Granularity.ALL_TOKENS, real_bert_tokenizer)[0]
    assert all_tokens_indices == [[i] for i in range(seq_len)]

    # TOKEN
    token_indices = Granularity.get_indices(tokens, Granularity.TOKEN, real_bert_tokenizer)[0]
    special = set(real_bert_tokenizer.all_special_ids)
    expected_token_pos = [i for i, tid in enumerate(tokens["input_ids"][0]) if int(tid) not in special]
    assert [idx[0] for idx in token_indices] == expected_token_pos

    # WORD
    word_indices = Granularity.get_indices(tokens, Granularity.WORD, real_bert_tokenizer)[0]
    # We know there are exactly 3 human words in *simple_text*.
    assert len(word_indices) == 3
    # First word should decode to "word"
    first_word_ids = [int(tokens["input_ids"][0][i]) for i in word_indices[0]]
    assert real_bert_tokenizer.decode(first_word_ids) == "word"


def test_low_level_granularities_matrices_and_decomposition(simple_text, real_bert_tokenizer):
    """Cross‑validate indices ⇄ matrices ⇄ decompositions for low‑level granularities."""

    tokens = real_bert_tokenizer(simple_text, return_tensors="pt", return_offsets_mapping=True)
    seq_len = tokens["input_ids"].shape[1]

    for gran in (Granularity.ALL_TOKENS, Granularity.TOKEN, Granularity.WORD):
        indices = Granularity.get_indices(tokens, gran, real_bert_tokenizer)[0]

        # Association matrix
        assoc = Granularity.get_association_matrix(tokens, gran, real_bert_tokenizer)[0]
        expected_mat = _build_expected_matrix(indices, seq_len)
        assert torch.equal(assoc, expected_mat)

        # Decomposition (ids)
        decomp_ids = Granularity.get_decomposition(tokens, gran, real_bert_tokenizer)[0]
        # Compare raw ids – order + content must match indices
        assert decomp_ids == [[int(tokens["input_ids"][0][i]) for i in grp] for grp in indices]

        # Decomposition (text)
        decomp_text = Granularity.get_decomposition(tokens, gran, real_bert_tokenizer, return_text=True)[0]
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


# ----------------------------------------------------------
# spaCy‑based granularities (SENTENCE )

needs_spacy = pytest.mark.skipif(
    not _HAS_SPACY,
    reason="spaCy  not available – skipping high‑level tests",
)


@needs_spacy
def test_spacy_granularities_indices(complex_text, real_bert_tokenizer):
    """Basic sanity checks on the hierarchy of spaCy granularities."""

    tokens = real_bert_tokenizer(complex_text, return_tensors="pt", return_offsets_mapping=True)

    sent_idx = Granularity.get_indices(tokens, Granularity.SENTENCE, real_bert_tokenizer)[0]

    # We know our handcrafted *complex_text*:
    #   • 4 clauses   (2 + 1 + 1)
    #   • 3 sentences (2 + 1)
    #   • 2 paragraphs
    assert len(sent_idx) == 3

    # Hierarchy sanity: same tokens regrouped, nothing lost / duplicated
    flat_sent = sorted(i for grp in sent_idx for i in grp)
    assert flat_sent == list(range(len(tokens["input_ids"][0])))


@needs_spacy
def test_spacy_granularities_matrices_and_decomposition(complex_text, real_bert_tokenizer):
    """Full round‑trip checks for SENTENCE ."""

    tokens = real_bert_tokenizer(complex_text, return_tensors="pt", return_offsets_mapping=True)
    seq_len = tokens["input_ids"].shape[1]

    gran = Granularity.SENTENCE

    indices = Granularity.get_indices(tokens, gran, real_bert_tokenizer)[0]

    # Association matrix
    assoc = Granularity.get_association_matrix(tokens, gran, real_bert_tokenizer)[0]
    expected_mat = _build_expected_matrix(indices, seq_len)
    assert torch.equal(assoc, expected_mat)

    # Decomposition (ids)
    decomp_ids = Granularity.get_decomposition(tokens, gran, real_bert_tokenizer)[0]
    assert decomp_ids == [[int(tokens["input_ids"][0][i]) for i in grp] for grp in indices]

    # Decomposition (text)
    decomp_text = Granularity.get_decomposition(tokens, gran, real_bert_tokenizer, return_text=True)[0]
    # Silent print for manual inspection when running directly
    print(f"\n[{gran.value}] decomposition:\n", decomp_text)

    # Each segment must be non‑empty & appear verbatim in the original text
    raw_text = real_bert_tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
    for segment in decomp_text:
        assert segment.strip() in raw_text

    # Join without spaces – coverage check (no char lost)
    joined = " ".join(seg.strip() for seg in decomp_text)
    assert joined == raw_text


# -----------------------------------------------------------------
# Manual run
# -----------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover – convenience only
    # Run tests when executed directly (no pytest needed)
    print("Running granularity tests…\n")
    sys.exit(pytest.main([__file__]))
