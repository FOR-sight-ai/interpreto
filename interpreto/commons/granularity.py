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
Definition of different granularity levels for explainers (tokens, words, sentences...)
"""

from __future__ import annotations

from enum import Enum

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from torch.nn.functional import interpolate
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

# TODO I dont know where to put this...
END_SENTENCE = (".", "?", "!")
END_PART_SENTENCE = (",", ";", ":", ".", "!", "?")
SENTENCE_SPLIT_EXCEPTIONS = (
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "St.",
    "Mt.",
    "etc.",
    "cf.",
    "i.e.",
    "e.g.",
    "vs.",
    "nb.",
    "env.",
    "approx.",
    "min.",
    "max.",
    "resp.",
    "ex.",
    "ref.",
    "Ph.D.",
    "M.Sc.",
    "B.Sc.",
    "U.S.",
    "U.K.",
    "E.U.",
)


class GranularityCombinationStrategy(Enum):
    pass


class GranularityAggregationStrategy(GranularityCombinationStrategy):
    """
    Enumeration of the available aggregation strategies for combining token-level
    scores into a single score for each unit of a higher-level granularity
    (e.g., word, sentence).

    This is used in explainability methods to reduce token-based attributions
    according to a defined granularity.

    Attributes:
        MEAN: Average of the token scores within each group.
        MAX: Maximum token score within each group.
        MIN: Minimum token score within each group.
        SUM: Sum of all token scores within each group.
        SIGNED_MAX: Selects the token with the highest absolute score and returns its signed value.
                        For example, given scores [3, -1, 7], returns 7; for [3, -1, -7], returns -7.
    """

    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    SIGNED_MAX = "signed_max"
    FIRST = "first"  # TODO: test
    LAST = "last"  # TODO: test

    def aggregate(  # noqa: PLR0911  # ignore too many return statements
        self, x: Float[torch.Tensor, "l d"], dim: int
    ) -> Float[torch.Tensor, "1 d"]:
        """
        Aggregate activations.
        Args:
            x (torch.Tensor): The tensor to aggregate, shape: (l, d).
        Returns:
            torch.Tensor: The aggregated tensor, shape (1, d).
        """
        match self:
            case GranularityAggregationStrategy.SUM:
                return x.sum(dim=dim, keepdim=True)
            case GranularityAggregationStrategy.MEAN:
                return x.mean(dim=dim, keepdim=True)
            case GranularityAggregationStrategy.MAX:
                return x.max(dim=dim, keepdim=True).values
            case GranularityAggregationStrategy.MIN:
                return x.min(dim=dim, keepdim=True).values
            case GranularityAggregationStrategy.SIGNED_MAX:
                return x.gather(dim, x.abs().max(dim=dim)[1].unsqueeze(dim))
            case GranularityAggregationStrategy.FIRST:
                # Select the first element along the aggregation dimension, keepdim=True
                return x.narrow(dim, start=0, length=1)
            case GranularityAggregationStrategy.LAST:
                # Select the last element along the aggregation dimension, keepdim=True
                return x.narrow(dim, start=x.size(dim) - 1, length=1)
            case _:
                raise NotImplementedError(f"Aggregation strategy {self} not implemented.")

    def unfold(self, x: Float[torch.Tensor, "1 d"], new_dim_length: int) -> Float[torch.Tensor, "{new_dim_length} d"]:
        """
        Unfold activations.
        Args:
            x (torch.Tensor): The tensor to unfold, shape: (1, d).
            new_dim_length (int): The new dimension length.
        Returns:
            torch.Tensor: The unfolded tensor, shape: (l, d).
        """
        match self:
            case GranularityAggregationStrategy.SUM:
                return (x / new_dim_length).repeat(new_dim_length, 1)
            case (
                GranularityAggregationStrategy.MEAN
                | GranularityAggregationStrategy.MAX
                | GranularityAggregationStrategy.MIN
                | GranularityAggregationStrategy.SIGNED_MAX
                | GranularityAggregationStrategy.FIRST
                | GranularityAggregationStrategy.LAST
            ):
                return x.repeat(new_dim_length, 1)
            case _:
                raise NotImplementedError(f"Aggregation strategy {self} not implemented.")


class GranularityResizeStrategy(GranularityCombinationStrategy):
    # TODO: evaluate if it needs an unfold function such as GranularityAggregationStrategy
    # NOTE: torch has no hamming/lanczos modes, and torchvision only offers them via the CPU-only
    # PIL backend, so they are unavailable for GPU tensors regardless of library.
    NEAREST = "nearest"  # nearest-neighbor — replicates values, keeps a binary mask binary
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    AREA = "area"  # box/area averaging — the energy-preserving downsample

    def resize(
        self,
        input: Float[torch.Tensor, "t h w"],
        output_size: tuple[int, int] | None = None,
        patch_size: int = 16,
    ) -> Float[torch.Tensor, "t gh gw"]:
        """
        Spatially resize per-channel maps, keeping the 2-D layout.

        Direction-agnostic: downsamples (pixel grid -> patch grid, for aggregating gradient
        attributions) when `output_size` is left `None`, or resizes to an explicit target (e.g.
        upsampling a perturbation mask back to pixel resolution) when `output_size` is given.

        Unlike `GranularityAggregationStrategy.aggregate` (which reduces a unit's pixels to a
        scalar), this keeps the 2-D layout and interpolates the whole grid. The `t` axis is treated
        as the channel dimension, so all maps are resized in one call and the per-channel axis is
        preserved (no broadcast across them). Runs on `x`'s device (CPU or GPU).

        Args:
            x (Float[torch.Tensor, "t h_in w_in"]): Per-channel maps to resize.
            patch_size (int): Patch side length; used only when `output_size` is `None`, to derive
                the downsample target ``(h_in // patch_size, w_in // patch_size)``.
            output_size (tuple[int, int] | None): Explicit target ``(h_out, w_out)``. When provided,
                it is used directly and `patch_size` is ignored; prefer computing it at the call site.

        Returns:
            Float[torch.Tensor, "t h_out w_out"]: Resized maps, same dtype/device as ``x``.
        """
        if output_size is None:
            _, h_in, w_in = input.shape
            assert h_in % patch_size == 0, "the height of the image must be divisble by the patch_size"
            assert w_in % patch_size == 0, "the width of the image must be divisble by the patch_size"
            output_size = (h_in // patch_size, w_in // patch_size)
        # interpolate expects 4-D (N, C, H, W); treat the maps as a single batch of `t` channels
        x4: Float[torch.Tensor, "1 t h_in w_in"] = input.unsqueeze(0)
        match self:
            case GranularityResizeStrategy.NEAREST:
                # nearest does not accept align_corners / antialias
                resized = interpolate(x4, size=output_size, mode="nearest")
            case GranularityResizeStrategy.BILINEAR:
                resized = interpolate(x4, size=output_size, mode="bilinear", align_corners=False, antialias=True)
            case GranularityResizeStrategy.BICUBIC:
                resized = interpolate(x4, size=output_size, mode="bicubic", align_corners=False, antialias=True)
            case GranularityResizeStrategy.AREA:
                # align_corners / antialias are not accepted with mode="area"
                resized = interpolate(x4, size=output_size, mode="area")
            case _:
                raise NotImplementedError(f"Resize strategy {self} not implemented.")
        return resized.squeeze(0)


class Granularity(Enum):
    # TODO: look if it's possible to refactor some functions in TextGranularity and ImageGranularity
    """
    Abstract Granularity class to harmonize typing.
    """

    pass


class TextGranularity(Granularity):
    """
    Enumerations of the different granularity levels supported for masking perturbations
    Allows to define token-wise masking, word-wise masking...
    """

    ALL_TOKENS = "all_tokens"  # All tokens, including special tokens like padding, eos, cls, etc.
    TOKEN = "token"  # Strictly tokens of the input
    WORD = "word"  # Words of the input
    SENTENCE = "sentence"  # Sentences of the input
    PART_SENTENCE = "part_sentence"  # Part of sentences, split on ",", ";", ":", ".", "!", "?"
    # PARAGRAPH = "paragraph"  # Not supported yet, the "\n\n" characters are replaced by spaces in many tokenizers.
    DEFAULT = ALL_TOKENS

    # @jaxtyped(typechecker=beartype)
    def get_indices(
        self,
        inputs: BatchEncoding,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None,
    ) -> list[list[list[int]]]:
        """
        Return *indices* of the tokens that correspond to the desired
        granularity for each samples.

        The result is a *list[list[list[int]]]* where each inner list contains the
        positions of the tokens that compose one granularity unit.
        The list hierarchy is as follows:

            - For each sample.

            - For each element for the granularity level. Thus, tokens, words, or sentences.

            - The inner list contains the positions of the tokens that compose one granularity unit.

        The granularity levels are:

            - ``ALL_TOKENS``: All tokens, including special tokens like [PAD], [EOS], [CLS], etc.

            - ``TOKEN``: Strictly tokens of the input.

            - ``WORD``: Tokens are grouped by word.

            - ``PART_SENTENCE``: Tokens are grouped by part of sentence, split on ",", ";", ":", ".", "!", "?".

            - ``SENTENCE``: Tokens are grouped by sentence.

        Args:
            inputs_mapping (BatchEncoding): Tokenized inputs, the output of
                `self.tokenizer("some_text", return_tensors="pt", return_offsets_mapping=True, truncation=True)`
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Hugging-Face tokenizer used downstream.

        Raises:
            NoWordIdsError: if *WORD* granularity is requested with a slow
                            tokenizer.
            NotImplementedError: if an unknown granularity is supplied.

        Examples:
            >>> from interpreto.commons.granularity import TextGranularity
            >>> raw_input_text = [
            ...     "Interpreto is magical. Or is it?",
            ...     "At least we try.",
            ... ]
            >>> input_text_with_special_tokens = [
            ...     "[CLS]|Inter|preto| is| magic|al|.| Or| is| it|?|[EOS]",
            ...     "[CLS]|At| least| we| try|.|[EOS]|[PAD]|[PAD]|[PAD]|[PAD]|[PAD]",
            ... ]
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> input_ids = tokenizer(raw_input_text, return_tensors="pt")["input_ids"]
            >>> TextGranularity.ALL_TOKENS.get_indices(input_ids, tokenizer=tokenizer)
            [[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
             [[12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23]]]
            >>> TextGranularity.TOKEN.get_indices(input_ids, tokenizer=tokenizer)
            [[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
             [[13], [14], [15], [16], [17]]]
            >>> TextGranularity.WORD.get_indices(input_ids, tokenizer=tokenizer)
            [[[1, 2], [3], [4, 5], [6], [7], [8], [9], [10]],
             [[13], [14], [15], [16], [17]]]
            >>> TextGranularity.SENTENCE.get_indices(input_ids, tokenizer=tokenizer)
            [[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10]],
             [[13, 14, 15, 16, 17]]]
        """

        match self or TextGranularity.DEFAULT:
            case TextGranularity.ALL_TOKENS:
                input_ids: Int[torch.Tensor, "n l"] = inputs["input_ids"]  # type: ignore
                return [TextGranularity.__all_tokens_get_indices(tokens_ids) for tokens_ids in input_ids]
            case TextGranularity.TOKEN:
                if tokenizer is None:
                    raise ValueError(
                        "Cannot get indices without a tokenizer if granularity is TOKEN."
                        + "Please provide a tokenizer or set granularity to ALL_TOKENS."
                    )

                special_ids = tokenizer.all_special_ids
                input_ids: Int[torch.Tensor, "n l"] = inputs["input_ids"]  # type: ignore
                return [TextGranularity.__token_get_indices(tokens_ids, special_ids) for tokens_ids in input_ids]
            case TextGranularity.WORD:
                if tokenizer is None:
                    raise ValueError(
                        "Cannot get indices without a tokenizer if granularity is WORD."
                        + "Please provide a tokenizer or set granularity to ALL_TOKENS."
                    )

                n_inputs = inputs["input_ids"].shape[0]  # type: ignore

                if TextGranularity.__word_ids_are_usable(tokenizer, inputs):
                    return [
                        TextGranularity.__word_get_indices_from_word_ids(inputs.word_ids(i)) for i in range(n_inputs)
                    ]

                return [
                    TextGranularity.__word_get_indices_from_input_ids(inputs["input_ids"][i], tokenizer)
                    for i in range(n_inputs)
                ]

            case TextGranularity.PART_SENTENCE | TextGranularity.SENTENCE:
                if self is TextGranularity.PART_SENTENCE:
                    split = END_PART_SENTENCE
                else:
                    split = END_SENTENCE
                if tokenizer is None:
                    raise ValueError(
                        "Cannot get indices without a tokenizer if granularity is PART_SENTENCE."
                        + "Please provide a tokenizer or set granularity to ALL_TOKENS."
                    )
                n_inputs = inputs["input_ids"].shape[0]  # type: ignore
                if tokenizer.is_fast and isinstance(inputs, BatchEncoding) and getattr(inputs, "encodings", None):
                    return [
                        TextGranularity.__sentence_get_indices_from_offsets(
                            inputs["input_ids"][i],  # type: ignore
                            inputs.encodings[i].offsets,  # type: ignore[attr-defined]
                            tokenizer,
                            split,  # type: ignore
                        )
                        for i in range(n_inputs)
                    ]
                return [
                    TextGranularity.__sentence_get_indices_from_input_ids(
                        inputs["input_ids"][i],  # type: ignore
                        tokenizer,
                        split,  # type: ignore
                    )
                    for i in range(n_inputs)
                ]

            case _:
                raise NotImplementedError(f"TextGranularity level {self} not implemented")

    @staticmethod
    def __all_tokens_get_indices(tokens_ids: torch.Tensor) -> list[list[int]]:
        """Indices for :pyattr:`ALL_TOKENS` – every position kept."""
        length = len(tokens_ids)
        return [[i] for i in range(length)]

    @staticmethod
    def __token_get_indices(tokens_ids: torch.Tensor, special_ids: list[int]) -> list[list[int]]:
        """Indices for :pyattr:`TOKEN` – skip special tokens."""
        return [[i] for i, tok_id in enumerate(tokens_ids) if tok_id not in special_ids]

    @staticmethod
    def __word_ids_are_usable(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, inputs: BatchEncoding) -> bool:
        """Return True when we have a fast-tokenizer and word ids provide meaningful word grouping."""
        if not tokenizer.is_fast:
            print("Tokenizer is not fast, cannot use word_ids for WORD granularity.")
            return False
        word_ids = inputs.word_ids(0)
        is_valid = isinstance(word_ids, list) and any(x is not None for x in word_ids)
        if is_valid:
            return True
        return False

    @staticmethod
    def __word_get_indices_from_word_ids(word_ids: list[int | None]) -> list[list[int]]:
        """Indices for :pyattr:`WORD` – group tokens belonging to the same word."""
        mapping: dict[int, list[int]] = {}
        for idx, wid in enumerate(word_ids):
            if wid is None:  # `None` for special tokens – ignore them
                continue
            mapping.setdefault(wid, []).append(idx)

        # Return groups ordered by word id (i.e. sentence order)
        return [mapping[k] for k in sorted(mapping)]

    @staticmethod
    def _starts_word(token: str) -> bool:
        return token.startswith((" ", "Ġ", "▁", "__"))

    @staticmethod
    def __word_get_indices_from_input_ids(
        input_ids: list[int] | torch.Tensor, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    ) -> list[list[int]]:
        """Indices for :pyattr:`WORD` – group tokens belonging to the same word."""
        special_ids = tokenizer.all_special_ids
        tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)

        indices: list[list[int]] = []
        current_word: list[int] = []
        for i, (token_id, token) in enumerate(zip(input_ids, tokens, strict=True)):
            # Skip special tokens
            if token_id in special_ids:
                continue

            # If token starts a new word, we put current to indices and initialize a new one
            if TextGranularity._starts_word(token):
                if current_word:
                    indices.append(current_word)
                current_word = [i]
            else:
                current_word.append(i)

        # If there's a word left, we put it in indices
        if current_word:
            indices.append(current_word)
        return indices

    # Mini functions for sentence splitting, to keep the main function clearer:
    @staticmethod
    def __next_non_special(start: int, ids, special_ids) -> int | None:
        """
        Find the next non-special token index after `start` (used to look ahead safely).
        """
        for j in range(start, len(ids)):
            if ids[j] not in special_ids:
                return j
        return None

    @staticmethod
    def __starts_with_space_marker(token: str) -> bool:
        """
        Detect "whitespace is part of the next token" across common tokenization schemes:
          - SentencePiece: '▁'
          - GPT2 BPE: 'Ġ'
          - Literal leading space: ' '
        """
        return token.startswith(("▁", "Ġ", " "))

    @staticmethod
    def __decode_one(tokenizer: PreTrainedTokenizer, tok_id: int) -> str:
        """
        Decode a single token id without normalizing spaces, to reliably detect leading whitespace/newlines.
        """
        return tokenizer.decode(
            [tok_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    @staticmethod
    def __build_sentence_exception_id_seqs(
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> list[list[int]]:
        """
        Build token-id sequences for exceptions (multiple casing + optional leading space),
        to robustly match both GPT-style (space in token) and WordPiece/BPE tokenizers.
        """
        seqs: list[list[int]] = []

        for ex in SENTENCE_SPLIT_EXCEPTIONS:  # type: ignore
            # Variants to be robust across cased/uncased and GPT-style leading-space tokens
            base_variants = {ex, ex.lower(), ex.upper()}
            variants = set(base_variants)
            variants.update({" " + v for v in base_variants})

            for v in variants:
                ids = tokenizer.encode(v, add_special_tokens=False)
                if ids:
                    seqs.append([int(x) for x in ids])

        # Deduplicate
        uniq: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for s in seqs:
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                uniq.append(s)

        # Optional: longer first (slightly faster / more specific first)
        uniq.sort(key=len, reverse=True)
        return uniq

    @staticmethod
    def __is_index_in_any_exception(
        ids: list[int],
        idx: int,
        exception_id_seqs: list[list[int]],
    ) -> bool:
        """
        Returns True if token position `idx` lies inside ANY matched exception sequence.
        This prevents splitting not only after the final '.', but also inside multi-dot
        abbreviations like 'U.S.' in the slow fallback mode.
        """
        for seq in exception_id_seqs:
            n = len(seq)
            if n == 0:
                continue

            # Try alignments where `idx` could correspond to any position inside `seq`
            for offset in range(n):
                start = idx - offset
                end = start + n
                if start < 0 or end > len(ids):
                    continue
                if ids[start:end] == seq:
                    return True

        return False

    @staticmethod
    def __sentence_get_indices_from_offsets(
        input_ids: torch.Tensor,
        offsets: list[tuple[int, int]],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        split: list[str],
    ) -> list[list[int]]:
        """
        Split ONLY when there is an end-of-sentence punctuation (., ?, !) followed by whitespace
        in the original text, EXCEPT if inside an exception (e.g., "U.S.", "Mr."...).

        For some tokenizers (e.g., GPT2 byte-level BPE), the whitespace can be included in the
        *next token span*, so offset gaps alone are not sufficient. We thus also check whether
        the next token *decodes* with a leading whitespace.
        """
        special_ids = tokenizer.all_special_ids
        ids = [int(x) for x in input_ids.tolist()]
        tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
        exception_id_seqs = TextGranularity.__build_sentence_exception_id_seqs(tokenizer)

        indices: list[list[int]] = []
        current_sentence: list[int] = []

        for i, (tok_id, tok_str) in enumerate(zip(ids, tokens, strict=True)):
            # Skip special tokens (CLS/SEP/PAD...):
            if tok_id in special_ids:
                continue
            current_sentence.append(i)

            j = TextGranularity.__next_non_special(i + 1, ids, special_ids)
            if j is None:
                continue

            # IMPORTANT: accept tokens like "Ġ!!", "...", "Ġ?", etc.
            if not any(p in tok_str for p in split):  # type: ignore
                # if not tok_str.endswith(split):  # type: ignore[arg-type]
                continue

            # Exception guard (only relevant for '.' abbreviations)
            if ("." in tok_str) and TextGranularity.__is_index_in_any_exception(ids, i, exception_id_seqs):
                continue

            curr_end = offsets[i][1]
            next_start = offsets[j][0]

            # 1) Offsets-based whitespace (works well for BERT-like tokenizers)
            has_whitespace_after = (next_start - curr_end) >= 1

            # 2) Decode-based whitespace (works well for GPT2-like tokenizers and newline tokens)
            if not has_whitespace_after:
                next_piece = TextGranularity.__decode_one(tokenizer, ids[j])  # type: ignore
                has_whitespace_after = bool(next_piece) and next_piece[0].isspace()

            # 3) Token-string space marker (fixes SentencePiece/LLaMA/Mistral: tokens like "▁Is")
            if not has_whitespace_after:
                has_whitespace_after = TextGranularity.__starts_with_space_marker(tokens[j])

            # If punctuation is followed by whitespace, start a new group
            if has_whitespace_after:
                indices.append(current_sentence)
                current_sentence = []

        if current_sentence:
            indices.append(current_sentence)

        return indices

    @staticmethod
    def __sentence_get_indices_from_input_ids(
        input_ids: list[int] | torch.Tensor,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        split: list[str],
    ) -> list[list[int]]:
        """
        Basic tokenizers (no offsets): simple split when we see ".", "?" or "!".
        Split on end punctuation directly, EXCEPT if inside an exception (e.g., "U.S.", "Mr."...).
        Avoid splitting multiple times for '...' when it is tokenized as ".", ".", ".".
        """
        special_ids = tokenizer.all_special_ids

        ids = (
            [int(x) for x in input_ids.tolist()]
            if isinstance(input_ids, torch.Tensor)
            else [int(x) for x in input_ids]
        )
        tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
        exception_id_seqs = TextGranularity.__build_sentence_exception_id_seqs(tokenizer)

        indices: list[list[int]] = []
        current_sentence: list[int] = []

        for i, (tok_id, tok_str) in enumerate(zip(ids, tokens, strict=True)):
            # Skip special tokens (CLS/SEP/PAD...):
            if tok_id in special_ids:
                continue
            current_sentence.append(i)

            # Fallback rule: split on end-of-sentence (".", "?", "!") punctuation suffix.
            if not any(p in tok_str for p in split):  # type: ignore
                continue

            # Exception guard (prevents splitting inside 'U.S.' etc.)
            if TextGranularity.__is_index_in_any_exception(ids, i, exception_id_seqs):
                continue

            # Avoid splitting 3 times for "..." when it is tokenized as ".", ".", "."
            j = TextGranularity.__next_non_special(i + 1, ids, special_ids)
            # if j is not None and tokens[j].startswith("."):
            #    continue  # still in a run of dots
            if j is not None:
                last_char = tok_str[-1]
                next_tok = tokens[j].lstrip("▁Ġ ")
                if next_tok.startswith(last_char):
                    continue

            indices.append(current_sentence)
            current_sentence = []

        if current_sentence:
            indices.append(current_sentence)

        return indices

    @jaxtyped(typechecker=beartype)
    def get_association_matrix(
        self,
        inputs: BatchEncoding,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        indices_list: list[list[list[int]]] | None = None,
    ) -> list[Bool[torch.Tensor, "g lp"]]:
        """
        Creates the matrix to pass from one granularity level to ALL_TOKENS granularity level (finally used by the perturbator)

        Args:
            inputs (BatchEncoding): Tokenized inputs, the output of `self.tokenizer("some_text", return_tensors="pt", return_offsets_mapping=True, truncation=True)`
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Hugging-Face tokenizer used downstream.
            indices_list (list[list[list[int]]] | None): Precomputed indices list from `get_indices` method to avoid recomputation.

        Raises:
            NotImplementedError: if granularity level is unknown, raises NotImplementedError

        Returns:
            list[torch.Tensor]: the list of matrices used to transform a specific granularity mask to a general mask that can be used on tokens.
                The list has ``n`` elements, each element is of shape ``(g, lp)``
                    ``g`` is the padded sequence length in the specific granularity,
                    and ``lp`` is the padded sequence length.
        """
        if indices_list is None:
            # get indices correspondence between granularity and ALL_TOKENS
            indices_list = self.get_indices(inputs, tokenizer)

        # iterate over the samples
        assoc_matrix_list: list[Bool[torch.Tensor, g, lp]] = []
        for indices in indices_list:
            g = len(indices)
            lp = inputs["input_ids"].shape[1]  # type: ignore

            # set to true matching positions in the matrix
            assoc_matrix: Bool[torch.Tensor, g, lp] = torch.zeros((g, lp), dtype=torch.bool)
            for j, gran_indices in enumerate(indices):
                assoc_matrix[j, gran_indices] = True
            assoc_matrix_list.append(assoc_matrix)

        return assoc_matrix_list

    @jaxtyped(typechecker=beartype)
    def get_decomposition(
        self,
        inputs: BatchEncoding,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        return_text: bool = False,
        raw_text: list[str] | None = None,
        indices_list: list[list[list[int]]] | None = None,
    ) -> list[list[list[int]]] | list[list[str]]:
        """
        Returns the token decomposition at the requested granularity level.
        Thus the a list of list of token indices is returned.

        This method groups token ids according to the chosen granularity. It can
        either keep every token, ignore special tokens or merge tokens that
        belong to the same word.

        Args:
            inputs (BatchEncoding): Tokenized inputs to decompose, the output of
                `self.tokenizer("some_text", return_tensors="pt", return_offsets_mapping=True, truncation=True)`
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Huggingface tokenizer used downstream.
            return_text (bool, optional):
                If True, the text corresponding to the token indices is returned.
                If False, the token ids are returned. Defaults to False.
            raw_text (list[str] | None):
                Optional list of original (pre-tokenization) texts, one per batch element.
                If provided and tokenizer is fast, SENTENCE text will be reconstructed by slicing
                the original string using offsets to preserve exact formatting (e.g., "url.com")
                and casing. Leading whitespace is stripped.
            indices_list (list[list[list[int]]] | None): Precomputed indices list from `get_indices` method to avoid recomputation.

        Returns:
            list[list[int]]: A nested list where the first level
                indexes the batch elements, the second level corresponds to groups of
                tokens and the last level contains the token ids inside each group.

        Raises:
            ValueError: If the tokenizer is not provided and return_text is True.
        """
        if not tokenizer and return_text:
            raise ValueError(
                "Tokenizer must be provided if return_text is True. Please provide a PreTrainedTokenizer or PreTrainedTokenizerFast instance."
            )

        if indices_list is None:
            # get indices correspondence between granularity and ALL_TOKENS
            indices_list = self.get_indices(inputs, tokenizer)

        all_decompositions: list[list] = []
        for i, indices in enumerate(indices_list):
            input_ids: Int[torch.Tensor, "l"] = inputs["input_ids"][i]  # type: ignore
            # convert indices to token ids
            decomposition: list = []
            for gran_indices in indices:
                ids = [int(input_ids[idx].item()) for idx in gran_indices]
                # TODO: additional testing of this, it might cause issues for the TopKInputs concept interpretation method
                if return_text:
                    text = tokenizer.decode(ids, skip_special_tokens=self is not TextGranularity.ALL_TOKENS)  # type: ignore
                    decomposition.append(text)
                # Proposition for exact recontruction but too costly (I keep only if one day is necessary
                # to have exact text reconstruction for sentences, but it is not the case for now):
                # if return_text:
                #     assert tokenizer is not None
                #     if (
                #         (self is TextGranularity.SENTENCE or self is TextGranularity.WORD)
                #         and raw_text is not None
                #         and tokenizer.is_fast
                #         and isinstance(inputs, BatchEncoding)
                #         and getattr(inputs, "encodings", None)
                #     ):
                #         # Offsets are aligned with token positions in the encoding
                #         offsets = inputs.encodings[i].offsets  # type: ignore[attr-defined]

                #         start = offsets[gran_indices[0]][0]
                #         end = offsets[gran_indices[-1]][1]

                #         # exact substring + remove leading whitespace
                #         text = raw_text[i][start:end].lstrip()
                #         decomposition.append(text)
                #     else:
                #         # Default: decode (works everywhere), but strip leading spaces for SENTENCE
                #         text = tokenizer.decode(ids, skip_special_tokens=self is not TextGranularity.ALL_TOKENS)
                #         if self is TextGranularity.SENTENCE:
                #             text = text.lstrip()
                #         decomposition.append(text)
                else:
                    decomposition.append(ids)
            all_decompositions.append(decomposition)

        return all_decompositions

    def granularity_score_aggregation(  # noqa: PLR0912  # ignore too many branches
        self,
        contribution: torch.Tensor,
        granularity_aggregation_strategy: GranularityAggregationStrategy | None = None,
        inputs: BatchEncoding | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        aggregate_inputs: bool = False,
        aggregate_targets: bool = False,
        indices_list: list[list[list[int]]] | None = None,
    ) -> Float[torch.Tensor, "t g"]:
        """
        Aggregate contribution according to the specified granularity.

        There are four possibilities:
        |                    | Perturbation     | Gradient                     |
        |--------------------|------------------|------------------------------|
        | **Classification** | No aggregation   | Aggregate inputs             |
        | **Generation**     | Aggregate targets| Aggregate inputs and targets |

        The four possibilities are encoded by `aggregate_inputs` and `aggregate_targets`.

        For classification, the targets are classes which are not subject to granularity.

        For perturbations, the granularity is already encoded in the perturbation masks.

        Args:
            contribution (torch.Tensor):
                The contribution to aggregate. Shape: (t, l)

            granularity_aggregation_strategy (GranularityAggregationStrategy):
                The aggregation method to use.
                It should be an attribute of `GranularityAggregationStrategy`. Choices are:
                    - `MEAN`: average of contribution

                    - `MAX`: maximum contribution

                    - `MIN`: minimum contribution

                    - `SUM`: sum of contribution

                    - `SIGNED_MAX`: contribution with the largest absolute value, preserving its sign

            inputs (BatchEncoding | None):
                In the case of generation, this should include the generated tokens.
                Required if granularity is not `ALL_TOKENS`.

            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast | None):
                Required for TOKEN/WORD-level filtering.

            aggregate_inputs (bool):
                If True, aggregate inputs. Used for gradient-based methods.

            aggregate_targets (bool):
                If True, aggregate targets. Used for generation tasks.

            indices_list (list[list[list[int]]] | None):
                Precomputed indices list from `get_indices` method to avoid recomputation.

        Returns:
            torch.Tensor: The aggregated contribution.
        """
        # Classification + Perturbation
        if not aggregate_targets and not aggregate_inputs:
            return contribution

        if self == TextGranularity.ALL_TOKENS:
            return contribution

        if inputs is None:
            raise ValueError("Inputs are required for non ALL_TOKENS granularity.")

        if indices_list is None:
            # extract indices of contribution to keep from inputs
            indices_list = self.get_indices(inputs, tokenizer)  # type: ignore

        if len(indices_list) > 1:
            raise ValueError(
                "`granularity_score_aggregation` do not support batched inputs. Please provide a single input."
            )
        sample_indices: list[list[int]] = indices_list[0]

        if aggregate_inputs:
            # Gradient-based methods
            match self:
                case TextGranularity.TOKEN:
                    # convert contribution to tensor for faster indexing
                    indices = torch.tensor(sample_indices).squeeze(1)
                    contribution = contribution[:, indices]
                case TextGranularity.WORD | TextGranularity.PART_SENTENCE | TextGranularity.SENTENCE:
                    # verify aggregation strategy is not None:
                    if granularity_aggregation_strategy is None:
                        raise ValueError(
                            "granularity_aggregation_strategy must be provided for WORD or SENTENCE granularity."
                        )
                    # iterate over granularity elements
                    aggregated_contribution: Float[torch.Tensor, "t g"] = torch.zeros(
                        (contribution.shape[0], len(sample_indices)),
                        dtype=contribution.dtype,
                        device=contribution.device,
                    )
                    for aggregation_index, token_indices in enumerate(sample_indices):
                        # extract token contribution for each word/sentence
                        tokens_contribution: Float[torch.Tensor, "t gi"] = contribution[:, token_indices]

                        if tokens_contribution.dim() == 1 or tokens_contribution.shape[1] == 1:
                            # if only one token, no aggregation needed
                            aggregated_contribution[:, [aggregation_index]] = tokens_contribution
                        else:
                            # aggregate token contribution for each word/sentence
                            aggregated_contribution[:, [aggregation_index]] = (
                                granularity_aggregation_strategy.aggregate(tokens_contribution, dim=1)
                            )
                    contribution = aggregated_contribution
                case _:
                    raise NotImplementedError(f"Invalid granularity for aggregation: {self}")

        if aggregate_targets:
            # Generation-based methods

            # extract the target indices from the inputs indices
            t = contribution.shape[0]
            l = inputs["input_ids"].shape[1]  # type: ignore

            if t >= l:
                raise ValueError(
                    "Cannot aggregate targets if the number of targets is greater than the number of inputs."
                    "The input_ids should include the generated tokens."
                    f"Got {t} targets and {l} inputs."
                )

            first_target_index = l - t
            first_target_granular_index = None
            for i, token_indices in enumerate(sample_indices):
                if first_target_index in token_indices:
                    first_target_granular_index = i
                    break

            if first_target_granular_index is None:
                raise ValueError(
                    "Cannot find first target token in the granularity token indices. "
                    "Try changing the granularity, or raise an issue on GitHub."
                )

            # keep only indices relate to the targets
            target_indices = sample_indices[first_target_granular_index:]

            # shift reference index to the first target
            target_indices = [
                [index - first_target_index for index in granular_indices if index >= first_target_index]
                for granular_indices in target_indices
            ]

            # same match case and operations
            # different indices and dimension on which to aggregate
            match self:
                case TextGranularity.TOKEN:
                    if len(target_indices) != contribution.shape[0]:
                        # convert contribution to tensor for faster indexing
                        indices = torch.tensor(target_indices).squeeze(1)
                        contribution = contribution[indices, :]
                case TextGranularity.WORD | TextGranularity.PART_SENTENCE | TextGranularity.SENTENCE:
                    # verify aggregation strategy is not None:
                    if granularity_aggregation_strategy is None:
                        raise ValueError(
                            "granularity_aggregation_strategy must be provided for WORD or SENTENCE granularity."
                        )
                    # iterate over granularity elements
                    aggregated_contribution: Float[torch.Tensor, "g lg"] = torch.zeros(
                        (len(target_indices), contribution.shape[1]),
                        dtype=contribution.dtype,
                        device=contribution.device,
                    )
                    for aggregation_index, token_indices in enumerate(target_indices):
                        # extract token contribution for each word/sentence
                        tokens_contribution: Float[torch.Tensor, "gi lg"] = contribution[token_indices, :]

                        if tokens_contribution.dim() == 1 or tokens_contribution.shape[0] == 1:
                            # if only one token, no aggregation needed
                            aggregated_contribution[[aggregation_index], :] = tokens_contribution
                        else:
                            # aggregate token contribution for each word/sentence
                            aggregated_contribution[[aggregation_index], :] = (
                                granularity_aggregation_strategy.aggregate(tokens_contribution, dim=0)
                            )
                    contribution = aggregated_contribution
                case _:
                    raise NotImplementedError(f"Invalid granularity for aggregation: {self}")

        return contribution


class ImageGranularity(Granularity):
    """
    Enumeration of granularity levels for image classification explainers.

    Standalone enum (not a subclass of `TextGranularity`) because Python forbids
    extending enums that already have members. Duck-typed with `TextGranularity`
    via the same method names (`get_indices`, `get_association_matrix`,
    `granularity_score_aggregation`, `get_decomposition`).

    Positions are flattened row-major: index `i` <-> `(row=i//W, col=i%W)`.
    For `vit-base-patch16-224`: PIXEL has `H*W = 50176` singleton units,
    PATCH has `196` units of `patch_size*patch_size = 256` pixel positions each.
    """

    PIXEL = "pixel"
    PATCH = "patch"

    #    def granularity_resize(
    #         self,
    #         contribution: torch.Tensor,
    #         granularity_resize_strategy: GranularityResizeStrategy | None = None,
    #         inputs: TensorMapping | None = None,
    #         patch_size: int = 16,
    #         aggregate_inputs: bool = False,
    #     ) -> Float[torch.Tensor, "t g"]:
    #         """
    #         Resize `contribution` of shape `(t, l=H*W)` to the chosen granularity.

    #         Image-native counterpart of `TextGranularity.granularity_score_aggregation`: instead of
    #         treating a unit as an unordered bag of pixels and reducing it to a scalar, it keeps the 2-D
    #         spatial layout and interpolates the whole grid (see `GranularityResizeStrategy`).

    #         This is only used to downsample, never to upsample. It is also only ever called by gradient methods.

    #         - PIXEL: identity — pixels are already the finest unit.
    #         - PATCH: reshape `(t, H*W)` → `(t, H, W)` (row-major), spatially resize to the patch grid,
    #           then reflatten row-major to `(t, g)`.

    #         Args:
    #             contribution (torch.Tensor): Per-pixel attribution scores of shape `(t, l)`.
    #             granularity_resize_strategy (GranularityResizeStrategy | None):
    #                 Interpolation mode for PATCH. Required for PATCH, ignored for PIXEL.
    #             inputs (TensorMapping | None): Required for PATCH to read H, W from `pixel_values`.
    #             patch_size (int): Patch side length; the pixel grid is reduced by this factor.
    #             aggregate_inputs (bool): If False (perturbation methods), returns `contribution`
    #                 unchanged — granularity is already encoded in the perturbation masks.
    #                 If True (gradient methods), resizes the per-pixel field to granularity units.

    #         Returns:
    #             torch.Tensor: Resized contribution of shape `(t, g)`.
    #         """
    #         # Perturbation methods: granularity is already in the masks, return as-is
    #         if not aggregate_inputs:
    #             return contribution

    #         match self:
    #             case ImageGranularity.PIXEL:
    #                 # pixels are the finest unit — nothing to resize
    #                 return contribution
    #             case ImageGranularity.PATCH:
    #                 if granularity_resize_strategy is None:
    #                     raise ValueError("granularity_resize_strategy must be provided for PATCH granularity.")
    #                 if inputs is None:
    #                     raise ValueError("Inputs are required for PATCH granularity to read H, W.")

    #                 t = contribution.shape[0]
    #                 h, w = int(inputs["pixel_values"].shape[-2]), int(inputs["pixel_values"].shape[-1])
    #                 # row-major reshape — matches get_indices (index = row*W + col) and the inference
    #                 # wrapper's flatten(start_dim=1)
    #                 grid: Float[torch.Tensor, "t h w"] = contribution.reshape(t, h, w)
    #                 # pass patch_size by keyword: leaving output_size=None lets resize derive the
    #                 # patch grid (h // patch_size, w // patch_size). Passing it positionally would bind
    #                 # patch_size to output_size and resize to a literal patch_size x patch_size grid.
    #                 resized: Float[torch.Tensor, "t hg wg"] = granularity_resize_strategy.resize(
    #                     grid, patch_size=patch_size
    #                 )
    #                 # row-major reflatten back to (t, g = hp*wp)
    #                 return resized.reshape(t, -1)
    #             case _:
    #                 raise NotImplementedError(f"Granularity {self} not implemented for resizing.")

    def resize_to_image(
        self,
        contribution: Float[torch.Tensor, "t g"],
        resize_strategy: GranularityResizeStrategy,
        inputs: TensorMapping,
        patch_size: int = 16,
    ) -> Float[torch.Tensor, "t h w"]:
        """
        Expand per-granularity-unit scores `(t, g)` back to a pixel-resolution map `(t, H, W)`.

        This function expands granularity units back up to the pixel grid, so
        the visualization receives a display-ready field and never has to interpolate itself. The
        interpolation policy (which `resize_strategy` to pass) is decided by the caller, since it
        depends on the method family (NEAREST for gradient methods, the  resize_    strategy for masking
        methods).

        - PIXEL: `g == H*W`, so this is a pure reshape `(t, g) -> (t, H, W)` (no interpolation).
        - PATCH: reshape `(t, g) -> (t, gh, gw)` (row-major), then spatially resize to `(H, W)`.

        Args:
            contribution (torch.Tensor): Per-unit attribution scores of shape `(t, g)`.
            resize_strategy (GranularityResizeStrategy): Interpolation mode for PATCH; ignored for PIXEL.
            inputs (TensorMapping): Required to read H, W from `pixel_values`.
            patch_size (int): Patch side length; the granularity grid is `(H // patch_size, W // patch_size)`.

        Returns:
            torch.Tensor: Pixel-resolution map of shape `(t, H, W)`.
        """
        t = contribution.shape[0]
        h_in, w_in = int(inputs["pixel_values"].shape[-2]), int(inputs["pixel_values"].shape[-1])
        match self:
            case ImageGranularity.PIXEL:
                # pixels are already the finest unit — row-major reshape, no interpolation
                return contribution.reshape(t, h_in, w_in)
            case ImageGranularity.PATCH:
                # divisibility is guaranteed upstream by `_validate_batch_feature`
                gh, gw = h_in // patch_size, w_in // patch_size
                grid: Float[torch.Tensor, "t gh gw"] = contribution.reshape(t, gh, gw)
                return resize_strategy.resize(grid, output_size=(h_in, w_in))
            case _:
                raise NotImplementedError(f"Granularity {self} not implemented for image resizing.")
