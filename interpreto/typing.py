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
Generic type annotations for Interpreto
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Generic, Protocol, TypeVar, runtime_checkable

import torch
from jaxtyping import Float, Int

TokenEmbedding = Float[torch.Tensor, "n ..."]
LatentActivations = Float[torch.Tensor, "n ..."]
ConceptsActivations = Float[torch.Tensor, "n cpt"]

T = TypeVar("T")
O_co = TypeVar("O_co", bound=object, covariant=True)

Nested = T | Iterable["Nested[T]"]
NestedIterable = Nested[Iterable[T]]
TensorMapping = MutableMapping[str, torch.Tensor]

SingleAttribution = (
    Float[torch.Tensor, "l"] | Float[torch.Tensor, "l c"] | Float[torch.Tensor, "l l_g"] | Float[torch.Tensor, "l l_t"]
)


@runtime_checkable
class HasWordIds(Protocol, Generic[O_co]):
    """
    Protocol for mapping having a word_ids method
    """

    def word_ids(self, index: int) -> Iterable[int | None]: ...


TensorMappingWithWordIds = HasWordIds[TensorMapping]

# Maybe consider NestedIterable rather that just iterable for model inputs ?
ModelInputs = str | TensorMapping | Iterable[str] | Iterable[TensorMapping]
GeneratedTarget = str | TensorMapping | torch.Tensor | Iterable[str | TensorMapping | torch.Tensor]
ClassificationTarget = (
    int | Int[torch.Tensor, "n"] | Int[torch.Tensor, "n t"] | Iterable[int] | Iterable[Int[torch.Tensor, "t"]]
)
TensorBaseline = torch.Tensor | float | int | None


class ConceptModelProtocol(Protocol):
    """Protocol for concept models."""

    @property
    def nb_concepts(self) -> int:
        """Number of concepts."""
        ...

    @property
    def fitted(self) -> bool:
        """Wether the concept model has been fitted."""
        ...

    def encode(self, x):
        """Encode the given activations using the concept model."""
        ...


class IncompatibilityError(TypeError):
    def __init__(self):
        message = (
            "Gradient-based methods require the model to be able to take inputs_embeds as input."
            + " However, it seems that the model does not support this feature."
            + " Please check that your model is compatible with the inputs_embeds parameter."
        )
        super().__init__(message)
