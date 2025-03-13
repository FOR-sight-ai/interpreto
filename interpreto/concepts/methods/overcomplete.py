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
Concept Bottleneck Explainer based on Overcomplete framework.
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum

import torch
from overcomplete import optimization as oc_opt
from overcomplete import sae as oc_sae
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from interpreto.commons.model_wrapping.model_with_split_points import ActivationSelectionStrategy, ModelWithSplitPoints
from interpreto.concepts.base import ConceptBottleneckExplainer, check_fitted
from interpreto.typing import LatentActivations


class SAELoss:
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    SAE loss functions should be callables supporting the following signature."""

    @staticmethod
    @abstractmethod
    def __call__(
        x: torch.Tensor, x_hat: torch.Tensor, pre_codes: torch.Tensor, codes: torch.Tensor, dictionary: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Original input to the `concept_model`.
            x_hat (torch.Tensor): Reconstructed input from the `concept_model`.
            pre_codes (torch.Tensor): Concept latents before the activation function.
            codes (torch.Tensor): Concept latents after the activation function.
            dictionary (torch.Tensor): Learned dictionary of the `concept_model`,
                with shape `(nb_concepts, input_size)`.
        """
        ...


class MSELoss(SAELoss):
    """Standard MSE reconstruction loss"""

    @staticmethod
    def __call__(x: torch.Tensor, x_hat: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (x - x_hat).square().mean()


class DeadNeuronsReanimationLoss(SAELoss):
    """Loss function promoting reanimation of dead neurons."""

    @staticmethod
    def __call__(
        x: torch.Tensor, x_hat: torch.Tensor, pre_codes: torch.Tensor, codes: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        loss = (x - x_hat).square().mean()
        # is dead of shape (k) (nb concepts) and is 1 iff
        # not a single code has fired in the batch
        is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
        # we push the pre_codes (before relu) towards the positive orthant
        reanim_loss = (pre_codes * is_dead[None, :]).mean()
        loss -= reanim_loss * 1e-3
        return loss


class OvercompleteSAEClasses(Enum):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    Overcomplete concepts encoder-decoder classes for dictionary learning
    derived from the [Overcomplete SAE](https://kempnerinstitute.github.io/overcomplete/saes/vanilla/) class.

    Valid classes are:

    * **SAE** from Cunningham et al. (2023)[^1] and Bricken et al. (2023)[^2]
        ([Overcomplete implementation](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/sae/base.py)).
    * **TopKSAE** from Gao et al. (2024)[^3]
        ([Overcomplete implementation](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/sae/topk_sae.py)).
    * **BatchTopKSAE** from Bussmann et al. (2024)[^4]
        ([Overcomplete implementation](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/sae/batchtopk_sae.py)).
    * **JumpReLUSAE** from Rajamanoharan et al. (2024)[^5]
        ([Overcomplete implementation](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/sae/jump_sae.py)).

    [^1]:
        Huben, R., Cunningham, H., Smith, L. R., Ewart, A., Sharkey, L. [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://openreview.net/forum?id=F76bwRSLeK).
        The Twelfth International Conference on Learning Representations, 2024.
    [^2]:
        Bricken, T. et al., [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features),
        Transformer Circuits Thread, 2023.
    [^3]:
        Gao, L. et al., [Scaling and evaluating sparse autoencoders](https://openreview.net/forum?id=tcsZt9ZNKD).
        The Thirteenth International Conference on Learning Representations, 2025.
    [^4]:
        Bussmann, B., Leask, P., Nanda, N. [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410).
        Arxiv Preprint, 2024.
    [^5]:
        Rajamanoharan, S. et al., [Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders](https://arxiv.org/abs/2407.14435).
        Arxiv Preprint, 2024.
    """

    SAE = oc_sae.SAE
    TopKSAE = oc_sae.TopKSAE
    BatchTopKSAE = oc_sae.BatchTopKSAE
    JumpReLUSAE = oc_sae.JumpSAE


class OvercompleteOptimClasses(Enum):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    Overcomplete optimization classes for dictionary learning
    derived from the [Overcomplete BaseOptimDictionaryLearning](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/optimization/base.py) class.

    Valid classes are:

    * **NMF** from Lee and Seung (1999)[^1] ([Overcomplete implementation](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/optimization/nmf.py)).
    * **SemiNMF** from Ding et al. (2008)[^2] ([Overcomplete implementation](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/optimization/semi_nmf.py)).
    * **ConvexNMF** from Ding et al. (2008)[^2]([Overcomplete implementation](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/optimization/convex_nmf.py)).

    Several methods sourced from `scikit-learn` are also available:

    * **PCA** from Pearson (1901)[^3]
    * **ICA** from Hyvarinen and Oja (2000)[^4]
    * **KMeans**
    * **DictionaryLearning** from Mairal et al. (2009)[^5]
    * **SparsePCA**
    * **SVD**

    [^1]:
        Lee, D., Seung, H. [Learning the parts of objects by non-negative matrix factorization](https://doi.org/10.1038/44565).
        Nature, 401, 1999, pp. 788–791.
    [^2]:
        C. H. Q. Ding, T. Li and M. I. Jordan, [Convex and Semi-Nonnegative Matrix Factorizations](https://ieeexplore.ieee.org/document/4685898).
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(1), 2010, pp. 45-55
    [^3]:
        K. Pearson, [On lines and planes of closest fit to systems of points in space](https://doi.org/10.1080/14786440109462720).
        Philosophical Magazine, 2(11), 1901, pp. 559-572.
    [^4]:
        A. Hyvarinen and E. Oja, [Independent Component Analysis: Algorithms and Applications](https://www.sciencedirect.com/science/article/pii/S0893608000000265),
        Neural Networks, 13(4-5), 2000, pp. 411-430.
    [^5]:
        J. Mairal, F. Bach, J. Ponce, G. Sapiro, [Online dictionary learning for sparse coding](https://www.di.ens.fr/~fbach/mairal_icml09.pdf)
        Proceedings of the 26th Annual International Conference on Machine Learning, 2009, pp. 689-696.
    """

    # NMF = oc_opt.NMF # TODO: Add treatment to manage activations to ensure they are strictly positive
    SemiNMF = oc_opt.SemiNMF
    ConvexNMF = oc_opt.ConvexNMF
    PCA = oc_opt.SkPCA
    ICA = oc_opt.SkICA
    KMeans = oc_opt.SkKMeans
    DictionaryLearning = oc_opt.SkDictionaryLearning
    SparsePCA = oc_opt.SkSparsePCA
    SVD = oc_opt.SkSVD


class SAELossClasses(Enum):
    """Overcomplete SAE loss functions."""

    MSE = MSELoss
    DeadNeuronsReanimation = DeadNeuronsReanimationLoss


# TODO: Rename, remove Overcomplete prefix
class OvercompleteSAE(ConceptBottleneckExplainer):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    Implementation of a concept explainer using a
    [overcomplete.sae.SAE](https://kempnerinstitute.github.io/overcomplete/saes/vanilla/) variant as `concept_model`.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
            It should have at least one split point on which `concept_model` can be fitted.
        split_point (str | None): The split point used to train the `concept_model`. Default: `None`, set only when
            the concept explainer is fitted.
        concept_model (oc_sae.SAE): An [Overcomplete SAE](https://kempnerinstitute.github.io/overcomplete/saes/vanilla/)
            variant for concept extraction.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `encode_activations` operation is differentiable.
        has_differentiable_concept_decoder (bool): Whether the `decode_concepts` operation is differentiable.
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        concept_model_class: type[oc_sae.SAE],
        *,
        nb_concepts: int,
        activation_select_strategy: str | ActivationSelectionStrategy = ActivationSelectionStrategy.ALL,
        activation_select_indices: int | list[int] | tuple[int] | None = None,
        split_point: str | None = None,
        encoder_module: nn.Module | str | None = None,
        dictionary_params: dict | None = None,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize the concept bottleneck explainer based on the Overcomplete SAE framework.

        Args:
            model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
                It should have at least one split point on which a concept explainer can be trained.
            concept_model_class (type[oc_sae.SAE]): One of the supported [Overcomplete SAE](https://kempnerinstitute.github.io/overcomplete/saes/vanilla/)
                variants. Supported classes are available in [interpreto.concepts.OvercompleteSAEClasses]().
            nb_concepts (int): Size of the SAE concept space.
            split_point (str | None): The split point used to train the `concept_model`. If None, tries to use the
                split point of `model_with_split_points` if a single one is defined.
            encoder_module (nn.Module | str | None): Encoder module to use for the `concept_module`.
            dictionary_params (dict | None): Dictionary parameters to use for the `concept_module`.
            device (torch.device | str): Device to use for the `concept_module`.
            **kwargs (dict): Additional keyword arguments to pass to the `concept_module`.
                See the Overcomplete documentation of the provided `concept_model_class` for more details.
        """
        if not issubclass(concept_model_class, oc_sae.SAE):
            raise ValueError(
                "ConceptEncoderDecoder must be a subclass of `overcomplete.sae.SAE`.\n"
                "Use `interpreto.concepts.methods.OvercompleteSAEClasses` to get the list of available SAE methods."
            )
        self.model_with_split_points = model_with_split_points
        self.split_point = split_point

        # TODO: this will be replaced with a scan and a better way to select how to pick activations based on model class
        activations = self.model_with_split_points.get_activations(
            self.model_with_split_points._example_input,
            select_strategy=activation_select_strategy,
            select_indices=activation_select_indices,
        )
        concept_model = concept_model_class(
            input_shape=activations[self.split_point].shape[-1],
            nb_concepts=nb_concepts,
            encoder_module=encoder_module,
            dictionary_params=dictionary_params,
            device=device,
            **kwargs,
        )
        super().__init__(model_with_split_points, concept_model, self.split_point)
        self.has_differentiable_concept_encoder = True
        self.has_differentiable_concept_decoder = True

    @property
    def device(self) -> torch.device:
        """Get the device on which the concept model is stored."""
        return next(self.concept_model.parameters()).device

    @device.setter
    def device(self, device: torch.device) -> None:
        """Set the device on which the concept model is stored."""
        self.concept_model.to(device)

    def fit(
        self,
        activations: LatentActivations | dict[str, LatentActivations],
        *,
        use_amp: bool = False,
        batch_size: int = 1024,
        criterion: type[SAELoss] = MSELoss,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        scheduler_class: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        lr: float = 1e-3,
        nb_epochs: int = 20,
        clip_grad: float | None = None,
        monitoring: int | None = None,
        device: torch.device | str = "cpu",
        max_nan_fallbacks: int | None = 5,
        overwrite: bool = False,
    ) -> dict:
        """Fit an Overcomplete SAE model on the given activations.

        Args:
            activations (torch.Tensor | dict[str, torch.Tensor]): The activations used for fitting the `concept_model`.
                If a dictionary is provided, the activation corresponding to `split_point` will be used.
            use_amp (bool): Whether to use automatic mixed precision for fitting.
            criterion (interpreto.concepts.SAELoss): Loss criterion for the training of the `concept_model`.
            optimizer_class (type[torch.optim.Optimizer]): Optimizer for the training of the `concept_model`.
            scheduler_class (type[torch.optim.lr_scheduler.LRScheduler] | None): Learning rate scheduler for the
                training of the `concept_model`.
            lr (float): Learning rate for the training of the `concept_model`.
            nb_epochs (int): Number of epochs for the training of the `concept_model`.
            clip_grad (float | None): Gradient clipping value for the training of the `concept_model`.
            monitoring (int | None): Monitoring frequency for the training of the `concept_model`.
            device (torch.device | str): Device to use for the training of the `concept_model`.
            max_nan_fallbacks (int | None): Maximum number of fallbacks to use when NaNs are encountered during
                training. Ignored if use_amp is False.
            overwrite (bool): Whether to overwrite the current model if it has already been fitted.
                Default: False.

        Returns:
            A dictionary with training history logs.
        """
        split_activations = self.prepare_fit(activations, overwrite=overwrite)
        dataloader = DataLoader(TensorDataset(split_activations.detach()), batch_size=batch_size, shuffle=True)
        optimizer_kwargs = {"lr": lr}
        optimizer = optimizer_class(self.concept_model.parameters(), **optimizer_kwargs)  # type: ignore
        train_params = {
            "model": self.concept_model,
            "dataloader": dataloader,
            "criterion": criterion(),
            "optimizer": optimizer,
            "nb_epochs": nb_epochs,
            "clip_grad": clip_grad,
            "monitoring": monitoring,
            "device": device,
        }
        if scheduler_class is not None:
            scheduler = scheduler_class(optimizer)
            train_params["scheduler"] = scheduler

        if use_amp:
            train_method = oc_sae.train.train_sae_amp
            train_params["max_nan_fallbacks"] = max_nan_fallbacks
        else:
            train_method = oc_sae.train_sae
        log = train_method(**train_params)
        self.is_fitted = True
        return log

    @check_fitted
    def encode_activations(self, activations: LatentActivations) -> torch.Tensor:  # ConceptActivations
        """Encode the given activations using the `concept_model` encoder.

        Args:
            activations (torch.Tensor): The activations to encode.

        Returns:
            The encoded concept activations.
        """
        # SAEs.encode returns both codes (concepts activations) and pre_codes (before relu)
        _, codes = super().encode_activations(activations.to(self.device))
        return codes

    @check_fitted
    def decode_concepts(self, concepts: torch.Tensor) -> torch.Tensor:
        """Decode the given concepts using the `concept_model` decoder.

        Args:
            concepts (torch.Tensor): The concepts to decode.

        Returns:
            The decoded concept activations.
        """
        return self.concept_model.decode(concepts.to(self.device))  # type: ignore


# TODO: Rename, remove Overcomplete prefix
class OvercompleteDictionaryLearning(ConceptBottleneckExplainer):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    Implementation of a concept explainer using an
    [overcomplete.optimization.BaseOptimDictionaryLearning](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/optimization/base.py)
        (NMF and PCA variants) as `concept_model`.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
            It should have at least one split point on which `concept_model` can be fitted.
        split_point (str | None): The split point used to train the `concept_model`. Default: `None`, set only when
            the concept explainer is fitted.
        concept_model (oc_sae.SAE): An [Overcomplete BaseOptimDictionaryLearning](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/optimization/base.py)
            variant for concept extraction.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `encode_activations` operation is differentiable.
        has_differentiable_concept_decoder (bool): Whether the `decode_concepts` operation is differentiable.
    """

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        concept_model_class: type[oc_opt.BaseOptimDictionaryLearning],
        *,
        nb_concepts: int,
        split_point: str | None = None,
        device: torch.device | str = "cpu",
        **kwargs,
    ):
        """
        Initialize the concept bottleneck explainer based on the Overcomplete BaseOptimDictionaryLearning framework.

        Args:
            model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
                It should have at least one split point on which a concept explainer can be trained.
            concept_model_class (type[oc_opt.BaseOptimDictionaryLearning]): One of the supported [Overcomplete BaseOptimDictionaryLearning](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/optimization/base.py)
                variants for concept extraction.
            nb_concepts (int): Size of the SAE concept space.
            split_point (str | None): The split point used to train the `concept_model`. If None, tries to use the
                split point of `model_with_split_points` if a single one is defined.
            device (torch.device | str): Device to use for the `concept_module`.
            **kwargs (dict): Additional keyword arguments to pass to the `concept_module`.
                See the Overcomplete documentation of the provided `concept_model_class` for more details.
        """
        if not issubclass(concept_model_class, oc_opt.BaseOptimDictionaryLearning):
            raise ValueError(
                "ConceptEncoderDecoder must be a subclass of `overcomplete.optimization.BaseOptimDictionaryLearning`.\n"
                "Use `interpreto.concepts.methods.OvercompleteOptimClasses` to get the list of available SAE methods."
            )
        self.model_with_split_points = model_with_split_points
        self.split_point = split_point
        if concept_model_class.__name__ == "ConvexNMF":
            # TODO: see if we can support the pgd solver or have an easy way to set parameters
            kwargs["solver"] = "mu"
        concept_model = concept_model_class(
            nb_concepts=nb_concepts,
            device=device,  # type: ignore
            **kwargs,
        )
        super().__init__(model_with_split_points, concept_model, self.split_point)
        self.has_differentiable_concept_encoder = False if "NMF" in concept_model_class.__name__ else True
        self.has_differentiable_concept_decoder = True

    def fit(self, activations: LatentActivations | dict[str, LatentActivations], *, overwrite: bool = False, **kwargs):
        """Fit an Overcomplete OptimDictionaryLearning model on the given activations.

        Args:
            activations (torch.Tensor | dict[str, torch.Tensor]): The activations used for fitting the `concept_model`.
                If a dictionary is provided, the activation corresponding to `split_point` will be used.
            overwrite (bool): Whether to overwrite the current model if it has already been fitted.
                Default: False.
            **kwargs (dict): Additional keyword arguments to pass to the `concept_model`.
                See the Overcomplete documentation of the provided `concept_model` for more details.
        """
        split_activations = self.prepare_fit(activations, overwrite=overwrite)
        self.concept_model.fit(split_activations, **kwargs)
        self.is_fitted = True
