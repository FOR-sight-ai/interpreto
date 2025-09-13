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
from typing import Generic, TypeVar

import torch
from overcomplete import optimization as oc_opt
from overcomplete import sae as oc_sae
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from interpreto.concepts.base import ConceptAutoEncoderExplainer, check_fitted
from interpreto.model_wrapping.model_with_split_points import ModelWithSplitPoints
from interpreto.typing import LatentActivations

# Type variables for covariant generics
_SAE_co = TypeVar("_SAE_co", bound=oc_sae.SAE, covariant=True)
_BODL_co = TypeVar("_BODL_co", bound=oc_opt.BaseOptimDictionaryLearning, covariant=True)


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


class SAELossClasses(Enum):
    """
    Enumeration of possible loss functions for SAEs.

    To pass as the `criterion` parameter of `SAEExplainer.fit()`.

    Attributes:
        MSE (type[SAELoss]): Mean Squared Error loss.
        DeadNeuronsReanimation (type[SAELoss]): Loss function promoting reanimation of dead neurons.
    """

    MSE = MSELoss
    DeadNeuronsReanimation = DeadNeuronsReanimationLoss


class SAEExplainer(ConceptAutoEncoderExplainer[oc_sae.SAE], Generic[_SAE_co]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    Implementation of a concept explainer using a
    [overcomplete.sae.SAE](https://kempnerinstitute.github.io/overcomplete/saes/vanilla/) variant as `concept_model`.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
            It should have at least one split point on which `concept_model` can be fitted.
        split_point (str | None): The split point used to train the `concept_model`. Default: `None`, set only when
            the concept explainer is fitted.
        concept_model (overcomplete.sae.SAE): An [Overcomplete SAE](https://kempnerinstitute.github.io/overcomplete/saes/vanilla/)
            variant for concept extraction.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `encode_activations` operation is differentiable.
        has_differentiable_concept_decoder (bool): Whether the `decode_concepts` operation is differentiable.

    Examples:
        >>> import datasets
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from interpreto import ModelWithSplitPoints
        >>> from interpreto.concepts import VanillaSAE
        >>> from interpreto.concepts.interpretations import TopKInputs
        >>> CLS_TOKEN = ModelWithSplitPoints.activation_granularities.CLS_TOKEN
        >>> WORD = ModelWithSplitPoints.activation_granularities.WORD
        ...
        >>> dataset = datasets.load_dataset("stanfordnlp/imdb")["train"]["text"][:1000]
        >>> repo_id = "Qwen/Qwen3-0.6B"
        >>> model = AutoModelForCausalLM.from_pretrained(repo_id, device_map="auto")
        >>> tokenizer = AutoTokenizer.from_pretrained(repo_id)
        ...
        >>> # 1. Split your model in two parts
        >>> splitted_model = ModelWithSplitPoints(
        >>>     model, tokenizer=tokenizer, split_points=[5],
        >>> )
        ...
        >>> # 2. Compute a dataset of activations
        >>> activations = splitted_model.get_activations(
        >>>     dataset, activation_granularity=WORD
        >>> )
        ...
        >>> # 3. Fit a concept model on the dataset
        >>> explainer = VanillaSAE(splitted_model, nb_concepts=100, device="cuda")
        >>> explainer.fit(activations, lr=1e-3, nb_epochs=20, batch_size=1024)
        ...
        >>> # 4. Interpret the concepts
        >>> interpreter = TopKInputs(
        >>>     concept_explainer=explainer,
        >>>     activation_granularity=WORD,
        >>> )
        >>> interpretations = interpreter.interpret(
        >>>     inputs=dataset, latent_activations=activations
        >>> )
        ...
        >>> # Print the interpretations
        >>> for id, words in interpretations.items():
        >>>     print(f"Concept {id}: {list(words.keys()) if words else None}")
    """

    has_differentiable_concept_encoder = True
    has_differentiable_concept_decoder = True

    @property
    @abstractmethod
    def concept_model_class(self) -> type[oc_sae.SAE]:
        """
        Defines the concept model class to use for the explainer.

        Returns:
            concept_model_class (type[overcomplete.sae.SAE]): One of the supported [Overcomplete SAE](https://kempnerinstitute.github.io/overcomplete/saes/vanilla/)
                variants. Supported classes are available in [interpreto.concepts.SAEExplainerClasses]().
        """
        raise NotImplementedError

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        *,
        nb_concepts: int,
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
            nb_concepts (int): Size of the SAE concept space.
            split_point (str | None): The split point used to train the `concept_model`. If None, tries to use the
                split point of `model_with_split_points` if a single one is defined.
            encoder_module (nn.Module | str | None): Encoder module to use to construct the SAE, see [Overcomplete SAE documentation](https://kempnerinstitute.github.io/overcomplete/saes/vanilla/).
            dictionary_params (dict | None): Dictionary parameters to use to construct the SAE, see [Overcomplete SAE documentation](https://kempnerinstitute.github.io/overcomplete/saes/vanilla/).
            device (torch.device | str): Device to use for the `concept_module`.
            **kwargs (dict): Additional keyword arguments to pass to the `concept_module`.
                See the Overcomplete documentation of the provided `concept_model_class` for more details.
        """
        if not issubclass(self.concept_model_class, oc_sae.SAE):
            raise ValueError(
                "ConceptEncoderDecoder must be a subclass of `overcomplete.sae.SAE`.\n"
                "Use `interpreto.concepts.methods.SAEExplainerClasses` to get the list of available SAE methods."
            )
        self.model_with_split_points = model_with_split_points
        self.split_point: str = split_point  # type: ignore

        # TODO: this will be replaced with a scan and a better way to select how to pick activations based on model class
        shapes = self.model_with_split_points.get_latent_shape()
        concept_model = self.concept_model_class(
            input_shape=shapes[self.split_point][-1],
            nb_concepts=nb_concepts,
            encoder_module=encoder_module,
            dictionary_params=dictionary_params,
            device=device,
            **kwargs,
        )
        super().__init__(model_with_split_points, concept_model, self.split_point)

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
        device: torch.device | str | None = None,
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
        if device is None:
            device = self.device
        split_activations = self._prepare_fit(activations, overwrite=overwrite)
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
        self.concept_model.fitted = True

        # Manually set `BatchTopKSAEConcepts` `.training` argument to `False`
        # Because overcomplete does not do it and it changes the `.encode()` method drastically.
        if hasattr(self.concept_model, "training"):
            self.concept_model.training = False
        return log

    @check_fitted
    def encode_activations(self, activations: LatentActivations) -> torch.Tensor:  # ConceptsActivations
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


class DictionaryLearningExplainer(ConceptAutoEncoderExplainer[oc_opt.BaseOptimDictionaryLearning], Generic[_BODL_co]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    Implementation of a concept explainer using an
    [overcomplete.optimization.BaseOptimDictionaryLearning](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/optimization/base.py)
        (NMF and PCA variants) as `concept_model`.

    Attributes:
        model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
            It should have at least one split point on which `concept_model` can be fitted.
        split_point (str | None): The split point used to train the `concept_model`. Default: `None`, set only when
            the concept explainer is fitted.
        concept_model (overcomplete.optimization.BaseOptimDictionaryLearning): An [Overcomplete BaseOptimDictionaryLearning](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/optimization/base.py)
            variant for concept extraction.
        is_fitted (bool): Whether the `concept_model` was fit on model activations.
        has_differentiable_concept_encoder (bool): Whether the `encode_activations` operation is differentiable.
        has_differentiable_concept_decoder (bool): Whether the `decode_concepts` operation is differentiable.

    Examples:
        >>> import datasets
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from interpreto import ModelWithSplitPoints
        >>> from interpreto.concepts import ICAConcepts
        >>> from interpreto.concepts.interpretations import TopKInputs
        >>> CLS_TOKEN = ModelWithSplitPoints.activation_granularities.CLS_TOKEN
        >>> WORD = ModelWithSplitPoints.activation_granularities.WORD
        ...
        >>> dataset = datasets.load_dataset("stanfordnlp/imdb")["train"]["text"][:1000]
        >>> repo_id = "Qwen/Qwen3-0.6B"
        >>> model = AutoModelForCausalLM.from_pretrained(repo_id, device_map="auto")
        >>> tokenizer = AutoTokenizer.from_pretrained(repo_id)
        ...
        >>> # 1. Split your model in two parts
        >>> splitted_model = ModelWithSplitPoints(
        >>>     model, tokenizer=tokenizer, split_points=[5],
        >>> )
        ...
        >>> # 2. Compute a dataset of activations
        >>> activations = splitted_model.get_activations(
        >>>     dataset, activation_granularity=WORD
        >>> )
        ...
        >>> # 3. Fit a concept model on the dataset
        >>> explainer = ICAConcepts(splitted_model, nb_concepts=20)
        >>> explainer.fit(activations)
        ...
        >>> # 4. Interpret the concepts
        >>> interpreter = TopKInputs(
        >>>     concept_explainer=explainer,
        >>>     activation_granularity=WORD,
        >>> )
        >>> interpretations = interpreter.interpret(
        >>>     inputs=dataset, latent_activations=activations
        >>> )
        ...
        >>> # Print the interpretations
        >>> for id, words in interpretations.items():
        >>>     print(f"Concept {id}: {list(words.keys())}")
    """

    @property
    @abstractmethod
    def concept_model_class(self) -> type[oc_opt.BaseOptimDictionaryLearning]:
        """
        Defines the concept model class to use for the explainer.

        Returns:
            concept_model_class (type[overcomplete.optimization.BaseOptimDictionaryLearning]): One of the supported [Overcomplete BaseOptimDictionaryLearning](https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/optimization/base.py)
                variants for concept extraction.
        """
        raise NotImplementedError

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
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
            nb_concepts (int): Size of the SAE concept space.
            split_point (str | None): The split point used to train the `concept_model`. If None, tries to use the
                split point of `model_with_split_points` if a single one is defined.
            device (torch.device | str): Device to use for the `concept_module`.
            **kwargs (dict): Additional keyword arguments to pass to the `concept_module`.
                See the Overcomplete documentation of the provided `concept_model_class` for more details.
        """
        concept_model = self.concept_model_class(
            nb_concepts=nb_concepts,
            device=device,  # type: ignore
            **kwargs,
        )
        super().__init__(model_with_split_points, concept_model, split_point)

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
        split_activations = self._prepare_fit(activations, overwrite=overwrite)
        self.concept_model.fit(split_activations, **kwargs)


class VanillaSAEConcepts(SAEExplainer[oc_sae.SAE]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    `ConceptAutoEncoderExplainer` with the Vanilla SAE from Cunningham et al. (2023)[^1] and Bricken et al. (2023)[^2] as concept model.

    Vanilla SAE implementation from [overcomplete.sae.SAE](https://kempnerinstitute.github.io/overcomplete/saes/vanilla/) class.

    [^1]:
        Huben, R., Cunningham, H., Smith, L. R., Ewart, A., Sharkey, L. [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://openreview.net/forum?id=F76bwRSLeK).
        The Twelfth International Conference on Learning Representations, 2024.
    [^2]:
        Bricken, T. et al., [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features),
        Transformer Circuits Thread, 2023.

    """

    @property
    def concept_model_class(self) -> type[oc_sae.SAE]:
        return oc_sae.SAE


class TopKSAEConcepts(SAEExplainer[oc_sae.TopKSAE]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    `ConceptAutoEncoderExplainer` with the TopK SAE from Gao et al. (2024)[^3] as concept model.

    TopK SAE implementation from [overcomplete.sae.TopKSAE](https://kempnerinstitute.github.io/overcomplete/saes/topk_sae/) class.

    [^3]:
        Gao, L. et al., [Scaling and evaluating sparse autoencoders](https://openreview.net/forum?id=tcsZt9ZNKD).
        The Thirteenth International Conference on Learning Representations, 2025.
    """

    @property
    def concept_model_class(self) -> type[oc_sae.TopKSAE]:
        return oc_sae.TopKSAE


class BatchTopKSAEConcepts(SAEExplainer[oc_sae.BatchTopKSAE]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    `ConceptAutoEncoderExplainer` with the BatchTopK SAE from Bussmann et al. (2024)[^4] as concept model.

    BatchTopK SAE implementation from [overcomplete.sae.BatchTopKSAE](https://kempnerinstitute.github.io/overcomplete/saes/batchtopk_sae/) class.

    [^4]:
        Bussmann, B., Leask, P., Nanda, N. [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410).
        Arxiv Preprint, 2024.
    """

    @property
    def concept_model_class(self) -> type[oc_sae.BatchTopKSAE]:
        return oc_sae.BatchTopKSAE


class JumpReLUSAEConcepts(SAEExplainer[oc_sae.JumpSAE]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    `ConceptAutoEncoderExplainer` with the JumpReLU SAE from Rajamanoharan et al. (2024)[^5] as concept model.

    JumpReLU SAE implementation from [overcomplete.sae.JumpReLUSAE](https://kempnerinstitute.github.io/overcomplete/saes/jump_sae/) class.

    [^5]:
        Rajamanoharan, S. et al., [Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders](https://arxiv.org/abs/2407.14435).
        Arxiv Preprint, 2024.
    """

    @property
    def concept_model_class(self) -> type[oc_sae.JumpSAE]:
        return oc_sae.JumpSAE


class NMFConcepts(DictionaryLearningExplainer[oc_opt.NMF]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    `ConceptAutoEncoderExplainer` with the NMF from Lee and Seung (1999)[^1] as concept model.

    NMF implementation from [overcomplete.optimization.NMF](https://kempnerinstitute.github.io/overcomplete/optimization/nmf/) class.

    [^1]:
        Lee, D., Seung, H. [Learning the parts of objects by non-negative matrix factorization](https://doi.org/10.1038/44565).
        Nature, 401, 1999, pp. 788–791.
    """

    has_differentiable_concept_decoder = True

    @property
    def concept_model_class(self) -> type[oc_opt.NMF]:
        return oc_opt.NMF

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
        *,
        nb_concepts: int,
        split_point: str | None = None,
        device: torch.device | str = "cpu",
        force_relu: bool = False,
        **kwargs,
    ):
        """
        Initialize the concept bottleneck explainer based on the Overcomplete BaseOptimDictionaryLearning framework.

        Args:
            model_with_split_points (ModelWithSplitPoints): The model to apply the explanation on.
                It should have at least one split point on which a concept explainer can be trained.
            nb_concepts (int): Size of the SAE concept space.
            split_point (str | None): The split point used to train the `concept_model`. If None, tries to use the
                split point of `model_with_split_points` if a single one is defined.
            device (torch.device | str): Device to use for the `concept_module`.
            force_relu (bool): Whether to force the activations to be positive.
            **kwargs (dict): Additional keyword arguments to pass to the `concept_module`.
                See the Overcomplete documentation of the provided `concept_model_class` for more details.
        """
        super().__init__(
            model_with_split_points,
            nb_concepts=nb_concepts,
            split_point=split_point,
            device=device,
            **kwargs,
        )
        self.force_relu = force_relu

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
        split_activations = self._prepare_fit(activations, overwrite=overwrite)
        if (split_activations < 0).any():
            if self.force_relu:
                split_activations = torch.nn.functional.relu(split_activations)
            else:
                raise ValueError(
                    "The activations should be positive. If you want to force the activations to be positive, "
                    "use the `NMFConcepts(..., force_relu=True)`."
                )
        self.concept_model.fit(split_activations, **kwargs)

    @check_fitted
    def encode_activations(self, activations: LatentActivations) -> torch.Tensor:  # ConceptsActivations
        """Encode the given activations using the `concept_model` encoder.

        Args:
            activations (LatentActivations): The activations to encode.

        Returns:
            The encoded concept activations.
        """
        self._sanitize_activations(activations)
        if (activations < 0).any():
            if self.force_relu:
                activations = torch.nn.functional.relu(activations)
            else:
                raise ValueError(
                    "The activations should be positive. If you want to force the activations to be positive, "
                    "use the `NMFConcepts(..., force_relu=True)`."
                )
        return self.concept_model.encode(activations)  # type: ignore


class SemiNMFConcepts(DictionaryLearningExplainer[oc_opt.SemiNMF]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    `ConceptAutoEncoderExplainer` with the SemiNMF from Ding et al. (2008)[^2] as concept model.

    SemiNMF implementation from [overcomplete.optimization.SemiNMF](https://kempnerinstitute.github.io/overcomplete/optimization/semi_nmf/) class.

    [^2]:
        C. H. Q. Ding, T. Li and M. I. Jordan, [Convex and Semi-Nonnegative Matrix Factorizations](https://ieeexplore.ieee.org/document/4685898).
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(1), 2010, pp. 45-55
    """

    has_differentiable_concept_decoder = True

    @property
    def concept_model_class(self) -> type[oc_opt.SemiNMF]:
        return oc_opt.SemiNMF


class ConvexNMFConcepts(DictionaryLearningExplainer[oc_opt.ConvexNMF]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    `ConceptAutoEncoderExplainer` with the ConvexNMF from Ding et al. (2008)[^2] as concept model.

    ConvexNMF implementation from [overcomplete.optimization.ConvexNMF](https://kempnerinstitute.github.io/overcomplete/optimization/convex_nmf/) class.

    [^2]:
        C. H. Q. Ding, T. Li and M. I. Jordan, [Convex and Semi-Nonnegative Matrix Factorizations](https://ieeexplore.ieee.org/document/4685898).
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(1), 2010, pp. 45-55
    """

    has_differentiable_concept_decoder = True

    @property
    def concept_model_class(self) -> type[oc_opt.ConvexNMF]:
        return oc_opt.ConvexNMF

    def __init__(
        self,
        model_with_split_points: ModelWithSplitPoints,
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
            nb_concepts (int): Size of the SAE concept space.
            split_point (str | None): The split point used to train the `concept_model`. If None, tries to use the
                split point of `model_with_split_points` if a single one is defined.
            device (torch.device | str): Device to use for the `concept_module`.
            **kwargs (dict): Additional keyword arguments to pass to the `concept_module`.
                See the Overcomplete documentation of the provided `concept_model_class` for more details.
        """
        kwargs["solver"] = "mu"  # TODO: see if we can support the pgd solver or have an easy way to set parameters
        super().__init__(
            model_with_split_points=model_with_split_points,
            nb_concepts=nb_concepts,
            split_point=split_point,
            device=device,
            **kwargs,
        )


class DictionaryLearningConcepts(DictionaryLearningExplainer[oc_opt.SkDictionaryLearning]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    `ConceptAutoEncoderExplainer` with the Dictionary Learning concepts from Mairal et al. (2009)[^5] as concept model.

    Dictionary Learning implementation from [overcomplete.optimization.SkDictionaryLearning](https://kempnerinstitute.github.io/overcomplete/optimization/sklearn/) class.

    [^5]:
        J. Mairal, F. Bach, J. Ponce, G. Sapiro, [Online dictionary learning for sparse coding](https://www.di.ens.fr/~fbach/mairal_icml09.pdf)
        Proceedings of the 26th Annual International Conference on Machine Learning, 2009, pp. 689-696.
    """

    @property
    def concept_model_class(self) -> type[oc_opt.SkDictionaryLearning]:
        return oc_opt.SkDictionaryLearning


class SparsePCAConcepts(DictionaryLearningExplainer[oc_opt.SkSparsePCA]):
    """Code: [:octicons-mark-github-24: `concepts/methods/overcomplete.py` ](https://github.com/FOR-sight-ai/interpreto/blob/dev/interpreto/concepts/methods/overcomplete.py)

    `ConceptAutoEncoderExplainer` with SparsePCA as concept model.

    SparsePCA implementation from [overcomplete.optimization.SkSparsePCA](https://kempnerinstitute.github.io/overcomplete/optimization/sklearn/) class.
    """

    @property
    def concept_model_class(self) -> type[oc_opt.SkSparsePCA]:
        return oc_opt.SkSparsePCA
