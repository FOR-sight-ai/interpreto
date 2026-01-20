"""
Module for Relaxed Archetypal SAE implementations.
For the implementation of the Relaxed Archetypal Dictionary, see archetypal_dictionary.py.
"""

import torch
from torch import nn

from .topk_sae import TopKSAE
from .jump_sae import JumpSAE
from .archetypal_dictionary import RelaxedArchetypalDictionary


class RATopKSAE(TopKSAE):
    """
    Relaxed Archetypal TopK SAE.

    This class implements a TopK SAE that utilizes a Relaxed Archetypal Dictionary.
    The dictionary atoms are initialized and constrained to be convex combinations
    of data points.

    For more information, see:
        - "Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in
          Large Vision Models" by T. Fel et al., ICML 2025 (https://arxiv.org/abs/2502.12892).

    Parameters
    ----------
    input_shape : int
        Dimensionality of the input data (excluding the batch dimension).
    nb_concepts : int
        Number of dictionary atoms (concepts).
    points : torch.Tensor
        The data points used to initialize/define the archetypes.
        Shape should be (num_points, input_shape).
    top_k : int
        Number of top activations to keep in the latent representation.
        By default, 10% sparsity is used.
    delta : float, optional
        Delta parameter for the archetypal dictionary, by default 1.0.
    use_multiplier : bool, optional
        Whether to use a learnable multiplier that parametrize the ball (e.g. if this parameter
        is 3 then the dictionary atoms are all on the ball of radius 3). By default True.
    **kwargs : dict, optional
        Additional arguments passed to the parent TopKSAE (e.g., encoder_module, device).
    """

    def __init__(self, input_shape, nb_concepts, points, top_k=None, delta=1.0, use_multiplier=True, **kwargs):
        assert isinstance(input_shape, int), "RATopKSAE input_shape must be an integer."

        super().__init__(input_shape=input_shape, nb_concepts=nb_concepts,
                         top_k=top_k, **kwargs)

        # enforce archetypal dictionary after the init of the parent class
        self.dictionary = RelaxedArchetypalDictionary(
            in_dimensions=input_shape,
            nb_concepts=nb_concepts,
            points=points,
            delta=delta,
            use_multiplier=use_multiplier,
            device=self.device
        )


class RAJumpSAE(JumpSAE):
    """
    Relaxed Archetypal Jump SAE.

    This class implements a Jump SAE that utilizes a Relaxed Archetypal Dictionary.
    The dictionary atoms are initialized and constrained to be convex combinations
    of data points.

    For more information, see:
        - "Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in
          Large Vision Models" by T. Fel et al., ICML 2025 (https://arxiv.org/abs/2502.12892).

    Parameters
    ----------
    input_shape : int
        Dimensionality of the input data (excluding the batch dimension).
    nb_concepts : int
        Number of dictionary atoms (concepts).
    points : torch.Tensor
        The data points used to initialize/define the archetypes.
        Shape should be (num_points, input_shape).
    bandwidth : float, optional
        Bandwidth parameter for the Jump SAE kernel, by default 1e-3.
    delta : float, optional
        Delta parameter for the archetypal dictionary, by default 1.0.
    use_multiplier : bool, optional
        Whether to use a learnable multiplier that parametrize the ball (e.g. if this parameter
        is 3 then the dictionary atoms are all on the ball of radius 3). By default True.
    **kwargs : dict, optional
        Additional arguments passed to the parent JumpSAE (e.g., encoder_module, device).
    """

    def __init__(self, input_shape, nb_concepts, points, bandwidth=1e-3, delta=1.0, use_multiplier=True, **kwargs):
        assert isinstance(input_shape, int), "RAJumpSAE input_shape must be an integer."

        super().__init__(input_shape=input_shape, nb_concepts=nb_concepts,
                         bandwidth=bandwidth, **kwargs)

        # enforce archetypal dictionary after the init of the parent class
        self.dictionary = RelaxedArchetypalDictionary(
            in_dimensions=input_shape,
            nb_concepts=nb_concepts,
            points=points,
            delta=delta,
            use_multiplier=use_multiplier,
            device=self.device
        )
