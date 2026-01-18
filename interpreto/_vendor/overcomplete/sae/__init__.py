"""
Sparse Autoencoder (SAE) module of Overcomplete.
"""

from .base import SAE
from .dictionary import DictionaryLayer
from .archetypal_dictionary import RelaxedArchetypalDictionary
from .optimizer import CosineScheduler
from .losses import mse_l1
from .train import train_sae
from .modules import MLPEncoder, AttentionEncoder, ResNetEncoder
from .factory import EncoderFactory
from .jump_sae import JumpSAE, jump_relu, heaviside
from .topk_sae import TopKSAE
from .rasae import RATopKSAE, RAJumpSAE
from .qsae import QSAE
from .batchtopk_sae import BatchTopKSAE
from .mp_sae import MpSAE
from .omp_sae import OMPSAE
