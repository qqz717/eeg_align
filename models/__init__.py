"""
EEG WaveletVQ Framework - Models Module
"""

from .wavelet_vq_tokenizer import WaveletVQEncoder, WaveletDecomposer, VectorQuantizer
from .parallel_encoders import ParallelFeatureEncoder, TemporalEncoder, SpatialEncoder, SemanticEncoder
from .cross_modal_alignment import CrossModalAligner, TextEncoder, ContrastiveLearning
from .unified_framework import UnifiedEEGFramework

__all__ = [
    'WaveletVQEncoder',
    'WaveletDecomposer', 
    'VectorQuantizer',
    'ParallelFeatureEncoder',
    'TemporalEncoder',
    'SpatialEncoder', 
    'SemanticEncoder',
    'CrossModalAligner',
    'TextEncoder',
    'ContrastiveLearning',
    'UnifiedEEGFramework'
]