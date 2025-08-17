"""
Core functionality for FIB-SEM data processing.

This module contains the fundamental components for data I/O, preprocessing,
and configuration management.
"""

from .data_io import load_fibsem_data, save_fibsem_data, FIBSEMData
from .config import FIBSEMConfig
from .preprocessing import (
    remove_noise, enhance_contrast, remove_artifacts,
    correct_drift, normalize_intensity
)

__all__ = [
    'load_fibsem_data', 'save_fibsem_data', 'FIBSEMData',
    'FIBSEMConfig',
    'remove_noise', 'enhance_contrast', 'remove_artifacts',
    'correct_drift', 'normalize_intensity'
]

