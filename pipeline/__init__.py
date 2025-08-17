"""
Pipeline framework for FIB-SEM analysis workflows.

This module provides the main orchestration framework that coordinates
the interaction between different analysis components.
"""

from .main_pipeline import FIBSEMPipeline, create_default_pipeline
from .batch_processor import BatchProcessor
from .visualization import FIBSEMVisualizer

__all__ = [
    'FIBSEMPipeline', 'create_default_pipeline',
    'BatchProcessor',
    'FIBSEMVisualizer'
]

