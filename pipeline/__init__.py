"""
Pipeline framework for FIB-SEM analysis workflows.

This module provides the main orchestration framework that coordinates
the interaction between different analysis components.
"""

from .main_pipeline import FIBSEMPipeline, create_default_pipeline

# Optional components; guard imports so the core pipeline remains usable
try:
    from .batch_processor import BatchProcessor  # type: ignore
except Exception:  # Module may not exist yet
    BatchProcessor = None  # type: ignore

try:
    from .visualization import FIBSEMVisualizer  # type: ignore
except Exception:
    FIBSEMVisualizer = None  # type: ignore

__all__ = ['FIBSEMPipeline', 'create_default_pipeline']
if BatchProcessor is not None:
    __all__.append('BatchProcessor')
if FIBSEMVisualizer is not None:
    __all__.append('FIBSEMVisualizer')

