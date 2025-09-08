"""
Pipeline framework for FIB-SEM analysis workflows.

This package exposes the main pipeline API while keeping optional
submodules truly optional for a smoother import experience.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
from pathlib import Path

# Import the main pipeline class eagerly
try:  # pragma: no cover - import guard
    from .main_pipeline import FIBSEMPipeline  # type: ignore[assignment]
except Exception:  # pragma: no cover - during incomplete installs
    FIBSEMPipeline = None  # type: ignore[assignment]



def create_default_pipeline(
    voxel_spacing: Optional[Tuple[float, float, float]] = None,
    config_path: Optional[Union[str, Path]] = None,
):
    """Light wrapper to build a default pipeline without hard-importing at module import time."""
    import importlib
    _mp = importlib.import_module(__name__ + ".main_pipeline")
    _fn = getattr(_mp, "create_default_pipeline")
    return _fn(voxel_spacing=voxel_spacing, config_path=config_path)


# Optional components; guard imports so the core pipeline remains usable
try:
    from .batch_processor import BatchProcessor  # type: ignore[attr-defined]
except Exception:
    BatchProcessor = None  # type: ignore[assignment]

try:
    from .visualization import FIBSEMVisualizer  # type: ignore[attr-defined]
except Exception:
    FIBSEMVisualizer = None  # type: ignore[assignment]

__all__ = ['FIBSEMPipeline', 'create_default_pipeline']
if BatchProcessor is not None:
    __all__.append('BatchProcessor')
if FIBSEMVisualizer is not None:
    __all__.append('FIBSEMVisualizer')

