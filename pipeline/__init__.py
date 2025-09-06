"""
Pipeline framework for FIB-SEM analysis workflows.

This module provides the main orchestration framework that coordinates
the interaction between different analysis components.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .main_pipeline import FIBSEMPipeline, create_default_pipeline
except ImportError:
    # Handle relative import issues by using absolute imports
    from main_pipeline import FIBSEMPipeline, create_default_pipeline

__all__ = [
    'FIBSEMPipeline', 'create_default_pipeline'
]

