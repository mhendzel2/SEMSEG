"""
Segmentation module for FIB-SEM data.

This module provides traditional and deep learning-based segmentation methods
for identifying structures in FIB-SEM datasets.

Features:
- Traditional methods: watershed, thresholding, morphology, region growing, 
  graph cuts, active contours, SLIC, Felzenszwalb, random walker
- Deep learning methods: U-Net 2D/3D, V-Net, Attention U-Net, nnU-Net, SAM3
- Parameter validation with informative error messages
- Unified API returning both segmentation masks and confidence/probability maps
"""

import numpy as np
import os
import logging
from typing import Dict, Any, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types and Parameter Validation
# =============================================================================

class SegmentationResult(NamedTuple):
    """Result container for segmentation operations.
    
    Attributes:
        labels: Integer label array where each unique value represents a distinct object
        confidence: Probability/confidence map (0-1 range) for the segmentation
        metadata: Additional information about the segmentation (method, parameters, etc.)
    """
    labels: np.ndarray
    confidence: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = {}


class TraditionalMethod(Enum):
    """Supported traditional segmentation methods."""
    WATERSHED = 'watershed'
    THRESHOLDING = 'thresholding'
    MORPHOLOGY = 'morphology'
    REGION_GROWING = 'region_growing'
    GRAPH_CUTS = 'graph_cuts'
    ACTIVE_CONTOURS = 'active_contours'
    SLIC = 'slic'
    FELZENSZWALB = 'felzenszwalb'
    RANDOM_WALKER = 'random_walker'


class DeepLearningMethod(Enum):
    """Supported deep learning segmentation methods."""
    UNET_2D = 'unet_2d'
    UNET_3D = 'unet_3d'
    VNET = 'vnet'
    ATTENTION_UNET = 'attention_unet'
    NNUNET = 'nnunet'
    SAM3 = 'sam3'


@dataclass
class WatershedParams:
    """Parameters for watershed segmentation."""
    min_distance: int = 20
    threshold_rel: float = 0.6
    use_3d_distance_transform: bool = True
    
    def __post_init__(self):
        if self.min_distance < 1:
            raise ValueError(f"min_distance must be >= 1, got {self.min_distance}")
        if not 0 < self.threshold_rel <= 1:
            raise ValueError(f"threshold_rel must be in (0, 1], got {self.threshold_rel}")


@dataclass
class ThresholdingParams:
    """Parameters for thresholding segmentation."""
    method: str = 'otsu'
    block_size: Optional[int] = None
    offset: float = 0
    
    VALID_METHODS = ['otsu', 'li', 'yen', 'isodata', 'minimum', 'mean', 'triangle', 'local']
    
    def __post_init__(self):
        if self.method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got {self.method}")
        if self.method == 'local' and self.block_size is None:
            raise ValueError("block_size required for local thresholding")
        if self.block_size is not None and self.block_size < 3:
            raise ValueError(f"block_size must be >= 3, got {self.block_size}")


@dataclass
class RegionGrowingParams:
    """Parameters for region growing segmentation."""
    seed_threshold: float = 0.5
    growth_threshold: float = 0.1
    connectivity: int = 1
    min_distance: int = 10
    max_iterations: int = 1000
    
    def __post_init__(self):
        if not 0 < self.seed_threshold <= 1:
            raise ValueError(f"seed_threshold must be in (0, 1], got {self.seed_threshold}")
        if not 0 < self.growth_threshold <= 1:
            raise ValueError(f"growth_threshold must be in (0, 1], got {self.growth_threshold}")
        if self.connectivity not in [1, 2, 3]:
            raise ValueError(f"connectivity must be 1, 2, or 3, got {self.connectivity}")
        if self.min_distance < 1:
            raise ValueError(f"min_distance must be >= 1, got {self.min_distance}")


@dataclass
class GraphCutsParams:
    """Parameters for graph cuts segmentation."""
    lambda_val: float = 1.0
    sigma: float = 10.0
    use_3d: bool = True
    
    def __post_init__(self):
        if self.lambda_val <= 0:
            raise ValueError(f"lambda_val must be > 0, got {self.lambda_val}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")


@dataclass
class ActiveContoursParams:
    """Parameters for active contours (Chan-Vese) segmentation."""
    iterations: int = 100
    smoothing: int = 3
    lambda1: float = 1.0
    lambda2: float = 1.0
    use_3d: bool = True
    
    def __post_init__(self):
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {self.smoothing}")


@dataclass 
class DeepLearningParams:
    """Base parameters for deep learning segmentation."""
    model_path: Optional[str] = None
    num_classes: int = 2
    threshold: float = 0.5
    batch_size: int = 1
    
    def __post_init__(self):
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if not 0 < self.threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}")


def validate_input_data(data: np.ndarray, method: str) -> None:
    """Validate input data for segmentation.
    
    Args:
        data: Input array to validate
        method: Segmentation method name for error messages
        
    Raises:
        ValueError: If data is invalid
    """
    if data is None:
        raise ValueError(f"Input data cannot be None for {method}")
    if not isinstance(data, np.ndarray):
        raise ValueError(f"Input data must be numpy array, got {type(data)}")
    if data.ndim not in [2, 3]:
        raise ValueError(f"Input data must be 2D or 3D, got {data.ndim}D")
    if data.size == 0:
        raise ValueError(f"Input data cannot be empty for {method}")
    if not np.isfinite(data).all():
        logger.warning(f"Input data contains non-finite values for {method}")


def check_dependencies(required: list) -> Dict[str, bool]:
    """Check if required dependencies are available.
    
    Args:
        required: List of module names to check
        
    Returns:
        Dictionary mapping module names to availability status
    """
    availability = {}
    for module in required:
        try:
            __import__(module)
            availability[module] = True
        except ImportError:
            availability[module] = False
    return availability


# =============================================================================
# Main Entry Points
# =============================================================================

def segment_traditional(data: np.ndarray, method: str, params: Dict[str, Any]) -> Union[np.ndarray, SegmentationResult]:
    """Apply traditional segmentation method.
    
    Args:
        data: Input image array (2D or 3D)
        method: Segmentation method name (see TraditionalMethod enum)
        params: Method-specific parameters
        
    Returns:
        SegmentationResult with labels, confidence map, and metadata if return_result=True in params,
        otherwise just the label array for backward compatibility.
    """
    validate_input_data(data, method)
    return_result = params.pop('return_result', False)
    
    method_map = {
        'watershed': _watershed_segmentation,
        'thresholding': _threshold_segmentation,
        'morphology': _morphological_segmentation,
        'region_growing': _region_growing_segmentation,
        'graph_cuts': _graph_cuts_segmentation,
        'active_contours': _active_contours_segmentation,
        'slic': _slic_segmentation,
        'felzenszwalb': _felzenszwalb_segmentation,
        'random_walker': _random_walker_segmentation,
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown traditional method: {method}. Available: {list(method_map.keys())}")
    
    result = method_map[method](data, params)
    
    if return_result:
        return result
    # Backward compatibility: return just labels
    if isinstance(result, SegmentationResult):
        return result.labels
    return result


def segment_deep_learning(data: np.ndarray, method: str, params: Dict[str, Any]) -> Union[np.ndarray, SegmentationResult]:
    """Apply deep learning segmentation method.
    
    Args:
        data: Input image array (2D or 3D)
        method: Segmentation method name (see DeepLearningMethod enum)
        params: Method-specific parameters
        
    Returns:
        SegmentationResult with labels, confidence map, and metadata if return_result=True in params,
        otherwise just the label array for backward compatibility.
    """
    validate_input_data(data, method)
    return_result = params.pop('return_result', False)
    
    method_map = {
        'unet_2d': _unet_2d_segmentation,
        'unet_3d': _unet_3d_segmentation,
        'vnet': _vnet_segmentation,
        'attention_unet': _attention_unet_segmentation,
        'nnunet': _nnunet_segmentation,
        'sam3': _sam3_segmentation,
    }
    
    if method not in method_map:
        available = list(method_map.keys())
        logger.warning(f"Deep learning method '{method}' not available. Supported: {available}")
        logger.info("Falling back to watershed segmentation")
        return _watershed_segmentation(data, {})
    
    # Check for TensorFlow dependency for most DL methods
    if method in ['unet_2d', 'unet_3d', 'vnet', 'attention_unet', 'nnunet']:
        deps = check_dependencies(['tensorflow'])
        if not deps.get('tensorflow', False):
            logger.warning(f"TensorFlow not available for {method}, falling back to watershed")
            return _watershed_segmentation(data, {})
    
    result = method_map[method](data, params)
    
    if return_result:
        return result
    # Backward compatibility: return just labels
    if isinstance(result, SegmentationResult):
        return result.labels
    return result


def get_available_methods() -> Dict[str, Dict[str, Any]]:
    """Get information about all available segmentation methods.
    
    Returns:
        Dictionary with method info, dependencies, and availability status
    """
    # Check dependencies
    deps = check_dependencies(['tensorflow', 'maxflow', 'torch', 'sam3'])
    
    methods = {
        'traditional': {
            'watershed': {'available': True, 'dependencies': [], 'description': '3D watershed using distance transforms'},
            'thresholding': {'available': True, 'dependencies': [], 'description': 'Automatic thresholding (Otsu, Li, Yen, etc.)'},
            'morphology': {'available': True, 'dependencies': [], 'description': 'Morphological operations with connected components'},
            'region_growing': {'available': True, 'dependencies': [], 'description': 'Seed-based region growing'},
            'graph_cuts': {'available': deps.get('maxflow', False), 'dependencies': ['maxflow'], 'description': 'Min-cut/max-flow graph-based segmentation'},
            'active_contours': {'available': True, 'dependencies': [], 'description': 'Morphological Chan-Vese level sets'},
            'slic': {'available': True, 'dependencies': [], 'description': 'SLIC superpixel/supervoxel segmentation'},
            'felzenszwalb': {'available': True, 'dependencies': [], 'description': 'Graph-based hierarchical segmentation'},
            'random_walker': {'available': True, 'dependencies': [], 'description': 'Random walker with automatic markers'},
        },
        'deep_learning': {
            'unet_2d': {'available': deps.get('tensorflow', False), 'dependencies': ['tensorflow'], 'description': '2D U-Net (slice-by-slice for 3D)'},
            'unet_3d': {'available': deps.get('tensorflow', False), 'dependencies': ['tensorflow'], 'description': 'True 3D U-Net with volumetric context'},
            'vnet': {'available': deps.get('tensorflow', False), 'dependencies': ['tensorflow'], 'description': 'V-Net with residual connections'},
            'attention_unet': {'available': deps.get('tensorflow', False), 'dependencies': ['tensorflow'], 'description': 'U-Net with attention gates'},
            'nnunet': {'available': deps.get('tensorflow', False), 'dependencies': ['tensorflow'], 'description': 'Self-configuring nnU-Net style'},
            'sam3': {'available': deps.get('torch', False) and deps.get('sam3', False), 'dependencies': ['torch', 'sam3'], 'description': 'Segment Anything Model 3'},
        }
    }
    
    return methods


# =============================================================================
# Traditional Segmentation Methods
# =============================================================================

def _watershed_segmentation(data: np.ndarray, params: Dict[str, Any]) -> SegmentationResult:
    """Perform true 3D watershed segmentation.
    
    Uses 3D distance transforms and local maxima detection for consistent
    volumetric segmentation instead of slice-by-slice processing.
    """
    from scipy import ndimage
    from skimage import segmentation, filters, morphology
    from skimage.feature import peak_local_max
    from skimage import measure
    
    # Validate and get parameters
    min_distance = params.get('min_distance', 20)
    threshold_rel = params.get('threshold_rel', 0.6)
    use_3d = params.get('use_3d_distance_transform', True)
    
    if min_distance < 1:
        raise ValueError(f"min_distance must be >= 1, got {min_distance}")
    if not 0 < threshold_rel <= 1:
        raise ValueError(f"threshold_rel must be in (0, 1], got {threshold_rel}")
    
    # Convert to binary if needed
    if data.dtype != bool:
        threshold = filters.threshold_otsu(data)
        binary = data > threshold
    else:
        binary = data
    
    # Compute distance transform (true 3D)
    distance = ndimage.distance_transform_edt(binary)
    
    if data.ndim == 3 and use_3d:
        # TRUE 3D WATERSHED: Find 3D local maxima for consistent volumetric markers
        logger.info("Performing true 3D watershed segmentation")
        
        # Find local maxima in 3D
        coordinates = peak_local_max(
            distance,
            min_distance=min_distance,
            threshold_abs=threshold_rel * distance.max(),
            exclude_border=False
        )
        
        # Create markers from local maxima
        markers = np.zeros_like(distance, dtype=np.int32)
        for idx, coord in enumerate(coordinates):
            markers[tuple(coord)] = idx + 1
        
        logger.info(f"Found {len(coordinates)} 3D markers for watershed")
        
    else:
        # 2D processing
        if data.ndim == 3:
            logger.info("Processing 3D data slice-by-slice (use_3d=False)")
            markers = np.zeros_like(distance, dtype=np.int32)
            label_counter = 1
            
            for i in range(distance.shape[0]):
                slice_distance = distance[i]
                if slice_distance.max() > 0:
                    coords = peak_local_max(
                        slice_distance,
                        min_distance=min_distance,
                        threshold_abs=threshold_rel * slice_distance.max()
                    )
                    for coord in coords:
                        markers[i, coord[0], coord[1]] = label_counter
                        label_counter += 1
        else:
            # 2D data
            coordinates = peak_local_max(
                distance,
                min_distance=min_distance,
                threshold_abs=threshold_rel * distance.max()
            )
            
            markers = np.zeros_like(distance, dtype=np.int32)
            for idx, coord in enumerate(coordinates):
                markers[tuple(coord)] = idx + 1
    
    # Apply watershed
    labels = segmentation.watershed(-distance, markers, mask=binary)
    
    # Create confidence map from distance transform (normalized)
    confidence = distance / (distance.max() + 1e-8)
    confidence = confidence * binary  # Zero outside mask
    
    metadata = {
        'method': 'watershed',
        'min_distance': min_distance,
        'threshold_rel': threshold_rel,
        'use_3d': use_3d,
        'num_markers': int(markers.max()),
        'num_labels': int(labels.max())
    }
    
    return SegmentationResult(labels=labels, confidence=confidence.astype(np.float32), metadata=metadata)

def _threshold_segmentation(data: np.ndarray, params: Dict[str, Any]) -> SegmentationResult:
    """Perform threshold-based segmentation with multiple methods."""
    from skimage import filters, measure

    method = params.get('method', 'otsu')
    block_size = params.get('block_size', None)
    offset = params.get('offset', 0)
    
    valid_methods = ['otsu', 'li', 'yen', 'isodata', 'minimum', 'mean', 'triangle', 'local']
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method}")

    threshold = None
    if method == 'otsu':
        threshold = filters.threshold_otsu(data)
    elif method == 'li':
        threshold = filters.threshold_li(data)
    elif method == 'yen':
        threshold = filters.threshold_yen(data)
    elif method == 'isodata':
        threshold = filters.threshold_isodata(data)
    elif method == 'minimum':
        threshold = filters.threshold_minimum(data)
    elif method == 'mean':
        threshold = filters.threshold_mean(data)
    elif method == 'triangle':
        threshold = filters.threshold_triangle(data)
    elif method == 'local':
        if block_size is None:
            block_size = 35
        # For 3D, apply to each slice
        if data.ndim == 3:
            binary = np.zeros_like(data, dtype=bool)
            for i in range(data.shape[0]):
                local_thresh = filters.threshold_local(data[i], block_size, offset=offset)
                binary[i] = data[i] > local_thresh
        else:
            local_thresh = filters.threshold_local(data, block_size, offset=offset)
            binary = data > local_thresh
    else:
        threshold = filters.threshold_otsu(data)

    if threshold is not None:
        binary = data > (threshold + offset)
    
    labels = measure.label(binary)
    
    # Confidence based on distance from threshold
    if threshold is not None:
        confidence = np.abs(data.astype(np.float32) - threshold)
        confidence = confidence / (confidence.max() + 1e-8)
    else:
        confidence = binary.astype(np.float32)
    
    metadata = {
        'method': 'thresholding',
        'threshold_method': method,
        'threshold_value': float(threshold) if threshold is not None else 'local',
        'offset': offset,
        'num_labels': int(labels.max())
    }

    return SegmentationResult(labels=labels, confidence=confidence, metadata=metadata)

def _morphological_segmentation(data: np.ndarray, params: Dict[str, Any]) -> SegmentationResult:
    """Perform morphology-based segmentation."""
    from skimage import morphology, measure, filters

    # First threshold the data
    threshold = filters.threshold_otsu(data)
    binary = data > threshold

    # Apply morphological operations
    operation = params.get('operation', 'opening')
    radius = params.get('radius', 3)
    
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")

    if data.ndim == 3:
        selem = morphology.ball(radius)
    else:
        selem = morphology.disk(radius)

    if operation == 'opening':
        processed = morphology.opening(binary, selem)
    elif operation == 'closing':
        processed = morphology.closing(binary, selem)
    elif operation == 'erosion':
        processed = morphology.erosion(binary, selem)
    elif operation == 'dilation':
        processed = morphology.dilation(binary, selem)
    else:
        processed = binary

    # Label connected components
    labels = measure.label(processed)
    
    # Confidence is binary (full confidence where segmented)
    confidence = processed.astype(np.float32)
    
    metadata = {
        'method': 'morphology',
        'operation': operation,
        'radius': radius,
        'num_labels': int(labels.max())
    }

    return SegmentationResult(labels=labels, confidence=confidence, metadata=metadata)


def _region_growing_segmentation(data: np.ndarray, params: Dict[str, Any]) -> SegmentationResult:
    """
    Perform optimized region growing segmentation using vectorized operations.
    
    This method uses morphological reconstruction instead of nested Python loops
    for significantly improved performance, especially on 3D volumes.
    """
    from skimage import measure, filters, morphology
    from skimage.morphology import reconstruction
    from scipy import ndimage
    from skimage.feature import peak_local_max
    
    # Get and validate parameters
    seed_threshold = params.get('seed_threshold', 0.5)
    growth_threshold = params.get('growth_threshold', 0.1)
    connectivity = params.get('connectivity', 1)
    min_distance = params.get('min_distance', 10)
    
    if not 0 < seed_threshold <= 1:
        raise ValueError(f"seed_threshold must be in (0, 1], got {seed_threshold}")
    if not 0 < growth_threshold <= 1:
        raise ValueError(f"growth_threshold must be in (0, 1], got {growth_threshold}")
    if connectivity not in [1, 2, 3]:
        raise ValueError(f"connectivity must be 1, 2, or 3, got {connectivity}")
    if min_distance < 1:
        raise ValueError(f"min_distance must be >= 1, got {min_distance}")
    
    # Normalize data to [0, 1]
    data_min, data_max = data.min(), data.max()
    if data_max > data_min:
        data_norm = (data.astype(np.float64) - data_min) / (data_max - data_min)
    else:
        data_norm = np.zeros_like(data, dtype=np.float64)
    
    logger.info("Using vectorized morphological reconstruction for region growing")
    
    # Create binary mask using thresholding
    threshold = filters.threshold_otsu(data)
    binary_mask = data > threshold
    
    # Compute distance transform for seed finding
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # Find seed points using local maxima in 3D
    if data.ndim == 3:
        coordinates = peak_local_max(
            distance,
            min_distance=min_distance,
            threshold_abs=seed_threshold * distance.max(),
            exclude_border=False
        )
    else:
        coordinates = peak_local_max(
            distance,
            min_distance=min_distance,
            threshold_abs=seed_threshold * distance.max()
        )
    
    if len(coordinates) == 0:
        logger.warning("No seeds found for region growing, returning empty segmentation")
        return SegmentationResult(
            labels=np.zeros_like(data, dtype=np.int32),
            confidence=np.zeros_like(data, dtype=np.float32),
            metadata={'method': 'region_growing', 'num_seeds': 0, 'num_labels': 0}
        )
    
    # Create marker image for reconstruction
    marker_image = np.zeros_like(data_norm)
    for coord in coordinates:
        marker_image[tuple(coord)] = data_norm[tuple(coord)]
    
    # Use morphological reconstruction for fast region growing
    # The mask limits growth based on intensity similarity
    intensity_mask = data_norm.copy()
    
    # Erode mask slightly based on growth threshold
    if data.ndim == 3:
        structure = ndimage.generate_binary_structure(3, connectivity)
    else:
        structure = ndimage.generate_binary_structure(2, connectivity)
    
    # Morphological reconstruction - this is the vectorized region growing
    reconstructed = reconstruction(marker_image, intensity_mask, method='dilation')
    
    # Threshold the reconstruction result
    region_mask = reconstructed > (1 - growth_threshold)
    region_mask = region_mask & binary_mask
    
    # Label connected components
    labels = measure.label(region_mask)
    
    # Confidence map based on reconstruction values
    confidence = reconstructed.astype(np.float32)
    confidence = confidence * region_mask  # Zero outside regions
    
    metadata = {
        'method': 'region_growing',
        'seed_threshold': seed_threshold,
        'growth_threshold': growth_threshold,
        'connectivity': connectivity,
        'num_seeds': len(coordinates),
        'num_labels': int(labels.max())
    }
    
    logger.info(f"Region growing found {len(coordinates)} seeds, {labels.max()} labels")
    
    return SegmentationResult(labels=labels, confidence=confidence, metadata=metadata)


def _graph_cuts_segmentation(data: np.ndarray, params: Dict[str, Any]) -> SegmentationResult:
    """
    Perform graph cuts segmentation using PyMaxflow.
    
    Supports both 2D and true 3D processing. For 3D, can optionally process
    slice-by-slice or use full 3D graph (memory intensive).
    """
    try:
        import maxflow
    except ImportError:
        logger.warning("PyMaxflow not available, falling back to watershed")
        result = _watershed_segmentation(data, {})
        if isinstance(result, SegmentationResult):
            return result
        return SegmentationResult(labels=result, confidence=None, metadata={'fallback': 'watershed'})
    
    # Get and validate parameters
    lambda_val = params.get('lambda', 1.0)
    sigma = params.get('sigma', 10.0)
    use_3d = params.get('use_3d', False)  # Full 3D is memory intensive
    
    if lambda_val <= 0:
        raise ValueError(f"lambda must be > 0, got {lambda_val}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    
    if data.ndim == 3:
        if use_3d:
            # True 3D graph cuts (memory intensive)
            logger.info("Performing true 3D graph cuts (may be memory intensive)")
            labels, confidence = _graph_cuts_3d(data, lambda_val, sigma)
        else:
            # Process slice by slice
            logger.info("Processing graph cuts slice-by-slice")
            result = np.zeros_like(data, dtype=np.int32)
            confidence = np.zeros_like(data, dtype=np.float32)
            max_label = 0
            for i in range(data.shape[0]):
                slice_labels, slice_conf = _graph_cuts_2d_with_conf(data[i], lambda_val, sigma)
                if slice_labels.max() > 0:
                    slice_labels[slice_labels > 0] += max_label
                    max_label = slice_labels.max()
                result[i] = slice_labels
                confidence[i] = slice_conf
            labels = result
    else:
        labels, confidence = _graph_cuts_2d_with_conf(data, lambda_val, sigma)
    
    metadata = {
        'method': 'graph_cuts',
        'lambda': lambda_val,
        'sigma': sigma,
        'use_3d': use_3d,
        'num_labels': int(labels.max())
    }
    
    return SegmentationResult(labels=labels, confidence=confidence, metadata=metadata)


def _graph_cuts_2d_with_conf(data: np.ndarray, lambda_val: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function for 2D graph cuts with confidence map."""
    import maxflow
    from skimage import filters, measure
    
    h, w = data.shape
    
    # Create graph
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((h, w))
    
    # Compute initial segmentation using Otsu
    threshold = filters.threshold_otsu(data)
    
    # Compute capacity arrays for confidence
    source_caps = np.maximum(0, data - threshold) * lambda_val
    sink_caps = np.maximum(0, threshold - data) * lambda_val
    
    # Add terminal edges (source/sink connections)
    for i in range(h):
        for j in range(w):
            g.add_tedge(nodeids[i, j], source_caps[i, j], sink_caps[i, j])
    
    # Add n-link edges (neighbor connections)
    for i in range(h):
        for j in range(w):
            if i < h - 1:  # Vertical edge
                weight = np.exp(-((data[i, j] - data[i + 1, j]) ** 2) / (2 * sigma ** 2))
                g.add_edge(nodeids[i, j], nodeids[i + 1, j], weight, weight)
            if j < w - 1:  # Horizontal edge
                weight = np.exp(-((data[i, j] - data[i, j + 1]) ** 2) / (2 * sigma ** 2))
                g.add_edge(nodeids[i, j], nodeids[i, j + 1], weight, weight)
    
    # Solve max flow
    g.maxflow()
    
    # Get segmentation result
    result = np.zeros((h, w), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            if g.get_segment(nodeids[i, j]) == 1:
                result[i, j] = 1
    
    # Label connected components
    labels = measure.label(result)
    
    # Confidence based on difference of capacities
    confidence = (source_caps - sink_caps)
    confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8)
    
    return labels, confidence.astype(np.float32)


def _graph_cuts_3d(data: np.ndarray, lambda_val: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """True 3D graph cuts implementation."""
    import maxflow
    from skimage import filters, measure
    
    d, h, w = data.shape
    
    # Create graph
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((d, h, w))
    
    threshold = filters.threshold_otsu(data)
    
    # Add terminal edges
    source_caps = np.maximum(0, data - threshold) * lambda_val
    sink_caps = np.maximum(0, threshold - data) * lambda_val
    
    for z in range(d):
        for y in range(h):
            for x in range(w):
                g.add_tedge(nodeids[z, y, x], source_caps[z, y, x], sink_caps[z, y, x])
    
    # Add n-link edges in all 3 directions
    for z in range(d):
        for y in range(h):
            for x in range(w):
                if z < d - 1:  # Z direction
                    weight = np.exp(-((data[z, y, x] - data[z + 1, y, x]) ** 2) / (2 * sigma ** 2))
                    g.add_edge(nodeids[z, y, x], nodeids[z + 1, y, x], weight, weight)
                if y < h - 1:  # Y direction
                    weight = np.exp(-((data[z, y, x] - data[z, y + 1, x]) ** 2) / (2 * sigma ** 2))
                    g.add_edge(nodeids[z, y, x], nodeids[z, y + 1, x], weight, weight)
                if x < w - 1:  # X direction
                    weight = np.exp(-((data[z, y, x] - data[z, y, x + 1]) ** 2) / (2 * sigma ** 2))
                    g.add_edge(nodeids[z, y, x], nodeids[z, y, x + 1], weight, weight)
    
    g.maxflow()
    
    result = np.zeros((d, h, w), dtype=np.int32)
    for z in range(d):
        for y in range(h):
            for x in range(w):
                if g.get_segment(nodeids[z, y, x]) == 1:
                    result[z, y, x] = 1
    
    labels = measure.label(result)
    confidence = (source_caps - sink_caps)
    confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8)
    
    return labels, confidence.astype(np.float32)


def _active_contours_segmentation(data: np.ndarray, params: Dict[str, Any]) -> SegmentationResult:
    """
    Perform true 3D active contours using morphological Chan-Vese.
    
    Uses 3D level set evolution for consistent volumetric segmentation.
    """
    from skimage import segmentation, filters, morphology, measure
    from scipy import ndimage
    
    # Get and validate parameters
    iterations = params.get('iterations', 100)
    smoothing = params.get('smoothing', 3)
    lambda1 = params.get('lambda1', 1.0)
    lambda2 = params.get('lambda2', 1.0)
    use_3d = params.get('use_3d', True)
    
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if smoothing < 0:
        raise ValueError(f"smoothing must be >= 0, got {smoothing}")
    
    # Create initial mask from threshold
    threshold = filters.threshold_otsu(data)
    initial_mask = data > threshold
    
    # Clean up initial mask
    if data.ndim == 3:
        selem = morphology.ball(2)
    else:
        selem = morphology.disk(2)
    initial_mask = morphology.binary_opening(initial_mask, selem)
    initial_mask = morphology.binary_closing(initial_mask, selem)
    
    if data.ndim == 3 and use_3d:
        # TRUE 3D ACTIVE CONTOURS
        logger.info("Performing true 3D morphological Chan-Vese")
        
        try:
            # morphological_chan_vese supports 3D data directly
            snake = segmentation.morphological_chan_vese(
                data,
                num_iter=iterations,
                init_level_set=initial_mask,
                smoothing=smoothing,
                lambda1=lambda1,
                lambda2=lambda2
            )
            
            labels = measure.label(snake)
            confidence = snake.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"3D active contours failed: {e}, falling back to slice-by-slice")
            use_3d = False
    
    if data.ndim == 3 and not use_3d:
        # Slice-by-slice fallback
        result = np.zeros_like(data, dtype=np.int32)
        confidence = np.zeros_like(data, dtype=np.float32)
        max_label = 0
        
        for i in range(data.shape[0]):
            slice_data = data[i]
            slice_mask = initial_mask[i]
            
            if np.any(slice_mask):
                try:
                    snake = segmentation.morphological_chan_vese(
                        slice_data,
                        num_iter=iterations,
                        init_level_set=slice_mask,
                        smoothing=smoothing,
                        lambda1=lambda1,
                        lambda2=lambda2
                    )
                    
                    slice_labels = measure.label(snake)
                    if slice_labels.max() > 0:
                        slice_labels[slice_labels > 0] += max_label
                        max_label = slice_labels.max()
                    result[i] = slice_labels
                    confidence[i] = snake.astype(np.float32)
                    
                except Exception as e:
                    logger.warning(f"Active contours failed for slice {i}: {e}")
                    result[i] = slice_mask.astype(np.int32)
                    confidence[i] = slice_mask.astype(np.float32)
        
        labels = result
    
    elif data.ndim == 2:
        try:
            snake = segmentation.morphological_chan_vese(
                data,
                num_iter=iterations,
                init_level_set=initial_mask,
                smoothing=smoothing,
                lambda1=lambda1,
                lambda2=lambda2
            )
            
            labels = measure.label(snake)
            confidence = snake.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Active contours failed: {e}")
            labels = measure.label(initial_mask)
            confidence = initial_mask.astype(np.float32)
    
    metadata = {
        'method': 'active_contours',
        'iterations': iterations,
        'smoothing': smoothing,
        'lambda1': lambda1,
        'lambda2': lambda2,
        'use_3d': use_3d,
        'num_labels': int(labels.max())
    }
    
    return SegmentationResult(labels=labels, confidence=confidence, metadata=metadata)


def _slic_segmentation(data: np.ndarray, params: Dict[str, Any]) -> SegmentationResult:
    """
    Perform SLIC (Simple Linear Iterative Clustering) superpixel segmentation.
    
    Works natively in 3D for consistent supervoxel generation.
    """
    from skimage import segmentation
    
    # Get and validate parameters
    n_segments = params.get('n_segments', 1000)
    compactness = params.get('compactness', 10.0)
    sigma = params.get('sigma', 1.0)
    
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")
    if compactness <= 0:
        raise ValueError(f"compactness must be > 0, got {compactness}")
    
    # Apply SLIC (works for both 2D and 3D natively)
    try:
        # New API (scikit-image >= 0.19)
        labels = segmentation.slic(
            data,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            channel_axis=None  # grayscale
        )
    except TypeError:
        # Old API fallback
        labels = segmentation.slic(
            data,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            multichannel=False,
            convert2lab=False
        )
    
    # Confidence is uniform within each superpixel (we use normalized data as proxy)
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
    confidence = data_norm.astype(np.float32)
    
    metadata = {
        'method': 'slic',
        'n_segments': n_segments,
        'compactness': compactness,
        'sigma': sigma,
        'num_labels': int(labels.max())
    }
    
    return SegmentationResult(labels=labels, confidence=confidence, metadata=metadata)


def _felzenszwalb_segmentation(data: np.ndarray, params: Dict[str, Any]) -> SegmentationResult:
    """
    Perform Felzenszwalb's graph-based segmentation.
    
    For 3D data, uses slice-by-slice processing with label merging.
    """
    from skimage import segmentation
    
    # Get and validate parameters
    scale = params.get('scale', 100)
    sigma = params.get('sigma', 0.5)
    min_size = params.get('min_size', 50)
    
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")
    
    if data.ndim == 3:
        # For 3D data, apply to each slice
        result = np.zeros_like(data, dtype=np.int32)
        max_label = 0
        
        for i in range(data.shape[0]):
            try:
                # New API
                slice_labels = segmentation.felzenszwalb(
                    data[i],
                    scale=scale,
                    sigma=sigma,
                    min_size=min_size,
                    channel_axis=None
                )
            except TypeError:
                # Old API
                slice_labels = segmentation.felzenszwalb(
                    data[i],
                    scale=scale,
                    sigma=sigma,
                    min_size=min_size,
                    multichannel=False
                )
            
            # Offset labels to avoid conflicts
            if slice_labels.max() > 0:
                slice_labels[slice_labels > 0] += max_label
                max_label = slice_labels.max()
            
            result[i] = slice_labels
        
        labels = result
    else:
        try:
            labels = segmentation.felzenszwalb(
                data,
                scale=scale,
                sigma=sigma,
                min_size=min_size,
                channel_axis=None
            )
        except TypeError:
            labels = segmentation.felzenszwalb(
                data,
                scale=scale,
                sigma=sigma,
                min_size=min_size,
                multichannel=False
            )
    
    # Confidence based on segment boundaries
    from skimage.segmentation import find_boundaries
    if data.ndim == 3:
        boundaries = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[0]):
            boundaries[i] = find_boundaries(labels[i], mode='inner').astype(np.float32)
    else:
        boundaries = find_boundaries(labels, mode='inner').astype(np.float32)
    
    confidence = 1.0 - boundaries
    
    metadata = {
        'method': 'felzenszwalb',
        'scale': scale,
        'sigma': sigma,
        'min_size': min_size,
        'num_labels': int(labels.max())
    }
    
    return SegmentationResult(labels=labels, confidence=confidence, metadata=metadata)


def _random_walker_segmentation(data: np.ndarray, params: Dict[str, Any]) -> SegmentationResult:
    """
    Perform random walker segmentation with automatic marker generation.
    
    Supports true 3D processing for consistent volumetric results.
    """
    from skimage import segmentation, filters, morphology, measure
    from scipy import ndimage
    
    # Get and validate parameters
    beta = params.get('beta', 130)
    mode = params.get('mode', 'cg_mg')
    use_3d = params.get('use_3d', True)
    
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    
    # Generate markers automatically
    threshold = filters.threshold_otsu(data)
    
    # Create structuring element based on dimensionality
    if data.ndim == 3:
        selem = morphology.ball(2)
    else:
        selem = morphology.disk(3)
    
    # Background markers (low intensity regions, eroded)
    bg_threshold = threshold * 0.5
    background = data < bg_threshold
    background = morphology.binary_erosion(background, selem)
    
    # Foreground markers (high intensity regions, eroded)
    fg_threshold = threshold * 1.5
    foreground = data > fg_threshold
    foreground = morphology.binary_erosion(foreground, selem)
    
    # Create marker array
    markers = np.zeros_like(data, dtype=np.int32)
    markers[background] = 1  # Background label
    markers[foreground] = 2  # Foreground label
    
    if not np.any(markers == 1) or not np.any(markers == 2):
        logger.warning("Could not generate sufficient markers for random walker, falling back to watershed")
        result = _watershed_segmentation(data, {})
        if isinstance(result, SegmentationResult):
            return result
        return SegmentationResult(labels=result, confidence=None, metadata={'fallback': 'watershed'})
    
    try:
        if data.ndim == 3 and use_3d:
            # True 3D random walker
            logger.info("Performing true 3D random walker segmentation")
            probabilities = segmentation.random_walker(
                data,
                markers,
                beta=beta,
                mode=mode,
                return_full_prob=True
            )
            
            # Get foreground probability as confidence
            if probabilities.ndim == 4:  # (2, d, h, w)
                confidence = probabilities[1].astype(np.float32)  # Foreground probability
                labels_raw = np.argmax(probabilities, axis=0) + 1
            else:
                confidence = (probabilities == 2).astype(np.float32)
                labels_raw = probabilities
            
            # Convert to instance labels
            foreground_mask = labels_raw == 2
            labels = measure.label(foreground_mask)
            
        elif data.ndim == 3:
            # Slice-by-slice processing
            result = np.zeros_like(data, dtype=np.int32)
            confidence = np.zeros_like(data, dtype=np.float32)
            max_label = 0
            
            for i in range(data.shape[0]):
                slice_data = data[i]
                slice_markers = markers[i]
                
                if np.any(slice_markers == 1) and np.any(slice_markers == 2):
                    try:
                        probs = segmentation.random_walker(
                            slice_data,
                            slice_markers,
                            beta=beta,
                            mode=mode,
                            return_full_prob=True
                        )
                        if probs.ndim == 3:
                            slice_conf = probs[1].astype(np.float32)
                            slice_raw = np.argmax(probs, axis=0) + 1
                        else:
                            slice_conf = (probs == 2).astype(np.float32)
                            slice_raw = probs
                        
                        slice_labels = measure.label(slice_raw == 2)
                        if slice_labels.max() > 0:
                            slice_labels[slice_labels > 0] += max_label
                            max_label = slice_labels.max()
                        result[i] = slice_labels
                        confidence[i] = slice_conf
                    except Exception:
                        result[i] = (slice_data > threshold).astype(np.int32)
                        confidence[i] = (slice_data > threshold).astype(np.float32)
                else:
                    result[i] = (slice_data > threshold).astype(np.int32)
                    confidence[i] = (slice_data > threshold).astype(np.float32)
            
            labels = result
            
        else:
            # 2D data
            probabilities = segmentation.random_walker(
                data,
                markers,
                beta=beta,
                mode=mode,
                return_full_prob=True
            )
            
            if probabilities.ndim == 3:
                confidence = probabilities[1].astype(np.float32)
                labels_raw = np.argmax(probabilities, axis=0) + 1
            else:
                confidence = (probabilities == 2).astype(np.float32)
                labels_raw = probabilities
            
            labels = measure.label(labels_raw == 2)
            
    except Exception as e:
        logger.warning(f"Random walker failed: {e}, falling back to watershed")
        result = _watershed_segmentation(data, {})
        if isinstance(result, SegmentationResult):
            return result
        return SegmentationResult(labels=result, confidence=None, metadata={'fallback': 'watershed'})
    
    metadata = {
        'method': 'random_walker',
        'beta': beta,
        'mode': mode,
        'use_3d': use_3d,
        'num_labels': int(labels.max())
    }
    
    return SegmentationResult(labels=labels, confidence=confidence, metadata=metadata)

# Deep Learning Segmentation Methods

def _unet_2d_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Perform 2D U-Net segmentation on 3D data slice by slice.
    
    Effective for fine-grained segmentation of cellular structures in SEM data.
    """
    try:
        import tensorflow as tf
        from .unet import unet_model, predict_slices
    except ImportError:
        logger.error("TensorFlow not available for U-Net segmentation")
        return _watershed_segmentation(data, {})
    
    # Get parameters
    model_path = params.get('model_path', None)
    input_size = params.get('input_size', (256, 256, 1))
    num_classes = params.get('num_classes', 2)
    threshold = params.get('threshold', 0.5)
    
    # Load or create model
    if model_path and os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded pre-trained U-Net model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")
            logger.info("Creating new U-Net model (untrained)")
            model = unet_model(input_size, num_classes)
    else:
        logger.warning("No pre-trained model specified, creating untrained model")
        model = unet_model(input_size, num_classes)
    
    if data.ndim == 3:
        # Use slice prediction for 3D data
        predictions = predict_slices(model, data)
    else:
        # For 2D data, add batch and channel dimensions
        data_input = np.expand_dims(np.expand_dims(data, 0), -1)
        predictions = model.predict(data_input)
        predictions = predictions[0, :, :, 1]  # Take foreground class
    
    # Threshold predictions
    binary_mask = predictions > threshold
    
    # Label connected components
    from skimage import measure
    labels = measure.label(binary_mask)
    
    return labels

def _unet_3d_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Perform true 3D U-Net segmentation.
    
    Best for volumetric segmentation where 3D context is important.
    """
    try:
        import tensorflow as tf
    except ImportError:
        logger.error("TensorFlow not available for 3D U-Net segmentation")
        return _watershed_segmentation(data, {})
    
    # Get parameters
    model_path = params.get('model_path', None)
    patch_size = params.get('patch_size', (64, 64, 64))
    num_classes = params.get('num_classes', 2)
    threshold = params.get('threshold', 0.5)
    overlap = params.get('overlap', 0.25)
    
    # Load or create 3D U-Net model
    if model_path and os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded pre-trained 3D U-Net model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load 3D model from {model_path}: {e}")
            model = _create_3d_unet_model(patch_size + (1,), num_classes)
    else:
        logger.warning("No pre-trained 3D model specified, creating untrained model")
        model = _create_3d_unet_model(patch_size + (1,), num_classes)
    
    # Ensure data is 3D
    if data.ndim == 2:
        data = np.expand_dims(data, 0)
    
    # Predict using sliding window approach for large volumes
    predictions = _predict_3d_sliding_window(model, data, patch_size, overlap)
    
    # Threshold and label
    binary_mask = predictions > threshold
    from skimage import measure
    labels = measure.label(binary_mask)
    
    return labels

def _vnet_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Perform V-Net segmentation (3D variant optimized for medical imaging).
    
    Excellent for 3D SEM data with complex volumetric structures.
    """
    try:
        import tensorflow as tf
    except ImportError:
        logger.error("TensorFlow not available for V-Net segmentation")
        return _watershed_segmentation(data, {})
    
    # Get parameters
    model_path = params.get('model_path', None)
    patch_size = params.get('patch_size', (64, 64, 64))
    num_classes = params.get('num_classes', 2)
    threshold = params.get('threshold', 0.5)
    
    # Load or create V-Net model
    if model_path and os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded pre-trained V-Net model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load V-Net model from {model_path}: {e}")
            model = _create_vnet_model(patch_size + (1,), num_classes)
    else:
        logger.warning("No pre-trained V-Net model specified, creating untrained model")
        model = _create_vnet_model(patch_size + (1,), num_classes)
    
    # Ensure data is 3D
    if data.ndim == 2:
        data = np.expand_dims(data, 0)
    
    # Predict
    predictions = _predict_3d_sliding_window(model, data, patch_size, 0.25)
    
    # Threshold and label
    binary_mask = predictions > threshold
    from skimage import measure
    labels = measure.label(binary_mask)
    
    return labels

def _attention_unet_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Perform Attention U-Net segmentation.
    
    Uses attention mechanisms to focus on relevant features for better boundary detection.
    """
    try:
        import tensorflow as tf
    except ImportError:
        logger.error("TensorFlow not available for Attention U-Net segmentation")
        return _watershed_segmentation(data, {})
    
    # Get parameters
    model_path = params.get('model_path', None)
    input_size = params.get('input_size', (256, 256, 1))
    num_classes = params.get('num_classes', 2)
    threshold = params.get('threshold', 0.5)
    
    # Load or create Attention U-Net model
    if model_path and os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded pre-trained Attention U-Net model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load Attention U-Net model from {model_path}: {e}")
            model = _create_attention_unet_model(input_size, num_classes)
    else:
        logger.warning("No pre-trained Attention U-Net model specified, creating untrained model")
        model = _create_attention_unet_model(input_size, num_classes)
    
    if data.ndim == 3:
        # Process slice by slice
        result = np.zeros_like(data, dtype=np.int32)
        
        for i in range(data.shape[0]):
            slice_data = data[i]
            slice_input = np.expand_dims(np.expand_dims(slice_data, 0), -1)
            slice_pred = model.predict(slice_input, verbose=0)
            slice_binary = slice_pred[0, :, :, 1] > threshold
            
            from skimage import measure
            slice_labels = measure.label(slice_binary)
            result[i] = slice_labels + (i * slice_labels.max() if slice_labels.max() > 0 else 0)
        
        return result
    else:
        # For 2D data
        data_input = np.expand_dims(np.expand_dims(data, 0), -1)
        predictions = model.predict(data_input, verbose=0)
        binary_mask = predictions[0, :, :, 1] > threshold
        
        from skimage import measure
        labels = measure.label(binary_mask)
        return labels

def _nnunet_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Perform nnU-Net-style segmentation with automatic preprocessing.
    
    Self-configuring neural network that adapts to dataset characteristics.
    """
    # This is a simplified version - full nnU-Net requires extensive setup
    logger.warning("Full nnU-Net implementation requires separate installation and training")
    logger.info("Using simplified adaptive U-Net approach instead")
    
    # Analyze data characteristics for adaptive parameters
    data_shape = data.shape
    data_spacing = params.get('spacing', [1.0, 1.0, 1.0])
    
    # Adaptive patch size based on data dimensions
    if data.ndim == 3:
        # For 3D data, use adaptive patch size
        target_spacing = [min(s, 3.0) for s in data_spacing]  # Limit to reasonable spacing
        patch_size = [min(64, s // 2) for s in data_shape]
        patch_size = [max(32, p) for p in patch_size]  # Minimum patch size
        
        # Use 3D U-Net with adaptive parameters
        adaptive_params = params.copy()
        adaptive_params.update({
            'patch_size': tuple(patch_size),
            'num_classes': params.get('num_classes', 2),
            'threshold': params.get('threshold', 0.5)
        })
        
        return _unet_3d_segmentation(data, adaptive_params)
    else:
        # For 2D data, use 2D U-Net
        adaptive_params = params.copy()
        adaptive_params.update({
            'input_size': (min(512, data_shape[0]), min(512, data_shape[1]), 1),
            'num_classes': params.get('num_classes', 2),
            'threshold': params.get('threshold', 0.5)
        })
        
        return _unet_2d_segmentation(data, adaptive_params)

def _sam3_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Perform segmentation using Meta's SAM3 (Segment Anything Model 3).

    Supports text prompts, box prompts, and point prompts.
    """
    try:
        # Import sam3 here to avoid hard dependency
        # This assumes sam3 is installed in the environment
        # or the user has cloned it to a path we can add.
        import torch
        # Note: SAM3 import structure might vary based on installation
        # We assume standard usage from documentation
        from sam3 import SAM3
    except ImportError:
        logger.error("SAM3 or PyTorch not installed. Please install sam3.")
        logger.info("Fallback to watershed due to missing dependencies.")
        return _watershed_segmentation(data, {})

    # Get prompts
    text_prompt = params.get('text_prompt', None)
    box_prompt = params.get('box_prompt', None) # Expects [x1, y1, x2, y2]
    point_prompt = params.get('point_prompt', None)

    # Check model path or auto-download
    model_type = params.get('model_type', 'vit_h') # default large model
    checkpoint = params.get('checkpoint_path', None)

    # Initialize model
    # This is a placeholder for the actual SAM3 initialization API
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = SAM3(checkpoint=checkpoint, model_type=model_type)
        sam.to(device=device)
    except Exception as e:
        logger.error(f"Failed to initialize SAM3 model: {e}")
        return _watershed_segmentation(data, {})

    # Preprocessing
    # SAM expects RGB images (H, W, 3) generally.
    # If 3D data is passed, we process slice-by-slice or use video tracking if SAM3 supports it natively.
    # For simplicity in this integration, we handle 2D slices or 3D volume slice-by-slice unless
    # we leverage SAM3's video capabilities (which require a different input format usually).

    result = np.zeros_like(data, dtype=np.int32)

    if data.ndim == 3:
        # Process each slice
        for i in range(data.shape[0]):
            slice_data = data[i]

            # Convert to RGB if grayscale
            if slice_data.ndim == 2:
                img_rgb = np.stack((slice_data,)*3, axis=-1)
            else:
                img_rgb = slice_data

            # Normalize to 0-255 uint8
            if img_rgb.dtype != np.uint8:
                if img_rgb.max() > img_rgb.min():
                    img_rgb = ((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min()) * 255).astype(np.uint8)
                else:
                    # Handle constant image
                    img_rgb = np.zeros_like(img_rgb, dtype=np.uint8)

            # Inference
            try:
                # SAM3 API placeholder
                masks = sam.predict(
                    image=img_rgb,
                    text_prompt=text_prompt,
                    box=box_prompt,
                    points=point_prompt
                )

                # Assume masks is a list or array of binary masks
                # We take the first one or combine them
                if isinstance(masks, list):
                    mask = masks[0]
                else:
                    mask = masks

                result[i] = mask.astype(np.int32)

            except Exception as e:
                logger.warning(f"SAM3 inference failed on slice {i}: {e}")
    else:
        # 2D data
        if data.ndim == 2:
            img_rgb = np.stack((data,)*3, axis=-1)
        else:
            img_rgb = data

        if img_rgb.dtype != np.uint8:
            if img_rgb.max() > img_rgb.min():
                img_rgb = ((img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min()) * 255).astype(np.uint8)
            else:
                # Handle constant image
                img_rgb = np.zeros_like(img_rgb, dtype=np.uint8)

        try:
            masks = sam.predict(
                image=img_rgb,
                text_prompt=text_prompt,
                box=box_prompt,
                points=point_prompt
            )

            if isinstance(masks, list):
                mask = masks[0]
            else:
                mask = masks

            result = mask.astype(np.int32)

        except Exception as e:
            logger.error(f"SAM3 inference failed: {e}")
            return _watershed_segmentation(data, {})

    from skimage import measure
    labels = measure.label(result)
    return labels

# Helper functions for deep learning models

def _create_3d_unet_model(input_shape: Tuple[int, ...], num_classes: int):
    """Create a 3D U-Net model."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow required for 3D U-Net")
    
    inputs = tf.keras.Input(input_shape)
    
    # Encoder path
    conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling3D((2, 2, 2))(conv1)
    
    conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling3D((2, 2, 2))(conv2)
    
    # Bottleneck
    conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    
    # Decoder path
    up4 = layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv3)
    merge4 = layers.concatenate([conv2, up4], axis=4)
    conv4 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge4)
    conv4 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv4)
    
    up5 = layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4)
    merge5 = layers.concatenate([conv1, up5], axis=4)
    conv5 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(merge5)
    conv5 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv5)
    
    outputs = layers.Conv3D(num_classes, (1, 1, 1), activation='softmax')(conv5)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model

def _create_vnet_model(input_shape: Tuple[int, ...], num_classes: int):
    """Create a V-Net model with residual connections."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow required for V-Net")
    
    inputs = tf.keras.Input(input_shape)
    
    # Left side of V-Net (encoder)
    conv1_1 = layers.Conv3D(16, (5, 5, 5), activation='relu', padding='same')(inputs)
    conv1_2 = layers.Conv3D(16, (5, 5, 5), activation='relu', padding='same')(conv1_1)
    add1 = layers.Add()([inputs, conv1_2])  # Residual connection
    down1 = layers.Conv3D(32, (2, 2, 2), strides=(2, 2, 2), activation='relu')(add1)
    
    conv2_1 = layers.Conv3D(32, (5, 5, 5), activation='relu', padding='same')(down1)
    conv2_2 = layers.Conv3D(32, (5, 5, 5), activation='relu', padding='same')(conv2_1)
    add2 = layers.Add()([down1, conv2_2])
    down2 = layers.Conv3D(64, (2, 2, 2), strides=(2, 2, 2), activation='relu')(add2)
    
    # Bottleneck
    conv3_1 = layers.Conv3D(64, (5, 5, 5), activation='relu', padding='same')(down2)
    conv3_2 = layers.Conv3D(64, (5, 5, 5), activation='relu', padding='same')(conv3_1)
    add3 = layers.Add()([down2, conv3_2])
    
    # Right side of V-Net (decoder)
    up4 = layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), activation='relu')(add3)
    concat4 = layers.concatenate([add2, up4])
    conv4_1 = layers.Conv3D(32, (5, 5, 5), activation='relu', padding='same')(concat4)
    conv4_2 = layers.Conv3D(32, (5, 5, 5), activation='relu', padding='same')(conv4_1)
    
    up5 = layers.Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), activation='relu')(conv4_2)
    concat5 = layers.concatenate([add1, up5])
    conv5_1 = layers.Conv3D(16, (5, 5, 5), activation='relu', padding='same')(concat5)
    conv5_2 = layers.Conv3D(16, (5, 5, 5), activation='relu', padding='same')(conv5_1)
    
    outputs = layers.Conv3D(num_classes, (1, 1, 1), activation='softmax')(conv5_2)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model

def _create_attention_unet_model(input_shape: Tuple[int, ...], num_classes: int):
    """Create an Attention U-Net model."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow required for Attention U-Net")
    
    def attention_block(x, gating, inter_channels):
        """Attention mechanism block."""
        theta_x = layers.Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(x)
        phi_g = layers.Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(gating)
        
        add_xg = layers.add([theta_x, phi_g])
        relu_xg = layers.Activation('relu')(add_xg)
        
        psi = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same')(relu_xg)
        sigmoid_xg = layers.Activation('sigmoid')(psi)
        
        # Upsample attention weights
        upsample_psi = layers.UpSampling2D(size=(x.shape[1] // sigmoid_xg.shape[1], 
                                                 x.shape[2] // sigmoid_xg.shape[2]))(sigmoid_xg)
        
        y = layers.multiply([upsample_psi, x])
        
        return y
    
    inputs = tf.keras.Input(input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    # Bottleneck
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    
    # Decoder with attention
    up4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3)
    att4 = attention_block(conv2, conv3, 64)
    merge4 = layers.concatenate([att4, up4])
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(merge4)
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    
    up5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4)
    att5 = attention_block(conv1, conv4, 32)
    merge5 = layers.concatenate([att5, up5])
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(conv5)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model

def _predict_3d_sliding_window(model, data: np.ndarray, patch_size: Tuple[int, int, int], 
                               overlap: float = 0.25) -> np.ndarray:
    """Predict on 3D data using sliding window approach."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required for 3D prediction")
    
    d, h, w = data.shape
    pd, ph, pw = patch_size
    
    # Calculate step sizes
    step_d = max(1, int(pd * (1 - overlap)))
    step_h = max(1, int(ph * (1 - overlap)))
    step_w = max(1, int(pw * (1 - overlap)))
    
    # Initialize prediction array
    predictions = np.zeros_like(data, dtype=np.float32)
    counts = np.zeros_like(data, dtype=np.float32)
    
    # Sliding window prediction
    for z in range(0, d - pd + 1, step_d):
        for y in range(0, h - ph + 1, step_h):
            for x in range(0, w - pw + 1, step_w):
                # Extract patch
                patch = data[z:z+pd, y:y+ph, x:x+pw]
                
                # Add batch and channel dimensions
                patch_input = np.expand_dims(np.expand_dims(patch, 0), -1)
                
                # Predict
                patch_pred = model.predict(patch_input, verbose=0)
                
                # Extract foreground probability
                if patch_pred.shape[-1] > 1:
                    patch_pred = patch_pred[0, :, :, :, 1]  # Foreground class
                else:
                    patch_pred = patch_pred[0, :, :, :, 0]
                
                # Add to predictions with overlap handling
                predictions[z:z+pd, y:y+ph, x:x+pw] += patch_pred
                counts[z:z+pd, y:y+ph, x:x+pw] += 1
    
    # Average overlapping predictions
    predictions = predictions / np.maximum(counts, 1)
    
    return predictions
