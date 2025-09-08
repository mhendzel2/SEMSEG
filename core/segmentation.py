"""
Segmentation module for FIB-SEM data.

This module provides traditional and deep learning-based segmentation methods
for identifying structures in FIB-SEM datasets.
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def segment_traditional(data: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
    """Apply traditional segmentation method."""
    if method == 'watershed':
        return _watershed_segmentation(data, params)
    elif method == 'thresholding':
        return _threshold_segmentation(data, params)
    elif method == 'morphology':
        return _morphological_segmentation(data, params)
    else:
        raise ValueError(f"Unknown traditional method: {method}")

def segment_deep_learning(data: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
    """Apply deep learning segmentation method."""
    # This is a simplified implementation - real version would load trained models
    logger.warning(f"Deep learning method {method} not fully implemented, using watershed fallback")
    return _watershed_segmentation(data, {})

def _watershed_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Perform watershed segmentation."""
    from scipy import ndimage
    from skimage import segmentation, filters
    try:
        from skimage.feature import peak_local_maxima
    except ImportError:
        # Fallback for older scikit-image versions
        def peak_local_maxima(image, min_distance=1, threshold_abs=None, **kwargs):
            from skimage.morphology import local_maxima
            from skimage.measure import label, regionprops

            # Find local maxima
            maxima = local_maxima(image)

            if threshold_abs is not None:
                maxima = maxima & (image >= threshold_abs)

            # Label and get centroids
            labeled_maxima = label(maxima)
            props = regionprops(labeled_maxima)

            # Return coordinates
            coords = [tuple(map(int, prop.centroid)) for prop in props]
            return coords

    # Get parameters
    min_distance = params.get('min_distance', 20)
    threshold_rel = params.get('threshold_rel', 0.6)

    # Convert to binary if needed
    if data.dtype != bool:
        threshold = filters.threshold_otsu(data)
        binary = data > threshold
    else:
        binary = data

    # Compute distance transform
    distance = ndimage.distance_transform_edt(binary)

    # Find local maxima as markers
    if data.ndim == 3:
        # For 3D data, find maxima in each slice
        markers = np.zeros_like(distance, dtype=np.int32)
        label_counter = 1

        for i in range(distance.shape[0]):
            slice_distance = distance[i]
            if slice_distance.max() > 0:
                local_maxima = peak_local_maxima(
                    slice_distance,
                    min_distance=min_distance,
                    threshold_abs=threshold_rel * slice_distance.max()
                )

                for coord in local_maxima:
                    markers[i, coord[0], coord[1]] = label_counter
                    label_counter += 1
    else:
        # For 2D data
        local_maxima = peak_local_maxima(
            distance,
            min_distance=min_distance,
            threshold_abs=threshold_rel * distance.max()
        )

        markers = np.zeros_like(distance, dtype=np.int32)
        for idx, coord in enumerate(local_maxima):
            markers[coord[0], coord[1]] = idx + 1

    # Apply watershed
    labels = segmentation.watershed(-distance, markers, mask=binary)

    return labels

def _threshold_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Perform threshold-based segmentation."""
    from skimage import filters, measure

    method = params.get('method', 'otsu')

    if method == 'otsu':
        threshold = filters.threshold_otsu(data)
    elif method == 'li':
        threshold = filters.threshold_li(data)
    elif method == 'yen':
        threshold = filters.threshold_yen(data)
    else:
        threshold = filters.threshold_otsu(data)

    binary = data > threshold
    labels = measure.label(binary)

    return labels

def _morphological_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Perform morphology-based segmentation."""
    from skimage import morphology, measure, filters

    # First threshold the data
    threshold = filters.threshold_otsu(data)
    binary = data > threshold

    # Apply morphological operations
    operation = params.get('operation', 'opening')
    radius = params.get('radius', 3)

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

    return labels
