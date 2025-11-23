"""
Segmentation module for FIB-SEM data.

This module provides traditional and deep learning-based segmentation methods
for identifying structures in FIB-SEM datasets.
"""

import numpy as np
import os
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def segment_traditional(data: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
    """Apply traditional segmentation method."""
    if method == 'watershed':
        return _watershed_segmentation(data, params)
    elif method == 'thresholding':
        return _threshold_segmentation(data, params)
    elif method == 'morphology':
        return _morphological_segmentation(data, params)
    elif method == 'region_growing':
        return _region_growing_segmentation(data, params)
    elif method == 'graph_cuts':
        return _graph_cuts_segmentation(data, params)
    elif method == 'active_contours':
        return _active_contours_segmentation(data, params)
    elif method == 'slic':
        return _slic_segmentation(data, params)
    elif method == 'felzenszwalb':
        return _felzenszwalb_segmentation(data, params)
    elif method == 'random_walker':
        return _random_walker_segmentation(data, params)
    else:
        raise ValueError(f"Unknown traditional method: {method}")

def segment_deep_learning(data: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
    """Apply deep learning segmentation method."""
    if method == 'unet_2d':
        return _unet_2d_segmentation(data, params)
    elif method == 'unet_3d':
        return _unet_3d_segmentation(data, params)
    elif method == 'vnet':
        return _vnet_segmentation(data, params)
    elif method == 'attention_unet':
        return _attention_unet_segmentation(data, params)
    elif method == 'nnunet':
        return _nnunet_segmentation(data, params)
    elif method == 'sam3':
        return _sam3_segmentation(data, params)
    else:
        # Fallback for unknown methods
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

def _region_growing_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Perform region growing segmentation.
    
    This method is particularly effective for 3D SEM data where structures
    have similar intensity values within connected regions.
    """
    from skimage import measure, filters
    from scipy import ndimage
    
    # Get parameters
    seed_threshold = params.get('seed_threshold', 0.5)
    growth_threshold = params.get('growth_threshold', 0.1)
    connectivity = params.get('connectivity', 1)
    
    # Normalize data
    data_norm = (data - data.min()) / (data.max() - data.min())
    
    # Find seed points using local maxima
    if data.ndim == 3:
        threshold = filters.threshold_otsu(data)
        binary = data > threshold
        distance = ndimage.distance_transform_edt(binary)
        
        # Use watershed to find initial seeds
        from skimage import segmentation
        seeds = segmentation.watershed(-distance, watershed_line=True)
        seeds = seeds * (distance > seed_threshold * distance.max())
    else:
        # For 2D data
        threshold = filters.threshold_otsu(data)
        binary = data > threshold
        distance = ndimage.distance_transform_edt(binary)
        
        # Find local maxima as seeds
        from skimage.feature import peak_local_maxima
        local_maxima = peak_local_maxima(
            distance,
            min_distance=params.get('min_distance', 10),
            threshold_abs=seed_threshold * distance.max()
        )
        
        seeds = np.zeros_like(data, dtype=np.int32)
        for idx, coord in enumerate(local_maxima):
            seeds[coord[0], coord[1]] = idx + 1
    
    # Perform region growing
    labels = np.zeros_like(data, dtype=np.int32)
    visited = np.zeros_like(data, dtype=bool)
    
    # Get all unique seed labels
    seed_labels = np.unique(seeds[seeds > 0])
    
    for seed_label in seed_labels:
        seed_mask = seeds == seed_label
        seed_coords = np.where(seed_mask)
        
        if len(seed_coords[0]) == 0:
            continue
            
        # Start with seed region
        current_region = seed_mask.copy()
        labels[current_region] = seed_label
        visited[current_region] = True
        
        # Get seed intensity statistics
        seed_intensity = data[seed_mask].mean()
        
        # Iteratively grow the region
        changed = True
        while changed:
            changed = False
            
            # Find boundary pixels
            if data.ndim == 3:
                structure = ndimage.generate_binary_structure(3, connectivity)
            else:
                structure = ndimage.generate_binary_structure(2, connectivity)
                
            dilated = ndimage.binary_dilation(current_region, structure)
            boundary = dilated & ~current_region & ~visited
            
            if not np.any(boundary):
                break
                
            # Check intensity similarity
            boundary_coords = np.where(boundary)
            boundary_intensities = data[boundary_coords]
            
            # Add pixels that meet the growth criterion
            similar_mask = np.abs(boundary_intensities - seed_intensity) < growth_threshold
            
            if np.any(similar_mask):
                new_coords = tuple(coord[similar_mask] for coord in boundary_coords)
                new_region = np.zeros_like(data, dtype=bool)
                new_region[new_coords] = True
                
                current_region |= new_region
                labels[new_region] = seed_label
                visited[new_region] = True
                changed = True
    
    return labels

def _graph_cuts_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Perform graph cuts segmentation using PyMaxflow.
    
    Effective for 3D SEM data with clear foreground/background separation.
    """
    try:
        import maxflow
    except ImportError:
        logger.warning("PyMaxflow not available, falling back to watershed")
        return _watershed_segmentation(data, {})
    
    # Get parameters
    lambda_val = params.get('lambda', 1.0)
    sigma = params.get('sigma', 10.0)
    
    # Flatten data for graph processing
    if data.ndim == 3:
        # For 3D, process slice by slice to manage memory
        result = np.zeros_like(data, dtype=np.int32)
        for i in range(data.shape[0]):
            slice_data = data[i]
            slice_result = _graph_cuts_2d(slice_data, lambda_val, sigma)
            result[i] = slice_result + (i * slice_result.max() if slice_result.max() > 0 else 0)
        return result
    else:
        return _graph_cuts_2d(data, lambda_val, sigma)

def _graph_cuts_2d(data: np.ndarray, lambda_val: float, sigma: float) -> np.ndarray:
    """Helper function for 2D graph cuts."""
    import maxflow
    from skimage import filters
    
    h, w = data.shape
    
    # Create graph
    g = maxflow.Graph[float]()
    
    # Add nodes
    nodeids = g.add_grid_nodes((h, w))
    
    # Compute initial segmentation using Otsu
    threshold = filters.threshold_otsu(data)
    
    # Add terminal edges (source/sink connections)
    for i in range(h):
        for j in range(w):
            intensity = data[i, j]
            
            # Source capacity (foreground)
            source_cap = lambda_val * max(0, intensity - threshold)
            
            # Sink capacity (background)
            sink_cap = lambda_val * max(0, threshold - intensity)
            
            g.add_tedge(nodeids[i, j], source_cap, sink_cap)
    
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
            if g.get_segment(nodeids[i, j]) == 1:  # Source segment
                result[i, j] = 1
    
    # Label connected components
    from skimage import measure
    labels = measure.label(result)
    
    return labels

def _active_contours_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Perform active contours (snakes) segmentation.
    
    Good for smooth boundary detection in 3D SEM data.
    """
    from skimage import segmentation, filters, morphology
    from scipy import ndimage
    
    # Get parameters
    alpha = params.get('alpha', 0.015)  # Length weight
    beta = params.get('beta', 10)      # Smoothness weight
    gamma = params.get('gamma', 0.001) # Time step
    iterations = params.get('iterations', 100)
    
    if data.ndim == 3:
        # For 3D data, use morphological snakes on each slice
        result = np.zeros_like(data, dtype=np.int32)
        
        for i in range(data.shape[0]):
            slice_data = data[i]
            
            # Initial contour using threshold
            threshold = filters.threshold_otsu(slice_data)
            initial_mask = slice_data > threshold
            
            # Clean up initial mask
            initial_mask = morphology.binary_opening(initial_mask)
            initial_mask = morphology.binary_closing(initial_mask)
            
            if np.any(initial_mask):
                try:
                    # Apply morphological active contours
                    snake = segmentation.morphological_chan_vese(
                        slice_data,
                        num_iter=iterations,
                        init_level_set=initial_mask,
                        smoothing=3,
                        lambda1=1,
                        lambda2=1
                    )
                    
                    # Label the result
                    from skimage import measure
                    labels = measure.label(snake)
                    result[i] = labels + (i * labels.max() if labels.max() > 0 else 0)
                    
                except Exception as e:
                    logger.warning(f"Active contours failed for slice {i}: {e}")
                    # Fallback to simple thresholding
                    result[i] = initial_mask.astype(np.int32)
        
        return result
    else:
        # For 2D data
        threshold = filters.threshold_otsu(data)
        initial_mask = data > threshold
        
        # Clean up initial mask
        initial_mask = morphology.binary_opening(initial_mask)
        initial_mask = morphology.binary_closing(initial_mask)
        
        try:
            snake = segmentation.morphological_chan_vese(
                data,
                num_iter=iterations,
                init_level_set=initial_mask,
                smoothing=3,
                lambda1=1,
                lambda2=1
            )
            
            from skimage import measure
            labels = measure.label(snake)
            return labels
            
        except Exception as e:
            logger.warning(f"Active contours failed: {e}")
            return initial_mask.astype(np.int32)

def _slic_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Perform SLIC (Simple Linear Iterative Clustering) superpixel segmentation.
    
    Excellent for creating oversegmented regions as preprocessing for other methods.
    """
    from skimage import segmentation
    
    # Get parameters
    n_segments = params.get('n_segments', 1000)
    compactness = params.get('compactness', 10.0)
    sigma = params.get('sigma', 1.0)
    
    # Apply SLIC
    if data.ndim == 3:
        # For 3D data, use 3D SLIC
        labels = segmentation.slic(
            data,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            multichannel=False,
            convert2lab=False
        )
    else:
        # For 2D data
        labels = segmentation.slic(
            data,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            multichannel=False,
            convert2lab=False
        )
    
    return labels

def _felzenszwalb_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Perform Felzenszwalb's graph-based segmentation.
    
    Good for hierarchical segmentation of 3D SEM structures.
    """
    from skimage import segmentation
    
    # Get parameters
    scale = params.get('scale', 100)
    sigma = params.get('sigma', 0.5)
    min_size = params.get('min_size', 50)
    
    if data.ndim == 3:
        # For 3D data, apply to each slice and then merge
        result = np.zeros_like(data, dtype=np.int32)
        max_label = 0
        
        for i in range(data.shape[0]):
            slice_data = data[i]
            slice_labels = segmentation.felzenszwalb(
                slice_data,
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
        
        return result
    else:
        # For 2D data
        labels = segmentation.felzenszwalb(
            data,
            scale=scale,
            sigma=sigma,
            min_size=min_size,
            multichannel=False
        )
        
        return labels

def _random_walker_segmentation(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Perform random walker segmentation.
    
    Excellent for weakly supervised segmentation of 3D SEM data.
    """
    from skimage import segmentation, filters, morphology
    from scipy import ndimage
    
    # Get parameters
    beta = params.get('beta', 130)  # Penalization coefficient
    mode = params.get('mode', 'cg_mg')  # Solver mode
    
    # Generate markers automatically
    threshold = filters.threshold_otsu(data)
    
    # Background markers (low intensity)
    bg_threshold = threshold * 0.5
    background = data < bg_threshold
    background = morphology.binary_erosion(background, morphology.disk(3) if data.ndim == 2 else morphology.ball(2))
    
    # Foreground markers (high intensity)
    fg_threshold = threshold * 1.5
    foreground = data > fg_threshold
    foreground = morphology.binary_erosion(foreground, morphology.disk(3) if data.ndim == 2 else morphology.ball(2))
    
    # Create marker array
    markers = np.zeros_like(data, dtype=np.int32)
    markers[background] = 1  # Background label
    markers[foreground] = 2  # Foreground label
    
    if not np.any(markers == 1) or not np.any(markers == 2):
        logger.warning("Could not generate sufficient markers for random walker, falling back to watershed")
        return _watershed_segmentation(data, {})
    
    try:
        if data.ndim == 3:
            # For 3D data, process slice by slice to manage memory
            result = np.zeros_like(data, dtype=np.int32)
            
            for i in range(data.shape[0]):
                slice_data = data[i]
                slice_markers = markers[i]
                
                if np.any(slice_markers == 1) and np.any(slice_markers == 2):
                    slice_labels = segmentation.random_walker(
                        slice_data,
                        slice_markers,
                        beta=beta,
                        mode=mode
                    )
                    result[i] = slice_labels
                else:
                    # Fallback to threshold if no markers
                    result[i] = (slice_data > threshold).astype(np.int32)
            
            return result
        else:
            # For 2D data
            labels = segmentation.random_walker(
                data,
                markers,
                beta=beta,
                mode=mode
            )
            
            return labels
            
    except Exception as e:
        logger.warning(f"Random walker failed: {e}, falling back to watershed")
        return _watershed_segmentation(data, {})

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
