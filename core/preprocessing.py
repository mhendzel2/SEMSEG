"""
Preprocessing module for FIB-SEM data.

This module provides comprehensive preprocessing tools for removing common
FIB-SEM artifacts and enhancing image quality for improved segmentation.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
import logging
from scipy import ndimage
from skimage import filters, exposure, restoration, morphology

logger = logging.getLogger(__name__)

def remove_noise(image: np.ndarray, 
                 method: str = 'gaussian',
                 **kwargs) -> np.ndarray:
    """
    Remove noise from FIB-SEM images.
    
    Args:
        image: Input image array
        method: Noise removal method ('gaussian', 'bilateral', 'median', 'wiener')
        **kwargs: Method-specific parameters
        
    Returns:
        Denoised image
    """
    logger.info(f"Applying {method} noise removal")
    
    if method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return filters.gaussian(image, sigma=sigma, preserve_range=True)
    
    elif method == 'bilateral':
        sigma_color = kwargs.get('sigma_color', 0.1)
        sigma_spatial = kwargs.get('sigma_spatial', 1.0)
        
        # Normalize image to [0, 1] range for bilateral filter
        img_normalized = image.astype(np.float64)
        img_normalized = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min())
        
        denoised = restoration.denoise_bilateral(
            img_normalized, 
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial
        )
        
        # Convert back to original range
        denoised = denoised * (image.max() - image.min()) + image.min()
        return denoised.astype(image.dtype)
    
    elif method == 'median':
        size = kwargs.get('size', 3)
        if image.ndim == 3:
            return ndimage.median_filter(image, size=size)
        else:
            return filters.median(image, morphology.disk(size))
    
    elif method == 'wiener':
        noise_variance = kwargs.get('noise_variance', None)
        return restoration.wiener(image, noise=noise_variance)
    
    else:
        raise ValueError(f"Unknown noise removal method: {method}")

def enhance_contrast(image: np.ndarray,
                     method: str = 'clahe',
                     **kwargs) -> np.ndarray:
    """
    Enhance contrast in FIB-SEM images.
    
    Args:
        image: Input image array
        method: Contrast enhancement method ('clahe', 'histogram_eq', 'adaptive_eq')
        **kwargs: Method-specific parameters
        
    Returns:
        Contrast-enhanced image
    """
    logger.info(f"Applying {method} contrast enhancement")
    
    if method == 'clahe':
        clip_limit = kwargs.get('clip_limit', 2.0)
        kernel_size = kwargs.get('kernel_size', (8, 8, 8))
        
        if image.ndim == 3:
            try:
                import mclahe
                # mclahe expects image to be in range [0, 1]
                img_normalized = image.astype(np.float32)
                img_normalized = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min())

                return mclahe.mclahe(img_normalized, kernel_size=np.array(kernel_size), clip_limit=clip_limit)
            except ImportError:
                logger.warning("mclahe not found, falling back to slice-by-slice CLAHE")
                result = np.zeros_like(image)
                for i in range(image.shape[0]):
                    result[i] = exposure.equalize_adapthist(
                        image[i],
                        clip_limit=clip_limit,
                        nbins=256
                    )
                return result
        else:
            return exposure.equalize_adapthist(
                image,
                clip_limit=clip_limit,
                nbins=256
            )
    
    elif method == 'histogram_eq':
        if image.ndim == 3:
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = exposure.equalize_hist(image[i])
            return result
        else:
            return exposure.equalize_hist(image)
    
    elif method == 'adaptive_eq':
        if image.ndim == 3:
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = exposure.equalize_adapthist(image[i])
            return result
        else:
            return exposure.equalize_adapthist(image)
    
    else:
        raise ValueError(f"Unknown contrast enhancement method: {method}")

def remove_artifacts(image: np.ndarray,
                     artifact_types: List[str] = ['curtaining', 'charging'],
                     **kwargs) -> np.ndarray:
    """
    Remove common FIB-SEM artifacts.
    
    Args:
        image: Input image array
        artifact_types: List of artifacts to remove
        **kwargs: Artifact-specific parameters
        
    Returns:
        Artifact-corrected image
    """
    result = image.copy()
    
    for artifact_type in artifact_types:
        logger.info(f"Removing {artifact_type} artifacts")
        
        if artifact_type == 'curtaining':
            result = _remove_curtaining(result, **kwargs)
        elif artifact_type == 'charging':
            result = _remove_charging(result, **kwargs)
        elif artifact_type == 'drift':
            result = _correct_drift(result, **kwargs)
        else:
            logger.warning(f"Unknown artifact type: {artifact_type}")
    
    return result

def _remove_curtaining(image: np.ndarray, **kwargs) -> np.ndarray:
    """Remove curtaining artifacts (vertical stripes)."""
    # Curtaining appears as vertical stripes in FIB-SEM images
    # We can reduce it by subtracting the column-wise median
    
    if image.ndim == 3:
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            slice_img = image[i].astype(np.float32)
            
            # Calculate column-wise median
            col_median = np.median(slice_img, axis=0)
            
            # Subtract smoothed version to avoid over-correction
            smoothed_median = filters.gaussian(col_median, sigma=5)
            correction = col_median - smoothed_median
            
            # Apply correction
            corrected = slice_img - correction[np.newaxis, :]
            result[i] = np.clip(corrected, 0, slice_img.max())
        
        return result.astype(image.dtype)
    else:
        slice_img = image.astype(np.float32)
        col_median = np.median(slice_img, axis=0)
        smoothed_median = filters.gaussian(col_median, sigma=5)
        correction = col_median - smoothed_median
        corrected = slice_img - correction[np.newaxis, :]
        return np.clip(corrected, 0, slice_img.max()).astype(image.dtype)

def _remove_charging(image: np.ndarray, **kwargs) -> np.ndarray:
    """Remove charging artifacts (bright spots/regions)."""
    # Charging artifacts appear as abnormally bright regions
    # We can detect and correct them using morphological operations
    
    threshold_factor = kwargs.get('threshold_factor', 2.0)
    
    if image.ndim == 3:
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            slice_img = image[i].astype(np.float32)
            
            # Detect charging artifacts
            mean_intensity = np.mean(slice_img)
            std_intensity = np.std(slice_img)
            threshold = mean_intensity + threshold_factor * std_intensity
            
            charging_mask = slice_img > threshold
            
            if np.any(charging_mask):
                # Inpaint charging regions
                corrected = restoration.inpaint_biharmonic(
                    slice_img, 
                    charging_mask,
                    multichannel=False
                )
                result[i] = corrected
            else:
                result[i] = slice_img
        
        return result.astype(image.dtype)
    else:
        slice_img = image.astype(np.float32)
        mean_intensity = np.mean(slice_img)
        std_intensity = np.std(slice_img)
        threshold = mean_intensity + threshold_factor * std_intensity
        
        charging_mask = slice_img > threshold
        
        if np.any(charging_mask):
            corrected = restoration.inpaint_biharmonic(
                slice_img,
                charging_mask,
                multichannel=False
            )
            return corrected.astype(image.dtype)
        else:
            return image

def _correct_drift(image: np.ndarray, **kwargs) -> np.ndarray:
    """Correct drift artifacts in image stacks."""
    if image.ndim != 3:
        logger.warning("Drift correction requires 3D image stack")
        return image
    
    # Simple drift correction using cross-correlation
    reference_slice = image[image.shape[0] // 2]  # Use middle slice as reference
    corrected = np.zeros_like(image)
    corrected[image.shape[0] // 2] = reference_slice
    
    for i in range(image.shape[0]):
        if i == image.shape[0] // 2:
            continue
        
        # Calculate cross-correlation with reference
        correlation = ndimage.correlate(reference_slice.astype(np.float32),
                                       image[i].astype(np.float32))
        
        # Find peak correlation (this is simplified - real implementation would be more robust)
        peak_pos = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        # Calculate shift (simplified)
        shift_y = peak_pos[0] - reference_slice.shape[0] // 2
        shift_x = peak_pos[1] - reference_slice.shape[1] // 2
        
        # Apply shift correction
        if abs(shift_y) < 10 and abs(shift_x) < 10:  # Only apply small corrections
            corrected[i] = ndimage.shift(image[i], [shift_y, shift_x], order=1)
        else:
            corrected[i] = image[i]  # Keep original if shift is too large
    
    return corrected

def correct_drift(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Correct drift in FIB-SEM image stacks.
    
    Args:
        image: 3D image stack
        **kwargs: Drift correction parameters
        
    Returns:
        Drift-corrected image stack
    """
    return _correct_drift(image, **kwargs)

def normalize_intensity(image: np.ndarray,
                        method: str = 'minmax',
                        **kwargs) -> np.ndarray:
    """
    Normalize image intensity values.
    
    Args:
        image: Input image array
        method: Normalization method ('minmax', 'zscore', 'percentile')
        **kwargs: Method-specific parameters
        
    Returns:
        Normalized image
    """
    logger.info(f"Applying {method} intensity normalization")
    
    if method == 'minmax':
        min_val = kwargs.get('min_val', image.min())
        max_val = kwargs.get('max_val', image.max())
        
        normalized = (image.astype(np.float32) - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)
    
    elif method == 'zscore':
        mean_val = kwargs.get('mean_val', image.mean())
        std_val = kwargs.get('std_val', image.std())
        
        return (image.astype(np.float32) - mean_val) / std_val
    
    elif method == 'percentile':
        low_percentile = kwargs.get('low_percentile', 2)
        high_percentile = kwargs.get('high_percentile', 98)
        
        p_low = np.percentile(image, low_percentile)
        p_high = np.percentile(image, high_percentile)
        
        normalized = (image.astype(np.float32) - p_low) / (p_high - p_low)
        return np.clip(normalized, 0, 1)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def preprocess_fibsem_data(image: np.ndarray,
                           steps: List[str] = ['noise_reduction', 'contrast_enhancement'],
                           parameters: Optional[Dict[str, Dict[str, Any]]] = None) -> np.ndarray:
    """
    Apply complete preprocessing pipeline to FIB-SEM data.
    
    Args:
        image: Input image array
        steps: List of preprocessing steps to apply
        parameters: Dictionary of parameters for each step
        
    Returns:
        Preprocessed image
    """
    if parameters is None:
        parameters = {}
    
    result = image.copy()
    
    logger.info(f"Starting preprocessing pipeline with steps: {steps}")
    
    for step in steps:
        step_params = parameters.get(step, {})
        
        if step == 'noise_reduction':
            result = remove_noise(result, **step_params)
        elif step == 'contrast_enhancement':
            result = enhance_contrast(result, **step_params)
        elif step == 'artifact_removal':
            result = remove_artifacts(result, **step_params)
        elif step == 'drift_correction':
            result = correct_drift(result, **step_params)
        elif step == 'intensity_normalization':
            result = normalize_intensity(result, **step_params)
        else:
            logger.warning(f"Unknown preprocessing step: {step}")
    
    logger.info("Preprocessing pipeline completed")
    return result

def get_preprocessing_recommendations(image: np.ndarray) -> Dict[str, Any]:
    """
    Analyze image and provide preprocessing recommendations.
    
    Args:
        image: Input image array
        
    Returns:
        Dictionary of recommended preprocessing steps and parameters
    """
    recommendations = {
        'steps': [],
        'parameters': {}
    }
    
    # Analyze image properties
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    intensity_range = image.max() - image.min()
    
    # Check for noise (high standard deviation relative to mean)
    noise_ratio = std_intensity / mean_intensity
    if noise_ratio > 0.3:
        recommendations['steps'].append('noise_reduction')
        recommendations['parameters']['noise_reduction'] = {
            'method': 'bilateral' if noise_ratio > 0.5 else 'gaussian',
            'sigma': min(2.0, noise_ratio)
        }
    
    # Check for poor contrast (low intensity range)
    if intensity_range < 100:  # Assuming 8-bit or similar
        recommendations['steps'].append('contrast_enhancement')
        recommendations['parameters']['contrast_enhancement'] = {
            'method': 'clahe',
            'clip_limit': 0.03
        }
    
    # Check for potential artifacts (analyze intensity distribution)
    hist, bins = np.histogram(image.flatten(), bins=256)
    
    # Look for unusual peaks that might indicate artifacts
    peak_indices = np.where(hist > np.mean(hist) + 3 * np.std(hist))[0]
    if len(peak_indices) > 2:  # More than expected number of peaks
        recommendations['steps'].append('artifact_removal')
        recommendations['parameters']['artifact_removal'] = {
            'artifact_types': ['curtaining', 'charging']
        }
    
    # Always recommend intensity normalization for consistency
    recommendations['steps'].append('intensity_normalization')
    recommendations['parameters']['intensity_normalization'] = {
        'method': 'percentile',
        'low_percentile': 2,
        'high_percentile': 98
    }
    
    return recommendations

