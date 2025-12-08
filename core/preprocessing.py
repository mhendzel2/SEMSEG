"""
Preprocessing module for FIB-SEM data.

This module provides comprehensive preprocessing tools for removing common
FIB-SEM artifacts and enhancing image quality for improved segmentation.

Features:
- Noise removal: Gaussian, bilateral, median, Wiener, non-local means
- Contrast enhancement: CLAHE, histogram equalization, adaptive
- Artifact correction: curtaining, charging, drift (with phase correlation)
- Intensity normalization: min-max, z-score, percentile
- Parameter validation with sensible defaults and ranges
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from scipy import ndimage
from skimage import filters, exposure, restoration, morphology

logger = logging.getLogger(__name__)


# =============================================================================
# Parameter Validation Classes
# =============================================================================

@dataclass
class NoiseRemovalParams:
    """Parameters for noise removal with validation."""
    method: str = 'gaussian'
    sigma: float = 1.0
    sigma_color: float = 0.1  # For bilateral
    sigma_spatial: float = 1.0  # For bilateral
    size: int = 3  # For median
    patch_size: int = 5  # For NL-means
    patch_distance: int = 6  # For NL-means
    h: Optional[float] = None  # For NL-means, auto if None
    
    VALID_METHODS = ['gaussian', 'bilateral', 'median', 'wiener', 'nl_means']
    
    def __post_init__(self):
        if self.method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got {self.method}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if self.sigma_color <= 0:
            raise ValueError(f"sigma_color must be > 0, got {self.sigma_color}")
        if self.size < 1 or self.size % 2 == 0:
            raise ValueError(f"size must be odd positive integer, got {self.size}")
        if self.patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {self.patch_size}")


@dataclass
class ContrastParams:
    """Parameters for contrast enhancement with validation."""
    method: str = 'clahe'
    clip_limit: float = 2.0
    kernel_size: Tuple[int, ...] = (8, 8, 8)
    nbins: int = 256
    
    VALID_METHODS = ['clahe', 'histogram_eq', 'adaptive_eq', 'rescale']
    
    def __post_init__(self):
        if self.method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got {self.method}")
        if self.clip_limit <= 0:
            raise ValueError(f"clip_limit must be > 0, got {self.clip_limit}")
        if self.nbins < 2:
            raise ValueError(f"nbins must be >= 2, got {self.nbins}")


@dataclass
class ArtifactParams:
    """Parameters for artifact removal with validation."""
    curtaining_enabled: bool = True
    charging_enabled: bool = True
    drift_enabled: bool = False
    curtaining_sigma: float = 5.0
    charging_threshold_factor: float = 2.0
    drift_method: str = 'phase_correlation'
    
    VALID_DRIFT_METHODS = ['phase_correlation', 'cross_correlation']
    
    def __post_init__(self):
        if self.curtaining_sigma <= 0:
            raise ValueError(f"curtaining_sigma must be > 0, got {self.curtaining_sigma}")
        if self.charging_threshold_factor <= 0:
            raise ValueError(f"charging_threshold_factor must be > 0, got {self.charging_threshold_factor}")
        if self.drift_method not in self.VALID_DRIFT_METHODS:
            raise ValueError(f"drift_method must be one of {self.VALID_DRIFT_METHODS}")


def validate_image(image: np.ndarray, name: str = "image") -> None:
    """Validate input image array."""
    if image is None:
        raise ValueError(f"{name} cannot be None")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be numpy array, got {type(image)}")
    if image.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if image.ndim not in [2, 3]:
        raise ValueError(f"{name} must be 2D or 3D, got {image.ndim}D")


# =============================================================================
# Noise Removal Functions
# =============================================================================

def remove_noise(image: np.ndarray, 
                 method: str = 'gaussian',
                 **kwargs) -> np.ndarray:
    """
    Remove noise from FIB-SEM images.
    
    Args:
        image: Input image array
        method: Noise removal method ('gaussian', 'bilateral', 'median', 'wiener', 'nl_means')
        **kwargs: Method-specific parameters
        
    Returns:
        Denoised image
    """
    logger.info(f"Applying {method} noise removal")
    
    if method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return filters.gaussian(image, sigma=sigma, preserve_range=True)
    
    elif method == 'nl_means':
        # Recommended for high-quality denoising, can be slow
        patch_size = kwargs.get('patch_size', 5)
        patch_distance = kwargs.get('patch_distance', 6)
        h = kwargs.get('h', 1.15 * np.std(image)) # h is a smoothing parameter

        # denoise_nl_means expects float images
        img_float = image.astype(np.float64)
        img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min())

        denoised = restoration.denoise_nl_means(
            img_float,
            patch_size=patch_size,
            patch_distance=patch_distance,
            h=h,
            preserve_range=True
        )

        # Convert back to original range and type
        denoised = (denoised * (image.max() - image.min())) + image.min()
        return denoised.astype(image.dtype)

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
    """
    Correct drift artifacts in image stacks using phase correlation.
    
    Uses robust phase correlation registration for accurate sub-pixel
    drift estimation between consecutive slices.
    """
    if image.ndim != 3:
        logger.warning("Drift correction requires 3D image stack")
        return image
    
    method = kwargs.get('drift_method', 'phase_correlation')
    max_shift = kwargs.get('max_shift', 50)  # Maximum allowed shift in pixels
    reference_mode = kwargs.get('reference_mode', 'previous')  # 'previous' or 'middle'
    
    logger.info(f"Correcting drift using {method} method")
    
    corrected = np.zeros_like(image, dtype=np.float32)
    shifts = []
    
    if reference_mode == 'middle':
        # Use middle slice as global reference
        ref_idx = image.shape[0] // 2
        reference = image[ref_idx].astype(np.float32)
        corrected[ref_idx] = reference
        
        # Forward pass (from middle to end)
        cumulative_shift = np.array([0.0, 0.0])
        for i in range(ref_idx + 1, image.shape[0]):
            shift = _compute_shift(reference, image[i], method, max_shift)
            cumulative_shift += shift
            shifts.append(cumulative_shift.copy())
            corrected[i] = _apply_shift(image[i], cumulative_shift)
        
        # Backward pass (from middle to start)
        cumulative_shift = np.array([0.0, 0.0])
        for i in range(ref_idx - 1, -1, -1):
            shift = _compute_shift(reference, image[i], method, max_shift)
            cumulative_shift += shift
            shifts.insert(0, cumulative_shift.copy())
            corrected[i] = _apply_shift(image[i], cumulative_shift)
    
    else:  # reference_mode == 'previous'
        # Progressive registration to previous slice
        corrected[0] = image[0].astype(np.float32)
        cumulative_shift = np.array([0.0, 0.0])
        
        for i in range(1, image.shape[0]):
            # Compute shift relative to previous (corrected) slice
            shift = _compute_shift(corrected[i-1], image[i], method, max_shift)
            cumulative_shift += shift
            shifts.append(cumulative_shift.copy())
            corrected[i] = _apply_shift(image[i], cumulative_shift)
    
    # Log shift statistics
    if shifts:
        shifts_arr = np.array(shifts)
        logger.info(f"Drift correction: max shift = {np.max(np.abs(shifts_arr)):.2f} pixels")
        logger.info(f"Drift correction: mean shift = {np.mean(np.abs(shifts_arr)):.2f} pixels")
    
    return corrected.astype(image.dtype)


def _compute_shift(reference: np.ndarray, moving: np.ndarray, 
                   method: str, max_shift: int) -> np.ndarray:
    """Compute shift between two images using specified method."""
    
    if method == 'phase_correlation':
        try:
            from skimage.registration import phase_cross_correlation
            
            shift, error, diffphase = phase_cross_correlation(
                reference.astype(np.float64),
                moving.astype(np.float64),
                upsample_factor=10  # Sub-pixel accuracy
            )
            
            # Validate shift magnitude
            if np.any(np.abs(shift) > max_shift):
                logger.warning(f"Detected shift {shift} exceeds max_shift {max_shift}, clamping")
                shift = np.clip(shift, -max_shift, max_shift)
            
            return np.array(shift)
            
        except ImportError:
            logger.warning("phase_cross_correlation not available, using cross_correlation")
            method = 'cross_correlation'
    
    if method == 'cross_correlation':
        # Traditional cross-correlation
        from scipy.signal import correlate2d
        
        # Normalize images
        ref_norm = (reference - reference.mean()) / (reference.std() + 1e-8)
        mov_norm = (moving - moving.mean()) / (moving.std() + 1e-8)
        
        # Compute cross-correlation
        correlation = correlate2d(ref_norm, mov_norm, mode='same')
        
        # Find peak
        peak = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        # Convert to shift (relative to center)
        center = np.array(correlation.shape) // 2
        shift = np.array(peak) - center
        
        # Validate shift
        if np.any(np.abs(shift) > max_shift):
            shift = np.clip(shift, -max_shift, max_shift)
        
        return shift.astype(np.float64)
    
    return np.array([0.0, 0.0])


def _apply_shift(image: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Apply sub-pixel shift to image using interpolation."""
    return ndimage.shift(image.astype(np.float32), shift, order=3, mode='constant', cval=0)


def correct_drift(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Correct drift in FIB-SEM image stacks using phase correlation.
    
    Args:
        image: 3D image stack
        drift_method: 'phase_correlation' (default) or 'cross_correlation'
        max_shift: Maximum allowed shift in pixels (default: 50)
        reference_mode: 'previous' (sequential) or 'middle' (global reference)
        
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

