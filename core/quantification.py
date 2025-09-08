"""
Quantification module for FIB-SEM data.

This module provides tools for quantifying morphological properties and
analyzing particles in segmented FIB-SEM datasets.
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def quantify_morphology(segmentation_result: np.ndarray,
                        voxel_spacing: tuple,
                        **kwargs) -> Dict[str, Any]:
    """
    Quantify morphological properties of segmented objects.

    Returns:
        Dictionary with morphological analysis results
    """
    if segmentation_result is None:
        return {'success': False, 'error': 'No segmentation available'}

    try:
        logger.info("Quantifying morphological properties")

        from skimage import measure

        # Get region properties
        props = measure.regionprops(segmentation_result)

        analyzed_labels = []
        volumes = []
        areas = []

        for prop in props:
            if prop.area > kwargs.get('min_size', 10):
                analyzed_labels.append(prop.label)

                # Calculate volume (area * voxel volume for 2D, or actual volume for 3D)
                if segmentation_result.ndim == 3:
                    voxel_volume = np.prod(voxel_spacing)
                    volume = prop.area * voxel_volume  # area is actually volume in 3D
                else:
                    voxel_area = voxel_spacing[1] * voxel_spacing[2]
                    volume = prop.area * voxel_area

                volumes.append(volume)
                areas.append(prop.area)

        result = {
            'success': True,
            'analyzed_labels': analyzed_labels,
            'morphological_analysis': {
                'volumes': volumes,
                'areas': areas,
                'num_objects': len(analyzed_labels)
            }
        }

        logger.info(f"Morphological quantification completed: {len(analyzed_labels)} objects")
        return result

    except Exception as e:
        logger.error(f"Error in morphological quantification: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def quantify_particles(segmentation_result: np.ndarray,
                       min_size: int = 50,
                       **kwargs) -> Dict[str, Any]:
    """
    Quantify particle properties.

    Args:
        min_size: Minimum particle size to analyze

    Returns:
        Dictionary with particle analysis results
    """
    if segmentation_result is None:
        return {'success': False, 'error': 'No segmentation available'}

    try:
        logger.info(f"Quantifying particles (min_size={min_size})")

        from skimage import measure

        props = measure.regionprops(segmentation_result)

        particles = []
        for prop in props:
            if prop.area >= min_size:
                particle_info = {
                    'label': prop.label,
                    'area': prop.area,
                    'centroid': prop.centroid,
                    'equivalent_diameter': prop.equivalent_diameter
                }
                particles.append(particle_info)

        result = {
            'success': True,
            'num_particles': len(particles),
            'particle_properties': particles
        }

        logger.info(f"Particle quantification completed: {len(particles)} particles")
        return result

    except Exception as e:
        logger.error(f"Error in particle quantification: {e}")
        return {
            'success': False,
            'error': str(e)
        }
