"""
Quantification module for FIB-SEM data.

This module provides comprehensive tools for quantifying morphological properties,
intensity statistics, and network topology in segmented FIB-SEM datasets.

Features:
- Volumetric measurements: volume, surface area, sphericity
- Shape descriptors: elongation, convexity, solidity, Euler characteristic
- Orientation analysis: principal axes, anisotropy
- Intensity statistics: mean, median, min, max, std, integrated density
- Distribution fitting: normal, log-normal, gamma, Weibull
- Network/topology analysis: skeletonization, node degree, tortuosity
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QuantificationParams:
    """Parameters for quantification analysis."""
    min_size: int = 10
    compute_volume: bool = True
    compute_surface_area: bool = True
    compute_shape_factors: bool = True
    compute_orientation: bool = True
    compute_intensity: bool = False
    compute_topology: bool = False
    fit_distributions: bool = False
    distributions_to_fit: List[str] = field(default_factory=lambda: ['normal', 'lognormal'])
    
    def __post_init__(self):
        if self.min_size < 1:
            raise ValueError(f"min_size must be >= 1, got {self.min_size}")


def quantify_morphology(segmentation_result: np.ndarray,
                        voxel_spacing: tuple,
                        raw_image: Optional[np.ndarray] = None,
                        **kwargs) -> Dict[str, Any]:
    """
    Comprehensive morphological quantification of segmented objects.
    
    Args:
        segmentation_result: Label array from segmentation
        voxel_spacing: Voxel dimensions in (z, y, x) order (nm or μm)
        raw_image: Optional intensity image for intensity-based measurements
        **kwargs: Additional parameters (see QuantificationParams)
        
    Returns:
        Dictionary with detailed morphological analysis results including:
        - volumes, surface_areas, sphericity
        - elongation, convexity, solidity
        - euler_characteristic, orientation
        - intensity_stats (if raw_image provided)
    """
    if segmentation_result is None:
        return {'success': False, 'error': 'No segmentation available'}

    try:
        logger.info("Quantifying morphological properties")
        
        from skimage import measure
        from scipy import ndimage
        
        # Get parameters
        min_size = kwargs.get('min_size', 10)
        compute_surface = kwargs.get('compute_surface_area', True)
        compute_shape = kwargs.get('compute_shape_factors', True)
        compute_orient = kwargs.get('compute_orientation', True)
        compute_intensity = kwargs.get('compute_intensity', raw_image is not None)
        
        # Calculate voxel dimensions
        if segmentation_result.ndim == 3:
            voxel_volume = float(np.prod(voxel_spacing))
            is_3d = True
        else:
            voxel_area = float(voxel_spacing[1] * voxel_spacing[2])
            voxel_volume = voxel_area  # For 2D, "volume" is area
            is_3d = False
        
        # Get region properties
        props = measure.regionprops(segmentation_result, intensity_image=raw_image)
        
        # Initialize result containers
        results = {
            'labels': [],
            'volumes': [],
            'centroids': [],
            'bounding_boxes': [],
        }
        
        if compute_surface:
            results['surface_areas'] = []
            results['sphericity'] = []
        
        if compute_shape:
            results['elongation'] = []
            results['solidity'] = []
            results['convexity'] = []
            results['euler_number'] = []
            results['eccentricity'] = []
        
        if compute_orient:
            results['orientation'] = []
            results['major_axis_length'] = []
            results['minor_axis_length'] = []
            if is_3d:
                results['anisotropy'] = []
        
        if compute_intensity and raw_image is not None:
            results['intensity_mean'] = []
            results['intensity_std'] = []
            results['intensity_min'] = []
            results['intensity_max'] = []
            results['intensity_median'] = []
            results['integrated_density'] = []
        
        for prop in props:
            if prop.area < min_size:
                continue
            
            results['labels'].append(prop.label)
            results['centroids'].append(prop.centroid)
            results['bounding_boxes'].append(prop.bbox)
            
            # Volume calculation
            volume = prop.area * voxel_volume
            results['volumes'].append(volume)
            
            # Surface area and sphericity
            if compute_surface:
                if is_3d:
                    surface_area = _compute_surface_area_3d(
                        segmentation_result == prop.label,
                        voxel_spacing
                    )
                else:
                    # For 2D, perimeter
                    surface_area = prop.perimeter * np.sqrt(voxel_spacing[1] * voxel_spacing[2])
                
                results['surface_areas'].append(surface_area)
                
                # Sphericity (3D) or circularity (2D)
                if is_3d:
                    # Sphericity = (36π * V²)^(1/3) / A
                    if surface_area > 0:
                        sphericity = (36 * np.pi * volume ** 2) ** (1/3) / surface_area
                        sphericity = min(1.0, sphericity)  # Clamp to [0, 1]
                    else:
                        sphericity = 0.0
                else:
                    # Circularity = 4π * A / P²
                    if surface_area > 0:
                        sphericity = 4 * np.pi * volume / (surface_area ** 2)
                        sphericity = min(1.0, sphericity)
                    else:
                        sphericity = 0.0
                
                results['sphericity'].append(sphericity)
            
            # Shape factors
            if compute_shape:
                # Elongation (ratio of major to minor axis)
                if hasattr(prop, 'axis_major_length') and hasattr(prop, 'axis_minor_length'):
                    major = prop.axis_major_length
                    minor = prop.axis_minor_length
                elif hasattr(prop, 'major_axis_length') and hasattr(prop, 'minor_axis_length'):
                    major = prop.major_axis_length
                    minor = prop.minor_axis_length
                else:
                    major = prop.equivalent_diameter
                    minor = prop.equivalent_diameter
                
                if minor > 0:
                    elongation = major / minor
                else:
                    elongation = 1.0
                results['elongation'].append(elongation)
                
                # Solidity (ratio of area to convex hull area)
                if hasattr(prop, 'solidity'):
                    results['solidity'].append(prop.solidity)
                else:
                    results['solidity'].append(1.0)
                
                # Convexity (perimeter of convex hull / perimeter)
                if hasattr(prop, 'convex_area') and prop.perimeter > 0:
                    # Approximate convexity
                    convexity = prop.area / prop.convex_area if prop.convex_area > 0 else 1.0
                else:
                    convexity = 1.0
                results['convexity'].append(convexity)
                
                # Euler number (topology)
                if hasattr(prop, 'euler_number'):
                    results['euler_number'].append(prop.euler_number)
                else:
                    results['euler_number'].append(1)
                
                # Eccentricity
                if hasattr(prop, 'eccentricity'):
                    results['eccentricity'].append(prop.eccentricity)
                else:
                    results['eccentricity'].append(0.0)
            
            # Orientation
            if compute_orient:
                if hasattr(prop, 'orientation'):
                    results['orientation'].append(np.degrees(prop.orientation))
                else:
                    results['orientation'].append(0.0)
                
                if hasattr(prop, 'axis_major_length'):
                    results['major_axis_length'].append(prop.axis_major_length * voxel_spacing[-1])
                    results['minor_axis_length'].append(prop.axis_minor_length * voxel_spacing[-1])
                elif hasattr(prop, 'major_axis_length'):
                    results['major_axis_length'].append(prop.major_axis_length * voxel_spacing[-1])
                    results['minor_axis_length'].append(prop.minor_axis_length * voxel_spacing[-1])
                else:
                    results['major_axis_length'].append(prop.equivalent_diameter * voxel_spacing[-1])
                    results['minor_axis_length'].append(prop.equivalent_diameter * voxel_spacing[-1])
                
                # Anisotropy for 3D
                if is_3d:
                    anisotropy = _compute_anisotropy_3d(segmentation_result == prop.label)
                    results['anisotropy'].append(anisotropy)
            
            # Intensity measurements
            if compute_intensity and raw_image is not None:
                mask = segmentation_result == prop.label
                intensities = raw_image[mask]
                
                results['intensity_mean'].append(float(np.mean(intensities)))
                results['intensity_std'].append(float(np.std(intensities)))
                results['intensity_min'].append(float(np.min(intensities)))
                results['intensity_max'].append(float(np.max(intensities)))
                results['intensity_median'].append(float(np.median(intensities)))
                results['integrated_density'].append(float(np.sum(intensities)))
        
        # Compute summary statistics
        summary = _compute_summary_statistics(results)
        
        result = {
            'success': True,
            'analyzed_labels': results['labels'],
            'num_objects': len(results['labels']),
            'morphological_analysis': results,
            'summary_statistics': summary
        }
        
        # Optional distribution fitting
        if kwargs.get('fit_distributions', False) and len(results['volumes']) > 5:
            distributions = kwargs.get('distributions_to_fit', ['normal', 'lognormal'])
            result['distribution_fits'] = _fit_distributions(
                results['volumes'], 
                distributions
            )
        
        logger.info(f"Morphological quantification completed: {len(results['labels'])} objects")
        return result

    except Exception as e:
        logger.error(f"Error in morphological quantification: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def _compute_surface_area_3d(binary_mask: np.ndarray, voxel_spacing: tuple) -> float:
    """
    Compute surface area of a 3D object using marching cubes.
    
    Args:
        binary_mask: Binary mask of the object
        voxel_spacing: Voxel dimensions (z, y, x)
        
    Returns:
        Surface area in squared units of voxel_spacing
    """
    try:
        from skimage import measure
        
        # Pad to avoid edge effects
        padded = np.pad(binary_mask, 1, mode='constant', constant_values=0)
        
        # Generate mesh using marching cubes
        verts, faces, _, _ = measure.marching_cubes(
            padded.astype(float), 
            level=0.5,
            spacing=voxel_spacing
        )
        
        # Calculate surface area from mesh triangles
        surface_area = measure.mesh_surface_area(verts, faces)
        
        return float(surface_area)
        
    except Exception as e:
        logger.warning(f"Marching cubes failed, using voxel approximation: {e}")
        # Fallback: count surface voxels
        from scipy import ndimage
        eroded = ndimage.binary_erosion(binary_mask)
        surface_voxels = np.sum(binary_mask) - np.sum(eroded)
        # Approximate surface area (6 faces per voxel, averaged)
        avg_face_area = (voxel_spacing[0] * voxel_spacing[1] + 
                         voxel_spacing[1] * voxel_spacing[2] + 
                         voxel_spacing[0] * voxel_spacing[2]) / 3
        return float(surface_voxels * avg_face_area * 2)


def _compute_anisotropy_3d(binary_mask: np.ndarray) -> float:
    """
    Compute anisotropy of a 3D object using eigenvalues of inertia tensor.
    
    Returns:
        Anisotropy value (0 = isotropic sphere, 1 = maximally anisotropic)
    """
    try:
        from skimage import measure
        
        props = measure.regionprops(binary_mask.astype(int))
        if not props:
            return 0.0
        
        prop = props[0]
        
        # Get inertia tensor eigenvalues
        if hasattr(prop, 'inertia_tensor_eigvals'):
            eigvals = np.array(prop.inertia_tensor_eigvals)
        else:
            # Compute manually
            coords = np.array(np.where(binary_mask)).T
            if len(coords) < 4:
                return 0.0
            
            centroid = coords.mean(axis=0)
            centered = coords - centroid
            
            inertia_tensor = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    if i == j:
                        other_axes = [k for k in range(3) if k != i]
                        inertia_tensor[i, i] = np.sum(
                            centered[:, other_axes[0]]**2 + centered[:, other_axes[1]]**2
                        )
                    else:
                        inertia_tensor[i, j] = -np.sum(centered[:, i] * centered[:, j])
            
            eigvals = np.linalg.eigvalsh(inertia_tensor)
        
        eigvals = np.sort(eigvals)[::-1]  # Descending order
        
        # Anisotropy: (λ1 - λ3) / λ1
        if eigvals[0] > 0:
            anisotropy = (eigvals[0] - eigvals[-1]) / eigvals[0]
        else:
            anisotropy = 0.0
        
        return float(np.clip(anisotropy, 0, 1))
        
    except Exception as e:
        logger.warning(f"Anisotropy computation failed: {e}")
        return 0.0


def _compute_summary_statistics(results: Dict[str, List]) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics for each measurement."""
    summary = {}
    
    for key, values in results.items():
        if key in ['labels', 'centroids', 'bounding_boxes']:
            continue
        
        if not values:
            continue
        
        arr = np.array(values, dtype=float)
        
        summary[key] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
            'count': len(arr)
        }
        
        # Add percentiles for volumes
        if key == 'volumes':
            summary[key]['p25'] = float(np.percentile(arr, 25))
            summary[key]['p75'] = float(np.percentile(arr, 75))
            summary[key]['total'] = float(np.sum(arr))
    
    return summary


def _fit_distributions(data: List[float], distributions: List[str]) -> Dict[str, Any]:
    """
    Fit statistical distributions to data.
    
    Args:
        data: List of measurements
        distributions: Distribution names to fit
        
    Returns:
        Dictionary with fitted parameters and goodness-of-fit metrics
    """
    try:
        from scipy import stats
        
        arr = np.array(data)
        arr = arr[arr > 0]  # Filter positive values for log distributions
        
        if len(arr) < 5:
            return {'error': 'Insufficient data points for distribution fitting'}
        
        fits = {}
        
        dist_map = {
            'normal': stats.norm,
            'lognormal': stats.lognorm,
            'gamma': stats.gamma,
            'weibull': stats.weibull_min,
            'exponential': stats.expon
        }
        
        for dist_name in distributions:
            if dist_name not in dist_map:
                continue
            
            try:
                dist = dist_map[dist_name]
                params = dist.fit(arr)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = stats.kstest(arr, dist_name, args=params)
                
                fits[dist_name] = {
                    'parameters': params,
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue)
                }
                
            except Exception as e:
                fits[dist_name] = {'error': str(e)}
        
        # Find best fit by p-value
        valid_fits = {k: v for k, v in fits.items() if 'ks_pvalue' in v}
        if valid_fits:
            best_fit = max(valid_fits.items(), key=lambda x: x[1]['ks_pvalue'])
            fits['best_fit'] = best_fit[0]
        
        return fits
        
    except ImportError:
        return {'error': 'scipy.stats not available for distribution fitting'}
    except Exception as e:
        return {'error': str(e)}


def quantify_particles(segmentation_result: np.ndarray,
                       min_size: int = 50,
                       max_size: Optional[int] = None,
                       voxel_spacing: Optional[tuple] = None,
                       raw_image: Optional[np.ndarray] = None,
                       **kwargs) -> Dict[str, Any]:
    """
    Comprehensive particle analysis with size filtering and statistics.

    Args:
        segmentation_result: Label array from segmentation
        min_size: Minimum particle size to analyze (in voxels)
        max_size: Maximum particle size to analyze (in voxels, None for no limit)
        voxel_spacing: Voxel dimensions for physical measurements
        raw_image: Optional intensity image for intensity measurements
        
    Returns:
        Dictionary with particle analysis results including distributions
    """
    if segmentation_result is None:
        return {'success': False, 'error': 'No segmentation available'}

    try:
        logger.info(f"Quantifying particles (min_size={min_size}, max_size={max_size})")

        from skimage import measure
        
        if voxel_spacing is None:
            voxel_spacing = (1.0, 1.0, 1.0) if segmentation_result.ndim == 3 else (1.0, 1.0)

        props = measure.regionprops(segmentation_result, intensity_image=raw_image)

        particles = []
        for prop in props:
            # Size filtering
            if prop.area < min_size:
                continue
            if max_size is not None and prop.area > max_size:
                continue
            
            # Basic properties
            particle_info = {
                'label': prop.label,
                'area_voxels': prop.area,
                'centroid': tuple(prop.centroid),
                'bbox': prop.bbox,
            }
            
            # Physical measurements
            if segmentation_result.ndim == 3:
                voxel_vol = float(np.prod(voxel_spacing))
                particle_info['volume'] = prop.area * voxel_vol
                particle_info['equivalent_diameter'] = prop.equivalent_diameter * voxel_spacing[-1]
            else:
                voxel_area = float(voxel_spacing[-2] * voxel_spacing[-1])
                particle_info['area'] = prop.area * voxel_area
                particle_info['equivalent_diameter'] = prop.equivalent_diameter * voxel_spacing[-1]
                particle_info['perimeter'] = prop.perimeter * voxel_spacing[-1]
            
            # Shape properties
            if hasattr(prop, 'solidity'):
                particle_info['solidity'] = prop.solidity
            if hasattr(prop, 'eccentricity'):
                particle_info['eccentricity'] = prop.eccentricity
            if hasattr(prop, 'orientation'):
                particle_info['orientation'] = np.degrees(prop.orientation)
            
            # Intensity properties (if raw image provided)
            if raw_image is not None:
                particle_info['mean_intensity'] = float(prop.mean_intensity)
                if hasattr(prop, 'intensity_max'):
                    particle_info['max_intensity'] = float(prop.intensity_max)
                    particle_info['min_intensity'] = float(prop.intensity_min)
            
            particles.append(particle_info)

        # Compute size distribution statistics
        if particles:
            sizes = [p.get('volume', p.get('area', p['area_voxels'])) for p in particles]
            size_stats = {
                'count': len(sizes),
                'mean': float(np.mean(sizes)),
                'std': float(np.std(sizes)),
                'median': float(np.median(sizes)),
                'min': float(np.min(sizes)),
                'max': float(np.max(sizes)),
                'total': float(np.sum(sizes))
            }
            
            # Size histogram
            hist, bin_edges = np.histogram(sizes, bins='auto')
            size_stats['histogram'] = {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        else:
            size_stats = {}

        result = {
            'success': True,
            'num_particles': len(particles),
            'particle_properties': particles,
            'size_statistics': size_stats,
            'filter_settings': {
                'min_size': min_size,
                'max_size': max_size
            }
        }

        logger.info(f"Particle quantification completed: {len(particles)} particles")
        return result

    except Exception as e:
        logger.error(f"Error in particle quantification: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def quantify_network(segmentation_result: np.ndarray,
                     voxel_spacing: Optional[tuple] = None,
                     **kwargs) -> Dict[str, Any]:
    """
    Analyze network topology using skeletonization.
    
    Computes connectivity graphs, node degree distribution, and tortuosity.
    
    Args:
        segmentation_result: Label array (typically binary or single-label for network)
        voxel_spacing: Voxel dimensions for physical measurements
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with network analysis results
    """
    if segmentation_result is None:
        return {'success': False, 'error': 'No segmentation available'}
    
    try:
        from skimage import morphology
        from scipy import ndimage
        
        logger.info("Analyzing network topology")
        
        if voxel_spacing is None:
            voxel_spacing = (1.0, 1.0, 1.0) if segmentation_result.ndim == 3 else (1.0, 1.0)
        
        # Convert to binary if labeled
        binary = segmentation_result > 0
        
        # Skeletonize
        if binary.ndim == 3:
            skeleton = morphology.skeletonize_3d(binary)
        else:
            skeleton = morphology.skeletonize(binary)
        
        # Count skeleton pixels
        skeleton_length_voxels = np.sum(skeleton)
        
        # Calculate physical length (approximate as voxel count * min spacing)
        min_spacing = min(voxel_spacing)
        skeleton_length_physical = skeleton_length_voxels * min_spacing
        
        # Find branch points and endpoints
        branch_points, endpoints = _find_skeleton_nodes(skeleton)
        
        num_branches = len(branch_points)
        num_endpoints = len(endpoints)
        
        # Estimate number of segments (Euler characteristic based)
        # For a skeleton: segments ≈ branch_points + 1 (for tree) or more for cycles
        num_segments = max(1, num_branches + 1)
        
        # Calculate tortuosity if we can identify paths
        tortuosity_stats = _compute_tortuosity(skeleton, voxel_spacing)
        
        # Node degree distribution
        degree_distribution = _compute_degree_distribution(skeleton, branch_points)
        
        result = {
            'success': True,
            'skeleton_length_voxels': int(skeleton_length_voxels),
            'skeleton_length_physical': float(skeleton_length_physical),
            'num_branch_points': num_branches,
            'num_endpoints': num_endpoints,
            'estimated_segments': num_segments,
            'branch_point_coords': [tuple(int(x) for x in bp) for bp in branch_points[:100]],  # Limit to 100
            'endpoint_coords': [tuple(int(x) for x in ep) for ep in endpoints[:100]],
            'tortuosity': tortuosity_stats,
            'degree_distribution': degree_distribution,
            'connectivity_ratio': float(num_branches / max(1, num_endpoints)) if num_endpoints > 0 else 0.0
        }
        
        logger.info(f"Network analysis completed: {num_branches} branch points, {num_endpoints} endpoints")
        return result
        
    except Exception as e:
        logger.error(f"Error in network quantification: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def _find_skeleton_nodes(skeleton: np.ndarray) -> Tuple[List, List]:
    """Find branch points and endpoints in a skeleton."""
    from scipy import ndimage
    
    # Create connectivity kernel
    if skeleton.ndim == 3:
        # 26-connectivity for 3D
        kernel = np.ones((3, 3, 3), dtype=int)
        kernel[1, 1, 1] = 0
    else:
        # 8-connectivity for 2D
        kernel = np.ones((3, 3), dtype=int)
        kernel[1, 1] = 0
    
    # Count neighbors for each skeleton pixel
    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
    neighbor_count = neighbor_count * skeleton  # Only skeleton pixels
    
    # Branch points: >= 3 neighbors
    branch_mask = (neighbor_count >= 3) & skeleton
    branch_points = np.array(np.where(branch_mask)).T.tolist()
    
    # Endpoints: exactly 1 neighbor
    endpoint_mask = (neighbor_count == 1) & skeleton
    endpoints = np.array(np.where(endpoint_mask)).T.tolist()
    
    return branch_points, endpoints


def _compute_degree_distribution(skeleton: np.ndarray, branch_points: List) -> Dict[str, Any]:
    """Compute node degree distribution."""
    from scipy import ndimage
    
    if not branch_points:
        return {'degrees': [], 'mean': 0, 'max': 0}
    
    # Create connectivity kernel
    if skeleton.ndim == 3:
        kernel = np.ones((3, 3, 3), dtype=int)
        kernel[1, 1, 1] = 0
    else:
        kernel = np.ones((3, 3), dtype=int)
        kernel[1, 1] = 0
    
    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
    
    degrees = []
    for bp in branch_points:
        bp = tuple(bp)
        if all(0 <= bp[i] < skeleton.shape[i] for i in range(skeleton.ndim)):
            degrees.append(int(neighbor_count[bp]))
    
    if degrees:
        return {
            'degrees': degrees,
            'mean': float(np.mean(degrees)),
            'max': int(np.max(degrees)),
            'min': int(np.min(degrees)),
            'histogram': dict(zip(*np.unique(degrees, return_counts=True)))
        }
    
    return {'degrees': [], 'mean': 0, 'max': 0}


def _compute_tortuosity(skeleton: np.ndarray, voxel_spacing: tuple) -> Dict[str, float]:
    """
    Compute tortuosity statistics for skeleton paths.
    
    Tortuosity = actual path length / straight-line distance
    """
    from scipy import ndimage
    
    # Label connected components of skeleton
    labeled, num_features = ndimage.label(skeleton)
    
    if num_features == 0:
        return {'mean': 1.0, 'min': 1.0, 'max': 1.0, 'std': 0.0}
    
    tortuosities = []
    
    for i in range(1, num_features + 1):
        component = labeled == i
        coords = np.array(np.where(component)).T
        
        if len(coords) < 2:
            continue
        
        # Path length (number of voxels * spacing)
        path_length = len(coords) * min(voxel_spacing)
        
        # Straight-line distance between endpoints
        # Find the two most distant points
        if len(coords) > 2:
            from scipy.spatial.distance import cdist
            # Sample if too many points
            if len(coords) > 1000:
                indices = np.random.choice(len(coords), 1000, replace=False)
                sample_coords = coords[indices]
            else:
                sample_coords = coords
            
            # Scale by voxel spacing
            scaled_coords = sample_coords * np.array(voxel_spacing)
            
            # Find max distance
            try:
                distances = cdist(scaled_coords, scaled_coords)
                max_dist = np.max(distances)
            except MemoryError:
                # Fallback: just use first and last
                max_dist = np.linalg.norm(
                    (coords[0] - coords[-1]) * np.array(voxel_spacing)
                )
        else:
            max_dist = np.linalg.norm(
                (coords[0] - coords[-1]) * np.array(voxel_spacing)
            )
        
        if max_dist > 0:
            tortuosity = path_length / max_dist
            tortuosities.append(tortuosity)
    
    if tortuosities:
        return {
            'mean': float(np.mean(tortuosities)),
            'min': float(np.min(tortuosities)),
            'max': float(np.max(tortuosities)),
            'std': float(np.std(tortuosities)),
            'count': len(tortuosities)
        }
    
    return {'mean': 1.0, 'min': 1.0, 'max': 1.0, 'std': 0.0, 'count': 0}


def quantify_intensity(segmentation_result: np.ndarray,
                       raw_image: np.ndarray,
                       voxel_spacing: Optional[tuple] = None,
                       **kwargs) -> Dict[str, Any]:
    """
    Compute intensity-based measurements for segmented objects.
    
    Args:
        segmentation_result: Label array from segmentation
        raw_image: Intensity image
        voxel_spacing: Voxel dimensions
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with intensity analysis results
    """
    if segmentation_result is None:
        return {'success': False, 'error': 'No segmentation available'}
    if raw_image is None:
        return {'success': False, 'error': 'No intensity image provided'}
    
    try:
        from skimage import measure
        
        logger.info("Computing intensity-based measurements")
        
        min_size = kwargs.get('min_size', 10)
        
        props = measure.regionprops(segmentation_result, intensity_image=raw_image)
        
        intensity_results = []
        
        for prop in props:
            if prop.area < min_size:
                continue
            
            mask = segmentation_result == prop.label
            intensities = raw_image[mask]
            
            result = {
                'label': prop.label,
                'mean': float(np.mean(intensities)),
                'median': float(np.median(intensities)),
                'std': float(np.std(intensities)),
                'min': float(np.min(intensities)),
                'max': float(np.max(intensities)),
                'integrated_density': float(np.sum(intensities)),
                'area_voxels': prop.area
            }
            
            # Percentiles
            result['p10'] = float(np.percentile(intensities, 10))
            result['p90'] = float(np.percentile(intensities, 90))
            
            # Coefficient of variation
            if result['mean'] > 0:
                result['cv'] = result['std'] / result['mean']
            else:
                result['cv'] = 0.0
            
            intensity_results.append(result)
        
        # Summary statistics across all objects
        if intensity_results:
            all_means = [r['mean'] for r in intensity_results]
            summary = {
                'global_mean': float(np.mean(all_means)),
                'global_std': float(np.std(all_means)),
                'num_objects': len(intensity_results)
            }
        else:
            summary = {'num_objects': 0}
        
        return {
            'success': True,
            'object_intensities': intensity_results,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"Error in intensity quantification: {e}")
        return {'success': False, 'error': str(e)}
