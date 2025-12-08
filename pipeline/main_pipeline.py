"""
Main pipeline for FIB-SEM analysis workflows.

This module provides the central orchestration framework that coordinates
all analysis components from data loading to final results.
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

from core.config import FIBSEMConfig
from core.data_io import load_fibsem_data, FIBSEMData
from core.preprocessing import preprocess_fibsem_data
from core import segmentation
from core import quantification

logger = logging.getLogger(__name__)

class FIBSEMPipeline:
    """Main pipeline for FIB-SEM data analysis."""
    
    def __init__(self, config: Optional[FIBSEMConfig] = None, 
                 voxel_spacing: Optional[Tuple[float, float, float]] = None):
        """
        Initialize FIB-SEM analysis pipeline.
        
        Args:
            config: Configuration object (creates default if None)
            voxel_spacing: Voxel spacing in (z, y, x) order in nanometers
        """
        self.config = config if config is not None else FIBSEMConfig()
        self.voxel_spacing = voxel_spacing or tuple(self.config.get('data.default_voxel_size', [10.0, 5.0, 5.0]))
        
        # Pipeline state
        self.data = None
        self.preprocessed_data = None
        self.segmentation_result = None
        self.processing_history = []
        
        logger.info(f"Initialized FIB-SEM pipeline with voxel spacing {self.voxel_spacing} nm")
    
    def load_data(self, data_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load FIB-SEM data from file.
        
        Args:
            data_path: Path to data file
            
        Returns:
            Dictionary with loading results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Loading data from {data_path}")
            self.data = load_fibsem_data(data_path, voxel_size=self.voxel_spacing)
            
            duration = time.time() - start_time
            
            result = {
                'success': True,
                'data': self.data,
                'duration': duration,
                'shape': self.data.shape,
                'voxel_size': self.data.voxel_size
            }
            
            self.processing_history.append({
                'step': 'load_data',
                'duration': duration,
                'parameters': {'data_path': str(data_path)},
                'result': {'shape': self.data.shape}
            })
            
            logger.info(f"Data loaded successfully: shape {self.data.shape}, duration {duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
    
    def preprocess_data(self, preprocessing_steps: Optional[List[str]] = None,
                        parameters: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Preprocess the loaded data.
        
        Args:
            preprocessing_steps: List of preprocessing steps to apply
            parameters: Parameters for each preprocessing step
            
        Returns:
            Dictionary with preprocessing results
        """
        if self.data is None:
            return {'success': False, 'error': 'No data loaded'}
        
        start_time = time.time()
        
        try:
            if preprocessing_steps is None:
                preprocessing_steps = self.config.get('preprocessing.default_steps', 
                                                    ['noise_reduction', 'contrast_enhancement'])
            
            logger.info(f"Preprocessing data with steps: {preprocessing_steps}")
            
            self.preprocessed_data = preprocess_fibsem_data(
                self.data.data,
                steps=preprocessing_steps,
                parameters=parameters
            )
            
            duration = time.time() - start_time
            
            result = {
                'success': True,
                'preprocessed_data': self.preprocessed_data,
                'preprocessing_steps': preprocessing_steps,
                'duration': duration
            }
            
            self.processing_history.append({
                'step': 'preprocess_data',
                'duration': duration,
                'parameters': {
                    'preprocessing_steps': preprocessing_steps,
                    'parameters': parameters
                }
            })
            
            logger.info(f"Preprocessing completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
    
    def segment_data(self, method: str = 'watershed',
                     method_type: str = 'traditional',
                     **kwargs) -> Dict[str, Any]:
        """
        Segment the preprocessed data.
        
        Args:
            method: Segmentation method name
            method_type: 'traditional' or 'deep_learning'
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary with segmentation results
        """
        if self.data is None:
            return {'success': False, 'error': 'No data loaded'}
        
        # Use preprocessed data if available, otherwise use original
        input_data = self.preprocessed_data if self.preprocessed_data is not None else self.data.data
        
        start_time = time.time()
        
        try:
            logger.info(f"Segmenting data using {method_type}.{method}")
            
            # Get method parameters from config
            config_params = self.config.get_segmentation_params(method, method_type)
            config_params.update(kwargs)  # Override with provided parameters
            
            if method_type == 'traditional':
                segmentation_result = segmentation.segment_traditional(input_data, method, config_params)
            elif method_type == 'deep_learning':
                segmentation_result = segmentation.segment_deep_learning(input_data, method, config_params)
            else:
                raise ValueError(f"Unknown method type: {method_type}")
            
            duration = time.time() - start_time
            
            # Store the full result object
            self.segmentation_result_full = segmentation_result
            
            # Extract labels for backward compatibility
            # SegmentationResult is a dataclass with .labels, .confidence, .metadata
            # segment_traditional/segment_deep_learning can return either SegmentationResult or ndarray
            from core.segmentation import SegmentationResult
            if isinstance(segmentation_result, SegmentationResult):
                labels = segmentation_result.labels
                confidence = segmentation_result.confidence
                metadata = segmentation_result.metadata
            else:
                # Legacy array return (when return_result=False, which is default)
                labels = np.asarray(segmentation_result)
                confidence = None
                metadata = {}
            
            # Store labels for backward compatibility with downstream methods
            self.segmentation_result = labels
            
            num_labels = int(len(np.unique(labels)) - 1)  # Exclude background
            
            result = {
                'success': True,
                'segmentation': labels,
                'confidence': confidence,
                'metadata': metadata,
                'method': method,
                'method_type': method_type,
                'parameters': config_params,
                'duration': duration,
                'num_labels': num_labels
            }
            
            self.processing_history.append({
                'step': 'segment_data',
                'duration': duration,
                'parameters': {
                    'method': method,
                    'method_type': method_type,
                    'parameters': config_params
                },
                'result': {'num_labels': num_labels}
            })
            
            logger.info(f"Segmentation completed: {result['num_labels']} labels, duration {duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in segmentation: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
    
    # Note: Segmentation methods have been consolidated in core/segmentation.py
    # This pipeline now uses the unified segment_traditional and segment_deep_learning
    # functions which provide consistent parameter validation and return types.
    
    def quantify_morphology(self, min_size: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Quantify morphological properties of segmented objects.
        
        Returns:
            Dictionary with morphological analysis results
        """
        if self.segmentation_result is None:
            return {'success': False, 'error': 'No segmentation available'}
        
        start_time = time.time()
        
        result = quantification.quantify_morphology(
            self.segmentation_result,
            self.voxel_spacing,
            **kwargs
        )

        duration = time.time() - start_time
        result['duration'] = duration

        if result['success']:
            self.processing_history.append({
                'step': 'quantify_morphology',
                'duration': duration,
                'result': {'num_objects': result['morphological_analysis']['num_objects']}
            })

        return result
    
    def quantify_particles(self, min_size: int = 50, **kwargs) -> Dict[str, Any]:
        """
        Quantify particle properties.
        
        Args:
            min_size: Minimum particle size to analyze
            
        Returns:
            Dictionary with particle analysis results
        """
        if self.segmentation_result is None:
            return {'success': False, 'error': 'No segmentation available'}
        
        start_time = time.time()
        
        result = quantification.quantify_particles(
            self.segmentation_result,
            min_size,
            **kwargs
        )

        duration = time.time() - start_time
        result['duration'] = duration

        if result['success']:
            self.processing_history.append({
                'step': 'quantify_particles',
                'duration': duration,
                'parameters': {'min_size': min_size},
                'result': {'num_particles': result['num_particles']}
            })

        return result
    
    def run_complete_pipeline(self, data_path: Union[str, Path],
                              segmentation_method: str = 'watershed',
                              segmentation_type: str = 'traditional',
                              output_dir: Optional[Union[str, Path]] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Args:
            data_path: Path to input data
            segmentation_method: Segmentation method to use
            segmentation_type: 'traditional' or 'deep_learning'
            output_dir: Optional output directory for results
            **kwargs: Additional parameters for pipeline steps
            
        Returns:
            Dictionary with complete pipeline results
        """
        pipeline_start = time.time()
        
        logger.info(f"Starting complete pipeline for {data_path}")
        
        results = {
            'pipeline_start_time': pipeline_start,
            'data_path': str(data_path),
            'segmentation_method': segmentation_method,
            'segmentation_type': segmentation_type
        }
        
        # Load data
        load_result = self.load_data(data_path)
        if not load_result['success']:
            results['error'] = f"Data loading failed: {load_result['error']}"
            return results
        results['data_loading'] = load_result
        
        # Preprocess data
        preprocessing_params = kwargs.get('preprocessing', {})
        preprocessing_steps = preprocessing_params.pop('steps', None)
        preprocess_result = self.preprocess_data(
            preprocessing_steps=preprocessing_steps,
            parameters=preprocessing_params
        )
        if not preprocess_result['success']:
            results['error'] = f"Preprocessing failed: {preprocess_result['error']}"
            return results
        results['preprocessing'] = preprocess_result
        
        # Segment data
        segmentation_params = kwargs.get('segmentation', {})
        segment_result = self.segment_data(
            method=segmentation_method,
            method_type=segmentation_type,
            **segmentation_params
        )
        if not segment_result['success']:
            results['error'] = f"Segmentation failed: {segment_result['error']}"
            return results
        results['segmentation_results'] = segment_result
        
        # Quantify morphology
        morphology_params = kwargs.get('morphology', {})
        morphology_result = self.quantify_morphology(**morphology_params)
        if morphology_result['success']:
            results['morphological_quantification'] = morphology_result
        
        # Quantify particles
        particle_params = kwargs.get('particles', {})
        particle_result = self.quantify_particles(**particle_params)
        if particle_result['success']:
            results['particle_quantification'] = particle_result
        
        # Calculate total duration
        results['pipeline_duration'] = time.time() - pipeline_start
        results['processing_history'] = self.processing_history
        
        logger.info(f"Complete pipeline finished in {results['pipeline_duration']:.2f}s")
        
        # Save results if output directory specified
        if output_dir is not None:
            self._save_results(results, output_dir)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], output_dir: Union[str, Path]) -> None:
        """Save pipeline results to output directory."""
        import json
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        results_file = output_dir / 'pipeline_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_for_json(results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization by converting numpy arrays."""
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            # Handle custom objects like FIBSEMData
            return self._prepare_for_json(obj.__dict__)
        else:
            try:
                # Try to convert to string if all else fails
                return str(obj)
            except:
                return None
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'data_loaded': self.data is not None,
            'data_preprocessed': self.preprocessed_data is not None,
            'segmentation_available': self.segmentation_result is not None,
            'processing_steps_completed': len(self.processing_history),
            'voxel_spacing': self.voxel_spacing
        }

def create_default_pipeline(voxel_spacing: Optional[Tuple[float, float, float]] = None,
                           config_path: Optional[Union[str, Path]] = None) -> FIBSEMPipeline:
    """
    Create a FIB-SEM pipeline with default configuration.
    
    Args:
        voxel_spacing: Voxel spacing in (z, y, x) order in nanometers
        config_path: Optional path to configuration file
        
    Returns:
        Configured FIBSEMPipeline instance
    """
    config = FIBSEMConfig(config_path) if config_path else FIBSEMConfig()
    
    if voxel_spacing is not None:
        config.set('data.default_voxel_size', list(voxel_spacing))
    
    return FIBSEMPipeline(config=config, voxel_spacing=voxel_spacing)

