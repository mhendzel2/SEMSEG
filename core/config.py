"""
Configuration management for FIB-SEM analysis workflows.

This module provides flexible parameter control and ensures reproducible
analysis workflows through hierarchical configuration files.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def _sanitize_config_path(user_path: str) -> str:
    """
    Sanitize a user-provided config file path to prevent path traversal attacks.
    
    Args:
        user_path: User-provided path string
        
    Returns:
        Sanitized path string (just the filename, no directory components)
        
    Raises:
        ValueError: If path contains invalid characters or patterns
    """
    if not user_path:
        return "fibsem_config.json"
    
    # Strip whitespace
    user_path = user_path.strip()
    
    # Reject path traversal sequences
    if ".." in user_path or "/" in user_path or "\\" in user_path:
        raise ValueError("Path must be a simple filename without directory components")
    
    # Validate filename characters (alphanumeric, underscore, hyphen, period)
    if not re.match(r'^[\w\-\.]+$', user_path):
        raise ValueError("Filename contains invalid characters")
    
    # Restrict to allowed extensions
    allowed_extensions = {'.json', '.yml', '.yaml'}
    if not any(user_path.lower().endswith(ext) for ext in allowed_extensions):
        raise ValueError(f"File must have one of these extensions: {allowed_extensions}")
    
    return user_path


class FIBSEMConfig:
    """Configuration manager for FIB-SEM analysis parameters."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._get_default_config()
        
        if config_path is not None:
            self.load_config(config_path)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        return {
            'data': {
                'default_voxel_size': [10.0, 5.0, 5.0],  # z, y, x in nm
                'supported_formats': ['.tif', '.tiff', '.h5', '.hdf5', '.npy'],
                'memory_limit': '75%',
                'use_memory_mapping': True
            },
            'preprocessing': {
                'default_steps': ['noise_reduction', 'contrast_enhancement'],
                'noise_reduction': {
                    'method': 'gaussian',
                    'sigma': 1.0
                },
                'contrast_enhancement': {
                    'method': 'clahe',
                    'clip_limit': 0.03,
                    'tile_grid_size': [8, 8]
                },
                'artifact_removal': {
                    'curtaining': True,
                    'charging': True,
                    'drift_correction': False
                }
            },
            'segmentation': {
                'traditional': {
                    'watershed': {
                        'min_distance': 20,
                        'threshold_rel': 0.6,
                        'watershed_line': True
                    },
                    'thresholding': {
                        'method': 'otsu',
                        'block_size': None,
                        'offset': 0
                    },
                    'morphology': {
                        'operation': 'opening',
                        'structuring_element': 'ball',
                        'radius': 3
                    },
                    'region_growing': {
                        'seed_threshold': 0.5,
                        'growth_threshold': 0.1,
                        'connectivity': 1,
                        'min_distance': 10
                    },
                    'graph_cuts': {
                        'lambda': 1.0,
                        'sigma': 10.0
                    },
                    'active_contours': {
                        'alpha': 0.015,
                        'beta': 10,
                        'gamma': 0.001,
                        'iterations': 100
                    },
                    'slic': {
                        'n_segments': 1000,
                        'compactness': 10.0,
                        'sigma': 1.0
                    },
                    'felzenszwalb': {
                        'scale': 100,
                        'sigma': 0.5,
                        'min_size': 50
                    },
                    'random_walker': {
                        'beta': 130,
                        'mode': 'cg_mg'
                    }
                },
                'deep_learning': {
                    'unet_2d': {
                        'model_path': None,
                        'input_size': [256, 256, 1],
                        'num_classes': 2,
                        'threshold': 0.5
                    },
                    'unet_3d': {
                        'model_path': None,
                        'patch_size': [64, 64, 64],
                        'num_classes': 2,
                        'threshold': 0.5,
                        'overlap': 0.25
                    },
                    'vnet': {
                        'model_path': None,
                        'patch_size': [64, 64, 64],
                        'num_classes': 2,
                        'threshold': 0.5
                    },
                    'attention_unet': {
                        'model_path': None,
                        'input_size': [256, 256, 1],
                        'num_classes': 2,
                        'threshold': 0.5
                    },
                    'nnunet': {
                        'spacing': [1.0, 1.0, 1.0],
                        'num_classes': 2,
                        'threshold': 0.5
                    },
                    'sam3': {
                        'model_type': 'vit_h',
                        'checkpoint_path': None,
                        'text_prompt': '',
                        'box_prompt': None,
                        'point_prompt': None
                    }
                }
            },
            'quantification': {
                'morphology': {
                    'compute_volume': True,
                    'compute_surface_area': True,
                    'compute_shape_factors': True,
                    'compute_topology': False,
                    'min_object_size': 10
                },
                'particles': {
                    'min_size': 50,
                    'max_size': None,
                    'connectivity': 3,
                    'compute_distributions': True
                },
                'network': {
                    'network_method': 'distance_transform',
                    'min_pore_size': 5,
                    'compute_transport': True,
                    'compute_connectivity': True
                },
                'statistics': {
                    'distributions_to_fit': ['normal', 'lognormal', 'gamma', 'weibull'],
                    'confidence_level': 0.95,
                    'bootstrap_samples': 1000
                }
            },
            'visualization': {
                'style': 'seaborn-v0_8',
                'figure_size': [10, 8],
                'dpi': 300,
                'save_format': 'png',
                'color_palette': 'viridis'
            },
            'processing': {
                'parallel': {
                    'max_workers': None,  # Auto-detect
                    'backend': 'threading',
                    'chunk_size': 'auto'
                },
                'gpu': {
                    'use_gpu': 'auto',
                    'device_id': 0,
                    'memory_fraction': 0.8
                },
                'logging': {
                    'level': 'INFO',
                    'save_logs': True,
                    'log_file': 'fibsem_analysis.log'
                }
            }
        }
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from {config_path}")
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            try:
                import yaml
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            except ImportError:
                raise ImportError("Please install PyYAML to load YAML configuration files")
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
        
        # Merge with default configuration
        self._merge_config(self.config, loaded_config)
        
        # Validate configuration
        self._validate_config()
    
    def save_config(self, config_path: Union[str, Path]) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Output path for configuration file
            
        Raises:
            ValueError: If path contains path traversal sequences
        """
        config_path = Path(config_path)
        
        # Security: Validate path to prevent path traversal attacks
        # Resolve the path and ensure it doesn't escape allowed directories
        try:
            resolved_path = config_path.resolve()
            # Check for path traversal attempts (e.g., ".." sequences)
            if ".." in str(config_path):
                raise ValueError("Path traversal sequences ('..') are not allowed in config paths")
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid path: {e}")
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving configuration to {config_path}")
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            try:
                import yaml
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            except ImportError:
                raise ImportError("Please install PyYAML to save YAML configuration files")
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
    
    def _merge_config(self, base_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries."""
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate voxel size
        voxel_size = self.config['data']['default_voxel_size']
        if not (isinstance(voxel_size, list) and len(voxel_size) == 3):
            raise ValueError("Voxel size must be a list of 3 values")
        
        if any(v <= 0 for v in voxel_size):
            raise ValueError("Voxel size values must be positive")
        
        # Validate memory limit
        memory_limit = self.config['data']['memory_limit']
        if isinstance(memory_limit, str) and memory_limit.endswith('%'):
            try:
                percent = float(memory_limit[:-1])
                if not 0 < percent <= 100:
                    raise ValueError("Memory limit percentage must be between 0 and 100")
            except ValueError:
                raise ValueError("Invalid memory limit format")
        
        # Validate segmentation parameters
        seg_config = self.config['segmentation']
        
        # Watershed parameters
        watershed = seg_config['traditional']['watershed']
        if watershed['min_distance'] <= 0:
            raise ValueError("Watershed min_distance must be positive")
        if not 0 <= watershed['threshold_rel'] <= 1:
            raise ValueError("Watershed threshold_rel must be between 0 and 1")
        
        # Deep learning parameters
        for method in ['multiresunet', 'wnet3d']:
            if method in seg_config['deep_learning']:
                params = seg_config['deep_learning'][method]
                patch_size = params['patch_size']
                if not (isinstance(patch_size, list) and len(patch_size) == 3):
                    raise ValueError(f"{method} patch_size must be a list of 3 values")
                if any(p <= 0 for p in patch_size):
                    raise ValueError(f"{method} patch_size values must be positive")
        
        logger.info("Configuration validation passed")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_segmentation_params(self, method: str, method_type: str = 'traditional') -> Dict[str, Any]:
        """
        Get segmentation parameters for specific method.
        
        Args:
            method: Segmentation method name
            method_type: 'traditional' or 'deep_learning'
            
        Returns:
            Dictionary of parameters for the method
        """
        try:
            return self.config['segmentation'][method_type][method].copy()
        except KeyError:
            logger.warning(f"No configuration found for {method_type}.{method}")
            return {}
    
    def get_quantification_params(self, analysis_type: str) -> Dict[str, Any]:
        """
        Get quantification parameters for specific analysis type.
        
        Args:
            analysis_type: Type of analysis ('morphology', 'particles', 'network', 'statistics')
            
        Returns:
            Dictionary of parameters for the analysis type
        """
        try:
            return self.config['quantification'][analysis_type].copy()
        except KeyError:
            logger.warning(f"No configuration found for quantification.{analysis_type}")
            return {}
    
    def setup_wizard(self) -> None:
        """Interactive setup wizard for basic configuration."""
        print("FIB-SEM Configuration Setup Wizard")
        print("=" * 40)
        
        # Get voxel size
        print("\n1. Voxel Size Configuration")
        print("Current voxel size (z, y, x in nm):", self.config['data']['default_voxel_size'])
        
        response = input("Update voxel size? (y/n): ").lower()
        if response == 'y':
            try:
                z_size = float(input("Z voxel size (nm): "))
                y_size = float(input("Y voxel size (nm): "))
                x_size = float(input("X voxel size (nm): "))
                self.config['data']['default_voxel_size'] = [z_size, y_size, x_size]
                print("✓ Voxel size updated")
            except ValueError:
                print("✗ Invalid input, keeping current values")
        
        # Memory configuration
        print("\n2. Memory Configuration")
        print("Current memory limit:", self.config['data']['memory_limit'])
        
        response = input("Update memory limit? (y/n): ").lower()
        if response == 'y':
            memory_limit = input("Memory limit (e.g., '75%' or '16GB'): ")
            self.config['data']['memory_limit'] = memory_limit
            print("✓ Memory limit updated")
        
        # Processing configuration
        print("\n3. Processing Configuration")
        print("Current max workers:", self.config['processing']['parallel']['max_workers'])
        
        response = input("Update parallel processing settings? (y/n): ").lower()
        if response == 'y':
            try:
                max_workers = input("Max workers (or 'auto'): ")
                if max_workers.lower() == 'auto':
                    self.config['processing']['parallel']['max_workers'] = None
                else:
                    self.config['processing']['parallel']['max_workers'] = int(max_workers)
                print("✓ Parallel processing settings updated")
            except ValueError:
                print("✗ Invalid input, keeping current values")
        
        print("\n✓ Configuration setup complete!")
        
        # Ask to save configuration
        response = input("Save configuration to file? (y/n): ").lower()
        if response == 'y':
            user_input = input("Configuration file name (default: fibsem_config.json): ")
            
            try:
                # Sanitize the user input to prevent path traversal
                config_path = _sanitize_config_path(user_input)
                self.save_config(config_path)
                print(f"✓ Configuration saved to {config_path}")
            except ValueError as e:
                print(f"✗ Error: {e}")
            except Exception as e:
                print(f"✗ Error saving configuration: {e}")

def create_default_config_file(output_path: Union[str, Path] = "fibsem_config.json") -> None:
    """
    Create a default configuration file.
    
    Args:
        output_path: Path for the output configuration file
    """
    config = FIBSEMConfig()
    config.save_config(output_path)
    print(f"Default configuration saved to {output_path}")

def load_config_from_dict(config_dict: Dict[str, Any]) -> FIBSEMConfig:
    """
    Create configuration from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        FIBSEMConfig object
    """
    config = FIBSEMConfig()
    config._merge_config(config.config, config_dict)
    config._validate_config()
    return config

