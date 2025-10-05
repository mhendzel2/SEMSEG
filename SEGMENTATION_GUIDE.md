# Advanced Segmentation Tools for 3D SEM Data

This document describes the comprehensive segmentation capabilities available for 3D scanning electron microscopy (SEM) image analysis in the SEMSEG framework.

## Overview

The segmentation module has been significantly enhanced with state-of-the-art methods specifically designed for 3D SEM data processing. These tools address the unique challenges of SEM data including noise, intensity variations, and complex 3D structures.

## Traditional Segmentation Methods

### 1. Watershed Segmentation
- **Use case**: Separating touching objects, cell segmentation
- **Parameters**: `min_distance`, `threshold_rel`, `watershed_line`
- **3D support**: Yes, processes slice-by-slice with 3D distance transform
- **Best for**: Separating closely packed structures

### 2. Region Growing
- **Use case**: Segmenting regions with similar intensity values
- **Parameters**: `seed_threshold`, `growth_threshold`, `connectivity`
- **3D support**: Yes, full 3D connectivity
- **Best for**: Homogeneous structures, organelles

### 3. Graph Cuts
- **Use case**: Binary segmentation with smooth boundaries
- **Parameters**: `lambda`, `sigma`
- **3D support**: Yes, slice-by-slice processing
- **Dependencies**: PyMaxflow (optional)
- **Best for**: Foreground/background separation

### 4. Active Contours (Snakes)
- **Use case**: Smooth boundary detection
- **Parameters**: `alpha`, `beta`, `gamma`, `iterations`
- **3D support**: Yes, morphological Chan-Vese implementation
- **Best for**: Smooth membrane structures

### 5. SLIC Superpixels
- **Use case**: Over-segmentation for hierarchical methods
- **Parameters**: `n_segments`, `compactness`, `sigma`
- **3D support**: Yes, true 3D superpixels
- **Best for**: Preprocessing for other methods

### 6. Felzenszwalb Graph-Based
- **Use case**: Hierarchical segmentation
- **Parameters**: `scale`, `sigma`, `min_size`
- **3D support**: Yes, slice-by-slice with label consistency
- **Best for**: Multi-scale structure detection

### 7. Random Walker
- **Use case**: Semi-supervised segmentation
- **Parameters**: `beta`, `mode`
- **3D support**: Yes, slice-by-slice processing
- **Best for**: Weakly supervised segmentation

## Deep Learning Methods

### 1. 2D U-Net
- **Use case**: Slice-by-slice segmentation
- **Parameters**: `model_path`, `input_size`, `num_classes`, `threshold`
- **Architecture**: Standard encoder-decoder with skip connections
- **Best for**: High-resolution 2D segmentation

### 2. 3D U-Net
- **Use case**: True volumetric segmentation
- **Parameters**: `model_path`, `patch_size`, `num_classes`, `threshold`, `overlap`
- **Architecture**: 3D convolutional encoder-decoder
- **Best for**: 3D context-dependent structures

### 3. V-Net
- **Use case**: Medical image segmentation with residual connections
- **Parameters**: `model_path`, `patch_size`, `num_classes`, `threshold`
- **Architecture**: V-shaped network with residual connections
- **Best for**: Complex 3D structures, detailed boundaries

### 4. Attention U-Net
- **Use case**: Segmentation with attention mechanisms
- **Parameters**: `model_path`, `input_size`, `num_classes`, `threshold`
- **Architecture**: U-Net with attention gates
- **Best for**: Fine-grained boundary detection

### 5. nnU-Net Style
- **Use case**: Self-configuring segmentation
- **Parameters**: `spacing`, `num_classes`, `threshold`
- **Architecture**: Adaptive network based on data characteristics
- **Best for**: Automated parameter selection

## Usage Examples

### Traditional Methods
```python
from core.segmentation import segment_traditional

# Region growing segmentation
params = {
    'seed_threshold': 0.5,
    'growth_threshold': 0.1,
    'connectivity': 1
}
labels = segment_traditional(data, 'region_growing', params)

# SLIC superpixels
params = {
    'n_segments': 1000,
    'compactness': 10.0,
    'sigma': 1.0
}
labels = segment_traditional(data, 'slic', params)
```

### Deep Learning Methods
```python
from core.segmentation import segment_deep_learning

# 3D U-Net segmentation
params = {
    'model_path': 'path/to/trained/model.h5',
    'patch_size': (64, 64, 64),
    'num_classes': 2,
    'threshold': 0.5,
    'overlap': 0.25
}
labels = segment_deep_learning(data, 'unet_3d', params)

# V-Net segmentation
params = {
    'model_path': 'path/to/vnet/model.h5',
    'patch_size': (64, 64, 64),
    'num_classes': 2,
    'threshold': 0.5
}
labels = segment_deep_learning(data, 'vnet', params)
```

### Pipeline Integration
```python
from pipeline.main_pipeline import FIBSEMPipeline

pipeline = FIBSEMPipeline()
pipeline.load_data('data/sample.tif')
pipeline.preprocess_data()

# Traditional segmentation
result = pipeline.segment_data(
    method='region_growing',
    method_type='traditional'
)

# Deep learning segmentation
result = pipeline.segment_data(
    method='unet_3d',
    method_type='deep_learning'
)
```

## Configuration

All segmentation methods can be configured through the configuration system:

```yaml
segmentation:
  traditional:
    region_growing:
      seed_threshold: 0.5
      growth_threshold: 0.1
      connectivity: 1
    slic:
      n_segments: 1000
      compactness: 10.0
      sigma: 1.0
  
  deep_learning:
    unet_3d:
      patch_size: [64, 64, 64]
      num_classes: 2
      threshold: 0.5
      overlap: 0.25
```

## Dependencies

### Required
- numpy >= 1.20.0
- scipy >= 1.7.0
- scikit-image >= 0.18.0

### Optional
- tensorflow >= 2.8.0 (for deep learning methods)
- PyMaxflow >= 1.2.13 (for graph cuts)

### Installation
```bash
pip install -r requirements.txt

# For GPU support
pip install tensorflow-gpu>=2.8.0

# For graph cuts
pip install PyMaxflow
```

## Performance Considerations

### Memory Management
- **3D Deep Learning**: Uses sliding window approach for large volumes
- **Traditional Methods**: Optimized for memory efficiency
- **Overlap Handling**: Automatic blending for sliding window predictions

### GPU Acceleration
- **TensorFlow**: Automatic GPU detection and usage
- **Scikit-image**: CPU optimized with parallel processing
- **Memory Limits**: Configurable through data loader

### Recommendations by Data Size
- **Small volumes (<512³)**: Use 3D methods directly
- **Medium volumes (512³-1024³)**: Use sliding window approach
- **Large volumes (>1024³)**: Consider superpixel preprocessing

## Best Practices

1. **Preprocessing**: Always apply noise reduction and contrast enhancement
2. **Method Selection**: Start with traditional methods for exploratory analysis
3. **Deep Learning**: Requires pre-trained models for best results
4. **Validation**: Use multiple methods and compare results
5. **Parameter Tuning**: Use configuration files for reproducibility

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce patch size or enable memory mapping
2. **Import errors**: Install optional dependencies
3. **Poor results**: Check preprocessing and parameter settings
4. **GPU issues**: Verify TensorFlow GPU installation

### Performance Optimization
1. Use appropriate data types (uint8/uint16 vs float32)
2. Enable memory mapping for large datasets
3. Optimize patch sizes for your GPU memory
4. Consider multi-scale approaches for speed