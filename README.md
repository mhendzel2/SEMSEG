# SEMSEG

3D FIB-SEM Segmentation and Quantification Program

## Overview

A comprehensive software package for automated analysis of three-dimensional Focused Ion Beam Scanning Electron Microscopy (FIB-SEM) datasets.

## Features

- **Data Loading**: Support for multiple file formats (TIFF, HDF5, NumPy arrays)
- **Preprocessing**: Noise reduction, contrast enhancement, and artifact removal  
- **Segmentation**: Traditional methods (watershed, thresholding, morphology)
- **Quantification**: Morphological analysis, particle analysis, and statistical measurements
- **Configuration**: Flexible parameter management with hierarchical configuration files

## Command Line Usage

For programmatic access:

```python
from SEMSEG import FIBSEMPipeline, FIBSEMConfig

# Create pipeline
config = FIBSEMConfig()
pipeline = FIBSEMPipeline(config=config, voxel_spacing=(10.0, 5.0, 5.0))

# Load and process data
result = pipeline.load_data('path/to/data.tif')
seg_result = pipeline.segment_data(method='watershed')
morph_result = pipeline.quantify_morphology()
```

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Author

Manus AI  
Version: 1.0  
Date: August 2025
