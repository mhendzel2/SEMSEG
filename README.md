# SEMSEG

3D FIB-SEM Segmentation and Quantification Program

## Overview

A comprehensive software package for automated analysis of three-dimensional Focused Ion Beam Scanning Electron Microscopy (FIB-SEM) datasets.

## Features

- **Data Loading**: Support for multiple file formats (TIFF, HDF5, NumPy arrays, remote Zarr datasets)
- **Preprocessing**: Advanced noise reduction, contrast enhancement, and FIB-SEM artifact removal  
- **Segmentation**: Comprehensive suite of traditional and deep learning methods
  - Traditional: watershed, region growing, graph cuts, active contours, SLIC, Felzenszwalb, random walker
  - Deep Learning: 2D/3D U-Net, V-Net, Attention U-Net, nnU-Net-style adaptive segmentation
- **Quantification**: Morphological analysis, particle analysis, and statistical measurements
- **Configuration**: Flexible parameter management with hierarchical configuration files
- **3D Support**: Full volumetric processing with memory-efficient sliding window approaches

## Command Line Usage

For programmatic access:

```python
from SEMSEG import FIBSEMPipeline, FIBSEMConfig

# Create pipeline
config = FIBSEMConfig()
pipeline = FIBSEMPipeline(config=config, voxel_spacing=(10.0, 5.0, 5.0))

# Load and process data
result = pipeline.load_data('path/to/data.tif')

# Traditional segmentation
seg_result = pipeline.segment_data(method='region_growing', method_type='traditional')

# Deep learning segmentation  
seg_result = pipeline.segment_data(method='unet_3d', method_type='deep_learning')

# Quantification
morph_result = pipeline.quantify_morphology()
```

## Segmentation Methods

### Traditional Methods
- **Watershed**: Separating touching objects
- **Region Growing**: Homogeneous structure segmentation  
- **Graph Cuts**: Binary segmentation with smooth boundaries
- **Active Contours**: Smooth boundary detection
- **SLIC**: Superpixel over-segmentation
- **Felzenszwalb**: Hierarchical graph-based segmentation
- **Random Walker**: Semi-supervised segmentation

### Deep Learning Methods
- **2D U-Net**: Slice-by-slice segmentation
- **3D U-Net**: True volumetric segmentation
- **V-Net**: Medical imaging optimized 3D network
- **Attention U-Net**: Focus on relevant features
- **nnU-Net Style**: Self-configuring adaptive segmentation

See [SEGMENTATION_GUIDE.md](SEGMENTATION_GUIDE.md) for detailed usage instructions.
```

See [SEGMENTATION_GUIDE.md](SEGMENTATION_GUIDE.md) for detailed usage instructions.

## 🚀 Quick Start

### 1. Install (First Time Only)
```powershell
# Create virtual environment and install
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Run the Program

**Choose your preferred method:**

#### 🖥️ GUI Mode (Easiest)
```powershell
.\venv\Scripts\python.exe launch_gui.py
```

#### 🔧 Command Line
```powershell
# Run on your data
.\venv\Scripts\python.exe -m __main__ --run "path\to\data.tif" --method region_growing

# See all options
.\venv\Scripts\python.exe -m __main__ --help
```

#### 🌐 Web Interface
```powershell
.\venv\Scripts\python.exe -m __main__ --web
```

#### 📝 Python Script
```python
import sys
sys.path.insert(0, r'c:\Users\mjhen\Github\SEMSEG')

from core.segmentation import segment_traditional
from core.data_io import load_fibsem_data

# Load your data
data = load_fibsem_data('your_data.tif')

# Segment
labels = segment_traditional(data.data, 'region_growing', {
    'seed_threshold': 0.5,
    'growth_threshold': 0.1
})
```

#### 🎮 Quick Demo
```powershell
.\venv\Scripts\python.exe quick_start.py
```

### 📚 Complete Documentation
- **[HOW_TO_RUN.md](HOW_TO_RUN.md)** - All methods to run the program
- **[SEGMENTATION_GUIDE.md](SEGMENTATION_GUIDE.md)** - Detailed method descriptions  
- **[INSTALLATION.md](INSTALLATION.md)** - Full installation and usage guide

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Author

Manus AI  
Version: 1.0  
Date: August 2025
