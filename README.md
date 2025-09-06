# SEMSEG

3D FIB-SEM Segmentation and Quantification Program with GUI

## Overview

A comprehensive software package for automated analysis of three-dimensional Focused Ion Beam Scanning Electron Microscopy (FIB-SEM) datasets. The program includes both command-line tools and a user-friendly graphical interface for data loading, segmentation, and visualization.

## Features

- **Data Loading**: Support for multiple file formats (TIFF, HDF5, NumPy arrays, raw binary)
- **Preprocessing**: Noise reduction, contrast enhancement, and artifact removal
- **Segmentation**: Traditional methods (watershed, thresholding, morphology) and deep learning approaches
- **Quantification**: Morphological analysis, particle analysis, and statistical measurements
- **Visualization**: Interactive GUI with slice navigation and overlay modes
- **Configuration**: Flexible parameter management with hierarchical configuration files

## GUI Application

### Quick Start

Launch the GUI application:

```bash
python launch_gui.py
```

Or from Python:

```python
from SEMSEG import run_gui
run_gui()
```

### GUI Features

The GUI provides four main tabs:

1. **Data Loading**
   - File browser for selecting data files
   - File information display (size, shape, data type)
   - Voxel size configuration
   - Data preview

2. **Segmentation**
   - Method selection (traditional vs. deep learning)
   - Preprocessing options (noise reduction, contrast enhancement, artifact removal)
   - Parameter adjustment for segmentation methods
   - Real-time progress tracking

3. **Visualization**
   - Side-by-side display of original and segmented data
   - Slice navigation for 3D datasets
   - Multiple display modes (original, segmented, overlay)
   - Interactive matplotlib controls

4. **Results**
   - Morphological and particle analysis
   - Statistical summaries
   - Results export and visualization

### Supported File Formats

- TIFF/TIF: Single or multi-page TIFF files
- HDF5: .h5 and .hdf5 files
- NumPy: .npy array files
- Raw binary: .raw files (with shape detection)

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

Required packages:
- numpy
- scipy
- scikit-image
- matplotlib
- pandas
- h5py
- tkinter (for GUI)

Install dependencies:
```bash
pip install numpy scipy scikit-image matplotlib pandas h5py
```

Note: tkinter is usually included with Python installations. On Ubuntu/Debian, install with:
```bash
sudo apt-get install python3-tk
```

## Demo

Run the GUI functionality demo:
```bash
python demo_gui.py
```

This will create sample data and demonstrate all GUI features without requiring a display.

## Author

Manus AI  
Version: 1.0  
Date: August 2025