# 🎯 SEMSEG - Quick Reference Card

## How to Initiate the Program

### ⚡ Fastest Methods

```powershell
# 1. GUI (Recommended for beginners)
.\venv\Scripts\python.exe launch_gui.py

# 2. Quick Demo
.\venv\Scripts\python.exe quick_start.py

# 3. Your Data (Command Line)
.\venv\Scripts\python.exe -m __main__ --run "your_data.tif"
```

---

## 📋 All Available Methods

### 1️⃣ GUI Mode
```powershell
.\venv\Scripts\python.exe launch_gui.py
```
✅ Point-and-click interface  
✅ Visual parameter adjustment  
✅ Perfect for exploration

### 2️⃣ Web Interface
```powershell
.\venv\Scripts\python.exe -m __main__ --web
```
✅ Browser-based  
✅ Modern UI  
✅ Remote access friendly

### 3️⃣ Command Line
```powershell
# Basic
.\venv\Scripts\python.exe -m __main__ --run "data.tif"

# With options
.\venv\Scripts\python.exe -m __main__ --run "data.tif" --method region_growing --type traditional

# With config
.\venv\Scripts\python.exe -m __main__ --run "data.tif" --config "config.yaml" --output "results/"
```
✅ Automation ready  
✅ Batch processing  
✅ Scriptable

### 4️⃣ Python Script
```python
import sys
sys.path.insert(0, r'c:\Users\mjhen\Github\SEMSEG')

from core.segmentation import segment_traditional
labels = segment_traditional(data, 'region_growing', params)
```
✅ Full control  
✅ Custom workflows  
✅ Research applications

### 5️⃣ Interactive Python
```powershell
.\venv\Scripts\python.exe
>>> import sys; sys.path.insert(0, r'c:\Users\mjhen\Github\SEMSEG')
>>> from core import segmentation
```
✅ Testing  
✅ Exploration  
✅ Development

---

## 🔧 Common Commands

```powershell
# Verify installation
.\venv\Scripts\python.exe verify_installation.py

# Run diagnostics
.\venv\Scripts\python.exe -m __main__ --diagnostics

# Test installation
.\venv\Scripts\python.exe -m __main__ --test

# Quick demo
.\venv\Scripts\python.exe quick_start.py

# Get help
.\venv\Scripts\python.exe -m __main__ --help
```

---

## 📊 Segmentation Methods

### Traditional (9 methods)
```powershell
--method watershed          # Default, good for separating objects
--method region_growing     # Homogeneous structures
--method graph_cuts         # Smooth boundaries
--method active_contours    # Membrane detection
--method slic              # Superpixels
--method felzenszwalb      # Hierarchical
--method random_walker     # Semi-supervised
--method thresholding      # Basic intensity
--method morphology        # Shape-based
```

### Deep Learning (5 methods)
```powershell
--method unet_2d           # Slice-by-slice
--method unet_3d           # Full 3D
--method vnet              # Medical imaging optimized
--method attention_unet    # With attention mechanism
--method nnunet            # Self-configuring
```

---

## 📂 File Support

```powershell
# Local files
--run "data.tif"                    # TIFF stack
--run "data.h5"                     # HDF5
--run "data.npy"                    # NumPy array

# Remote data
--run "s3://bucket/data.zarr"       # S3 Zarr
--run "oo:jrc_hela-2"               # OpenOrganelle
```

---

## 💡 First-Time User Path

```powershell
# Step 1: Quick demo
.\venv\Scripts\python.exe quick_start.py

# Step 2: Try GUI
.\venv\Scripts\python.exe launch_gui.py

# Step 3: Your data (simple)
.\venv\Scripts\python.exe -m __main__ --run "your_data.tif"

# Step 4: Your data (optimized)
.\venv\Scripts\python.exe -m __main__ --run "your_data.tif" --method region_growing --output "results/"
```

---

## 📖 Documentation Quick Links

| Document | Purpose |
|----------|---------|
| `HOW_TO_RUN.md` | **Detailed guide to all run methods** |
| `SEGMENTATION_GUIDE.md` | Method descriptions & parameters |
| `INSTALLATION.md` | Setup & usage examples |
| `README.md` | Project overview |

---

## 🚨 Troubleshooting

**Problem: Module not found**
```python
# Solution: Add to path
import sys
sys.path.insert(0, r'c:\Users\mjhen\Github\SEMSEG')
```

**Problem: PowerShell execution errors**
```powershell
# Solution: Use full venv path
.\venv\Scripts\python.exe script.py
```

**Problem: GUI won't start**
```powershell
# Solution: Check tkinter
.\venv\Scripts\python.exe -c "import tkinter; print('OK')"
```

---

## 🎓 Example Workflows

### Workflow 1: Quick Analysis
```powershell
.\venv\Scripts\python.exe -m __main__ --run "data.tif" --method watershed
```

### Workflow 2: High-Quality Segmentation
```powershell
.\venv\Scripts\python.exe -m __main__ --run "data.tif" --method region_growing --output "results/"
```

### Workflow 3: Deep Learning
```powershell
.\venv\Scripts\python.exe -m __main__ --run "data.tif" --method unet_3d --type deep_learning
```

### Workflow 4: Custom Script
```python
# my_workflow.py
import sys
sys.path.insert(0, r'c:\Users\mjhen\Github\SEMSEG')

from core.data_io import load_fibsem_data
from core.preprocessing import preprocess_fibsem_data  
from core.segmentation import segment_traditional

data = load_fibsem_data('data.tif')
preprocessed = preprocess_fibsem_data(data.data, steps=['noise_reduction'])
labels = segment_traditional(preprocessed, 'region_growing', {
    'seed_threshold': 0.5,
    'growth_threshold': 0.1
})
print(f"Found {labels.max()} objects")
```

Run: `.\venv\Scripts\python.exe my_workflow.py`

---

## ⚙️ Configuration Example

Create `my_config.yaml`:
```yaml
segmentation:
  traditional:
    region_growing:
      seed_threshold: 0.5
      growth_threshold: 0.1
      connectivity: 1
    watershed:
      min_distance: 20
      threshold_rel: 0.6
```

Use: `.\venv\Scripts\python.exe -m __main__ --run data.tif --config my_config.yaml`

---

**Made by**: Manus AI  
**Version**: 1.0  
**Last Updated**: October 5, 2025

---

Need more help? Run: `.\venv\Scripts\python.exe quick_start.py --help-quick`
