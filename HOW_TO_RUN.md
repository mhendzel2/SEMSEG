# How to Initiate SEMSEG

There are **5 different ways** to run the SEMSEG program, depending on your needs:

---

## 1. 🖥️ **GUI Mode (Graphical Interface)**

### Launch the Desktop GUI:
```powershell
# Using the venv Python
.\venv\Scripts\python.exe launch_gui.py

# Or using the main module
.\venv\Scripts\python.exe -m __main__ --gui
```

The GUI provides:
- File picker for selecting SEM data
- Visual parameter adjustment
- Interactive visualization
- Point-and-click workflow

---

## 2. 🌐 **Web Interface (Streamlit)**

### Launch the Web UI:
```powershell
# Method 1: Using the main module
.\venv\Scripts\python.exe -m __main__ --web

# Method 2: Direct streamlit command (requires streamlit installed)
.\venv\Scripts\python.exe -m streamlit run webui.py
```

The web interface provides:
- Browser-based access
- Modern, responsive UI
- Easy parameter configuration
- Suitable for remote access

---

## 3. 🔧 **Command Line Interface**

### Run Pipeline from Command Line:
```powershell
# Basic usage with default watershed segmentation
.\venv\Scripts\python.exe -m __main__ --run "path\to\data.tif"

# Specify segmentation method
.\venv\Scripts\python.exe -m __main__ --run "path\to\data.tif" --method region_growing --type traditional

# With custom configuration file
.\venv\Scripts\python.exe -m __main__ --run "path\to\data.tif" --config "config.yaml" --output "results/"

# Deep learning segmentation
.\venv\Scripts\python.exe -m __main__ --run "path\to\data.tif" --method unet_3d --type deep_learning
```

### Available CLI Options:
- `--run PATH` - Path to data file (.tif/.h5/.npy) or remote Zarr
- `--method METHOD` - Segmentation method (default: watershed)
- `--type TYPE` - Segmentation type: traditional or deep_learning
- `--config PATH` - Optional config file (JSON or YAML)
- `--output DIR` - Output directory for results

### CLI Utilities:
```powershell
# Run diagnostics
.\venv\Scripts\python.exe -m __main__ --diagnostics

# Test installation
.\venv\Scripts\python.exe -m __main__ --test
```

---

## 4. 📝 **Python Script/Interactive**

### Create a Python Script:

```python
# my_segmentation.py
import sys
sys.path.insert(0, r'c:\Users\mjhen\Github\SEMSEG')

from core.config import FIBSEMConfig
from core.data_io import load_fibsem_data
from core.segmentation import segment_traditional, segment_deep_learning
from core.preprocessing import preprocess_fibsem_data

# Load data
print("Loading data...")
data = load_fibsem_data('path/to/your/data.tif', voxel_size=(10, 5, 5))

# Preprocess
print("Preprocessing...")
preprocessed = preprocess_fibsem_data(
    data.data,
    steps=['noise_reduction', 'contrast_enhancement'],
    noise_reduction={'method': 'gaussian', 'sigma': 1.0},
    contrast_enhancement={'method': 'clahe'}
)

# Segment with region growing
print("Segmenting...")
labels = segment_traditional(
    preprocessed,
    method='region_growing',
    params={
        'seed_threshold': 0.5,
        'growth_threshold': 0.1,
        'connectivity': 1
    }
)

print(f"Segmentation complete! Found {labels.max()} objects")
```

### Run Your Script:
```powershell
.\venv\Scripts\python.exe my_segmentation.py
```

### Interactive Python Session:
```powershell
# Start Python
.\venv\Scripts\python.exe

# Then in Python:
>>> import sys
>>> sys.path.insert(0, r'c:\Users\mjhen\Github\SEMSEG')
>>> from core import segmentation
>>> # Now you can use all functions interactively
```

---

## 5. 🔬 **Using the Complete Pipeline**

### Full Pipeline Approach:

```python
# full_pipeline.py
import sys
sys.path.insert(0, r'c:\Users\mjhen\Github\SEMSEG')

from core.config import FIBSEMConfig
# Note: Pipeline has relative import issues, use direct imports instead

# Import core modules
from core.data_io import load_fibsem_data
from core.preprocessing import preprocess_fibsem_data
from core.segmentation import segment_traditional
from core.quantification import quantify_morphology

# 1. Configuration
print("Setting up configuration...")
config = FIBSEMConfig()

# 2. Load data
print("Loading data...")
data = load_fibsem_data('path/to/data.tif', voxel_size=(10, 5, 5))

# 3. Preprocess
print("Preprocessing...")
preprocessed = preprocess_fibsem_data(
    data.data,
    steps=['noise_reduction', 'contrast_enhancement'],
    noise_reduction={'method': 'gaussian', 'sigma': 1.0},
    contrast_enhancement={'method': 'clahe'}
)

# 4. Segment
print("Segmenting...")
segmentation_params = config.get_segmentation_params('region_growing', 'traditional')
labels = segment_traditional(preprocessed, 'region_growing', segmentation_params)

# 5. Quantify
print("Quantifying...")
results = quantify_morphology(labels, voxel_size=data.voxel_size)

print(f"Analysis complete!")
print(f"Found {results['num_objects']} objects")
print(f"Total volume: {results.get('total_volume', 'N/A')}")
```

### Run:
```powershell
.\venv\Scripts\python.exe full_pipeline.py
```

---

## 📊 **Quick Reference - All Methods**

| Method | Command | Use Case |
|--------|---------|----------|
| **GUI** | `.\venv\Scripts\python.exe launch_gui.py` | Interactive, visual workflow |
| **Web** | `.\venv\Scripts\python.exe -m __main__ --web` | Browser-based, remote access |
| **CLI** | `.\venv\Scripts\python.exe -m __main__ --run data.tif` | Batch processing, automation |
| **Script** | `.\venv\Scripts\python.exe my_script.py` | Custom workflows, research |
| **Interactive** | `.\venv\Scripts\python.exe` | Testing, exploration |

---

## 🎯 **Recommended Workflow for First-Time Users**

1. **Start with GUI** to understand the workflow:
   ```powershell
   .\venv\Scripts\python.exe launch_gui.py
   ```

2. **Try CLI** for a quick test:
   ```powershell
   .\venv\Scripts\python.exe -m __main__ --run "your_data.tif" --method region_growing
   ```

3. **Create custom scripts** for your specific analysis:
   - Copy examples from `SEGMENTATION_GUIDE.md`
   - Modify parameters for your data
   - Automate repetitive tasks

---

## 🆘 **Troubleshooting**

### PowerShell Execution Errors?
Always use the full path to the venv Python:
```powershell
.\venv\Scripts\python.exe script.py
```

### Module Import Errors?
Add the SEMSEG directory to your Python path:
```python
import sys
sys.path.insert(0, r'c:\Users\mjhen\Github\SEMSEG')
```

### GUI Not Working?
Check that tkinter is available (should be by default with Python):
```powershell
.\venv\Scripts\python.exe -c "import tkinter; print('OK')"
```

### Web UI Not Working?
Install streamlit (optional dependency):
```powershell
.\venv\Scripts\pip install streamlit
```

---

## 📚 **Next Steps**

- Read `SEGMENTATION_GUIDE.md` for detailed method descriptions
- Check `INSTALLATION.md` for usage examples
- Review `README.md` for project overview
- Run `verify_installation.py` to test your setup

---

## 💡 **Pro Tips**

1. **Start Simple**: Begin with traditional methods (watershed, region_growing) before trying deep learning
2. **Preprocess First**: Always apply noise reduction and contrast enhancement
3. **Iterate Parameters**: Use the GUI to find good parameters, then automate with scripts
4. **Save Configurations**: Create YAML config files for reproducibility
5. **Monitor Memory**: For large 3D volumes, use sliding window approaches (automatic for 3D U-Net)

---

**Need Help?** Check the documentation files or run:
```powershell
.\venv\Scripts\python.exe -m __main__ --help
```
