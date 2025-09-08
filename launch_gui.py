#!/usr/bin/env python3
"""
FIB-SEM GUI Launcher

Simple launcher script for the FIB-SEM Segmentation and Quantification GUI.
"""

import sys
import os
from pathlib import Path

# Add the package directory to Python path
package_dir = Path(__file__).parent
sys.path.insert(0, str(package_dir))

def main():
    """Launch the FIB-SEM GUI application."""
    try:
        # Import and run the GUI
        from gui.main_gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"Error importing GUI module: {e}")
        print("Please ensure all required packages are installed:")
        print("pip install numpy scipy scikit-image matplotlib h5py pandas")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()