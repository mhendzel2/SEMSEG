"""
Main GUI Application for FIB-SEM Segmentation and Quantification.

This module provides a user-friendly graphical interface for loading data,
performing segmentation, and visualizing results.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
import numpy as np
import threading
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.config import FIBSEMConfig
from core.data_io import load_fibsem_data, get_file_info, FIBSEMData, load_subvolume
from pipeline.main_pipeline import FIBSEMPipeline


class FIBSEMGUIApp:
    """Main GUI application for FIB-SEM analysis."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("FIB-SEM Segmentation and Quantification Tool")
        self.root.geometry("1200x800")
        
        # Initialize data and pipeline
        self.config = FIBSEMConfig()
        self.pipeline = None
        self.current_data = None
        self.segmentation_result = None
        self.current_slice_index = 0
        self.roi_selection = {}
        self.rect_selector = None
        
        # Setup the GUI
        self.setup_gui()
        
        # Status variables
        self.processing = False
    
    def setup_gui(self):
        """Setup the main GUI layout."""
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.setup_data_tab()
        self.setup_segmentation_tab()
        self.setup_visualization_tab()
        self.setup_results_tab()
        
        # Status bar
        self.setup_status_bar(main_frame)
    
    def setup_data_tab(self):
        """Setup the data loading tab."""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data Loading")
        
        # File selection section
        file_section = ttk.LabelFrame(data_frame, text="Data File Selection", padding=10)
        file_section.pack(fill=tk.X, padx=10, pady=5)
        
        # File path display
        self.file_path_var = tk.StringVar()
        file_path_frame = ttk.Frame(file_section)
        file_path_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_path_frame, text="Selected file:").pack(side=tk.LEFT)
        file_path_entry = ttk.Entry(file_path_frame, textvariable=self.file_path_var, state="readonly")
        file_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Browse button
        browse_button = ttk.Button(file_section, text="Browse...", command=self.browse_file)
        browse_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Or label
        ttk.Label(file_section, text="OR").pack(side=tk.LEFT, padx=5)

        # OpenOrganelle section
        oo_section = ttk.LabelFrame(data_frame, text="Load from OpenOrganelle.org", padding=10)
        oo_section.pack(fill=tk.X, padx=10, pady=5)

        oo_frame = ttk.Frame(oo_section)
        oo_frame.pack(fill=tk.X, pady=5)

        ttk.Label(oo_frame, text="Dataset ID:").pack(side=tk.LEFT)
        self.oo_id_var = tk.StringVar()
        oo_id_entry = ttk.Entry(oo_frame, textvariable=self.oo_id_var)
        oo_id_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # File information section
        info_section = ttk.LabelFrame(data_frame, text="File Information", padding=10)
        info_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # File info text widget
        self.file_info_text = tk.Text(info_section, height=6, state=tk.DISABLED)
        scrollbar_info = ttk.Scrollbar(info_section, orient=tk.VERTICAL, command=self.file_info_text.yview)
        self.file_info_text.configure(yscrollcommand=scrollbar_info.set)
        
        self.file_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_info.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Voxel size configuration
        voxel_section = ttk.LabelFrame(data_frame, text="Voxel Size Configuration", padding=10)
        voxel_section.pack(fill=tk.X, padx=10, pady=5)
        
        voxel_frame = ttk.Frame(voxel_section)
        voxel_frame.pack()
        
        ttk.Label(voxel_frame, text="Z (nm):").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.voxel_z_var = tk.StringVar(value="10.0")
        ttk.Entry(voxel_frame, textvariable=self.voxel_z_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(voxel_frame, text="Y (nm):").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.voxel_y_var = tk.StringVar(value="5.0")
        ttk.Entry(voxel_frame, textvariable=self.voxel_y_var, width=10).grid(row=0, column=3, padx=5)
        
        ttk.Label(voxel_frame, text="X (nm):").grid(row=0, column=4, padx=5, sticky=tk.W)
        self.voxel_x_var = tk.StringVar(value="5.0")
        ttk.Entry(voxel_frame, textvariable=self.voxel_x_var, width=10).grid(row=0, column=5, padx=5)
        
        # ROI Selection Section
        roi_section = ttk.LabelFrame(data_frame, text="Region of Interest (ROI) Selection", padding=10)
        roi_section.pack(fill=tk.X, padx=10, pady=5)

        roi_frame = ttk.Frame(roi_section)
        roi_frame.pack(fill=tk.X)

        # Z-axis selection
        ttk.Label(roi_frame, text="Z-slice range:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.roi_z_start_var = tk.StringVar(value="0")
        ttk.Entry(roi_frame, textvariable=self.roi_z_start_var, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(roi_frame, text="to").grid(row=0, column=2)
        self.roi_z_end_var = tk.StringVar(value="0")
        ttk.Entry(roi_frame, textvariable=self.roi_z_end_var, width=8).grid(row=0, column=3, padx=2)

        # X/Y range display (updated by RectangleSelector)
        self.roi_xy_var = tk.StringVar(value="X/Y range: (not selected)")
        ttk.Label(roi_frame, textvariable=self.roi_xy_var).grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky=tk.W)

        # Load buttons
        load_buttons_frame = ttk.Frame(data_frame)
        load_buttons_frame.pack(pady=10)
        ttk.Button(load_buttons_frame, text="Load Low-Res Preview", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_buttons_frame, text="Load High-Res ROI", command=self.load_roi).pack(side=tk.LEFT, padx=5)
    
    def setup_segmentation_tab(self):
        """Setup the segmentation tab."""
        seg_frame = ttk.Frame(self.notebook)
        self.notebook.add(seg_frame, text="Segmentation")
        
        # Method selection
        method_section = ttk.LabelFrame(seg_frame, text="Segmentation Method", padding=10)
        method_section.pack(fill=tk.X, padx=10, pady=5)
        
        # Method type
        method_type_frame = ttk.Frame(method_section)
        method_type_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(method_type_frame, text="Method Type:").pack(side=tk.LEFT)
        self.method_type_var = tk.StringVar(value="traditional")
        method_type_combo = ttk.Combobox(method_type_frame, textvariable=self.method_type_var,
                                       values=["traditional", "deep_learning"], state="readonly")
        method_type_combo.pack(side=tk.LEFT, padx=(5, 0))
        method_type_combo.bind("<<ComboboxSelected>>", self.on_method_type_change)
        
        # Specific method
        method_frame = ttk.Frame(method_section)
        method_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="watershed")
        self.method_combo = ttk.Combobox(method_frame, textvariable=self.method_var, state="readonly")
        self.method_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Parameters section
        params_section = ttk.LabelFrame(seg_frame, text="Parameters", padding=10)
        params_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Parameters notebook
        self.params_notebook = ttk.Notebook(params_section)
        self.params_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Preprocessing tab
        preprocess_frame = ttk.Frame(self.params_notebook)
        self.params_notebook.add(preprocess_frame, text="Preprocessing")
        
        # Preprocessing options
        self.noise_reduction_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preprocess_frame, text="Noise Reduction", 
                       variable=self.noise_reduction_var).pack(anchor=tk.W, pady=2)
        
        self.contrast_enhancement_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preprocess_frame, text="Contrast Enhancement", 
                       variable=self.contrast_enhancement_var).pack(anchor=tk.W, pady=2)
        
        self.artifact_removal_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(preprocess_frame, text="Artifact Removal", 
                       variable=self.artifact_removal_var).pack(anchor=tk.W, pady=2)
        
        # Segmentation parameters tab
        seg_params_frame = ttk.Frame(self.params_notebook)
        self.params_notebook.add(seg_params_frame, text="Segmentation")
        
        # Dynamic parameter widgets will be added here
        self.param_widgets = {}
        self.param_frame = ttk.Frame(seg_params_frame)
        self.param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize method options and parameters
        self.on_method_type_change()
        
        # Run segmentation button
        ttk.Button(seg_frame, text="Run Segmentation", command=self.run_segmentation).pack(pady=10)
    
    def setup_visualization_tab(self):
        """Setup the visualization tab."""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Visualization")
        
        # Control panel
        control_panel = ttk.Frame(viz_frame)
        control_panel.pack(fill=tk.X, padx=10, pady=5)
        
        # Display mode selection
        ttk.Label(control_panel, text="Display:").pack(side=tk.LEFT)
        self.display_mode_var = tk.StringVar(value="original")
        display_combo = ttk.Combobox(control_panel, textvariable=self.display_mode_var,
                                   values=["original", "segmented", "overlay"], state="readonly")
        display_combo.pack(side=tk.LEFT, padx=(5, 10))
        display_combo.bind("<<ComboboxSelected>>", self.update_visualization)
        
        # Slice navigation
        ttk.Label(control_panel, text="Slice:").pack(side=tk.LEFT)
        self.slice_var = tk.IntVar(value=0)
        self.slice_scale = ttk.Scale(control_panel, from_=0, to=100, orient=tk.HORIZONTAL,
                                   variable=self.slice_var, command=self.on_slice_change)
        self.slice_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        
        self.slice_label = ttk.Label(control_panel, text="0/0")
        self.slice_label.pack(side=tk.LEFT)
        
        # Matplotlib figure
        self.viz_figure, self.viz_axes = plt.subplots(1, 2, figsize=(10, 5))
        self.viz_figure.tight_layout()
        
        self.viz_canvas = FigureCanvasTkAgg(self.viz_figure, viz_frame)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Navigation toolbar
        viz_toolbar = NavigationToolbar2Tk(self.viz_canvas, viz_frame)
        viz_toolbar.update()

        # Add RectangleSelector for ROI
        self.rect_selector = RectangleSelector(
            self.viz_axes[0], self.on_roi_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        self.rect_selector.set_active(False)
    
    def setup_results_tab(self):
        """Setup the results and analysis tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        
        # Analysis controls
        analysis_section = ttk.LabelFrame(results_frame, text="Analysis Options", padding=10)
        analysis_section.pack(fill=tk.X, padx=10, pady=5)
        
        analysis_buttons_frame = ttk.Frame(analysis_section)
        analysis_buttons_frame.pack()
        
        ttk.Button(analysis_buttons_frame, text="Morphological Analysis",
                  command=self.run_morphological_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(analysis_buttons_frame, text="Particle Analysis",
                  command=self.run_particle_analysis).pack(side=tk.LEFT, padx=5)
        
        # Results display
        results_section = ttk.LabelFrame(results_frame, text="Analysis Results", padding=10)
        results_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Results text widget
        self.results_text = tk.Text(results_section, state=tk.DISABLED)
        scrollbar_results = ttk.Scrollbar(results_section, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar_results.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_results.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_status_bar(self, parent):
        """Setup the status bar."""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=(0, 10))
    
    def browse_file(self):
        """Open file browser to select data file."""
        filetypes = [
            ("All supported", "*.tif *.tiff *.h5 *.hdf5 *.npy"),
            ("TIFF files", "*.tif *.tiff"),
            ("HDF5 files", "*.h5 *.hdf5"),
            ("NumPy files", "*.npy"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select FIB-SEM data file",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
            self.oo_id_var.set("")  # Clear OpenOrganelle ID
            self.show_file_info(filename)
    
    def show_file_info(self, file_path):
        """Display information about the selected file."""
        try:
            info = get_file_info(file_path)
            
            info_text = f"File: {info['file_path']}\n"
            info_text += f"Size: {info['file_size'] / (1024**2):.2f} MB\n"
            info_text += f"Format: {info['format']}\n"
            
            if 'shape' in info:
                info_text += f"Shape: {info['shape']}\n"
            if 'dtype' in info:
                info_text += f"Data type: {info['dtype']}\n"
            
            self.file_info_text.config(state=tk.NORMAL)
            self.file_info_text.delete(1.0, tk.END)
            self.file_info_text.insert(1.0, info_text)
            self.file_info_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file information: {str(e)}")
    
    def load_data(self):
        """Load data from file or OpenOrganelle."""
        file_path = self.file_path_var.get()
        oo_id = self.oo_id_var.get()

        if not file_path and not oo_id:
            messagebox.showerror("Error", "Please select a data file or enter an OpenOrganelle dataset ID.")
            return
        
        if oo_id:
            load_path = f"oo:{oo_id}"
            self.file_path_var.set("") # Clear file path
        else:
            load_path = file_path

        try:
            # Get voxel size
            voxel_z = float(self.voxel_z_var.get())
            voxel_y = float(self.voxel_y_var.get())
            voxel_x = float(self.voxel_x_var.get())
            voxel_size = (voxel_z, voxel_y, voxel_x)
            
            self.status_var.set("Loading data...")
            self.progress_bar.start()
            
            # Load data in a separate thread to avoid blocking GUI
            def load_thread():
                try:
                    self.pipeline = FIBSEMPipeline(config=self.config, voxel_spacing=voxel_size)
                    result = self.pipeline.load_data(load_path)
                    
                    if result['success']:
                        self.current_data = result['data']
                        self.root.after(0, self.on_data_loaded)
                    else:
                        self.root.after(0, lambda: self.on_load_error(result['error']))
                        
                except Exception as e:
                    self.root.after(0, lambda: self.on_load_error(str(e)))
            
            threading.Thread(target=load_thread, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid voxel size values. Please enter valid numbers.")
    
    def on_data_loaded(self):
        """Handle successful data loading."""
        self.progress_bar.stop()
        self.status_var.set(f"Data loaded: {self.current_data.shape}")
        
        # Update slice navigation
        if len(self.current_data.shape) == 3:
            max_slice = self.current_data.shape[0] - 1
            self.slice_scale.configure(to=max_slice)
            self.slice_var.set(max_slice // 2)  # Set to middle slice
            self.slice_label.config(text=f"{self.slice_var.get()}/{max_slice}")
        else:
            self.slice_scale.configure(to=0)
            self.slice_var.set(0)
            self.slice_label.config(text="0/0")
        
        # Update ROI selection widgets
        self.roi_z_start_var.set("0")
        self.roi_z_end_var.set(str(max_slice))
        self.rect_selector.set_active(True)

        # Update visualization
        self.update_visualization()
        
        messagebox.showinfo("Success", f"Data loaded successfully!\nShape: {self.current_data.shape}")
    
    def on_load_error(self, error_msg):
        """Handle data loading error."""
        self.progress_bar.stop()
        self.status_var.set("Ready")
        messagebox.showerror("Error", f"Failed to load data: {error_msg}")

    def load_roi(self):
        """Load the selected high-resolution Region of Interest."""
        oo_id = self.oo_id_var.get()
        if not oo_id:
            messagebox.showerror("Error", "ROI loading only works with OpenOrganelle datasets.")
            return

        if 'x' not in self.roi_selection or 'y' not in self.roi_selection:
            messagebox.showerror("Error", "Please select an X/Y region in the visualization tab first.")
            return

        try:
            z_start = int(self.roi_z_start_var.get())
            z_end = int(self.roi_z_end_var.get())

            # Create slice objects
            roi_slices = (
                slice(z_start, z_end),
                slice(self.roi_selection['y'][0], self.roi_selection['y'][1]),
                slice(self.roi_selection['x'][0], self.roi_selection['x'][1])
            )

            self.status_var.set("Loading high-resolution ROI...")
            self.progress_bar.start()

            def load_roi_thread():
                try:
                    # Assume the preview is the lowest resolution (-1)
                    subvolume = load_subvolume(
                        dataset_path=f"oo:{oo_id}",
                        roi_slices=roi_slices,
                        preview_resolution_level=-1
                    )

                    # Replace current data with the new subvolume
                    self.current_data = subvolume
                    self.root.after(0, self.on_data_loaded)

                except Exception as e:
                    self.root.after(0, lambda: self.on_load_error(f"Failed to load ROI: {str(e)}"))

            threading.Thread(target=load_roi_thread, daemon=True).start()

        except ValueError:
            messagebox.showerror("Error", "Invalid Z-slice range. Please enter valid numbers.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def on_method_type_change(self, event=None):
        """Handle method type selection change."""
        method_type = self.method_type_var.get()
        
        if method_type == "traditional":
            methods = ["watershed", "thresholding", "morphology"]
        else:
            methods = ["multiresunet", "wnet3d"]
        
        self.method_combo['values'] = methods
        if methods:
            self.method_var.set(methods[0])
        
        self.update_parameter_widgets()
    
    def update_parameter_widgets(self):
        """Update parameter input widgets based on selected method."""
        # Clear existing parameter widgets
        for widget in self.param_widgets.values():
            widget.destroy()
        self.param_widgets.clear()
        
        method_type = self.method_type_var.get()
        method = self.method_var.get()
        
        if not method:
            return
        
        # Get default parameters from config
        params = self.config.get_segmentation_params(method, method_type)
        
        row = 0
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                # Numeric parameter
                label = ttk.Label(self.param_frame, text=f"{param_name.replace('_', ' ').title()}:")
                label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
                
                var = tk.StringVar(value=str(param_value))
                entry = ttk.Entry(self.param_frame, textvariable=var, width=15)
                entry.grid(row=row, column=1, padx=5, pady=2)
                
                self.param_widgets[param_name] = var
                row += 1
                
            elif isinstance(param_value, bool):
                # Boolean parameter
                var = tk.BooleanVar(value=param_value)
                check = ttk.Checkbutton(self.param_frame, text=param_name.replace('_', ' ').title(),
                                      variable=var)
                check.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
                
                self.param_widgets[param_name] = var
                row += 1
    
    def run_segmentation(self):
        """Run the segmentation process."""
        if self.current_data is None:
            messagebox.showerror("Error", "Please load data first.")
            return
        
        if self.processing:
            messagebox.showwarning("Warning", "Processing already in progress.")
            return
        
        self.processing = True
        self.status_var.set("Running segmentation...")
        self.progress_bar.start()
        
        def segmentation_thread():
            try:
                # Get preprocessing options
                preprocessing_steps = []
                if self.noise_reduction_var.get():
                    preprocessing_steps.append('noise_reduction')
                if self.contrast_enhancement_var.get():
                    preprocessing_steps.append('contrast_enhancement')
                if self.artifact_removal_var.get():
                    preprocessing_steps.append('artifact_removal')
                
                # Run preprocessing if any steps selected
                if preprocessing_steps:
                    preprocess_result = self.pipeline.preprocess_data(preprocessing_steps=preprocessing_steps)
                    if not preprocess_result['success']:
                        raise Exception(f"Preprocessing failed: {preprocess_result['error']}")
                
                # Get segmentation parameters
                method_type = self.method_type_var.get()
                method = self.method_var.get()
                
                # Collect parameters from widgets
                seg_params = {}
                for param_name, widget_var in self.param_widgets.items():
                    try:
                        if isinstance(widget_var, tk.BooleanVar):
                            seg_params[param_name] = widget_var.get()
                        else:
                            value = widget_var.get()
                            # Try to convert to number if possible
                            try:
                                if '.' in value:
                                    seg_params[param_name] = float(value)
                                else:
                                    seg_params[param_name] = int(value)
                            except ValueError:
                                seg_params[param_name] = value
                    except Exception:
                        pass  # Skip invalid parameters
                
                # Run segmentation
                result = self.pipeline.segment_data(method=method, method_type=method_type, **seg_params)
                
                if result['success']:
                    self.segmentation_result = result['segmentation']
                    self.root.after(0, self.on_segmentation_complete, result)
                else:
                    self.root.after(0, lambda: self.on_segmentation_error(result['error']))
                    
            except Exception as e:
                self.root.after(0, lambda: self.on_segmentation_error(str(e)))
        
        threading.Thread(target=segmentation_thread, daemon=True).start()
    
    def on_segmentation_complete(self, result):
        """Handle successful segmentation completion."""
        self.processing = False
        self.progress_bar.stop()
        self.status_var.set(f"Segmentation complete: {result['num_labels']} objects found")
        
        # Update visualization
        self.update_visualization()
        
        # Show results
        results_text = f"Segmentation Results:\n"
        results_text += f"Method: {result['method_type']}.{result['method']}\n"
        results_text += f"Number of objects: {result['num_labels']}\n"
        results_text += f"Processing time: {result['duration']:.2f} seconds\n"
        
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)
        self.results_text.config(state=tk.DISABLED)
        
        messagebox.showinfo("Success", f"Segmentation completed!\n{result['num_labels']} objects found.")
    
    def on_segmentation_error(self, error_msg):
        """Handle segmentation error."""
        self.processing = False
        self.progress_bar.stop()
        self.status_var.set("Ready")
        messagebox.showerror("Error", f"Segmentation failed: {error_msg}")
    
    def update_visualization(self, event=None):
        """Update the visualization display."""
        if self.current_data is None:
            return
        
        # Clear axes
        for ax in self.viz_axes:
            ax.clear()
        
        # Get current slice
        slice_idx = self.slice_var.get()
        
        # Get data slice
        if len(self.current_data.shape) == 3:
            original_slice = self.current_data.data[slice_idx]
        else:
            original_slice = self.current_data.data
        
        # Display original data
        self.viz_axes[0].imshow(original_slice, cmap='gray')
        self.viz_axes[0].set_title('Original Data')
        self.viz_axes[0].axis('off')
        
        # Display segmentation or overlay
        display_mode = self.display_mode_var.get()
        
        if self.segmentation_result is not None:
            if len(self.segmentation_result.shape) == 3:
                seg_slice = self.segmentation_result[slice_idx]
            else:
                seg_slice = self.segmentation_result
            
            if display_mode == "segmented":
                self.viz_axes[1].imshow(seg_slice, cmap='tab20')
                self.viz_axes[1].set_title('Segmentation')
            elif display_mode == "overlay":
                self.viz_axes[1].imshow(original_slice, cmap='gray')
                # Create overlay with transparency
                masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
                self.viz_axes[1].imshow(masked_seg, cmap='tab20', alpha=0.5)
                self.viz_axes[1].set_title('Overlay')
            else:
                self.viz_axes[1].imshow(original_slice, cmap='gray')
                self.viz_axes[1].set_title('Original Data')
        else:
            self.viz_axes[1].imshow(original_slice, cmap='gray')
            self.viz_axes[1].set_title('Original Data')
        
        self.viz_axes[1].axis('off')
        
        # Update canvas
        self.viz_canvas.draw()
    
    def on_slice_change(self, value):
        """Handle slice navigation change."""
        if self.current_data is None:
            return
        
        slice_idx = int(float(value))
        max_slice = self.current_data.shape[0] - 1 if len(self.current_data.shape) == 3 else 0
        self.slice_label.config(text=f"{slice_idx}/{max_slice}")
        self.update_visualization()

    def on_roi_select(self, eclick, erelease):
        """Callback for the RectangleSelector."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        self.roi_selection['x'] = sorted((x1, x2))
        self.roi_selection['y'] = sorted((y1, y2))

        # Update the label in the data tab
        self.roi_xy_var.set(f"X range: {self.roi_selection['x'][0]}-{self.roi_selection['x'][1]}, Y range: {self.roi_selection['y'][0]}-{self.roi_selection['y'][1]}")
    
    def run_morphological_analysis(self):
        """Run morphological analysis on segmentation results."""
        if self.segmentation_result is None:
            messagebox.showerror("Error", "Please run segmentation first.")
            return
        
        self.status_var.set("Running morphological analysis...")
        self.progress_bar.start()
        
        def analysis_thread():
            try:
                result = self.pipeline.quantify_morphology()
                self.root.after(0, lambda: self.on_analysis_complete("Morphological", result))
            except Exception as e:
                self.root.after(0, lambda: self.on_analysis_error(str(e)))
        
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def run_particle_analysis(self):
        """Run particle analysis on segmentation results."""
        if self.segmentation_result is None:
            messagebox.showerror("Error", "Please run segmentation first.")
            return
        
        self.status_var.set("Running particle analysis...")
        self.progress_bar.start()
        
        def analysis_thread():
            try:
                result = self.pipeline.quantify_particles()
                self.root.after(0, lambda: self.on_analysis_complete("Particle", result))
            except Exception as e:
                self.root.after(0, lambda: self.on_analysis_error(str(e)))
        
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def on_analysis_complete(self, analysis_type, result):
        """Handle analysis completion."""
        self.progress_bar.stop()
        self.status_var.set("Ready")
        
        if result['success']:
            # Format results for display
            results_text = f"{analysis_type} Analysis Results:\n"
            results_text += "=" * 40 + "\n"
            
            if analysis_type == "Morphological":
                morph_data = result['morphological_analysis']
                results_text += f"Number of objects: {morph_data['num_objects']}\n"
                if morph_data['volumes']:
                    volumes = morph_data['volumes']
                    results_text += f"Volume statistics:\n"
                    results_text += f"  Mean: {np.mean(volumes):.2f} nm³\n"
                    results_text += f"  Std: {np.std(volumes):.2f} nm³\n"
                    results_text += f"  Min: {np.min(volumes):.2f} nm³\n"
                    results_text += f"  Max: {np.max(volumes):.2f} nm³\n"
            
            elif analysis_type == "Particle":
                results_text += f"Number of particles: {result['num_particles']}\n"
                if result['particle_properties']:
                    particles = result['particle_properties']
                    areas = [p['area'] for p in particles]
                    results_text += f"Area statistics:\n"
                    results_text += f"  Mean: {np.mean(areas):.2f} pixels\n"
                    results_text += f"  Std: {np.std(areas):.2f} pixels\n"
                    results_text += f"  Min: {np.min(areas):.2f} pixels\n"
                    results_text += f"  Max: {np.max(areas):.2f} pixels\n"
            
            results_text += f"\nProcessing time: {result['duration']:.2f} seconds\n"
            
            # Append to results display
            self.results_text.config(state=tk.NORMAL)
            self.results_text.insert(tk.END, "\n" + results_text)
            self.results_text.config(state=tk.DISABLED)
            self.results_text.see(tk.END)
            
            messagebox.showinfo("Success", f"{analysis_type} analysis completed!")
        else:
            messagebox.showerror("Error", f"{analysis_type} analysis failed: {result['error']}")
    
    def on_analysis_error(self, error_msg):
        """Handle analysis error."""
        self.progress_bar.stop()
        self.status_var.set("Ready")
        messagebox.showerror("Error", f"Analysis failed: {error_msg}")


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = FIBSEMGUIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()