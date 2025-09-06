"""
GUI Demo Script - Test GUI functionality without opening windows.

This script demonstrates the GUI functionality by creating sample data,
running analysis, and generating visualization plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from core.config import FIBSEMConfig
from core.data_io import load_fibsem_data, get_file_info
from pipeline.main_pipeline import FIBSEMPipeline

def demo_gui_functionality():
    """Demonstrate GUI functionality without actual GUI."""
    print("FIB-SEM GUI Functionality Demo")
    print("=" * 50)
    
    # 1. Data Loading Demo
    print("\n1. Data Loading Demo:")
    sample_file = 'test_data/sample_fibsem_data.npy'
    
    if not os.path.exists(sample_file):
        print(f"✗ Sample file not found: {sample_file}")
        return
    
    # Get file information
    info = get_file_info(sample_file)
    print(f"✓ File: {info['file_path']}")
    print(f"✓ Size: {info['file_size'] / (1024**2):.2f} MB")
    print(f"✓ Shape: {info.get('shape', 'unknown')}")
    print(f"✓ Data type: {info.get('dtype', 'unknown')}")
    
    # Load data
    config = FIBSEMConfig()
    pipeline = FIBSEMPipeline(config=config, voxel_spacing=(10.0, 5.0, 5.0))
    result = pipeline.load_data(sample_file)
    
    if not result['success']:
        print(f"✗ Data loading failed: {result['error']}")
        return
    
    data = result['data']
    print(f"✓ Data loaded successfully: {data.shape}")
    
    # 2. Preprocessing Demo
    print("\n2. Preprocessing Demo:")
    preprocessing_steps = ['noise_reduction', 'contrast_enhancement']
    preprocess_result = pipeline.preprocess_data(preprocessing_steps=preprocessing_steps)
    
    if preprocess_result['success']:
        print(f"✓ Preprocessing completed: {preprocessing_steps}")
    else:
        print(f"✗ Preprocessing failed: {preprocess_result['error']}")
    
    # 3. Segmentation Demo
    print("\n3. Segmentation Demo:")
    seg_result = pipeline.segment_data(method='watershed', method_type='traditional')
    
    if not seg_result['success']:
        print(f"✗ Segmentation failed: {seg_result['error']}")
        return
    
    print(f"✓ Segmentation completed:")
    print(f"  Method: {seg_result['method_type']}.{seg_result['method']}")
    print(f"  Objects found: {seg_result['num_labels']}")
    print(f"  Processing time: {seg_result['duration']:.2f} seconds")
    
    # 4. Analysis Demo
    print("\n4. Analysis Demo:")
    
    # Morphological analysis
    morph_result = pipeline.quantify_morphology()
    if morph_result['success']:
        print(f"✓ Morphological analysis:")
        print(f"  Objects analyzed: {morph_result['num_objects']}")
        if morph_result['object_properties']:
            volumes = [p['volume_nm3'] for p in morph_result['object_properties'] if 'volume_nm3' in p]
            if volumes:
                print(f"  Volume range: {np.min(volumes):.1f} - {np.max(volumes):.1f} nm³")
                print(f"  Mean volume: {np.mean(volumes):.1f} nm³")
    
    # Particle analysis
    particle_result = pipeline.quantify_particles(min_size=50)
    if particle_result['success']:
        print(f"✓ Particle analysis:")
        print(f"  Particles found: {particle_result['num_particles']}")
    
    # 5. Visualization Demo
    print("\n5. Visualization Demo:")
    
    # Create output directory
    output_dir = Path('gui_demo_output')
    output_dir.mkdir(exist_ok=True)
    
    # Get middle slice for visualization
    middle_slice = data.shape[0] // 2
    original_slice = data.data[middle_slice]
    segmented_slice = seg_result['segmentation'][middle_slice]
    
    # Create visualization plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    axes[0].imshow(original_slice, cmap='gray')
    axes[0].set_title(f'Original Data (Slice {middle_slice})')
    axes[0].axis('off')
    
    # Segmentation
    axes[1].imshow(segmented_slice, cmap='tab20')
    axes[1].set_title(f'Segmentation ({seg_result["num_labels"]} objects)')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(original_slice, cmap='gray')
    masked_seg = np.ma.masked_where(segmented_slice == 0, segmented_slice)
    axes[2].imshow(masked_seg, cmap='tab20', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = output_dir / 'gui_demo_visualization.png'
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: {viz_file}")
    
    # Create analysis summary plot
    if morph_result['success'] and morph_result['object_properties']:
        volumes = [p['volume_nm3'] for p in morph_result['object_properties'] if 'volume_nm3' in p]
        areas = [p['num_voxels'] for p in morph_result['object_properties']]

        if volumes:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Volume histogram
            ax1.hist(volumes, bins=20, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Volume (nm³)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Object Volume Distribution')
            ax1.grid(True, alpha=0.3)

            # Volume vs area scatter
            if len(areas) == len(volumes):
                ax2.scatter(areas, volumes, alpha=0.6)
                ax2.set_xlabel('Area (voxels)')
                ax2.set_ylabel('Volume (nm³)')
                ax2.set_title('Volume vs Area')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        analysis_file = output_dir / 'gui_demo_analysis.png'
        plt.savefig(analysis_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Analysis plots saved: {analysis_file}")
    
    # 6. Summary
    print("\n6. Demo Summary:")
    print("✓ GUI functionality successfully demonstrated:")
    print("  - Data loading and file information display")
    print("  - Preprocessing with noise reduction and contrast enhancement")
    print("  - Watershed segmentation with parameter configuration")
    print("  - Morphological and particle analysis")
    print("  - Visualization with slice navigation and overlay modes")
    print("  - Results display and analysis plots")
    
    print(f"\nDemo outputs saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    demo_gui_functionality()