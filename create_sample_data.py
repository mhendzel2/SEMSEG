"""
Script to generate sample FIB-SEM data for testing the GUI.
"""

import numpy as np
import os
from pathlib import Path

def create_sample_data():
    """Create sample 3D FIB-SEM data for testing."""
    # Create test data directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Generate synthetic 3D data (50 slices, 256x256 pixels)
    np.random.seed(42)  # For reproducible results
    
    # Create a basic structure with some objects
    data = np.random.randint(50, 150, (50, 256, 256), dtype=np.uint8)
    
    # Add some circular objects
    center_z, center_y, center_x = 25, 128, 128
    
    for i in range(50):
        for y in range(256):
            for x in range(256):
                # Create spherical objects
                dist = np.sqrt((i - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
                if dist < 20:
                    data[i, y, x] = 220
                elif dist < 30:
                    data[i, y, x] = 180
                
                # Add some smaller objects
                for obj_center in [(10, 64, 64), (40, 192, 192), (30, 64, 192)]:
                    obj_z, obj_y, obj_x = obj_center
                    dist = np.sqrt((i - obj_z)**2 + (y - obj_y)**2 + (x - obj_x)**2)
                    if dist < 15:
                        data[i, y, x] = 200
    
    # Add some noise
    noise = np.random.normal(0, 10, data.shape)
    data = np.clip(data.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Save as numpy file
    output_file = test_dir / "sample_fibsem_data.npy"
    np.save(output_file, data)
    
    print(f"Sample data created: {output_file}")
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Value range: {data.min()} - {data.max()}")
    
    return str(output_file)

if __name__ == "__main__":
    create_sample_data()