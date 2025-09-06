"""
Data Input/Output module for FIB-SEM datasets.

This module provides unified interfaces for reading and writing FIB-SEM data
in various formats including TIFF stacks, HDF5 files, and raw binary data.
"""

import numpy as np
import os
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import logging
import requests
import zarr
import shutil
import s3fs

# Set up logging
logger = logging.getLogger(__name__)

class FIBSEMData:
    """Container class for FIB-SEM data with metadata."""
    
    def __init__(self, data: np.ndarray, voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Initialize FIB-SEM data container.
        
        Args:
            data: 3D numpy array containing the image data
            voxel_size: Tuple of (z, y, x) voxel dimensions in micrometers
        """
        self.data = data
        self.voxel_size = voxel_size
        self.metadata = {}
        
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of the data."""
        return self.data.shape
    
    @property
    def volume_um3(self) -> float:
        """Calculate total volume in cubic micrometers."""
        voxel_volume = self.voxel_size[0] * self.voxel_size[1] * self.voxel_size[2]
        return np.prod(self.shape) * voxel_volume
    
    def get_slice(self, axis: int, index: int) -> np.ndarray:
        """Get a 2D slice along specified axis."""
        if axis == 0:
            return self.data[index, :, :]
        elif axis == 1:
            return self.data[:, index, :]
        elif axis == 2:
            return self.data[:, :, index]
        else:
            raise ValueError("Axis must be 0, 1, or 2")

def download_openorganelle_data(dataset_name: str, cache_dir: Union[str, Path] = "cache") -> Path:
    """
    Download data from OpenOrganelle.org.
    Args:
        dataset_name: Name of the dataset (e.g., "jrc_hela-2")
        cache_dir: Directory to cache downloaded data
    Returns:
        Path to the downloaded zarr dataset
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    zarr_path = cache_dir / f"{dataset_name}.zarr"

    if zarr_path.exists():
        logger.info(f"Dataset '{dataset_name}' found in cache.")
        return zarr_path

    # URL from https://www.openorganelle.org/datasets
    url = f"https://janelia-cosem-datasets.s3.amazonaws.com/{dataset_name}/{dataset_name}.zarr"

    logger.info(f"Downloading dataset '{dataset_name}' from {url}")

    # This is a simplified download. A real implementation would handle this better.
    # For now, we assume the user has `s3fs` installed and zarr can handle the s3 URL.
    try:
        s3 = s3fs.S3FileSystem(anon=True)

        # Recursively download the directory
        s3.get(f"janelia-cosem-datasets/{dataset_name}/{dataset_name}.zarr", str(zarr_path), recursive=True)

        logger.info(f"Successfully downloaded and cached '{dataset_name}' to {zarr_path}")

    except ImportError:
        logger.error("s3fs is required to download from OpenOrganelle. Please install with 'pip install s3fs'")
        raise
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        if zarr_path.exists():
            shutil.rmtree(zarr_path)
        raise

    return zarr_path


def load_fibsem_data(file_path: Union[str, Path], 
                     voxel_size: Optional[Tuple[float, float, float]] = None) -> FIBSEMData:
    """
    Load FIB-SEM data from various file formats or OpenOrganelle.
    
    Args:
        file_path: Path to the data file or OpenOrganelle ID (e.g., "oo:jrc_hela-2")
        voxel_size: Optional voxel size (z, y, x) in micrometers
        
    Returns:
        FIBSEMData object containing the loaded data
    """
    if isinstance(file_path, str) and file_path.startswith("oo:"):
        dataset_name = file_path.split(":")[1]
        try:
            local_path = download_openorganelle_data(dataset_name)
            data = zarr.open(str(local_path), mode='r')
            # Convert to numpy array in memory
            data = np.array(data)
        except Exception as e:
            raise IOError(f"Failed to load OpenOrganelle dataset '{dataset_name}': {e}")
    else:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading FIB-SEM data from {file_path}")

        # Determine file format and load accordingly
        if file_path.suffix.lower() in ['.tif', '.tiff']:
            data = _load_tiff_stack(file_path)
        elif file_path.suffix.lower() in ['.h5', '.hdf5']:
            data = _load_hdf5(file_path)
        elif file_path.suffix.lower() == '.npy':
            data = _load_numpy(file_path)
        elif file_path.suffix.lower() == '.raw':
            data = _load_raw_binary(file_path)
        else:
            # Try to load as numpy array first
            try:
                data = np.load(file_path)
            except:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Ensure data is 3D
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D data, got {data.ndim}D")
    
    # Set default voxel size if not provided
    if voxel_size is None:
        voxel_size = (1.0, 1.0, 1.0)
    
    logger.info(f"Loaded data with shape {data.shape} and voxel size {voxel_size}")
    
    return FIBSEMData(data, voxel_size)

def _load_tiff_stack(file_path: Path) -> np.ndarray:
    """Load TIFF stack using tifffile or PIL."""
    try:
        import tifffile
        return tifffile.imread(file_path)
    except ImportError:
        try:
            from PIL import Image
            import numpy as np
            
            # Load multi-page TIFF
            img = Image.open(file_path)
            frames = []
            
            try:
                while True:
                    frames.append(np.array(img))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            
            return np.stack(frames, axis=0)
            
        except ImportError:
            raise ImportError("Please install tifffile or PIL to load TIFF files")

def _load_hdf5(file_path: Path) -> np.ndarray:
    """Load data from HDF5 file."""
    try:
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            # Try common dataset names
            dataset_names = ['data', 'image', 'volume', 'array']
            
            for name in dataset_names:
                if name in f:
                    return f[name][:]
            
            # If no common names found, use the first dataset
            keys = list(f.keys())
            if keys:
                return f[keys[0]][:]
            else:
                raise ValueError("No datasets found in HDF5 file")
                
    except ImportError:
        raise ImportError("Please install h5py to load HDF5 files")

def _load_numpy(file_path: Path) -> np.ndarray:
    """Load numpy array file."""
    return np.load(file_path)

def _load_raw_binary(file_path: Path) -> np.ndarray:
    """Load raw binary data (requires shape information)."""
    # This is a simplified version - in practice, you'd need shape info
    data = np.fromfile(file_path, dtype=np.uint8)
    
    # Try to guess dimensions (this is very basic)
    total_size = len(data)
    
    # Common FIB-SEM dimensions
    possible_shapes = [
        (100, 512, 512),
        (200, 512, 512),
        (500, 1024, 1024),
        (1000, 2048, 2048)
    ]
    
    for shape in possible_shapes:
        if np.prod(shape) == total_size:
            return data.reshape(shape)
    
    # If no standard shape works, make it cubic-ish
    cube_size = int(np.cbrt(total_size))
    if cube_size ** 3 == total_size:
        return data.reshape(cube_size, cube_size, cube_size)
    
    raise ValueError("Cannot determine shape for raw binary data")

def save_fibsem_data(data: Union[FIBSEMData, np.ndarray], 
                     file_path: Union[str, Path],
                     format: str = 'auto') -> None:
    """
    Save FIB-SEM data to file.
    
    Args:
        data: FIBSEMData object or numpy array to save
        file_path: Output file path
        format: Output format ('tiff', 'hdf5', 'numpy', or 'auto')
    """
    file_path = Path(file_path)
    
    # Extract numpy array if FIBSEMData object
    if isinstance(data, FIBSEMData):
        array_data = data.data
        voxel_size = data.voxel_size
        metadata = data.metadata
    else:
        array_data = data
        voxel_size = (1.0, 1.0, 1.0)
        metadata = {}
    
    # Determine format
    if format == 'auto':
        format = _determine_format_from_extension(file_path)
    
    logger.info(f"Saving data with shape {array_data.shape} to {file_path} (format: {format})")
    
    # Create output directory if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save according to format
    if format == 'tiff':
        _save_tiff(array_data, file_path, metadata)
    elif format == 'hdf5':
        _save_hdf5(array_data, file_path, voxel_size, metadata)
    elif format == 'numpy':
        _save_numpy(array_data, file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

def _determine_format_from_extension(file_path: Path) -> str:
    """Determine save format from file extension."""
    ext = file_path.suffix.lower()
    
    if ext in ['.tif', '.tiff']:
        return 'tiff'
    elif ext in ['.h5', '.hdf5']:
        return 'hdf5'
    elif ext == '.npy':
        return 'numpy'
    else:
        return 'numpy'  # Default to numpy

def _save_tiff(data: np.ndarray, file_path: Path, metadata: Dict[str, Any]) -> None:
    """Save data as TIFF stack."""
    try:
        import tifffile
        tifffile.imwrite(file_path, data, metadata=metadata)
    except ImportError:
        try:
            from PIL import Image
            
            # Convert to uint8 if needed
            if data.dtype != np.uint8:
                data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
            
            # Save as multi-page TIFF
            images = [Image.fromarray(slice_data) for slice_data in data]
            images[0].save(file_path, save_all=True, append_images=images[1:])
            
        except ImportError:
            raise ImportError("Please install tifffile or PIL to save TIFF files")

def _save_hdf5(data: np.ndarray, file_path: Path, 
               voxel_size: Tuple[float, float, float],
               metadata: Dict[str, Any]) -> None:
    """Save data as HDF5 file."""
    try:
        import h5py
        
        with h5py.File(file_path, 'w') as f:
            # Save main data
            dset = f.create_dataset('data', data=data, compression='gzip')
            
            # Save voxel size
            dset.attrs['voxel_size_z'] = voxel_size[0]
            dset.attrs['voxel_size_y'] = voxel_size[1]
            dset.attrs['voxel_size_x'] = voxel_size[2]
            
            # Save metadata
            for key, value in metadata.items():
                try:
                    dset.attrs[key] = value
                except (TypeError, ValueError):
                    # Skip metadata that can't be saved as HDF5 attributes
                    pass
                    
    except ImportError:
        raise ImportError("Please install h5py to save HDF5 files")

def _save_numpy(data: np.ndarray, file_path: Path) -> None:
    """Save data as numpy array."""
    np.save(file_path, data)

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a FIB-SEM data file without loading it.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Dictionary containing file information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    info = {
        'file_path': str(file_path),
        'file_size': file_path.stat().st_size,
        'format': file_path.suffix.lower()
    }
    
    try:
        # Try to get shape information without loading full data
        if file_path.suffix.lower() in ['.tif', '.tiff']:
            try:
                import tifffile
                with tifffile.TiffFile(file_path) as tif:
                    info['shape'] = tif.series[0].shape
                    info['dtype'] = tif.series[0].dtype
            except ImportError:
                pass
                
        elif file_path.suffix.lower() in ['.h5', '.hdf5']:
            try:
                import h5py
                with h5py.File(file_path, 'r') as f:
                    # Get info from first dataset
                    keys = list(f.keys())
                    if keys:
                        dset = f[keys[0]]
                        info['shape'] = dset.shape
                        info['dtype'] = dset.dtype
            except ImportError:
                pass
                
        elif file_path.suffix.lower() == '.npy':
            # For numpy files, we can get info without loading
            with open(file_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
                info['shape'] = shape
                info['dtype'] = dtype
                
    except Exception as e:
        logger.warning(f"Could not get detailed file info: {e}")
    
    return info

