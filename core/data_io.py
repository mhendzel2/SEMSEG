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



def load_fibsem_data(file_path: Union[str, Path],
                     voxel_size: Optional[Tuple[float, float, float]] = None,
                     resolution_level: Optional[int] = -1) -> Union[FIBSEMData, zarr.Group]:
    """
    Load FIB-SEM data from various file formats or OpenOrganelle.

    For OpenOrganelle datasets, this function opens a remote connection
    and can load a specific resolution level or return the Zarr group.

    Args:
        file_path: Path to the data file or OpenOrganelle ID (e.g., "oo:jrc_hela-2")
        voxel_size: Optional voxel size (z, y, x) in micrometers
        resolution_level: For multiscale datasets, the resolution level to load.
                          -1 loads the lowest resolution (default).
                          Other integers select from the available resolutions (e.g., 0 for highest).
                          If None, the remote Zarr group is returned without loading data.

    Returns:
        FIBSEMData object, or a Zarr group if resolution_level is None for an OpenOrganelle dataset.
    """
    if isinstance(file_path, str) and file_path.startswith("oo:"):
        dataset_name = file_path.split(":")[1]
        try:
            s3 = s3fs.S3FileSystem(anon=True)
            url = f"s3://janelia-cosem-datasets/{dataset_name}/{dataset_name}.zarr"
            store = s3fs.S3Map(root=url, s3=s3, check=False)
            zarr_group = zarr.open(store, mode='r')

            if resolution_level is None:
                return zarr_group

            # Handle multiscale data
            if 'multiscales' in zarr_group.attrs:
                datasets = zarr_group.attrs['multiscales'][0]['datasets']

                if resolution_level == -1:
                    # Load the lowest resolution (last in the list)
                    level_path = datasets[-1]['path']
                elif resolution_level < len(datasets):
                    level_path = datasets[resolution_level]['path']
                else:
                    raise ValueError(f"Resolution level {resolution_level} is out of bounds for dataset {dataset_name} with {len(datasets)} levels.")

                data = np.array(zarr_group[level_path])
            else:
                # If no multiscales, treat as a single array
                data = np.array(zarr_group)

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


def load_subvolume(
    dataset_path: str,
    roi_slices: Tuple[slice, slice, slice],
    preview_resolution_level: int = -1,
) -> FIBSEMData:
    """
    Load a sub-volume from a remote OpenOrganelle dataset.

    This function translates ROI coordinates from a lower-resolution preview
    to the highest resolution and downloads only the selected region.

    Args:
        dataset_path: The OpenOrganelle dataset identifier (e.g., "oo:jrc_hela-2").
        roi_slices: A tuple of slices for (z, y, x) defining the ROI on the preview resolution.
        preview_resolution_level: The resolution level on which the ROI was defined.
                                  -1 corresponds to the lowest resolution.

    Returns:
        A FIBSEMData object containing the high-resolution sub-volume.
    """
    if not (isinstance(dataset_path, str) and dataset_path.startswith("oo:")):
        raise ValueError("This function is designed for OpenOrganelle datasets only.")

    # Open the remote Zarr group without loading any data
    zarr_group = load_fibsem_data(dataset_path, resolution_level=None)

    if not isinstance(zarr_group, zarr.Group) or 'multiscales' not in zarr_group.attrs:
        raise ValueError("Dataset is not a valid multiscale OME-Zarr.")

    multiscale_meta = zarr_group.attrs['multiscales'][0]
    datasets = multiscale_meta['datasets']

    # Get metadata for the highest resolution (level 0)
    high_res_meta = datasets[0]
    high_res_path = high_res_meta['path']
    high_res_transforms = high_res_meta.get('coordinateTransformations', [{'type': 'scale', 'scale': [1, 1, 1]}])
    high_res_scale = next(t['scale'] for t in high_res_transforms if t['type'] == 'scale')

    # Get metadata for the preview resolution
    if preview_resolution_level == -1:
        preview_res_meta = datasets[-1]
    elif preview_resolution_level < len(datasets):
        preview_res_meta = datasets[preview_resolution_level]
    else:
        raise ValueError(f"Invalid preview resolution level: {preview_resolution_level}")

    preview_res_transforms = preview_res_meta.get('coordinateTransformations', [{'type': 'scale', 'scale': [1, 1, 1]}])
    preview_res_scale = next(t['scale'] for t in preview_res_transforms if t['type'] == 'scale')

    # Calculate the scaling factor between the preview and high-res data
    # Assuming axes are ordered (z, y, x)
    scale_factors = [
        preview_s / high_s for preview_s, high_s in zip(preview_res_scale, high_res_scale)
    ]

    # Scale the ROI slices
    scaled_slices = []
    for i, s in enumerate(roi_slices):
        start = int(s.start * scale_factors[i]) if s.start is not None else 0
        stop = int(s.stop * scale_factors[i]) if s.stop is not None else -1
        scaled_slices.append(slice(start, stop))

    # Get the highest resolution data array
    high_res_array = zarr_group[high_res_path]

    # Slice the array to get the sub-volume. This performs the partial download.
    subvolume_data = np.array(high_res_array[tuple(scaled_slices)])

    # For voxel size, we should use the one from the high-resolution data
    voxel_size = tuple(high_res_scale)

    return FIBSEMData(data=subvolume_data, voxel_size=voxel_size)

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

