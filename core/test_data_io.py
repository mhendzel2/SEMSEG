import unittest
from unittest import mock
import numpy as np
import zarr
from pathlib import Path

from .data_io import load_fibsem_data, load_subvolume, FIBSEMData

def create_mock_ome_zarr_group():
    """Creates an in-memory Zarr group simulating a multiscale OME-Zarr dataset."""
    store = zarr.storage.MemoryStore()
    root = zarr.group(store=store)

    # Create multiscale metadata
    root.attrs['multiscales'] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"}
            ],
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]
                },
                {
                    "path": "s1",
                    "coordinateTransformations": [{"type": "scale", "scale": [2.0, 2.0, 2.0]}]
                },
                {
                    "path": "s2",
                    "coordinateTransformations": [{"type": "scale", "scale": [4.0, 4.0, 4.0]}]
                }
            ]
        }
    ]

    # Create data arrays for each resolution
    root.create_array("s0", data=np.zeros((100, 100, 100), dtype=np.uint8)) # High-res
    root.create_array("s1", data=np.zeros((50, 50, 50), dtype=np.uint8))   # Mid-res
    root.create_array("s2", data=np.zeros((25, 25, 25), dtype=np.uint8))   # Low-res

    return root

class TestRemoteDataIO(unittest.TestCase):

    @mock.patch('core.data_io.s3fs.S3FileSystem')
    @mock.patch('core.data_io.zarr.open')
    def test_load_fibsem_data_remote(self, mock_zarr_open, mock_s3fs):
        """Test loading data from a remote OpenOrganelle Zarr store."""
        mock_zarr_group = create_mock_ome_zarr_group()
        mock_zarr_open.return_value = mock_zarr_group

        dataset_id = "oo:jrc_hela-2"

        # 1. Test loading lowest resolution by default
        fibsem_data = load_fibsem_data(dataset_id)
        self.assertIsInstance(fibsem_data, FIBSEMData)
        self.assertEqual(fibsem_data.data.shape, (25, 25, 25)) # s2 shape

        # 2. Test loading a specific resolution (highest)
        fibsem_data_s0 = load_fibsem_data(dataset_id, resolution_level=0)
        self.assertEqual(fibsem_data_s0.data.shape, (100, 100, 100)) # s0 shape

        # 3. Test returning the Zarr group itself
        zarr_group = load_fibsem_data(dataset_id, resolution_level=None)
        self.assertIsInstance(zarr_group, zarr.Group)
        self.assertIn('s0', zarr_group)
        self.assertIn('s1', zarr_group)
        self.assertIn('s2', zarr_group)

    @mock.patch('core.data_io.s3fs.S3FileSystem')
    @mock.patch('core.data_io.zarr.open')
    @mock.patch('zarr.core.array.Array.__getitem__')
    def test_load_subvolume(self, mock_array_getitem, mock_zarr_open, mock_s3fs):
        """Test loading a sub-volume from a remote dataset."""
        mock_zarr_group = create_mock_ome_zarr_group()
        mock_zarr_open.return_value = mock_zarr_group

        # Make the mocked __getitem__ return a correctly shaped array
        mock_array_getitem.return_value = np.zeros((20, 20, 20))

        dataset_id = "oo:jrc_hela-2"
        # ROI on the lowest resolution (s2, shape 25x25x25)
        roi_slices = (slice(10, 15), slice(10, 15), slice(10, 15))

        # Load subvolume from this ROI
        subvolume_data = load_subvolume(
            dataset_path=dataset_id,
            roi_slices=roi_slices,
            preview_resolution_level=-1 # s2
        )

        # Verify that zarr.open was called to get the group
        mock_zarr_open.assert_called_once()

        # Verify the slicing on the high-resolution array (s0)
        # Scale factor from s2 to s0 is 4.0/1.0 = 4
        expected_slice = (
            slice(10 * 4, 15 * 4), # 40:60
            slice(10 * 4, 15 * 4), # 40:60
            slice(10 * 4, 15 * 4)  # 40:60
        )

        # Check that the array was sliced with the correct scaled coordinates
        mock_array_getitem.assert_called_once_with(expected_slice)

        # Check the shape of the returned data
        self.assertEqual(subvolume_data.data.shape, (20, 20, 20))

if __name__ == '__main__':
    unittest.main()
