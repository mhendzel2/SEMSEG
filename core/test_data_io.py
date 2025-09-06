import unittest
from unittest import mock
import tempfile
import shutil
from pathlib import Path
import numpy as np
import zarr

from .data_io import download_openorganelle_data, load_fibsem_data

class TestDataIO(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @mock.patch('core.data_io.s3fs.S3FileSystem')
    def test_download_openorganelle_data(self, mock_s3fs):
        """Test downloading data from OpenOrganelle.org with mocking."""

        # Configure the mock
        mock_s3_instance = mock_s3fs.return_value

        def create_dummy_zarr(s3_path, local_path, recursive):
            # Simulate the download by creating a dummy zarr store
            zarr_path = Path(local_path)
            zarr.save_array(str(zarr_path), np.zeros((10, 10)))

        mock_s3_instance.get.side_effect = create_dummy_zarr

        dataset_name = "test_dataset"
        zarr_path = download_openorganelle_data(dataset_name, cache_dir=self.test_dir)

        # Check that the path is correct and exists
        self.assertTrue(zarr_path.exists())
        self.assertEqual(zarr_path.name, f"{dataset_name}.zarr")

        # Check if it's a valid zarr store
        data = zarr.open(str(zarr_path), mode='r')
        self.assertIsInstance(data, zarr.Array)
        self.assertEqual(data.shape, (10, 10))

    @mock.patch('core.data_io.download_openorganelle_data')
    def test_load_data_openorganelle(self, mock_download):
        """Test loading data using the 'oo:' prefix with mocking."""

        dataset_name = "test_dataset"
        dummy_zarr_path = Path(self.test_dir) / f"{dataset_name}.zarr"
        zarr.save_array(str(dummy_zarr_path), np.arange(100).reshape(10, 10))

        mock_download.return_value = dummy_zarr_path

        dataset_id = f"oo:{dataset_name}"
        fibsem_data = load_fibsem_data(dataset_id)

        # Check that download was called
        mock_download.assert_called_once_with(dataset_name)

        # Check the loaded data
        self.assertIsInstance(fibsem_data.data, np.ndarray)
        self.assertEqual(fibsem_data.data.shape, (1, 10, 10))
        np.testing.assert_array_equal(fibsem_data.data, np.arange(100).reshape(1, 10, 10))

if __name__ == '__main__':
    unittest.main()
