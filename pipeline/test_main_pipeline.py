import unittest
import numpy as np
from .main_pipeline import FIBSEMPipeline
from core.data_io import FIBSEMData

class TestMainPipeline(unittest.TestCase):

    def test_active_contour_segmentation(self):
        """Test the Active Contour segmentation method in the pipeline."""
        # Create a sample 3D image with a simple object
        img_data = np.zeros((10, 50, 50), dtype=np.uint8)
        img_data[3:7, 15:35, 15:35] = 255 # A simple block object
        fibsem_data = FIBSEMData(img_data)

        # Initialize the pipeline
        pipeline = FIBSEMPipeline()
        pipeline.data = fibsem_data # Manually set the data

        # Run segmentation with active_contour
        result = pipeline.segment_data(method='active_contour', method_type='traditional')

        # Check that the segmentation was successful
        self.assertTrue(result['success'])
        self.assertIn('segmentation', result)

        segmentation_result = result['segmentation']

        # Check the output shape
        self.assertEqual(segmentation_result.shape, img_data.shape)

        # Check that some labels were created (more than just background)
        self.assertGreater(len(np.unique(segmentation_result)), 1)

    def test_active_contour_with_initial_contour(self):
        """Test Active Contour with a provided initial contour."""
        img_data = np.zeros((10, 50, 50), dtype=np.uint8)
        img_data[3:7, 15:35, 15:35] = 255
        fibsem_data = FIBSEMData(img_data)

        pipeline = FIBSEMPipeline()
        pipeline.data = fibsem_data

        # Define a simple rectangular initial contour
        initial_contour = np.array([[10, 10], [10, 40], [40, 40], [40, 10]])

        result = pipeline.segment_data(
            method='active_contour',
            method_type='traditional',
            initial_contour=initial_contour
        )

        self.assertTrue(result['success'])
        self.assertEqual(result['segmentation'].shape, img_data.shape)
        self.assertGreater(len(np.unique(result['segmentation'])), 1)

if __name__ == '__main__':
    unittest.main()
