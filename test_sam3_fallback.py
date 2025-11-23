
import unittest
import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from core.segmentation import segment_deep_learning

class TestSAM3Fallback(unittest.TestCase):
    def test_sam3_fallback(self):
        # Create dummy data
        data = np.zeros((10, 100, 100), dtype=np.uint8)
        data[:, 30:70, 30:70] = 200 # Create a bright square

        # Params
        params = {
            'text_prompt': 'square',
            'model_type': 'vit_h'
        }

        print("Testing SAM3 segmentation (expecting fallback)...")
        # This should log an error/info and return a watershed result, not crash
        result = segment_deep_learning(data, 'sam3', params)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, data.shape)
        # Watershed should find the object
        self.assertTrue(len(np.unique(result)) > 1)
        print("SAM3 fallback test passed.")

if __name__ == '__main__':
    unittest.main()
