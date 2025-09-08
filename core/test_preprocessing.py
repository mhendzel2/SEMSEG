import unittest
import numpy as np
from .preprocessing import remove_noise

class TestPreprocessing(unittest.TestCase):

    def test_remove_noise_nl_means(self):
        """Test the Non-Local Means noise removal method."""
        # Create a sample image with noise
        img = np.zeros((10, 20, 20), dtype=np.uint8)
        img[5, 10, 10] = 255
        noisy_img = img + np.random.randint(0, 20, size=img.shape, dtype=np.uint8)

        # Apply NL-Means denoising
        denoised_img = remove_noise(noisy_img, method='nl_means')

        # Check that the output has the same shape and type
        self.assertEqual(denoised_img.shape, noisy_img.shape)
        self.assertEqual(denoised_img.dtype, noisy_img.dtype)

        # Check that the noise has been reduced (standard deviation should be lower)
        self.assertLess(np.std(denoised_img), np.std(noisy_img))

if __name__ == '__main__':
    unittest.main()
