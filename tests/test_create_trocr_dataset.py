import os
import unittest
from unittest.mock import MagicMock
from flow_inference.image_processing import ImageHandler
from flow_inference.create_trocr_dataset import TrOCRInferenceDataset


class TestTrOCRInferenceDataset(unittest.TestCase):

    def setUp(self):
        # Define paths to test image files
        current_dir: str = os.path.dirname(os.path.realpath(__file__))
        self.test_image_path = os.path.join(current_dir, '..', 'test_data', 'images', '1_0054.0.png')
        self.image_paths = [self.test_image_path, self.test_image_path]

        # Mock the ImageHandler and its handle_image method
        self.mock_image_handler = MagicMock(spec=ImageHandler)
        self.mock_image_handler.handle_image.return_value = [0.1, 0.2, 0.3]  # Example pixel values

        # Initialize the TrOCRInferenceDataset with mocked handler
        self.dataset = TrOCRInferenceDataset(self.image_paths, self.mock_image_handler)

    def test_len(self):
        # Check if dataset length matches number of provided image paths
        self.assertEqual(len(self.dataset), 2)

    def test_getitem(self):
        # Retrieve the first item from the dataset
        item = self.dataset[0]

        # Check if 'pixel_values' and 'file_name' keys exist in the returned item
        self.assertIn('pixel_values', item)
        self.assertIn('file_name', item)

        # Check if pixel values are correctly retrieved
        self.assertEqual(item['pixel_values'], [0.1, 0.2, 0.3])

        # Check if the file name matches the expected test image path
        self.assertEqual(item['file_name'], self.test_image_path)

        # Ensure the ImageHandler's handle_image was called with the correct file path
        self.mock_image_handler.handle_image.assert_called_with(self.test_image_path)


if __name__ == '__main__':
    unittest.main()