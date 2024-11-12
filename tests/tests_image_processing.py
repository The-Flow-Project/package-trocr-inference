import unittest
import torch
import os
from PIL import Image
from transformers import TrOCRProcessor
from flow_inference.image_processing import ImageProcessor


class TrOCRProcessorHandler:
    def __init__(self, model_name):
        self.processor = TrOCRProcessor.from_pretrained(model_name)

    def __call__(self, image, return_tensors="pt"):
        return self.processor(image, return_tensors=return_tensors)


class TestImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        processor = TrOCRProcessorHandler('microsoft/trocr-base-handwritten')
        cls.image_processor = ImageProcessor(processor)
        current_dir: str = os.path.dirname(os.path.realpath(__file__))
        cls.test_image_path = os.path.join(current_dir, '..', 'test_data', 'images', '1_0054.0.png')
        cls.invalid_image_path = os.path.join(current_dir, '..', 'test_data', 'images', 'invalid.jpg')

    def test_convert_to_rgb_valid(self):
        """Test converting a loaded image to RGB mode."""
        rgb_image = self.image_processor.convert_to_rgb(self.test_image_path)
        self.assertEqual(rgb_image.mode, "RGB")

    def test_convert_to_rgb_invalid(self):
        """Test converting a non-image input raises ValueError."""
        with self.assertRaises(ValueError):
            self.image_processor.convert_to_rgb("not_an_image")

    def test_normalize_image(self):
        """Test normalizing an image returns expected tensor shape."""
        image = self.image_processor.convert_to_rgb(self.test_image_path)
        pixel_values = self.image_processor.normalize_image(image)
        self.assertIsInstance(pixel_values, torch.Tensor)
        self.assertEqual(pixel_values.shape[1:], (3, 384, 384))  # Shape may vary based on processor

    def test_process_image(self):
        """Test full processing pipeline on a valid image."""
        with Image.open(self.test_image_path) as image:
            pixel_values = self.image_processor.process_image(image)
            self.assertIsInstance(pixel_values, torch.Tensor)

    def test_process_from_path(self):
        """Test end-to-end processing from an image path."""
        pixel_values = self.image_processor.process_from_path(self.test_image_path)
        self.assertIsInstance(pixel_values, torch.Tensor)

    def test_process_batch(self):
        """Test batch processing of multiple images."""
        pixel_values_batch = self.image_processor.process_batch([self.test_image_path, self.test_image_path])
        self.assertIsInstance(pixel_values_batch, torch.Tensor)
        self.assertEqual(pixel_values_batch.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
