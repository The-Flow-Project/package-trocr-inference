import unittest
import torch
import os
from PIL import Image, UnidentifiedImageError
from transformers import TrOCRProcessor
from flow_inference.image_processing import ImageHandler


class TestImageHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        cls.image_handler = ImageHandler(
            processor=processor,
            image_size=(384, 384),
            do_resize=True,
            aspect_ratio_resize=False
        )
        current_dir: str = os.path.dirname(os.path.realpath(__file__))
        cls.test_image_path = os.path.join(current_dir, '..', 'test_data', 'images', '1_0054.0.png')
        cls.invalid_image_path = os.path.join(current_dir, '..', 'test_data', 'images', 'invalid.jpg')

    def test_load_image_valid(self):
        """Test that a valid image is loaded and converted to RGB."""
        image = self.image_handler.load_image(self.test_image_path)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")

    def test_load_image_invalid(self):
        """Test that loading an invalid image raises an exception."""
        with self.assertRaises((OSError, UnidentifiedImageError)):
            self.image_handler.load_image(self.invalid_image_path)

    def test_resize_with_aspect_ratio(self):
        """Test that the image is resized while preserving aspect ratio."""
        # Use an image with different dimensions to check resizing
        large_image = Image.new('RGB', (800, 600), color='blue')
        handler_with_aspect_ratio = ImageHandler(
            processor=self.image_handler.processor,
            image_size=(384, 384),
            do_resize=True,
            aspect_ratio_resize=True
        )
        resized_image = handler_with_aspect_ratio.resize_with_aspect_ratio(large_image)
        self.assertEqual(resized_image.size, (384, 384))

    def test_process_image(self):
        """Test processing a valid image into a torch.Tensor."""
        with Image.open(self.test_image_path) as image:
            pixel_values = self.image_handler.process_image(image)
            self.assertIsInstance(pixel_values, torch.Tensor)
            self.assertEqual(pixel_values.shape, torch.Size([3, 384, 384]))

    def test_handle_image_full_pipeline(self):
        """Test the full pipeline: load, resize, and process an image."""
        pixel_values = self.image_handler.handle_image(self.test_image_path)
        self.assertIsInstance(pixel_values, torch.Tensor)
        self.assertEqual(pixel_values.shape, torch.Size([3, 384, 384]))

    def test_handle_image_with_aspect_ratio(self):
        """Test the pipeline when aspect ratio resizing is enabled."""
        handler_with_aspect_ratio = ImageHandler(
            processor=self.image_handler.processor,
            image_size=(384, 384),
            do_resize=True,
            aspect_ratio_resize=True
        )
        pixel_values = handler_with_aspect_ratio.handle_image(self.test_image_path)
        self.assertIsInstance(pixel_values, torch.Tensor)
        self.assertEqual(pixel_values.shape, torch.Size([3, 384, 384]))

    def test_process_batch(self):
        """Test processing a batch of images."""
        image_paths = [self.test_image_path, self.test_image_path]
        pixel_values_list = []

        for path in image_paths:
            pixel_values = self.image_handler.handle_image(path)
            pixel_values_list.append(pixel_values)

        batch_tensor = torch.stack(pixel_values_list)
        self.assertIsInstance(batch_tensor, torch.Tensor)
        self.assertEqual(batch_tensor.shape, torch.Size([2, 3, 384, 384]))


if __name__ == "__main__":
    unittest.main()
