import io
import unittest
import torch
import os
from PIL import Image, UnidentifiedImageError
from transformers import TrOCRProcessor
from flow_inference.image_processing import ImageHandler


class TestImageHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        cls.image_handler = ImageHandler(
            processor=cls.processor,
            target_image_size=(384, 384)
        )

        # Create a simple RGB test image
        cls.sample_image = Image.new("RGB", (512, 512), color="red")

        # Convert it to bytes (simulate Hugging Face dataset format)
        with io.BytesIO() as buffer:
            cls.sample_image.save(buffer, format="PNG")
            cls.image_bytes = buffer.getvalue()

        # Create record dict similar to dataset row
        cls.sample_record = {
            "Image": cls.image_bytes,
            "filename": "test_image.png",
            "line_id": "L0001"
        }

    def test_load_image_valid(self):
        """Test that a valid image is loaded and converted to RGB."""
        image = self.image_handler.load_image_from_bytes(self.image_bytes)
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")

    def test_load_image_invalid(self):
        """Test that loading an invalid image raises an exception."""
        invalid_bytes = b"not_a_real_image"
        with self.assertRaises(IOError):
            self.image_handler.load_image_from_bytes(invalid_bytes)

    def test_resize_with_aspect_ratio(self):
        """Test that the image is resized while preserving aspect ratio."""
        # Use an image with different dimensions to check resizing
        large_image = Image.new('RGB', (800, 600), color='blue')
        handler_with_aspect_ratio = ImageHandler(
            processor=self.image_handler.processor,
            target_image_size=(384, 384)
        )
        resized_image = handler_with_aspect_ratio.resize_with_aspect_ratio(large_image)
        self.assertEqual(resized_image.size, (384, 384))

    def test_process_image(self):
        """Test processing a valid image into a torch.Tensor."""
        tensor = self.image_handler.process_image(self.sample_image)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([3, 384, 384]))

    def test_handle_image_full_pipeline(self):
        """Test the full pipeline: load, resize, and process an image."""
        tensor = self.image_handler.handle_image(self.sample_record)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([3, 384, 384]))

    def test_handle_image_no_resizing(self):
        """Test the pipeline when resizing is disabled by setting `target_image_size=None`."""
        handler_no_resize = ImageHandler(processor=self.processor, target_image_size=None)
        tensor = handler_no_resize.handle_image(self.sample_record)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dim(), 3)
        self.assertEqual(tensor.size(0), 3)

    def test_process_batch(self):
        """Test processing a batch of images."""
        records = [self.sample_record, self.sample_record]
        tensors = [self.image_handler.handle_image(r) for r in records]
        batch = torch.stack(tensors)
        self.assertEqual(batch.shape, torch.Size([2, 3, 384, 384]))

if __name__ == "__main__":
    unittest.main()
