"""Load, normalize, resize, and process images for TrOCR inference."""
# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
import io
from typing import Tuple, Dict, Any
from PIL import Image, ImageOps
from transformers import TrOCRProcessor
import torch
from flow_inference.utils.logging.inference_logger import logger


# ===============================================================================
# CLASS
# ===============================================================================
class ImageHandler:
    """Prepare document line images for TrOCR inference.

    The handler converts raw image inputs to RGB PIL images, optionally resizes
    them to a configured target size, and applies a TrOCR processor to produce
    pixel tensors for model inference.
    """

    def __init__(
            self, 
            processor: TrOCRProcessor,
            target_image_size: Tuple[int, int] = None
            ):
        """Initialize the image handler.

        Args:
            processor: TrOCR processor used to convert images into model tensors.
            target_image_size: Optional target image size as ``(width, height)``.
        """
        self.processor = processor
        self.target_image_size = target_image_size

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> Image:
        """Load an RGB image from raw bytes.

        Args:
            image_bytes: Encoded image bytes.

        Returns:
            PIL image converted to RGB mode.

        Raises:
            OSError: If the bytes cannot be decoded as an image.
        """
        try:
            return Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            raise IOError(f"Failed to load image from bytes: {e}")

    def resize_with_aspect_ratio(self, image: Image) -> Image:
        """Resize an image to the target size while preserving aspect ratio.

        The resized image is padded to ``target_image_size`` so that the output
        has exactly the configured dimensions.

        Args:
            image: PIL image to resize and pad.

        Returns:
            Resized and padded RGB image.

        Raises:
            ValueError: If no target image size is configured or resizing fails.
        """
        try:
            image.thumbnail(self.target_image_size, Image.Resampling.LANCZOS)

            padded_image = ImageOps.pad(
                image,
                self.target_image_size,
                method=Image.Resampling.LANCZOS,
                color=(1, 1, 1)  # white padding
            )
            return padded_image
        except ValueError as e:
            raise ValueError(f"Invalid value encountered during resizing: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during resizing: {str(e)}")

    def process_image(self, image: Image) -> torch.Tensor:
        """Convert a PIL image into a TrOCR pixel tensor.

        Args:
            image: PIL image to process.

        Returns:
            Pixel tensor expected by the TrOCR model.

        Raises:
            ValueError: If the processor cannot process the image.
        """
        try:
            return self.processor(image, return_tensors='pt').pixel_values.squeeze()
        except ValueError as e:
            raise ValueError(f"Error processing image: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during image processing: {str(e)}")

    def handle_image(self, record: Dict[str, Any]) -> torch.Tensor:
        """Extract, normalize, resize, and process an image record.

        Args:
            record: Dataset record containing an ``image`` field and optional
                metadata such as ``filename`` and ``line_id``.

        Returns:
            Pixel tensor ready for TrOCR inference.

        Raises:
            ValueError: If the record does not contain an image field.
            TypeError: If the image value has an unsupported type.
        """
        filename = record.get("filename", "<unknown>")

        try:
            # Step 1: extract and normalize image data
            image_data = record.get("image")
            if image_data is None:
                raise ValueError("Record does not contain an 'image' field.")

            # Unwrap Hugging Face Image objects
            if isinstance(image_data, dict) and "bytes" in image_data:
                image_data = image_data["bytes"]

            # Convert to PIL
            if isinstance(image_data, bytes):
                image = self.load_image_from_bytes(image_data)
            elif isinstance(image_data, Image.Image):
                image = image_data.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(image_data)}")

            # Step 2: optional resizing
            if self.target_image_size:
                if image.size[0] < self.target_image_size[0] or image.size[1] < self.target_image_size[1]:
                    logger.debug(f"Resizing with padding for {filename} (original size {image.size})")
                    image = self.resize_with_aspect_ratio(image)
                else:
                    logger.debug(f"Resizing image {filename} to {self.target_image_size}")
                    image = image.resize(self.target_image_size, Image.Resampling.LANCZOS)

            # Step 3: process through the processor
            processed_image = self.process_image(image)
            logger.info(f"Successfully processed image: {filename}")
            return processed_image

        except Exception as e:
            logger.error(f"Error processing record {filename}: {e}")
            raise
