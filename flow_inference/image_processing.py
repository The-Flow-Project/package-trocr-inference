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
    """
    A utility class to handle image loading, resizing, and processing.
    """
    def __init__(
            self, 
            processor: TrOCRProcessor,
            target_image_size: Tuple[int, int] = None
            ):
        """

        :param processor: a TrOCRProcessor instance.
        :param target_image_size: the size the image should have.
        """
        self.processor = processor
        self.target_image_size = target_image_size

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> Image:
        """
        Load an image from raw bytes and convert it to RGB.
        """
        try:
            return Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            raise IOError(f"Failed to load image from bytes: {e}")

    def resize_with_aspect_ratio(self, image: Image) -> Image:
        """
        Resize the image while maintaining the aspect ratio.

        :param image: The image to apply the padding to.
        :return: The padded image.
        """
        try:
            image.thumbnail(self.target_image_size, Image.Resampling.LANCZOS)

            # Create a new image with a white background and paste the resized image into it
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
        """
        Process an image using the provided processor (resize, normalize, etc.).

        :param: The PIL image to process.
        :return: The processed image as a torch.Tensor.
        """
        try:
            return self.processor(image, return_tensors='pt').pixel_values.squeeze()
        except ValueError as e:
            raise ValueError(f"Error processing image: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during image processing: {str(e)}")

    def handle_image(self, record: Dict[str, Any]) -> torch.Tensor:
        """
        Process a single record containing image bytes and metadata.

        Expected record format examples:
        {
            "image": bytes | {"bytes": bytes, "path": None} | PIL.Image,
            "filename": str,
            "line_id": str,
            ...
        }
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
