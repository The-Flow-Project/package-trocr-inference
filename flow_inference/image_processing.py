# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from typing import Tuple
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
    def load_image(file_name: str) -> Image:
        """
        Load an image and convert it to RGB.

        :param file_name: The filename of the image to load.
        :return: The loaded image as PIL image.
        """
        try:
            return Image.open(file_name).convert('RGB')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image file '{file_name}' not found. {str(e)}")
        except IOError as e:
            raise IOError(f"IO error occurred while loading the image '{file_name}'. {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error occurred while loading the image '{file_name}': {str(e)}")

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

    def handle_image(self, file_name: str) -> torch.Tensor:
        """
        Full pipeline to load, resize (optionally), and process an image.

        :param file_name: The filename of the image to be processed.
        :return: The processed image as a torch.Tensor.
        """
        try:
            image = self.load_image(file_name)

            if self.target_image_size:
                if image.size[0] < self.target_image_size[0] or image.size[1] < self.target_image_size[1]:
                    logger.info(f"Resizing with aspect ratio due to image size: {image.size}")
                    image = self.resize_with_aspect_ratio(image)
                else:
                    logger.info(f"Resizing image to {self.target_image_size}")
                    image = image.resize(self.target_image_size, Image.Resampling.LANCZOS)

            processed_image = self.process_image(image)

            logger.info(f"Successfully processed image: {file_name}")
            return processed_image
        except FileNotFoundError as e:
            logger.error(f"Image file not found: {file_name}. {str(e)}")
            raise
        except IOError as e:
            logger.error(f"IO error occurred while loading image: {file_name}. {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"Value error occurred while processing image: {file_name}. {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during image handling for {file_name}: {str(e)}")
            raise
