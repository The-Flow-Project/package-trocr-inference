from typing import Tuple
from PIL import Image, ImageOps
from transformers import TrOCRProcessor
import torch


class ImageHandler:
    """
    A utility class to handle image loading, resizing, and processing.
    """
    def __init__(
            self, 
            processor: TrOCRProcessor, 
            image_size: Tuple[int, int] = (384, 384), 
            do_resize: bool = False, 
            aspect_ratio_resize: bool = False
            ):
        """

        :param processor: a TrOCRProcessor instance.
        :param image_size: the size the image should have.
        :param do_resize: whether to resize. Defaults to False.
        :param aspect_ratio_resize: whether to resize (padding if necessary). Defaults to False.
        """
        self.processor = processor
        self.image_size = image_size
        self.do_resize = do_resize
        self.aspect_ratio_resize = aspect_ratio_resize

    @staticmethod
    def load_image(file_name: str) -> Image:
        """
        Load an image and convert it to RGB.

        :param file_name: The filename of the image to load.
        :return: The loaded image as PIL image.
        """
        return Image.open(file_name).convert('RGB')

    def resize_with_aspect_ratio(self, image: Image) -> Image:
        """
        Resize the image while maintaining the aspect ratio.

        :param image: The image to apply the padding to.
        :return: The padded image.
        """
        image.thumbnail(self.image_size, Image.Resampling.LANCZOS)

        # Create a new image with a white background and paste the resized image into it
        padded_image = ImageOps.pad(
            image, 
            self.image_size, 
            method=Image.Resampling.LANCZOS, 
            color=(1, 1, 1)  # white padding
        )
        return padded_image

    def process_image(self, image: Image) -> torch.Tensor:
        """
        Process an image using the provided processor (resize, normalize, etc.).

        :param: The PIL image to process.
        :return: The processed image as a torch.Tensor.
        """
        return self.processor(image, return_tensors='pt').pixel_values.squeeze()

    def handle_image(self, file_name: str) -> torch.Tensor:
        """
        Full pipeline to load, resize (optionally), and process an image.

        :param file_name: The filename of the image to be processed.
        :return: The processed image as a torch.Tensor.
        """
        image = self.load_image(file_name)

        if self.do_resize and self.aspect_ratio_resize:
            if image.size[0] < self.image_size[0] or image.size[1] < self.image_size[1]:
                image = self.resize_with_aspect_ratio(image)
        elif self.do_resize:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        return self.process_image(image)
