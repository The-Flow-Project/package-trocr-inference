from PIL import Image
import torch


class ImageProcessor:
    def __init__(self, processor):
        self.processor = processor

    def convert_to_rgb(self, image) -> Image.Image:
        """Convert a PIL Image to RGB mode. Accepts both image paths and PIL Image objects."""
        if isinstance(image, str):
            # If image is a path, open it
            try:
                with Image.open(image) as img:
                    return img.convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to load image from path '{image}': {e}")
        elif isinstance(image, Image.Image):
            # If it's already a PIL Image, convert it
            return image.convert("RGB")
        else:
            raise ValueError("Input must be either a valid image path or a PIL Image.")

    def normalize_image(self, image):
        """Normalize the image and return tensor values using the processor."""
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        return pixel_values

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Process the image (RGB conversion, normalization)."""
        image = self.convert_to_rgb(image)  # Ensure the image is RGB
        pixel_values = self.normalize_image(image)
        return pixel_values

    def process_from_path(self, image_path: str) -> torch.Tensor:
        """Process an image from a file path."""
        with Image.open(image_path) as image:
            return self.process_image(image)

    def process_batch(self, image_paths):
        """Process a batch of images given a list of file paths."""
        batch = [self.process_from_path(path) for path in image_paths]
        return torch.stack(batch)
