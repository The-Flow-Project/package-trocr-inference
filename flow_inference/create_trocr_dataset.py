# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from torch.utils.data import Dataset
from typing import List
from flow_inference.image_processing import ImageHandler
from flow_inference.utils.logging.inference_logger import logger


# ===============================================================================
# CLASS
# ===============================================================================
class TrOCRInferenceDataset(Dataset):
    """
    Dataset class for TrOCR inference.
    """

    def __init__(self, file_names: List[str], image_handler: ImageHandler):
        """
        :param file_names: List of image file paths.
        :param image_handler: An instance of ImageHandler for processing images.
        """
        self.file_names = file_names
        self.image_handler = image_handler

    def __len__(self) -> int:
        """
        Get the size of the dataset.
        :return: Number of files to be processed.
        """
        return len(self.file_names)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a processed image and its file name.
        :param idx: Index of the image to retrieve.
        """
        file_name = self.file_names[idx]
        logger.debug(f"Fetching image: {file_name} at index: {idx}")
        try:
            pixel_values = self.image_handler.handle_image(file_name)
            encoding = {'pixel_values': pixel_values, 'file_name': file_name}
            logger.debug(f"Successfully processed image: {file_name}")
            return encoding
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_name}. Error: {e}")
            raise
        except ValueError as e:
            logger.error(f"Value error while processing image: {file_name}. Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while processing image: {file_name}. Error: {e}")
            raise
