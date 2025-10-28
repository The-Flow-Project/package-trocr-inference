# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from torch.utils.data import Dataset
from typing import List, Dict
from PIL import Image
from flow_inference.image_processing import ImageHandler
from flow_inference.utils.logging.inference_logger import logger


# ===============================================================================
# CLASS
# ===============================================================================
class TrOCRInferenceDataset(Dataset):
    """
    Dataset class for TrOCR inference using Hugging Face records
    (in-memory PIL images) instead of file paths.
    """

    def __init__(self, records: List[Dict], image_handler: ImageHandler):
        """
        :param records: List of dicts, each containing:
                        {'image': PIL.Image, 'filename': str, ...}
        :param image_handler: An instance of ImageHandler for processing images.
        """
        self.records = records
        self.image_handler = image_handler

    def __len__(self) -> int:
        """
        Get the size of the dataset.
        :return: Number of records to be processed.
        """
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve and process an in-memory image and its filename.
        :param idx: Index of the image to retrieve.
        """
        record = self.records[idx]
        file_name = record.get("filename") or f"line_{idx}.png"
        image = record.get("image")

        logger.debug(f"Fetching in-memory image: {file_name} at index: {idx}")

        if image is None or not isinstance(image, Image.Image):
            logger.error(f"Invalid or missing image at index {idx} ({file_name})")
            raise ValueError(f"Invalid or missing image at index {idx} ({file_name})")

        try:
            pixel_values = self.image_handler.process_image(image)
            encoding = {'pixel_values': pixel_values, 'file_name': file_name}
            logger.debug(f"Successfully processed image: {file_name}")
            return encoding
        except ValueError as e:
            logger.error(f"Value error while processing image: {file_name}. Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while processing image: {file_name}. Error: {e}")
            raise