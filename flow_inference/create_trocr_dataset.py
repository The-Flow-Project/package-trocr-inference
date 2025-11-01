# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from torch.utils.data import Dataset
from typing import List, Dict
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
        filename = record.get("filename") or f"line_{idx}.png"

        logger.debug(f"Fetching in-memory image: {filename} at index: {idx}")

        try:
            pixel_values = self.image_handler.handle_image(record)

            return {
                "pixel_values": pixel_values,
                "filename": filename
            }
        except ValueError as e:
            logger.error(f"Value error while processing image: {filename}. Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while processing image: {filename}. Error: {e}")
            raise