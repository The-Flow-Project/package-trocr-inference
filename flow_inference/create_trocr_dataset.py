"""Provide a PyTorch dataset wrapper for TrOCR inference records."""

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
    """Represent Hugging Face records as a PyTorch dataset for TrOCR inference.

    The dataset receives in-memory records, processes each record's image through
    an ``ImageHandler``, and returns pixel tensors together with line-level
    metadata required to map predictions back to their source document.
    """

    def __init__(self, records: List[Dict], image_handler: ImageHandler):
        """Initialize the inference dataset.

        Args:
            records: Hugging Face-style records containing image data and metadata.
            image_handler: Image handler used to normalize and process record images.
        """
        self.records = records
        self.image_handler = image_handler

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        """Process and return one inference item.

        Args:
            idx: Index of the record to retrieve.

        Returns:
            Dictionary containing processed pixel values and source metadata.

        Raises:
            ValueError: If the image record cannot be processed.
            Exception: If an unexpected image-processing error occurs.
        """
        record = self.records[idx]
        filename = record.get("filename")
        region_id = record.get("region_id")
        line_id = record.get("line_id")
        project_name = record.get("project_name", "")

        logger.debug(f"Fetching in-memory image: {filename} at index: {idx}")

        try:
            pixel_values = self.image_handler.handle_image(record)

            return {
                "pixel_values": pixel_values,
                "filename": filename,
                "region_id": region_id,
                "line_id": line_id,
                "project_name": project_name,
            }
        except ValueError as e:
            logger.error(f"Value error while processing image: {filename}. Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while processing image: {filename}. Error: {e}")
            raise