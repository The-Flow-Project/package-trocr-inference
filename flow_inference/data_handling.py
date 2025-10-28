# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from typing import Optional, List, Dict, Union
from datasets import load_dataset
from PIL import Image
import pandas as pd
import io
from flow_inference.utils.logging.inference_logger import logger


# ===============================================================================
# CLASS
# ===============================================================================
class HuggingFaceDataHandler:
    """
    Lightweight loader that mirrors HuggingFacePreprocessor's dataset fetching
    behavior, but skips XML parsing and conversion steps.
    """

    def __init__(self,
                 dataset_id: str,
                 huggingface_token: Optional[str] = None):
        """
        Initialize the loader.

        :param dataset_id: e.g., "my-org/my-preprocessed-dataset"
        :param huggingface_token: Hugging Face authentication token (for private datasets)
        """
        self.dataset_id = dataset_id
        self.huggingface_token = huggingface_token
        self.dataset: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self.state: str = 'initialized'

    # ---------------------------------------------------------------------------
    # DOWNLOAD
    # ---------------------------------------------------------------------------
    def download(self) -> None:
        """
        Download the dataset from Hugging Face Hub using the datasets library.
        """
        try:
            logger.info(f"Downloading dataset from Hugging Face: {self.dataset_id}")
            self.dataset = load_dataset(self.dataset_id, split="train", token=self.huggingface_token)
            self.state = 'downloaded'
            logger.info(f"Successfully loaded dataset: {self.dataset_id}")
            logger.info(f"Columns: {self.dataset.column_names}")
        except Exception as e:
            self.state = 'failed'
            logger.error(f"Failed to download dataset {self.dataset_id}: {e}")
            raise e

    # ---------------------------------------------------------------------------
    # CONVERSION
    # ---------------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the loaded dataset to a pandas DataFrame.
        """
        if self.dataset is None:
            logger.error("Dataset not loaded. Call download() first.")
            raise RuntimeError("Dataset not loaded. Call download() first.")

        logger.info("Converting Hugging Face dataset to pandas DataFrame...")
        self.df = self.dataset.to_pandas()
        self.state = 'converted'
        logger.info("Dataset converted to DataFrame successfully.")
        return self.df

    # ---------------------------------------------------------------------------
    # IMAGE EXTRACTION
    # ---------------------------------------------------------------------------
    def get_image_records(self) -> List[Dict[str, Union[Image.Image, str]]]:
        """
        Extract images and metadata from the dataset for inference.

        :return: A list of dicts with keys: image, xml, filename, project
        """
        if self.df is None:
            self.to_dataframe()

        logger.info("Extracting images and metadata for inference...")
        records = []

        for _, row in self.df.iterrows():
            img = row.get("image")
            if isinstance(img, (bytes, bytearray)):
                img = Image.open(io.BytesIO(img)).convert("RGB")

            records.append({
                "image": img,
                "xml": row.get("xml"),
                "filename": row.get("filename"),
                "project": row.get("project"),
            })

        logger.info(f"Extracted {len(records)} image records for inference.")
        self.state = 'ready'
        return records
