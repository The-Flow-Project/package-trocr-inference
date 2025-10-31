# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from typing import List, Dict
from datasets import load_dataset, Split
import pandas as pd
from datasets.exceptions import *
from flow_inference.utils.logging.inference_logger import logger


# ===============================================================================
# CLASS
# ===============================================================================
class HuggingFaceDataHandler:
    """
    Download and convert Hugging Face datasets
    """

    def __init__(self,
                 dataset_name: str,
                 huggingface_token: str | None = None,
                 split: str | Split | None = None):
        """
        Initialize the dataset loader.

        :param dataset_name: e.g., "my-org/my-preprocessed-dataset"
        :param huggingface_token: Hugging Face authentication token (for private datasets)
        """
        self.dataset_name = dataset_name
        self.huggingface_token = huggingface_token
        self.split = split
        self.dataset: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self.state: str = 'initialized'

    # ---------------------------------------------------------------------------
    # DOWNLOAD DATASETS
    # ---------------------------------------------------------------------------
    def download(self) -> None:
        """
        Download the dataset from Hugging Face Hub using the Datasets library.
        """
        try:
            logger.info(f"Downloading dataset: {self.dataset_name} (split={self.split})")

            if self.split:
                # dataset has explicit split
                self.dataset = load_dataset(
                    self.dataset_name, split=self.split, token=self.huggingface_token
                )
            else:
                # load full dataset (no split argument)
                self.dataset = load_dataset(self.dataset_name, token=self.huggingface_token)

            self.state = "downloaded"
            logger.info(f"Successfully loaded dataset: {self.dataset_name}")
        except DatasetNotFoundError as e:
            self.state = "failed"
            logger.error(f"Dataset not found: '{self.dataset_name}'. Check spelling or visibility/access rights.")
            raise
        except UnexpectedDownloadedFileError as e:
            self.state = "failed"
            logger.error(f"Some of the downloaded files did not match the requirements.")
            raise
        except UnexpectedSplitsError as e:
            self.state = "failed"
            logger.error(f"The expected split of the downloaded files is missing.")
            raise
        except DatasetsError as e:
            self.state = "failed"
            logger.error(f"Something else went wrong when trying to access the dataset on Hugging Face.")
            raise

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

    def convert_df_into_dict_list(self) -> List[Dict[str, object]]:
        """
        Convert the loaded Hugging Face dataset (self.df) into a list of dictionaries.

        - Includes all columns as they exist in the dataset.
        - Does not modify or interpret any values (no image conversions).
        - Works for any dataset schema.
        """
        if self.df is None:
            self.to_dataframe()

        logger.info("Converting DataFrame rows into dictionaries...")
        records = [row.to_dict() for _, row in self.df.iterrows()]

        logger.info(f"Converted {len(records)} records with {len(self.df.columns)} columns.")
        self.state = "ready"
        return records


