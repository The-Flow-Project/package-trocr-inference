# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from typing import List, Dict
from datasets import load_dataset, Split, Dataset, DatasetDict
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
        self.dataset: Optional[Dict[str, Dataset]] = None
        self.df: Optional[Dict[str, pd.DataFrame]] = None
        self.state: str = 'initialized'

    # ---------------------------------------------------------------------------
    # DOWNLOAD DATASETS
    # ---------------------------------------------------------------------------
    def download_hf_dataset(self) -> None:
        """
        Download the dataset from Hugging Face Hub using the Datasets library.
        """
        logger.info(f"Downloading all splits for dataset: {self.dataset_name}")

        try:
            hf_dataset = load_dataset(self.dataset_name,
                                      token=self.huggingface_token,
                                      data_dir="data")

            # Case 1 — dataset has splits
            if isinstance(hf_dataset, DatasetDict):
                self.dataset = dict(hf_dataset)
                self.state = "downloaded_all"

            # Case 2 — dataset has no splits
            else:
                self.dataset = {"default": hf_dataset}
                self.state = "downloaded_default"

            logger.info(f"Successfully loaded dataset with splits: {list(self.dataset.keys())}")

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
    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Convert the loaded dataset to a pandas DataFrame.
        """
        if self.dataset is None:
            logger.error("Dataset not loaded. Call download() first.")
            raise RuntimeError("Dataset not loaded. Call download() first.")

        dfs = {}
        for split_name, split_data in self.dataset.items():
            logger.info(f"Converting split '{split_name}' to DataFrame...")
            dfs[split_name] = split_data.to_pandas()
        self.state = 'converted'
        logger.info("Dataset converted to DataFrame successfully.")
        return dfs

    def convert_to_list_of_dicts(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        """
        Convert the loaded Hugging Face dataset (self.df) into a list of dictionaries.

        - Includes all columns as they exist in the dataset.
        - Does not modify or interpret any values (no image conversions).
        - Works for any dataset schema.
        """
        """
        Convert all DataFrames into list-of-dicts.
        """
        recs: Dict[str, List[Dict]] = {}
        for split_name, df in dfs.items():
            logger.info(f"Converting DataFrame for split '{split_name}' into list of dicts...")
            recs[split_name] = df.to_dict(orient="records")
        return recs

    def convert_df_into_hf_dataset(self) -> Dataset:
        """
        Convert the internal pandas DataFrame (self.df) back into a Hugging Face Dataset.
        """
        if self.df is None:
            logger.error("No DataFrame found. Cannot convert to Hugging Face Dataset.")
            raise RuntimeError("DataFrame not available. Please generate or load it first.")

        logger.info("Converting pandas DataFrame back to Hugging Face Dataset...")
        only_df = next(iter(self.df.values()))
        hf_dataset = Dataset.from_pandas(only_df)
        logger.info("Conversion successful.")
        return hf_dataset

    # ---------------------------------------------------------------------------
    # PUSH UPDATED DATASET TO HUGGING FACE HUB
    # ---------------------------------------------------------------------------
    def push_to_hub(
            self,
            upload_repo_name: str,
            private: bool = True,
            commit_message: str = "Upload updated dataset"
    ):
        """
        Upload the entire dataset (with all its splits) as a single dataset.
        This produces the correct HF Hub layout with:
          - data/train-*
          - data/test-*
          - dataset_infos.json
        """

        if self.dataset is None:
            raise RuntimeError("Dataset not loaded.")

        if self.df is None:
            raise RuntimeError("No DataFrames stored. Cannot push to hub.")

        # Build a DatasetDict
        ds_dict = DatasetDict()
        for split_name, df in self.df.items():
            ds_dict[split_name] = Dataset.from_pandas(df)

        # Now push the DatasetDict as a single dataset
        ds_dict.push_to_hub(
            repo_id=upload_repo_name,
            token=self.huggingface_token,
            private=private,
            commit_message=commit_message,
        )

        logger.info(f"Uploaded dataset with splits to HF Hub: {upload_repo_name}")
        self.state = "pushed"

    def upload_file(self, repo_name: str, target_path: str, content_bytes: bytes):
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=content_bytes,
            path_in_repo=target_path,
            repo_id=repo_name,
            repo_type="dataset",
            token=self.huggingface_token,
        )


