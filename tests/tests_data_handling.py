import os
import unittest
from unittest.mock import patch
from datasets import DatasetDict, Dataset
from datasets.exceptions import (
    DatasetNotFoundError,
    UnexpectedDownloadedFileError,
    UnexpectedSplitsError,
    DatasetsError,
)
from flow_inference.data_handling import HuggingFaceDataHandler
from dotenv import load_dotenv
from huggingface_hub import HfApi
import pandas as pd


class TestHuggingFaceDataHandler(unittest.TestCase):
    def setUp(self):
        self.handler = HuggingFaceDataHandler(dataset_name="my-org/my-dataset")
        load_dotenv()
        self.hf_token_read = os.getenv("HUGGINGFACE_TOKEN_READ")
        self.hf_download_repo_name = os.getenv("HUGGINGFACE_DOWNLOAD_REPO_NAME")
        self.write_token = os.getenv("HUGGINGFACE_TOKEN_READ_WRITE")
        self.test_repo = os.getenv("HUGGINGFACE_TEST_UPLOAD_REPO_NAME")

    # -----------------------------------------------------------------------
    # MOCK TEST HUGGING FACE DOWNLOAD
    # -----------------------------------------------------------------------

    @patch("flow_inference.data_handling.load_dataset")
    def test_download_success(self, mock_load_dataset):
        """download_hf_dataset() loads all splits or wraps a single-split dataset."""

        # Case 1: dataset has multiple splits → DatasetDict
        fake_splits = DatasetDict({
            "train": Dataset.from_dict({"x": [1]}),
            "test": Dataset.from_dict({"x": [2]}),
        })
        mock_load_dataset.return_value = fake_splits

        handler = HuggingFaceDataHandler("my-org/my-dataset", huggingface_token="TOKEN123")
        handler.download_hf_dataset()

        mock_load_dataset.assert_called_once_with("my-org/my-dataset", token="TOKEN123")

        self.assertEqual(handler.state, "downloaded_all")
        self.assertEqual(list(handler.dataset.keys()), ["train", "test"])

        mock_load_dataset.reset_mock()

        # Case 2: dataset has a single split → Dataset
        fake_single = Dataset.from_dict({"x": [1]})
        mock_load_dataset.return_value = fake_single

        handler = HuggingFaceDataHandler("my-org/my-dataset", huggingface_token="TOKEN123")
        handler.download_hf_dataset()

        mock_load_dataset.assert_called_once_with("my-org/my-dataset", token="TOKEN123")

        self.assertEqual(handler.state, "downloaded_default")
        self.assertIn("default", handler.dataset)

    @patch("flow_inference.data_handling.load_dataset")
    def test_download_failure(self, mock_load_dataset):
        error_cases = [
            (DatasetNotFoundError("Dataset not found"), DatasetNotFoundError),
            (UnexpectedDownloadedFileError("Unexpected file format"), UnexpectedDownloadedFileError),
            (UnexpectedSplitsError("Missing split"), UnexpectedSplitsError),
            (DatasetsError("General HF dataset error"), DatasetsError),
        ]

        for raised_exc, expected_exc in error_cases:
            with self.subTest(exc_type=expected_exc.__name__):
                mock_load_dataset.side_effect = raised_exc

                with self.assertRaises(expected_exc):
                    self.handler.download_hf_dataset()

                self.assertEqual(self.handler.state, "failed")
                mock_load_dataset.reset_mock()

    @patch("flow_inference.data_handling.load_dataset")
    def test_download_success_with_token(self, mock_load_dataset):
        self.handler.huggingface_token = "hf_ABC123"

        fake_dataset = DatasetDict({
            "train": Dataset.from_dict({"x": [1]}),
            "test": Dataset.from_dict({"x": [2]}),
        })
        mock_load_dataset.return_value = fake_dataset

        self.handler.download_hf_dataset()

        mock_load_dataset.assert_called_once_with(
            "my-org/my-dataset", token="hf_ABC123"
        )

        self.assertEqual(self.handler.state, "downloaded_all")
        self.assertIsInstance(self.handler.dataset, dict)
        self.assertIn("train", self.handler.dataset)
        self.assertIn("test", self.handler.dataset)
        self.assertIsInstance(self.handler.dataset["train"], Dataset)
        self.assertIsInstance(self.handler.dataset["test"], Dataset)

    # -----------------------------------------------------------------------
    # INTEGRATION TEST — REAL DATASET
    # -----------------------------------------------------------------------

    def test_real_hf_dataset_download_and_convert(self):
        if not self.hf_token_read or not self.hf_download_repo_name:
            self.skipTest("Missing Hugging Face credentials.")

        handler = HuggingFaceDataHandler(
            dataset_name=self.hf_download_repo_name,
            huggingface_token=self.hf_token_read,
        )

        handler.download_hf_dataset()
        self.assertIn(handler.state, {"downloaded_all", "downloaded_default"})
        self.assertIsNotNone(handler.dataset)

        dfs = handler.to_dataframe()
        self.assertGreater(len(dfs), 0)

        for df in dfs.values():
            self.assertFalse(df.empty)

        dicts = handler.convert_to_list_of_dicts(dfs)
        self.assertGreater(len(dicts), 0)

    # -----------------------------------------------------------------------
    # INTEGRATION TEST — PUSH TO HUB
    # -----------------------------------------------------------------------

    def test_push_to_hub(self):
        if not self.write_token or not self.test_repo:
            self.skipTest("Missing Hugging Face WRITE token or test repo name.")

        api = HfApi()
        api.create_repo(
            repo_id=self.test_repo,
            private=True,
            exist_ok=True,
            token=self.write_token,
        )

        # Multi-split DataFrames
        df_train = pd.DataFrame({"filename": ["a"], "text": ["hello"]})
        df_test = pd.DataFrame({"filename": ["c"], "text": ["test"]})

        handler = HuggingFaceDataHandler(
            dataset_name=self.test_repo,
            huggingface_token=self.write_token,
        )

        # REQUIRED: dataset must be a DatasetDict now
        handler.dataset = DatasetDict({
            "train": Dataset.from_pandas(df_train),
            "test": Dataset.from_pandas(df_test),
        })

        handler.df = {
            "train": df_train,
            "test": df_test,
        }

        handler.push_to_hub(
            upload_repo_name=self.test_repo,
            private=True,
            commit_message="Unit test upload"
        )

        self.assertEqual(handler.state, "pushed")

        files = api.list_repo_files(
            repo_id=self.test_repo,
            repo_type="dataset",
            token=self.write_token
        )

        files_lower = [f.lower() for f in files]

        self.assertTrue(any("train" in f for f in files_lower))
        self.assertTrue(any("test" in f for f in files_lower))


if __name__ == '__main__':
    unittest.main()
