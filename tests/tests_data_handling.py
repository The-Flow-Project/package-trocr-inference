import os
import unittest
from unittest.mock import patch, MagicMock
from datasets.exceptions import DatasetNotFoundError, UnexpectedDownloadedFileError, UnexpectedSplitsError, \
    DatasetsError
from flow_inference.data_handling import HuggingFaceDataHandler
from dotenv import load_dotenv


class TestHuggingFaceDataHandler(unittest.TestCase):
    def setUp(self):
        self.handler = HuggingFaceDataHandler(dataset_name="my-org/my-dataset")
        load_dotenv()
        self.hf_token_read = os.getenv("HUGGINGFACE_TOKEN_READ")
        self.hf_download_repo_name = os.getenv("HUGGINGFACE_DOWNLOAD_REPO_NAME")

    # -----------------------------------------------------------------------
    # MOCK TEST HUGGING FACE DOWNLOAD
    # -----------------------------------------------------------------------

    @patch("flow_inference.data_handling.load_dataset")
    def test_download_success_with_various_splits(self, mock_load_dataset):
        """Test download() works for train, test, and no split."""
        fake_dataset = MagicMock()
        fake_dataset.column_names = ["col1", "col2"]
        mock_load_dataset.return_value = fake_dataset

        # test cases
        cases = [
            ("train", {"split": "train"}),
            ("test", {"split": "test"}),
            (None, {})  # no split provided
        ]

        for split_value, expected_kwargs in cases:
            with self.subTest(split=split_value):
                handler = HuggingFaceDataHandler("my-org/my-dataset", split=split_value)
                handler.download()

                mock_load_dataset.assert_called_once_with(
                    handler.dataset_name,
                    token=handler.huggingface_token,
                    **expected_kwargs
                )

                self.assertEqual(handler.state, "downloaded")
                self.assertIs(handler.dataset, fake_dataset)

                mock_load_dataset.reset_mock()

    @patch("flow_inference.data_handling.load_dataset")
    def test_download_failure(self, mock_load_dataset):
        """
        Ensure download() handles all known Hugging Face Datasets exceptions correctly.
        """
        error_cases = [
            (DatasetNotFoundError("Dataset not found"), DatasetNotFoundError),
            (UnexpectedDownloadedFileError("Unexpected file format"), UnexpectedDownloadedFileError),
            (UnexpectedSplitsError("Missing split"), UnexpectedSplitsError),
            (DatasetsError("General HF dataset error"), DatasetsError)
        ]

        for raised_exc, expected_exc in error_cases:
            with self.subTest(exc_type=expected_exc.__name__):
                mock_load_dataset.side_effect = raised_exc

                with self.assertRaises(expected_exc):
                    self.handler.download()

                self.assertEqual(self.handler.state, "failed")
                mock_load_dataset.reset_mock()

    @patch("flow_inference.data_handling.load_dataset")
    def test_download_success_with_token(self, mock_load_dataset):
        self.handler.huggingface_token = "hf_ABC123"
        fake_dataset = MagicMock()
        mock_load_dataset.return_value = fake_dataset

        self.handler.download()

        mock_load_dataset.assert_called_once_with(
            "my-org/my-dataset", token="hf_ABC123"
        )

        self.assertEqual(self.handler.state, "downloaded")
        self.assertIsNotNone(self.handler.dataset)

    # -----------------------------------------------------------------------
    # INTEGRATION TEST — REAL DATASET
    # -----------------------------------------------------------------------

    def test_real_hf_dataset_download_and_convert(self):
        """
        Integration test using a real dataset on Hugging Face Hub.
        Requires a valid HUGGINGFACE_TOKEN_READ in the .env file.
        Skips automatically if token is missing.
        """

        handler = HuggingFaceDataHandler(
            dataset_name=self.hf_download_repo_name,
            huggingface_token=self.hf_token_read,
            split="train"
        )

        # Step 1 Download
        handler.download()
        self.assertEqual(handler.state, "downloaded")
        self.assertIsNotNone(handler.dataset)

        print("\nState after download:", handler.state)
        print("Dataset type:", type(handler.dataset))

        # Step 2 Convert to DataFrame
        df = handler.to_dataframe()
        self.assertEqual(handler.state, "converted")
        self.assertFalse(df.empty)

        print("State after to_dataframe:", handler.state)
        print("DataFrame shape:", df.shape)
        print(df.head())

        # Step 3 Convert to list of dicts
        records = handler.convert_df_into_dict_list()
        self.assertEqual(handler.state, "ready")
        self.assertTrue(isinstance(records, list))
        self.assertTrue(len(records) > 0)

        print("State after convert_df_into_dict_list:", handler.state)
        print("Number of records:", len(records))
        print("First record:", records[0])

if __name__ == '__main__':
    unittest.main()
