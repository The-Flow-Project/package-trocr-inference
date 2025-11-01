import os
import unittest
import pandas as pd
from flow_inference.inference import Inference
from dotenv import load_dotenv


class TestInference(unittest.TestCase):

    def setUp(self):
        """Set up an Inference instance configured for a small HF dataset."""
        load_dotenv()
        self.hf_repo_name = os.getenv("HUGGINGFACE_DOWNLOAD_REPO_NAME")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN_READ")


        self.inference = Inference(
            hf_repo_name=self.hf_repo_name,
            hf_token=self.hf_token,
            trocr_model="microsoft/trocr-small-handwritten",
            stop_on_fail=False,
        )

    def test_perform_inference_returns_dataframe(self):
        """Run the full inference pipeline and ensure it returns a valid DataFrame."""
        from flow_inference.data_handling import HuggingFaceDataHandler

        # Temporarily limit dataset to 3 records for faster testing
        original_convert = HuggingFaceDataHandler.convert_df_into_dict_list
        HuggingFaceDataHandler.convert_df_into_dict_list = (
            lambda self: original_convert(self)[:3]
        )

        try:
            result_df = self.inference.perform_inference()
        finally:
            HuggingFaceDataHandler.convert_df_into_dict_list = original_convert

        # Check that a DataFrame is returned
        self.assertIsInstance(result_df, pd.DataFrame, "Expected result to be a pandas DataFrame")

        # Check it has some data
        self.assertFalse(result_df.empty, "Resulting DataFrame is empty")

        # Check that at least one inference column was added
        inference_cols = [col for col in result_df.columns if col.startswith("inference_")]
        self.assertTrue(len(inference_cols) > 0, "No inference column found in DataFrame")

        # Check that the inference column has some non-null entries
        inferred_values = result_df[inference_cols[0]].dropna()
        self.assertTrue(len(inferred_values) > 0, "Inference column is empty")
    def test_run_inference_returns_dict(self):
        """Ensure run_inference() produces a dictionary of results."""
        from flow_inference.data_handling import HuggingFaceDataHandler

        handler = HuggingFaceDataHandler(
            dataset_name=self.hf_repo_name,
            huggingface_token=self.hf_token,
            split="train"
        )
        handler.download()
        handler.to_dataframe()
        records = handler.convert_df_into_dict_list()[:2]

        self.inference.statusManager.initialize_status(len(records))

        result_dict = self.inference.run_inference(records)

        self.assertIsInstance(result_dict, dict, "Expected inference result to be a dictionary")
        self.assertTrue(len(result_dict) > 0, "Inference result dictionary is empty")


if __name__ == "__main__":
    unittest.main()
