import os
import unittest
import pandas as pd
from flow_inference.inference import Inference
from dotenv import load_dotenv

from flow_inference.model_handling import ModelManager


class TestInference(unittest.TestCase):

    def setUp(self):
        """Set up an Inference instance configured for a small HF dataset."""
        load_dotenv()
        self.download_repo_name = os.getenv("HUGGINGFACE_DOWNLOAD_REPO_NAME")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN_READ")
        self.write_token = os.getenv("HUGGINGFACE_TOKEN_READ_WRITE")
        self.test_repo = os.getenv("HUGGINGFACE_TEST_UPLOAD_REPO_NAME")

        self.inference = Inference(
            download_repo_name=self.download_repo_name,
            hf_token=self.write_token,
            trocr_model="microsoft/trocr-small-handwritten",
            stop_on_fail=False,
            push_to_hub=False
        )

    def test_perform_inference_returns_dataframe(self):
        """Run the full inference pipeline and ensure it returns a dict of DataFrames."""
        from flow_inference.data_handling import HuggingFaceDataHandler

        # limit dataset to 3 per split
        original_convert = HuggingFaceDataHandler.convert_to_list_of_dicts

        def limited_convert(self, dfs):
            full = original_convert(self, dfs)
            return {split: recs[:3] for split, recs in full.items()}

        HuggingFaceDataHandler.convert_to_list_of_dicts = limited_convert

        try:
            result = self.inference.perform_inference()
        finally:
            HuggingFaceDataHandler.convert_to_list_of_dicts = original_convert

        # must return a dict
        self.assertIsInstance(result, dict, "Expected dict of DataFrames")
        self.assertGreater(len(result), 0, "No splits returned")

        for split, df in result.items():
            with self.subTest(split=split):

                self.assertIsInstance(df, pd.DataFrame)
                self.assertFalse(df.empty, "Returned DataFrame is empty")

                inference_cols = [c for c in df.columns if c.startswith("inference_")]
                self.assertGreater(len(inference_cols), 0, "No inference column found")
                inference_col = inference_cols[0]

                if split in self.inference.requested_splits or split == "default":
                    inferred = df[df[inference_col] != ""]

                    # at least one inferred line
                    self.assertGreater(
                        len(inferred),
                        0,
                        f"No inference results written for split '{split}'"
                    )

                    # ensure no conflicting inference per (filename, line_id)
                    for (filename, line_id), group in inferred.groupby(["filename", "line_id"]):
                        unique_vals = group[inference_col].unique()
                        self.assertEqual(
                            len(unique_vals),
                            1,
                            f"Multiple inference values for ({filename}, {line_id}) in split '{split}'"
                        )

                # non-requested split: must contain ONLY empty strings
                else:
                    unique_vals = set(df[inference_col].unique())
                    self.assertEqual(
                        unique_vals,
                        {""},
                        f"Non-requested split '{split}' should have only empty strings"
                    )

    def test_run_inference_returns_dict(self):
        """Ensure run_inference() produces a dictionary of results."""
        from flow_inference.data_handling import HuggingFaceDataHandler

        handler = HuggingFaceDataHandler(
            dataset_name=self.download_repo_name,
            huggingface_token=self.hf_token
        )

        handler.download_hf_dataset()
        dfs = handler.to_dataframe()
        records_dict = handler.convert_to_list_of_dicts(dfs)

        # pick a split (train preferred)
        if "train" in records_dict:
            records = records_dict["train"][:2]
        else:
            records = next(iter(records_dict.values()))[:2]

        self.inference.statusManager.initialize_status(len(records))

        model_manager = ModelManager()
        processor = model_manager.load_processor(self.inference.trocr_model)
        model = model_manager.load_model(self.inference.trocr_model)

        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)

        result_dict = self.inference.run_inference(
            records=records,
            model=model,
            processor=processor,
            device=model_manager.device
        )

        self.assertIsInstance(result_dict, dict, "Expected inference result to be a dictionary")
        self.assertGreater(len(result_dict), 0, "Inference result dictionary is empty")
        for key, value in result_dict.items():
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)

            filename, line_id = key

            self.assertIsInstance(filename, str)
            self.assertIsInstance(line_id, str)
            self.assertIsInstance(value, str)

            valid_keys = {(r["filename"], r["line_id"]) for r in records}
            self.assertIn(
                (filename, line_id),
                valid_keys,
                "Inference result key is not a (filename, line_id) pair from input records"
            )

    def test_full_inference_with_upload(self):
        """
        Full integration test:
        - downloads dataset
        - runs inference
        - writes results back into dataframe
        - pushes updated dataset to HF Hub
        """

        if not self.write_token or not self.test_repo:
            self.skipTest("Missing WRITE token or upload repo name.")

        from huggingface_hub import HfApi
        api = HfApi()

        # Ensure repo exists
        api.create_repo(
            repo_id=self.test_repo,
            token=self.write_token,
            exist_ok=True,
            repo_type="dataset",
            private=True,
        )

        # limit dataset size
        from flow_inference.data_handling import HuggingFaceDataHandler
        original_convert = HuggingFaceDataHandler.convert_to_list_of_dicts

        def limited_convert(self, dfs):
            full = original_convert(self, dfs)
            return {split: recs[:3] for split, recs in full.items()}

        HuggingFaceDataHandler.convert_to_list_of_dicts = limited_convert

        try:
            inference = Inference(
                download_repo_name=self.download_repo_name,
                hf_token=self.write_token,
                trocr_model="microsoft/trocr-small-handwritten",
                stop_on_fail=False,
                push_to_hub=True,
                upload_repo_name=self.test_repo
            )

            result = inference.perform_inference()

        finally:
            HuggingFaceDataHandler.convert_to_list_of_dicts = original_convert

        # Verify inference output
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

        # Verify all returned splits have inference columns
        for split, df in result.items():
            with self.subTest(split=split):

                self.assertFalse(df.empty)

                inference_cols = [c for c in df.columns if c.startswith("inference_")]
                self.assertGreater(len(inference_cols), 0)

                inference_col = inference_cols[0]

                # line-level correctness checks
                if "line_id" in df.columns:
                    inferred = df[df[inference_col] != ""]

                    # Requested splits: must contain inference
                    if split in inference.requested_splits or split == "default":

                        self.assertGreater(
                            len(inferred),
                            0,
                            f"No inference results written for requested split '{split}'"
                        )

                        # One unique inference value per line_id
                        for (filename, line_id), group in inferred.groupby(["filename", "line_id"]):
                            unique_vals = group[inference_col].unique()
                            self.assertEqual(
                                len(unique_vals),
                                1,
                                f"Multiple inference values found for ({filename}, {line_id}) in split '{split}'"
                            )

                    # Unrequested splits: must contain ONLY empty strings
                    else:
                        self.assertEqual(
                            len(inferred),
                            0,
                            f"Unrequested split '{split}' should not contain inference results"
                        )

        # Verify upload to HF Hub
        files = api.list_repo_files(
            repo_id=self.test_repo,
            repo_type="dataset",
            token=self.write_token
        )

        files_lower = [f.lower() for f in files]

        self.assertTrue(
            any("train" in f for f in files_lower),
            "Uploaded repo missing train split parquet"
        )

        self.assertTrue(
            any("parquet" in f for f in files_lower),
            "No parquet files uploaded"
        )


if __name__ == "__main__":
    unittest.main()
