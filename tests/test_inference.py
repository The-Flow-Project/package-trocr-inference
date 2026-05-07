import os
import unittest
from unittest.mock import patch

import pandas as pd
from flow_inference.inference import Inference
from dotenv import load_dotenv

from flow_inference.model_handling import ModelManager


class TestInference(unittest.TestCase):

    def setUp(self):
        """Set up an Inference instance configured for a small HF dataset."""
        load_dotenv()
        self.download_repo_name = os.getenv("TEST_REPO_PUBLIC_EXTERNAL")
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

    @staticmethod
    def _key_columns_for_df(df: pd.DataFrame) -> list[str]:
        key = ["project_name", "filename", "line_id"]

        if "line_augmentation" in df.columns:
            key.append("line_augmentation")

        return key

    @staticmethod
    def _is_original_row(df: pd.DataFrame) -> pd.Series:
        if "line_augmentation" not in df.columns:
            return pd.Series(True, index=df.index)

        return (
            df["line_augmentation"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("original")
        )

    @staticmethod
    def _limit_records_for_test(records: list[dict], limit: int = 10) -> list[dict]:
        if not records:
            return records

        if not any("line_augmentation" in record for record in records):
            return records[:limit]

        originals = [
            record
            for record in records
            if str(record.get("line_augmentation", "")).strip().lower() == "original"
        ]

        return originals[:limit]

    def test_perform_inference_returns_dataframe(self):
        """Run the full inference pipeline and ensure it returns a dict of DataFrames."""
        from flow_inference.data_handling import HuggingFaceDataHandler

        # limit dataset to 3 original rows per split when augmented
        original_convert = HuggingFaceDataHandler.convert_to_list_of_dicts

        def limited_convert(dfs):
            full = original_convert(dfs)
            return {
                split: self._limit_records_for_test(recs, limit=10)
                for split, recs in full.items()
            }

        with patch.object(HuggingFaceDataHandler,
                          "convert_to_list_of_dicts",
                          staticmethod(limited_convert)):
            result = self.inference.perform_inference()

        # must return a dict
        self.assertIsInstance(result, dict, "Expected dict of DataFrames")
        self.assertGreater(len(result), 0, "No splits returned")

        for split, df in result.items():
            with self.subTest(split=split):

                self.assertIsInstance(df, pd.DataFrame)
                self.assertFalse(df.empty, "Returned DataFrame is empty")

                inference_cols = [c for c in df.columns if c.startswith("inference_")]
                self.assertGreater(len(inference_cols), 0, "No inference column found")
                inference_col = sorted(inference_cols)[-1]

                if split in self.inference.requested_splits or split == "default":
                    inferred = df[df[inference_col].fillna("").astype(str).str.strip() != ""]

                    # at least one inferred line
                    self.assertGreater(
                        len(inferred),
                        0,
                        f"No inference results written for split '{split}'"
                    )

                    self.assertFalse(
                        inferred[inference_col].fillna("").astype(str).str.strip().eq("").any(),
                        f"Some inferred rows have empty inference values in split '{split}'"
                    )

                    # if augmented, only original rows may receive inference
                    if "line_augmentation" in df.columns:
                        non_original = df[~self._is_original_row(df)]
                        non_original_non_empty = (
                            non_original[inference_col]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                            .ne("")
                        )

                        self.assertFalse(
                            non_original_non_empty.any(),
                            f"Non-original augmented rows received inference in split '{split}'"
                        )

                # non-requested split: must contain ONLY empty strings
                else:
                    non_empty = df[inference_col].fillna("").astype(str).str.strip().ne("")
                    self.assertFalse(
                        non_empty.any(),
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
            records = records_dict["train"]
        else:
            records = next(iter(records_dict.values()))

        records = self._limit_records_for_test(records, limit=2)

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
            self.assertEqual(len(key), 3)

            project, filename, line_id = key

            self.assertIsInstance(project, str)
            self.assertIsInstance(filename, str)
            self.assertIsInstance(line_id, str)
            self.assertIsInstance(value, list)
            self.assertGreater(len(value), 0)

            for prediction in value:
                self.assertIsInstance(prediction, str)

            valid_keys = {
                (str(r["project_name"]), str(r["filename"]), str(r["line_id"]))
                for r in records
            }

            self.assertIn(
                (str(project), str(filename), str(line_id)),
                valid_keys,
                "Inference result key is not a (project_name, filename, line_id) tuple from filtered input records"
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

        def limited_convert(dfs):
            full = original_convert(dfs)
            return {
                split: self._limit_records_for_test(recs, limit=10)
                for split, recs in full.items()
            }

        with patch.object(HuggingFaceDataHandler, "convert_to_list_of_dicts", staticmethod(limited_convert)):
            inference = Inference(
                download_repo_name=self.download_repo_name,
                hf_token=self.write_token,
                trocr_model="microsoft/trocr-small-handwritten",
                stop_on_fail=False,
                push_to_hub=True,
                upload_repo_name=self.test_repo
            )

            result = inference.perform_inference()

        # Verify inference output
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

        # Verify all returned splits have inference columns
        for split, df in result.items():
            with self.subTest(split=split):

                self.assertFalse(df.empty)

                inference_cols = [c for c in df.columns if c.startswith("inference_")]
                self.assertGreater(len(inference_cols), 0)

                inference_col = sorted(inference_cols)[-1]

                # line-level correctness checks
                if "line_id" in df.columns:
                    inferred = df[df[inference_col].fillna("").astype(str).str.strip() != ""]

                    # Requested splits: must contain inference
                    if split in inference.requested_splits or split == "default":

                        self.assertGreater(
                            len(inferred),
                            0,
                            f"No inference results written for requested split '{split}'"
                        )

                        self.assertFalse(
                            inferred[inference_col]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                            .eq("")
                            .any(),
                            f"Some inferred rows have empty inference values in split '{split}'"
                        )

                        # if augmented, only original rows may receive inference
                        if "line_augmentation" in df.columns:
                            non_original = df[~self._is_original_row(df)]
                            non_original_non_empty = (
                                non_original[inference_col]
                                .fillna("")
                                .astype(str)
                                .str.strip()
                                .ne("")
                            )

                            self.assertFalse(
                                non_original_non_empty.any(),
                                f"Non-original augmented rows received inference in split '{split}'"
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

    def test_filter_records_for_inference_without_line_augmentation_keeps_all_records(self):
        records = [
            {"project_name": "p", "filename": "f", "line_id": "1"},
            {"project_name": "p", "filename": "f", "line_id": "2"},
        ]

        filtered = self.inference._filter_records_for_inference(records)

        self.assertEqual(filtered, records)

    def test_filter_records_for_inference_with_line_augmentation_keeps_only_original(self):
        records = [
            {
                "project_name": "p",
                "filename": "f",
                "line_id": "1",
                "line_augmentation": "original",
            },
            {
                "project_name": "p",
                "filename": "f",
                "line_id": "1",
                "line_augmentation": "rotation",
            },
            {
                "project_name": "p",
                "filename": "f",
                "line_id": "2",
                "line_augmentation": " ORIGINAL ",
            },
            {
                "project_name": "p",
                "filename": "f",
                "line_id": "3",
                "line_augmentation": None,
            },
        ]

        filtered = self.inference._filter_records_for_inference(records)

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["line_id"], "1")
        self.assertEqual(filtered[1]["line_id"], "2")
        self.assertTrue(
            all(
                str(record["line_augmentation"]).strip().lower() == "original"
                for record in filtered
            )
        )

    def test_write_inference_to_dataframe_preserves_augmented_rows_and_writes_only_original(self):
        df = pd.DataFrame({
            "project_name": ["p", "p", "p"],
            "filename": ["f", "f", "f"],
            "line_id": ["1", "1", "2"],
            "line_augmentation": ["original", "rotation", "original"],
            "text": ["", "", ""],
        })

        inferred_lines = {
            ("p", "f", "1"): ["prediction for line 1"],
            ("p", "f", "2"): ["prediction for line 2"],
        }

        updated = self.inference.write_inference_to_dataframe(
            inferred_lines=inferred_lines,
            original_df=df,
        )

        self.assertEqual(len(updated), len(df))

        inference_cols = [c for c in updated.columns if c.startswith("inference_")]
        self.assertEqual(len(inference_cols), 1)
        inference_col = inference_cols[0]

        original_l1 = updated[
            (updated["line_id"] == "1")
            & (updated["line_augmentation"] == "original")
            ]
        augmented_l1 = updated[
            (updated["line_id"] == "1")
            & (updated["line_augmentation"] == "rotation")
            ]

        self.assertEqual(
            original_l1.iloc[0][inference_col],
            "prediction for line 1",
        )

        self.assertEqual(
            augmented_l1.iloc[0][inference_col],
            "",
        )

        self.assertEqual(
            updated.loc[updated["line_id"] == "2", inference_col].iloc[0],
            "prediction for line 2",
        )

    def test_run_inference_with_empty_records_returns_empty_dict(self):
        self.inference.statusManager.initialize_status(0)

        result = self.inference.run_inference(
            records=[],
            model=None,
            processor=None,
            device="cpu",
        )

        self.assertEqual(result, {})

    def test_write_inference_to_dataframe_writes_duplicate_predictions_separately(self):
        df = pd.DataFrame({
            "project_name": ["p", "p"],
            "filename": ["f", "f"],
            "line_id": ["1", "1"],
            "text": ["", ""],
        })

        inferred_lines = {
            ("p", "f", "1"): ["prediction A", "prediction B"],
        }

        updated = self.inference.write_inference_to_dataframe(
            inferred_lines=inferred_lines,
            original_df=df,
        )

        inference_cols = [c for c in updated.columns if c.startswith("inference_")]
        self.assertEqual(len(inference_cols), 1)
        inference_col = inference_cols[0]

        self.assertEqual(
            updated[inference_col].tolist(),
            ["prediction A", "prediction B"],
        )

    def test_write_inference_to_dataframe_writes_duplicate_original_augmented_rows_separately(self):
        df = pd.DataFrame({
            "project_name": ["p", "p", "p"],
            "filename": ["f", "f", "f"],
            "line_id": ["1", "1", "1"],
            "line_augmentation": ["original", "original", '{"rotation": 1}'],
            "text": ["", "", ""],
        })

        inferred_lines = {
            ("p", "f", "1"): ["prediction A", "prediction B"],
        }

        updated = self.inference.write_inference_to_dataframe(
            inferred_lines=inferred_lines,
            original_df=df,
        )

        inference_cols = [c for c in updated.columns if c.startswith("inference_")]
        inference_col = inference_cols[0]

        original_values = updated.loc[
            updated["line_augmentation"] == "original",
            inference_col,
        ].tolist()

        augmented_values = updated.loc[
            updated["line_augmentation"] != "original",
            inference_col,
        ].tolist()

        self.assertEqual(original_values, ["prediction A", "prediction B"])
        self.assertEqual(augmented_values, [""])

    def test_filter_records_for_inference_without_augmentation_keeps_duplicate_records(self):
        records = [
            {"project_name": "p", "filename": "f", "line_id": "1"},
            {"project_name": "p", "filename": "f", "line_id": "1"},
            {"project_name": "p", "filename": "f", "line_id": "2"},
        ]

        filtered = self.inference._filter_records_for_inference(records)

        self.assertEqual(filtered, records)
        self.assertEqual(len(filtered), 3)

    def test_filter_records_for_inference_with_augmentation_keeps_duplicate_original_records(self):
        records = [
            {
                "project_name": "p",
                "filename": "f",
                "line_id": "1",
                "line_augmentation": "original",
            },
            {
                "project_name": "p",
                "filename": "f",
                "line_id": "1",
                "line_augmentation": "original",
            },
            {
                "project_name": "p",
                "filename": "f",
                "line_id": "1",
                "line_augmentation": '{"rotation": 1}',
            },
        ]

        filtered = self.inference._filter_records_for_inference(records)

        self.assertEqual(len(filtered), 2)
        self.assertTrue(
            all(
                record["line_augmentation"].strip().lower() == "original"
                for record in filtered
            )
        )
        self.assertEqual(
            [(record["project_name"], record["filename"], record["line_id"]) for record in filtered],
            [("p", "f", "1"), ("p", "f", "1")],
        )


if __name__ == "__main__":
    unittest.main()