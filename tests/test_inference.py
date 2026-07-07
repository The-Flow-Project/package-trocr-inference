import os
import unittest
import warnings
from unittest.mock import Mock, patch
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import pandas as pd
from huggingface_hub import HfApi

from flow_inference.data_handling import HuggingFaceDataHandler
from flow_inference.inference import Inference
from flow_inference.model_handling import ModelManager


class TestInference(unittest.TestCase):

    def setUp(self):
        """Set up environment variables only. Do not load the model here."""
        self.download_repo_name = os.getenv("HUGGINGFACE_DOWNLOAD_REPO_NAME")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN_READ")
        self.write_token = os.getenv("HUGGINGFACE_TOKEN_READ_WRITE")
        self.test_repo = os.getenv("HUGGINGFACE_TEST_UPLOAD_REPO_NAME")

    @staticmethod
    def _limit_records_for_test(
            records: list[dict],
            limit: int = 10,
    ) -> list[dict]:
        return records[:limit]

    def _unique_test_repo_name(self, suffix: str) -> str:
        if not self.test_repo:
            self.skipTest("Missing HUGGINGFACE_TEST_UPLOAD_REPO_NAME.")

        namespace = self.test_repo.rsplit("/", 1)[0]
        safe_suffix = suffix.replace("_", "-")
        return f"{namespace}/test-inference-{safe_suffix}-{uuid4().hex[:8]}"

    @staticmethod
    def _lightweight_inference() -> Inference:
        """
        Create an Inference instance without running __init__.

        This is useful for pure helper-method tests such as
        write_inference_to_dataframe(), where loading TrOCR would only slow the
        test down and is unrelated to what is being checked.
        """
        inference = object.__new__(Inference)
        inference.trocr_model = "microsoft/trocr-small-printed"
        inference.statusManager = Mock()
        return inference

    @patch("flow_inference.inference.HuggingFaceDataHandler")
    @patch("flow_inference.inference.ModelManager")
    def test_perform_inference_returns_dataframe(
            self,
            mock_model_manager_cls,
            mock_handler_cls,
    ):
        """perform_inference() should return DataFrames with predictions."""
        mock_model_manager = mock_model_manager_cls.return_value
        mock_model_manager.device = "cpu"
        mock_model_manager.load_processor.return_value = object()
        mock_model_manager.load_model.return_value = object()

        df = pd.DataFrame({
            "project_name": ["p", "p"],
            "filename": ["f", "f"],
            "region_id": ["r", "r"],
            "line_id": ["1", "2"],
            "text": ["hello", "world"],
        })
        records = df.to_dict(orient="records")

        mock_loader = mock_handler_cls.return_value
        mock_loader.to_dataframe.return_value = {"train": df}
        mock_loader.convert_to_list_of_dicts.return_value = {
            "train": records,
        }

        inference = Inference(
            download_repo_name="dummy/repo",
            hf_token=None,
            trocr_model="dummy-model",
            stop_on_fail=False,
            push_to_hub=False,
            splits=["train"],
        )

        with patch.object(
                inference,
                "run_inference",
                return_value={
                    ("p", "f", "r", "1"): ["prediction 1"],
                    ("p", "f", "r", "2"): ["prediction 2"],
                },
        ):
            result = inference.perform_inference()

        self.assertIsInstance(result, dict, "Expected dict of DataFrames")
        self.assertIn("train", result)

        result_df = result["train"]
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertFalse(result_df.empty, "Returned DataFrame is empty")

        inference_cols = [
            c for c in result_df.columns
            if c.startswith("inference_")
        ]
        self.assertEqual(len(inference_cols), 1, "Expected one inference column")

        inference_col = inference_cols[0]
        self.assertEqual(
            result_df[inference_col].tolist(),
            ["prediction 1", "prediction 2"],
        )

    def test_run_inference_returns_dict(self):
        """Ensure run_inference() produces a dictionary of real OCR results."""
        if not self.download_repo_name or not self.hf_token:
            self.skipTest("Missing Hugging Face credentials or repo name.")

        handler = HuggingFaceDataHandler(
            dataset_name=self.download_repo_name,
            huggingface_token=self.hf_token,
        )

        handler.download_hf_dataset()
        dfs = handler.to_dataframe()
        records_dict = handler.convert_to_list_of_dicts(dfs)

        if "train" in records_dict:
            records = records_dict["train"]
        else:
            records = next(iter(records_dict.values()))

        records = self._limit_records_for_test(records, limit=2)

        inference = Inference(
            download_repo_name=self.download_repo_name,
            hf_token=self.write_token,
            trocr_model="microsoft/trocr-small-printed",
            stop_on_fail=False,
            push_to_hub=False,
        )
        inference.statusManager.initialize_status(len(records))

        model_manager = ModelManager()
        processor = model_manager.load_processor(inference.trocr_model)
        model = model_manager.load_model(inference.trocr_model)

        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)

        result_dict = inference.run_inference(
            records=records,
            model=model,
            processor=processor,
            device=model_manager.device,
        )

        self.assertIsInstance(result_dict, dict)
        self.assertGreater(len(result_dict), 0)

        valid_keys = {
            (
                str(r.get("project_name", "")),
                str(r["filename"]),
                str(r["region_id"]),
                str(r["line_id"]),
            )
            for r in records
        }

        for key, value in result_dict.items():
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 4)

            project, filename, region_id, line_id = key

            self.assertIsInstance(project, str)
            self.assertIsInstance(filename, str)
            self.assertIsInstance(region_id, str)
            self.assertIsInstance(line_id, str)
            self.assertIsInstance(value, list)
            self.assertGreater(len(value), 0)

            for prediction in value:
                self.assertIsInstance(prediction, str)

            self.assertIn(
                (str(project), str(filename), str(region_id), str(line_id)),
                valid_keys,
                "Inference result key is not a "
                "(project_name, filename, region_id, line_id) tuple "
                "from input records",
            )

    def test_full_inference_with_upload(self):
        """
        Full integration test:
        - downloads dataset
        - runs inference on a limited number of records
        - writes results back into dataframe
        - pushes updated dataset to a fresh HF Hub target repo
        - verifies parquet files + README exist
        - never uploads to the source repo
        """
        if (
            not self.write_token
            or not self.hf_token
            or not self.test_repo
            or not self.download_repo_name
        ):
            self.skipTest("Missing Hugging Face credentials or repo names.")

        target_repo = self._unique_test_repo_name("full-upload")
        self.assertNotEqual(target_repo, self.download_repo_name)

        original_convert = HuggingFaceDataHandler.convert_to_list_of_dicts

        def limited_convert(dfs):
            full = original_convert(dfs)
            return {
                split: self._limit_records_for_test(recs, limit=10)
                for split, recs in full.items()
            }

        with patch.object(
            HuggingFaceDataHandler,
            "convert_to_list_of_dicts",
            staticmethod(limited_convert),
        ):
            inference = Inference(
                download_repo_name=self.download_repo_name,
                hf_token=self.write_token,
                trocr_model="microsoft/trocr-small-printed",
                stop_on_fail=False,
                push_to_hub=True,
                upload_repo_name=target_repo,
                upload_mode="new_repo",
                private_repo=True,
            )

            result = inference.perform_inference()

        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

        written_inference_col = None

        for split, df in result.items():
            with self.subTest(split=split):
                self.assertFalse(df.empty)

                inference_cols = [
                    c for c in df.columns
                    if c.startswith("inference_")
                ]
                self.assertGreater(len(inference_cols), 0)

                inference_col = sorted(inference_cols)[-1]
                written_inference_col = inference_col

                if "line_id" in df.columns:
                    inferred = df[
                        df[inference_col]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .ne("")
                    ]

                    if split in inference.requested_splits or split == "default":
                        self.assertGreater(
                            len(inferred),
                            0,
                            f"No inference results written for requested split '{split}'",
                        )
                    else:
                        self.assertEqual(
                            len(inferred),
                            0,
                            f"Unrequested split '{split}' should not contain inference results",
                        )

        api = HfApi()
        files = api.list_repo_files(
            repo_id=target_repo,
            repo_type="dataset",
            token=self.write_token,
        )

        files_lower = [f.lower() for f in files]

        self.assertIn("README.md", files)
        self.assertTrue(
            any("train" in f for f in files_lower),
            "Uploaded repo missing train split parquet",
        )
        self.assertTrue(
            any(f.endswith(".parquet") for f in files_lower),
            "No parquet files uploaded",
        )

        verify_handler = HuggingFaceDataHandler(
            dataset_name=target_repo,
            huggingface_token=self.hf_token,
            split=None,
        )
        verify_handler.download_hf_dataset()
        verify_dfs = verify_handler.to_dataframe()

        self.assertGreater(len(verify_dfs), 0)
        self.assertIsNotNone(written_inference_col)

        found_uploaded_inference_col = False
        found_uploaded_prediction = False

        for df in verify_dfs.values():
            if written_inference_col in df.columns:
                found_uploaded_inference_col = True
                values = df[written_inference_col].fillna("").astype(str).str.strip()
                if values.ne("").any():
                    found_uploaded_prediction = True

        self.assertTrue(
            found_uploaded_inference_col,
            f"Uploaded repo missing inference column '{written_inference_col}'",
        )
        self.assertTrue(
            found_uploaded_prediction,
            "Uploaded inference column exists but contains no non-empty prediction.",
        )

    def test_write_inference_to_dataframe_writes_original_and_augmented_rows(self):
        df = pd.DataFrame({
            "project_name": ["p", "p", "p"],
            "filename": ["f", "f", "f"],
            "region_id": ["r", "r", "r"],
            "line_id": ["1", "1", "2"],
            "line_augmentation": [
                "original",
                "rotation",
                "original",
            ],
            "text": ["", "", ""],
        })

        inferred_lines = {
            ("p", "f", "r", "1"): [
                "prediction for original line 1",
                "prediction for rotated line 1",
            ],
            ("p", "f", "r", "2"): [
                "prediction for line 2",
            ],
        }

        inference = self._lightweight_inference()
        updated = inference.write_inference_to_dataframe(
            inferred_lines=inferred_lines,
            original_df=df,
        )

        inference_cols = [
            col for col in updated.columns
            if col.startswith("inference_")
        ]
        self.assertEqual(len(inference_cols), 1)
        inference_col = inference_cols[0]

        self.assertEqual(
            updated[inference_col].tolist(),
            [
                "prediction for original line 1",
                "prediction for rotated line 1",
                "prediction for line 2",
            ],
        )

    @patch("flow_inference.inference.ModelManager")
    def test_run_inference_with_empty_records_returns_empty_dict(
            self,
            mock_model_manager_cls,
    ):
        mock_model_manager = mock_model_manager_cls.return_value
        mock_model_manager.device = "cpu"
        mock_model_manager.load_processor.return_value = object()
        mock_model_manager.load_model.return_value = object()

        inference = Inference(
            download_repo_name="dummy/repo",
            hf_token=None,
            trocr_model="dummy-model",
            stop_on_fail=False,
            push_to_hub=False,
        )
        inference.statusManager.initialize_status(0)

        result = inference.run_inference(
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
            "region_id": ["r", "r"],
            "line_id": ["1", "1"],
            "text": ["", ""],
        })

        inferred_lines = {
            ("p", "f", "r", "1"): ["prediction A", "prediction B"],
        }

        inference = self._lightweight_inference()
        updated = inference.write_inference_to_dataframe(
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
            "region_id": ["r", "r", "r"],
            "line_augmentation": [
                "original",
                "original",
                '{"rotation": 1}',
            ],
            "text": ["", "", ""],
        })

        inferred_lines = {
            ("p", "f", "r", "1"): [
                "prediction A",
                "prediction B",
                "prediction C",
            ],
        }

        inference = self._lightweight_inference()
        updated = inference.write_inference_to_dataframe(
            inferred_lines=inferred_lines,
            original_df=df,
        )

        inference_col = next(
            col for col in updated.columns
            if col.startswith("inference_")
        )

        self.assertEqual(
            updated[inference_col].tolist(),
            [
                "prediction A",
                "prediction B",
                "prediction C",
            ],
        )

    @patch("flow_inference.inference.HuggingFaceDataHandler")
    @patch("flow_inference.inference.ModelManager")
    def test_perform_inference_refuses_source_repo_upload_by_default(
            self,
            mock_model_manager_cls,
            mock_handler_cls,
    ):
        """
        If push_to_hub=True and no upload_repo_name is provided,
        Inference would target the source repo. This must be refused unless
        allow_source_repo_update=True.
        """
        mock_model_manager = mock_model_manager_cls.return_value
        mock_model_manager.device = "cpu"
        mock_model_manager.load_processor.return_value = object()
        mock_model_manager.load_model.return_value = object()

        mock_loader = mock_handler_cls.return_value

        df = pd.DataFrame({
            "project_name": ["p"],
            "filename": ["f"],
            "region_id": ["r"],
            "line_id": ["1"],
            "text": [""],
        })

        mock_loader.to_dataframe.return_value = {"train": df}
        mock_loader.convert_to_list_of_dicts.return_value = {
            "train": [
                {
                    "project_name": "p",
                    "filename": "f",
                    "region_id": "r",
                    "line_id": "1",
                    "text": "",
                }
            ]
        }

        mock_loader.push_to_hub.side_effect = RuntimeError(
            "Refusing to upload into the source dataset repo"
        )

        inference = Inference(
            download_repo_name="same/source-repo",
            hf_token="TOKEN",
            trocr_model="microsoft/trocr-small-printed",
            stop_on_fail=False,
            push_to_hub=True,
            upload_repo_name=None,
            upload_mode="new_repo",
            allow_source_repo_update=False,
        )

        with patch.object(
                inference,
                "run_inference",
                return_value={("p", "f", "r", "1"): ["pred"]},
        ):
            with self.assertRaisesRegex(
                    RuntimeError,
                    "Refusing to upload into the source dataset repo",
            ):
                inference.perform_inference()

        mock_loader.push_to_hub.assert_called_once()

    @patch("flow_inference.inference.HuggingFaceDataHandler")
    @patch("flow_inference.inference.ModelManager")
    def test_perform_inference_passes_all_rows_to_inference(
            self,
            mock_model_manager_cls,
            mock_handler_cls,
    ):
        mock_model_manager = mock_model_manager_cls.return_value
        mock_model_manager.device = "cpu"
        mock_model_manager.load_processor.return_value = object()
        mock_model_manager.load_model.return_value = object()

        df = pd.DataFrame({
            "project_name": ["p", "p"],
            "filename": ["f", "f"],
            "region_id": ["r", "r"],
            "line_id": ["1", "1"],
            "line_augmentation": ["original", '{"rotation": 1}'],
            "text": ["hello", "hello"],
        })

        records = df.to_dict(orient="records")

        mock_loader = mock_handler_cls.return_value
        mock_loader.to_dataframe.return_value = {"train": df}
        mock_loader.convert_to_list_of_dicts.return_value = {
            "train": records,
        }

        inference = Inference(
            download_repo_name="dummy/repo",
            hf_token=None,
            trocr_model="dummy-model",
            push_to_hub=False,
            splits=["train"],
        )

        predictions = {
            ("p", "f", "r", "1"): [
                "original prediction",
                "augmented prediction",
            ]
        }

        with patch.object(
                inference,
                "run_inference",
                return_value=predictions,
        ) as mock_run:
            result = inference.perform_inference()

        mock_run.assert_called_once()

        passed_records = mock_run.call_args.kwargs["records"]

        self.assertEqual(passed_records, records)
        self.assertEqual(len(passed_records), 2)
        self.assertEqual(
            passed_records[1]["line_augmentation"],
            '{"rotation": 1}',
        )

        inference_col = next(
            col
            for col in result["train"].columns
            if col.startswith("inference_")
        )

        self.assertEqual(
            result["train"][inference_col].tolist(),
            [
                "original prediction",
                "augmented prediction",
            ],
        )


if __name__ == "__main__":
    unittest.main()
