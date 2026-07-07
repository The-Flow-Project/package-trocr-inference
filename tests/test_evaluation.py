import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
from dotenv import load_dotenv
from flow_inference.evaluation import Evaluation


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        """Set up an Evaluation instance configured for testing."""
        load_dotenv()
        self.evaluation_repo_name = os.getenv("EVALUATION_REPO")
        self.write_token = os.getenv("HUGGINGFACE_TOKEN_READ_WRITE")

        self.evaluator = Evaluation(
            evaluation_repo_name=self.evaluation_repo_name,
            hf_token=self.write_token,
            splits=None
        )

    # -------------------------------------------------------------
    # UNIT TEST: SELECTING SPLITS - DEFAULT BEHAVIOR
    # -------------------------------------------------------------
    def test_select_splits_default(self):
        dfs = {
            "train": pd.DataFrame({"a": [1]}),
            "test": pd.DataFrame({"a": [2]}),
        }
        out = self.evaluator.select_splits(dfs)
        self.assertEqual(out["a"].iloc[0], 2)

    # -------------------------------------------------------------
    # UNIT TEST: SELECTING SPLITS - REQUESTED SPLIT
    # -------------------------------------------------------------
    def test_select_splits_requested(self):
        self.evaluator.splits = ["train"]
        dfs = {
            "train": pd.DataFrame({"a": [10]}),
            "test": pd.DataFrame({"a": [20]}),
        }
        out = self.evaluator.select_splits(dfs)
        self.assertEqual(out["a"].iloc[0], 10)

    # -------------------------------------------------------------
    # UNIT TEST: GROUND TRUTH EXTRACTION - BASIC CASE
    # -------------------------------------------------------------
    def test_extract_ground_truth(self):
        df = pd.DataFrame({"text": ["hello", "world"]})
        gt = self.evaluator._extract_ground_truth(df)
        self.assertEqual(gt, ["hello", "world"])

    # -------------------------------------------------------------
    # UNIT TEST: GROUND TRUTH EXTRACTION - MISSING 'text' COLUMN
    # -------------------------------------------------------------
    def test_extract_ground_truth_missing(self):
        df = pd.DataFrame({"other": [1]})
        with self.assertRaises(ValueError):
            self.evaluator._extract_ground_truth(df)

    # -------------------------------------------------------------
    # UNIT TEST: FINDING THE LATEST INFERENCE COLUMN - BASIC CASE
    # -------------------------------------------------------------
    def test_find_latest_inference_column(self):
        df = pd.DataFrame({
            "text": ["a"],
            "inference_2024-01-01_model_x": ["1"],
            "inference_2025-01-01_model_x": ["2"],
        })
        col = self.evaluator._find_latest_inference_column(df)
        self.assertEqual(col, "inference_2025-01-01_model_x")

    # -------------------------------------------------------------
    # UNIT TEST: FINDING THE LATEST INFERENCE COLUMN - MISSING CASE
    # -------------------------------------------------------------
    def test_find_latest_inference_missing(self):
        df = pd.DataFrame({"text": ["a"]})
        with self.assertRaises(ValueError):
            self.evaluator._find_latest_inference_column(df)

    # -------------------------------------------------------------
    # UNIT TEST: HYPOTHESIS EXTRACTION - BASIC CASE
    # -------------------------------------------------------------
    def test_extract_hypothesis(self):
        df = pd.DataFrame({"col": ["a", "b"]})
        out = self.evaluator._extract_hypothesis(df, "col")
        self.assertEqual(out, ["a", "b"])

    # -------------------------------------------------------------
    # UNIT TEST: HYPOTHESIS EXTRACTION - MISSING COLUMN
    # -------------------------------------------------------------
    def test_extract_hypothesis_missing(self):
        df = pd.DataFrame({"col": ["a"]})
        with self.assertRaises(ValueError):
            self.evaluator._extract_hypothesis(df, "nothing")

    # -------------------------------------------------------------
    # UNIT TEST: CER COMPUTATION
    # -------------------------------------------------------------
    def test_compute_cer(self):
        cer = self.evaluator.compute_cer(["abc"], ["abd"])
        self.assertIsInstance(cer, float)

    # -------------------------------------------------------------
    # UNIT TEST: OUTPUT FILE CREATION
    # -------------------------------------------------------------
    def test_create_output_files(self):
        files = self.evaluator.create_output_files(
            groundtruth=["a", "b"],
            hypothesis=["a", "c"],
            cer_score=0.25,
            timestamp="2026-04-19T22:00:00",
        )

        self.assertIn("gt.txt", files)
        self.assertIn("hypothesis.txt", files)
        self.assertIn("evaluation_report.json", files)

        self.assertIsInstance(files["gt.txt"], bytes)
        self.assertEqual(files["gt.txt"].decode(), "a\nb")

        report_text = files["evaluation_report.json"].decode("utf-8")
        self.assertIn('"timestamp": "2026-04-19T22:00:00"', report_text)
        self.assertIn('"cer": 0.25', report_text)

    # -------------------------------------------------------------
    # UNIT TEST: EVALUATION ROW FILTERING NO OVERLAP
    # -------------------------------------------------------------
    def test_evaluation_fails_when_no_gt_inference_overlap(self):
        df = pd.DataFrame({
            "text": ["", "", "hello", "world"],
            "inference_x": ["abc", "def", "", ""],
        })

        evaluator = Evaluation("dummy", None)

        with self.assertRaises(RuntimeError):
            evaluator._filter_eval_rows(df, "inference_x")

    # -------------------------------------------------------------
    # UNIT TEST: CER COMPUTATION ON VALID OVERLAP
    # -------------------------------------------------------------
    def test_evaluation_computes_cer_on_valid_overlap(self):
        df = pd.DataFrame({
            "text": ["hello", "world", "ignored"],
            "inference_x": ["hallo", "world", ""],
            "line_augmentation": [
                "original",
                '{"rotation": 1}',
                '{"blur": 1}',
            ],
        })

        evaluator = Evaluation("dummy", None)

        df_eval = evaluator._filter_eval_rows(df, "inference_x")

        self.assertEqual(len(df_eval), 2)
        self.assertEqual(
            df_eval["line_augmentation"].tolist(),
            ["original", '{"rotation": 1}'],
        )

        gt = evaluator._extract_ground_truth(df_eval)
        hyp = evaluator._extract_hypothesis(df_eval, "inference_x")

        cer = evaluator.compute_cer(gt, hyp)

        self.assertGreaterEqual(cer, 0.0)
        self.assertLessEqual(cer, 1.0)

    # -------------------------------------------------------------
    # UNIT TEST: UPLOAD RESULTS CREATES CORRECT PATHS
    # -------------------------------------------------------------
    def test_upload_results_returns_evaluation_path(self):
        evaluator = Evaluation("dummy_repo", None)
        evaluator.data_handler.upload_file = MagicMock()

        files = {
            "gt.txt": b"a\nb",
            "hypothesis.txt": b"a\nc",
        }

        evaluation_path = evaluator.upload_results(files, "2026-04-19T22_00_00")

        self.assertEqual(evaluation_path, "evaluation/2026-04-19T22_00_00/")
        self.assertEqual(evaluator.data_handler.upload_file.call_count, 2)

        calls = evaluator.data_handler.upload_file.call_args_list
        target_paths = [call.kwargs["target_path"] for call in calls]
        self.assertIn("evaluation/2026-04-19T22_00_00/gt.txt", target_paths)
        self.assertIn("evaluation/2026-04-19T22_00_00/hypothesis.txt", target_paths)

    # -------------------------------------------------------------
    # UNIT TEST: README UPLOAD
    # -------------------------------------------------------------
    @patch("flow_inference.evaluation.HuggingFaceReadmeBuilder")
    def test_upload_readme(self, mock_builder_cls):
        evaluator = Evaluation("dummy_repo", None, splits=["test"])

        evaluator.data_handler.dataset = MagicMock()
        evaluator.data_handler.parquet_paths = {"test": ["dummy.parquet"]}
        evaluator.data_handler.upload_file = MagicMock()

        mock_builder = mock_builder_cls.from_handler.return_value
        mock_builder.render.return_value = "README CONTENT"

        dfs = {
            "test": pd.DataFrame({
                "project_name": ["p1"],
                "text": ["hello"],
                "inference_x": ["hallo"],
            })
        }

        evaluator.upload_readme(
            dfs=dfs,
            inference_col="inference_x",
            cer_score=0.123,
            eval_rows=1,
            timestamp="2026-04-19T22:00:00",
            evaluation_path="evaluation/2026-04-19T22_00_00/",
        )

        mock_builder_cls.from_handler.assert_called_once()
        evaluator.data_handler.upload_file.assert_called_once_with(
            repo_name="dummy_repo",
            target_path="README.md",
            content_bytes=b"README CONTENT",
        )

    # --------------------------------------------------
    # INTEGRATION TEST: PERFORM EVALUATION AND UPLOAD
    # --------------------------------------------------
    def test_perform_evaluation_integration(self):
        """
        Integration test:
        - runs evaluation
        - uploads evaluation files into repo
        - regenerates README
        - verifies all files exists
        """
        if not self.evaluation_repo_name or not self.write_token:
            self.skipTest("Missing HF integration configuration")

        evaluator = Evaluation(
            evaluation_repo_name=self.evaluation_repo_name,
            hf_token=self.write_token,
            splits=None
        )
        files = evaluator.perform_evaluation()

        # basic output checks
        self.assertIn("gt.txt", files)
        self.assertIn("hypothesis.txt", files)
        self.assertIn("evaluation_report.json", files)

        # verify repo contents
        from huggingface_hub import HfApi
        api = HfApi()

        repo_files = api.list_repo_files(
            repo_id=self.evaluation_repo_name,
            repo_type="dataset",
            token=self.write_token
        )

        # check if evaluation files exist
        self.assertTrue(any(f.startswith("evaluation/") and f.endswith("gt.txt") for f in repo_files))
        self.assertTrue(any(f.startswith("evaluation/") and f.endswith("hypothesis.txt") for f in repo_files))
        self.assertTrue(any(f.startswith("evaluation/") and f.endswith("evaluation_report.json") for f in repo_files))

        # check if README exists
        self.assertIn("README.md", repo_files)

        # download repo snapshot and inspect content
        tmp_dir = tempfile.mkdtemp(prefix="verify_eval_")
        try:
            local_path = api.snapshot_download(
                repo_id=self.evaluation_repo_name,
                repo_type="dataset",
                token=self.write_token,
                local_dir=tmp_dir,
            )

            local_path = Path(local_path)

            # check README content
            readme_path = local_path / "README.md"
            self.assertTrue(readme_path.exists())

            readme_text = readme_path.read_text(encoding="utf-8")

            self.assertIn("## Evaluation Results", readme_text)
            self.assertIn("CER", readme_text)
            self.assertIn("Evaluation Files", readme_text)

            # check evaluation report
            reports = list(local_path.rglob("evaluation_report.json"))
            self.assertTrue(reports, "No evaluation_report.json found")

            report = json.loads(reports[0].read_text())

            self.assertIn("cer", report)
            self.assertIn("timestamp", report)
            self.assertGreaterEqual(report["cer"], 0.0)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_restore_image_feature_casts_image_column(self):
        from datasets import Dataset, Image as DatasetImage

        ds = Dataset.from_dict({
            "image": [{"bytes": b"fake-image-bytes", "path": None}],
            "text": ["hello"],
        })

        restored = self.evaluator._restore_image_feature(ds)

        self.assertIsInstance(restored.features["image"], DatasetImage)

    def test_filter_eval_rows_includes_original_and_augmented_rows(self):
        df = pd.DataFrame({
            "text": [
                "ground truth original",
                "ground truth rotated",
                "ground truth blurred",
                "missing prediction",
            ],
            "inference_x": [
                "prediction original",
                "prediction rotated",
                "prediction blurred",
                "",
            ],
            "line_augmentation": [
                "original",
                '{"rotation": 1}',
                '{"blur": 2}',
                "original",
            ],
        })

        df_eval = self.evaluator._filter_eval_rows(
            df=df,
            inference_col="inference_x",
        )

        self.assertEqual(len(df_eval), 3)

        self.assertEqual(
            df_eval["line_augmentation"].tolist(),
            [
                "original",
                '{"rotation": 1}',
                '{"blur": 2}',
            ],
        )


if __name__ == "__main__":
    unittest.main()