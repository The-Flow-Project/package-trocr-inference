import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from datasets import DatasetDict, Dataset
from dotenv import load_dotenv

from flow_inference.data_handling import HuggingFaceDataHandler


class TestHuggingFaceDataHandler(unittest.TestCase):
    def setUp(self):
        self.handler = HuggingFaceDataHandler(dataset_name="my-org/my-dataset")
        load_dotenv()
        self.hf_token_read = os.getenv("HUGGINGFACE_TOKEN_READ")
        self.hf_download_repo_name = os.getenv("HUGGINGFACE_DOWNLOAD_REPO_NAME")
        self.write_token = os.getenv("HUGGINGFACE_TOKEN_READ_WRITE")
        self.test_repo = os.getenv("HUGGINGFACE_TEST_UPLOAD_REPO_NAME")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _write_parquet(path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    # -----------------------------------------------------------------------
    # MOCK TEST HUGGING FACE DOWNLOAD
    # -----------------------------------------------------------------------
    @patch("flow_inference.data_handling.load_dataset")
    @patch("flow_inference.data_handling.snapshot_download")
    @patch("flow_inference.data_handling.HfApi")
    def test_download_success(self, mock_hfapi, mock_snapshot_download, mock_load_dataset):
        """
        AUTO mode:
        - snapshot_download must use data/**/*.parquet
        - train/test are detected from local folder structure
        """
        fake_info = mock_hfapi.return_value.dataset_info.return_value
        fake_info.sha = "fake_sha_123"

        fake_splits = DatasetDict({
            "train": Dataset.from_dict({"x": [1]}),
            "test": Dataset.from_dict({"x": [2]}),
        })
        mock_load_dataset.return_value = fake_splits

        # Make snapshot_download "create" the expected parquet layout inside local_dir
        def _fake_snapshot_download(*args, **kwargs):
            local_dir = Path(kwargs["local_dir"])
            df_train = pd.DataFrame({"line_id": ["a", "b"], "x": [1, 2]})
            df_test = pd.DataFrame({"line_id": ["c", "d"], "x": [3, 4]})

            self._write_parquet(local_dir / "data/train/docA/train_file.parquet", df_train)
            self._write_parquet(local_dir / "data/test/docB/test_file.parquet", df_test)
            return str(local_dir)

        mock_snapshot_download.side_effect = _fake_snapshot_download

        handler = HuggingFaceDataHandler(
            dataset_name="my-org/my-dataset",
            huggingface_token="TOKEN123",
            revision="main",
            split=None,  # AUTO mode
        )

        handler.download_hf_dataset()

        # dataset_info must be used to resolve sha
        mock_hfapi.return_value.dataset_info.assert_called_once_with(
            repo_id="my-org/my-dataset",
            revision="main",
            token="TOKEN123",
        )

        # snapshot_download must download ALL parquet so layout can be detected
        self.assertTrue(mock_snapshot_download.called)
        sd_kwargs = mock_snapshot_download.call_args.kwargs
        self.assertEqual(sd_kwargs["allow_patterns"], ["data/**/*.parquet"])

        # load_dataset should be called with parquet + explicit file lists
        self.assertTrue(mock_load_dataset.called)
        args, kwargs = mock_load_dataset.call_args
        self.assertEqual(args[0], "parquet")
        self.assertIn("data_files", kwargs)
        self.assertIn("train", kwargs["data_files"])
        self.assertIn("test", kwargs["data_files"])

        train_files = kwargs["data_files"]["train"]
        test_files = kwargs["data_files"]["test"]

        self.assertTrue(any(str(p).endswith(".parquet") for p in train_files))
        self.assertTrue(any(str(p).endswith(".parquet") for p in test_files))

        self.assertEqual(handler.state, "downloaded_all")
        self.assertEqual(set(handler.dataset.keys()), {"train", "test"})
        self.assertIn("train", handler.parquet_paths)
        self.assertIn("test", handler.parquet_paths)

    @patch("flow_inference.data_handling.load_dataset")
    @patch("flow_inference.data_handling.snapshot_download")
    @patch("flow_inference.data_handling.HfApi")
    def test_download_explicit_train_test(self, mock_hfapi, mock_snapshot_download, mock_load_dataset):
        """
        EXPLICIT mode: split=["train","test"]

        Must:
          - snapshot_download only train + test parquet paths
          - error if either split missing
          - load_dataset(parquet, data_files={train,test})
        """
        fake_info = mock_hfapi.return_value.dataset_info.return_value
        fake_info.sha = "fake_sha_456"

        fake_splits = DatasetDict({
            "train": Dataset.from_dict({"x": [1]}),
            "test": Dataset.from_dict({"x": [2]}),
        })
        mock_load_dataset.return_value = fake_splits

        # Fake HF download layout
        def _fake_snapshot_download(*args, **kwargs):
            local_dir = Path(kwargs["local_dir"])

            # Must create BOTH train and test because explicit mode demands it
            df_train = pd.DataFrame({"line_id": ["t1", "t2"], "x": [1, 2]})
            df_test = pd.DataFrame({"line_id": ["s1", "s2"], "x": [3, 4]})

            self._write_parquet(local_dir / "data/train/docA/train.parquet", df_train)
            self._write_parquet(local_dir / "data/test/docB/test.parquet", df_test)
            return str(local_dir)

        mock_snapshot_download.side_effect = _fake_snapshot_download

        handler = HuggingFaceDataHandler(
            dataset_name="my-org/my-dataset",
            huggingface_token="TOKEN123",
            revision="main",
            split=["train", "test"],  # EXPLICIT
        )

        handler.download_hf_dataset()

        # snapshot_download must only request train+test parquet
        sd_kwargs = mock_snapshot_download.call_args.kwargs
        self.assertIn("data/train/**/*.parquet", sd_kwargs["allow_patterns"])
        self.assertIn("data/test/**/*.parquet", sd_kwargs["allow_patterns"])
        self.assertNotIn("data/**/*.parquet", sd_kwargs["allow_patterns"])

        # load_dataset called correctly
        args, kwargs = mock_load_dataset.call_args
        self.assertEqual(args[0], "parquet")
        self.assertIn("train", kwargs["data_files"])
        self.assertIn("test", kwargs["data_files"])

        self.assertEqual(set(handler.dataset.keys()), {"train", "test"})
        self.assertIn("train", handler.parquet_paths)
        self.assertIn("test", handler.parquet_paths)

    @patch("flow_inference.data_handling.load_dataset")
    @patch("flow_inference.data_handling.snapshot_download")
    @patch("flow_inference.data_handling.HfApi")
    def test_download_success_with_token(self, mock_hfapi, mock_snapshot_download, mock_load_dataset):
        """
        Same as above, but uses self.handler with a token and verifies calls.
        """
        self.handler.huggingface_token = "hf_ABC123"
        self.handler.revision = "main"
        # IMPORTANT: set splits explicitly or use default; both are acceptable
        self.handler.requested_splits = {"train", "test", "default"}

        fake_info = mock_hfapi.return_value.dataset_info.return_value
        fake_info.sha = "fake_sha_456"

        fake_dataset = DatasetDict({
            "train": Dataset.from_dict({"x": [1]}),
            "test": Dataset.from_dict({"x": [2]}),
        })
        mock_load_dataset.return_value = fake_dataset

        def _fake_snapshot_download(*args, **kwargs):
            local_dir = Path(kwargs["local_dir"])
            df_train = pd.DataFrame({"line_id": ["t1"], "x": [1]})
            df_test = pd.DataFrame({"line_id": ["s1"], "x": [2]})
            self._write_parquet(local_dir / "data/train/docX/a.parquet", df_train)
            self._write_parquet(local_dir / "data/test/docY/b.parquet", df_test)
            return str(local_dir)

        mock_snapshot_download.side_effect = _fake_snapshot_download

        self.handler.download_hf_dataset()

        mock_hfapi.return_value.dataset_info.assert_called_once_with(
            repo_id="my-org/my-dataset",
            revision="main",
            token="hf_ABC123",
        )

        # Assert load_dataset called correctly
        args, kwargs = mock_load_dataset.call_args
        self.assertEqual(args[0], "parquet")
        self.assertIn("data_files", kwargs)
        self.assertIn("train", kwargs["data_files"])
        self.assertIn("test", kwargs["data_files"])

        self.assertEqual(self.handler.state, "downloaded_all")
        self.assertIsInstance(self.handler.dataset, DatasetDict)
        self.assertIn("train", self.handler.dataset)
        self.assertIn("test", self.handler.dataset)

    # -----------------------------------------------------------------------
    # INTEGRATION TEST — REAL DATASET
    # -----------------------------------------------------------------------
    def test_real_hf_dataset_download_and_convert(self):
        """
        Real integration: downloads actual dataset and converts to DataFrames.

        Skips if env vars are missing.
        """
        if not self.hf_token_read or not self.hf_download_repo_name:
            self.skipTest("Missing Hugging Face credentials.")

        handler = HuggingFaceDataHandler(
            dataset_name=self.hf_download_repo_name,
            huggingface_token=self.hf_token_read,
            split=None,  # auto (train/test/default as available)
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
    # UNIT TEST — PUSH TO HUB (structure-preserving commit)
    # -----------------------------------------------------------------------
    @patch("flow_inference.data_handling.HfApi")
    def test_push_to_hub(self, mock_hfapi):
        """
        Tests the new push_to_hub:
        - updates parquet files in-place based on key_col (line_id)
        - computes repo path relative to local snapshot root (preserves structure)
        - uses HfApi.create_commit with CommitOperationAdd operations
        """
        # Create a local snapshot tree
        tmp_root = Path(tempfile.mkdtemp(prefix="hf_test_snapshot_"))
        local_root = tmp_root / "snapshot"
        (local_root / "data/train/docA").mkdir(parents=True, exist_ok=True)
        (local_root / "data/test/docB").mkdir(parents=True, exist_ok=True)

        # Base parquet contents (before inference)
        train_path = local_root / "data/train/docA/train_file.parquet"
        test_path = local_root / "data/test/docB/test_file.parquet"

        df_train_orig = pd.DataFrame({
            "line_id": ["L1", "L2", "L3"],
            "text": ["", "", ""],
        })
        df_test_orig = pd.DataFrame({
            "line_id": ["T1", "T2", "T3"],
            "text": ["", "", ""],
        })

        self._write_parquet(train_path, df_train_orig)
        self._write_parquet(test_path, df_test_orig)

        # Handler prepared as if download_hf_dataset() had already run
        handler = HuggingFaceDataHandler(
            dataset_name="fake/repo",
            huggingface_token="WRITE_TOKEN",
            split=["train", "test"],
        )
        handler._local_root = local_root
        handler.parquet_paths = {
            "train": [str(train_path)],
            "test": [str(test_path)],
        }

        # Inference results in df (note: only some keys updated)
        handler.df = {
            "train": pd.DataFrame({
                "line_id": ["L2", "L3"],
                "text": ["hello", "world"],
                "inference_col": ["pred2", "pred3"],
            }),
            "test": pd.DataFrame({
                "line_id": ["T1"],
                "text": ["test-hi"],
                "inference_col": ["predT1"],
            }),
        }

        handler.push_to_hub(
            upload_repo_name="fake/upload",
            commit_message="Unit test commit",
            key_col="line_id",
        )

        # Verify parquet files were updated correctly in-place
        updated_train = pd.read_parquet(train_path).set_index("line_id")
        updated_test = pd.read_parquet(test_path).set_index("line_id")

        # L1 untouched, L2 & L3 updated
        self.assertEqual(updated_train.loc["L1", "text"], "")
        self.assertEqual(updated_train.loc["L2", "text"], "hello")
        self.assertEqual(updated_train.loc["L3", "text"], "world")
        self.assertEqual(updated_train.loc["L2", "inference_col"], "pred2")
        self.assertEqual(updated_train.loc["L3", "inference_col"], "pred3")

        # T2/T3 untouched, T1 updated
        self.assertEqual(updated_test.loc["T2", "text"], "")
        self.assertEqual(updated_test.loc["T1", "text"], "test-hi")
        self.assertEqual(updated_test.loc["T1", "inference_col"], "predT1")

        # Verify create_commit called with structure-preserving paths
        mock_hfapi.return_value.create_commit.assert_called_once()
        cc_kwargs = mock_hfapi.return_value.create_commit.call_args.kwargs

        self.assertEqual(cc_kwargs["repo_id"], "fake/upload")
        self.assertEqual(cc_kwargs["repo_type"], "dataset")
        self.assertEqual(cc_kwargs["commit_message"], "Unit test commit")
        self.assertEqual(cc_kwargs["token"], "WRITE_TOKEN")

        ops = cc_kwargs["operations"]
        # Extract path_in_repo from CommitOperationAdd objects
        paths_in_repo = sorted([op.path_in_repo for op in ops])

        self.assertIn("data/train/docA/train_file.parquet", paths_in_repo)
        self.assertIn("data/test/docB/test_file.parquet", paths_in_repo)

        self.assertEqual(handler.state, "pushed")


if __name__ == "__main__":
    unittest.main()
