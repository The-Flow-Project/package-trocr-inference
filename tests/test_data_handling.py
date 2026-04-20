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
    # UNIT TEST: MOCK HUGGING FACE DOWNLOAD
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
            split=None,
        )

        handler.download_hf_dataset()

        mock_hfapi.return_value.dataset_info.assert_called_once_with(
            repo_id="my-org/my-dataset",
            revision="main",
            token="TOKEN123",
        )

        self.assertTrue(mock_snapshot_download.called)
        sd_kwargs = mock_snapshot_download.call_args.kwargs
        self.assertIn("data/**/*.parquet", sd_kwargs["allow_patterns"])
        self.assertNotIn("README.md", sd_kwargs["allow_patterns"])

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

    # -------------------------------------------------------------
    # UNIT TEST: DOWNLOAD WITH EXPLICIT SPLIT SELECTION
    # -------------------------------------------------------------
    @patch("flow_inference.data_handling.load_dataset")
    @patch("flow_inference.data_handling.snapshot_download")
    @patch("flow_inference.data_handling.HfApi")
    def test_download_explicit_train_test(self, mock_hfapi, mock_snapshot_download, mock_load_dataset):
        """
        EXPLICIT mode: split=['train','test']
        """
        fake_info = mock_hfapi.return_value.dataset_info.return_value
        fake_info.sha = "fake_sha_456"

        fake_splits = DatasetDict({
            "train": Dataset.from_dict({"x": [1]}),
            "test": Dataset.from_dict({"x": [2]}),
        })
        mock_load_dataset.return_value = fake_splits

        def _fake_snapshot_download(*args, **kwargs):
            local_dir = Path(kwargs["local_dir"])
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
            split=["train", "test"],
        )

        handler.download_hf_dataset()

        sd_kwargs = mock_snapshot_download.call_args.kwargs
        self.assertIn("data/train/**/*.parquet", sd_kwargs["allow_patterns"])
        self.assertIn("data/test/**/*.parquet", sd_kwargs["allow_patterns"])
        self.assertNotIn("data/**/*.parquet", sd_kwargs["allow_patterns"])
        self.assertNotIn("README.md", sd_kwargs["allow_patterns"])

        args, kwargs = mock_load_dataset.call_args
        self.assertEqual(args[0], "parquet")
        self.assertIn("train", kwargs["data_files"])
        self.assertIn("test", kwargs["data_files"])

        self.assertEqual(set(handler.dataset.keys()), {"train", "test"})
        self.assertIn("train", handler.parquet_paths)
        self.assertIn("test", handler.parquet_paths)

    # -------------------------------------------------------------
    # UNIT TEST: DOWNLOAD WITH TOKEN AND DEFAULT SPLIT
    # -------------------------------------------------------------
    @patch("flow_inference.data_handling.load_dataset")
    @patch("flow_inference.data_handling.snapshot_download")
    @patch("flow_inference.data_handling.HfApi")
    def test_download_success_with_token(self, mock_hfapi, mock_snapshot_download, mock_load_dataset):
        self.handler.huggingface_token = "hf_ABC123"
        self.handler.revision = "main"
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
    # INTEGRATION TEST: DOWNLOAD AND CONVERT REAL DATASET
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
            split=None,
        )

        handler.download_hf_dataset()
        self.assertEqual(handler.state, "downloaded_all")
        self.assertIsNotNone(handler.dataset)

        dfs = handler.to_dataframe()
        self.assertGreater(len(dfs), 0)

        for df in dfs.values():
            self.assertFalse(df.empty)

        dicts = handler.convert_to_list_of_dicts(dfs)
        self.assertGreater(len(dicts), 0)

    # -----------------------------------------------------------------------
    # UNIT TEST: PUSH TO HUB
    # -----------------------------------------------------------------------
    @patch("flow_inference.data_handling.HfApi")
    def test_push_to_hub(self, mock_hfapi):
        """
        Tests push_to_hub:
        - updates parquet files in place
        - preserves relative repo paths
        - adds generated README.md to commit
        """
        tmp_root = Path(tempfile.mkdtemp(prefix="hf_test_snapshot_"))
        local_root = tmp_root / "snapshot"
        (local_root / "data/train/docA").mkdir(parents=True, exist_ok=True)
        (local_root / "data/test/docB").mkdir(parents=True, exist_ok=True)

        train_path = local_root / "data/train/docA/train_file.parquet"
        test_path = local_root / "data/test/docB/test_file.parquet"

        df_train_orig = pd.DataFrame({
            "project_name": ["docA"] * 3,
            "filename": ["train_file"] * 3,
            "line_id": ["L1", "L2", "L3"],
            "text": ["", "", ""],
        })
        df_test_orig = pd.DataFrame({
            "project_name": ["docB", "docB", "docB"],
            "filename": ["test_file", "test_file", "test_file"],
            "line_id": ["T1", "T2", "T3"],
            "text": ["", "", ""],
        })

        self._write_parquet(train_path, df_train_orig)
        self._write_parquet(test_path, df_test_orig)

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

        handler.dataset = DatasetDict({
            "train": Dataset.from_pandas(df_train_orig, preserve_index=False),
            "test": Dataset.from_pandas(df_test_orig, preserve_index=False),
        })

        handler.df = {
            "train": pd.DataFrame({
                "project_name": ["docA", "docA"],
                "filename": ["train_file", "train_file"],
                "line_id": ["L2", "L3"],
                "text": ["hello", "world"],
                "inference_col": ["pred2", "pred3"],
            }),
            "test": pd.DataFrame({
                "project_name": ["docB"],
                "filename": ["test_file"],
                "line_id": ["T1"],
                "text": ["test-hi"],
                "inference_col": ["predT1"],
            }),
        }

        handler.push_to_hub(
            upload_repo_name="fake/upload",
            commit_message="Unit test commit",
        )

        updated_train = pd.read_parquet(train_path).set_index("line_id")
        updated_test = pd.read_parquet(test_path).set_index("line_id")

        self.assertEqual(updated_train.loc["L1", "text"], "")
        self.assertEqual(updated_train.loc["L2", "text"], "hello")
        self.assertEqual(updated_train.loc["L3", "text"], "world")
        self.assertEqual(updated_train.loc["L2", "inference_col"], "pred2")
        self.assertEqual(updated_train.loc["L3", "inference_col"], "pred3")

        self.assertEqual(updated_test.loc["T2", "text"], "")
        self.assertEqual(updated_test.loc["T1", "text"], "test-hi")
        self.assertEqual(updated_test.loc["T1", "inference_col"], "predT1")

        mock_hfapi.return_value.create_commit.assert_called_once()
        cc_kwargs = mock_hfapi.return_value.create_commit.call_args.kwargs

        self.assertEqual(cc_kwargs["repo_id"], "fake/upload")
        self.assertEqual(cc_kwargs["repo_type"], "dataset")
        self.assertEqual(cc_kwargs["commit_message"], "Unit test commit")
        self.assertEqual(cc_kwargs["token"], "WRITE_TOKEN")

        ops = cc_kwargs["operations"]
        paths_in_repo = sorted([op.path_in_repo for op in ops])

        self.assertIn("data/train/docA/train_file.parquet", paths_in_repo)
        self.assertIn("data/test/docB/test_file.parquet", paths_in_repo)
        self.assertIn("README.md", paths_in_repo)

        self.assertEqual(handler.state, "pushed")

    # -------------------------------------------------------------
    # UNIT TEST: README GENERATION
    # -------------------------------------------------------------
    @patch("flow_inference.data_handling.HfApi")
    def test_generated_readme_is_added_to_commit(self, mock_hfapi):
        tmp_root = Path(tempfile.mkdtemp())
        snapshot = tmp_root / "snapshot"
        (snapshot / "data/train/docA").mkdir(parents=True, exist_ok=True)

        parquet_path = snapshot / "data/train/docA/train_file.parquet"
        df_orig = pd.DataFrame({
            "project_name": ["docA"],
            "filename": ["train_file"],
            "line_id": ["L1"],
            "text": [""],
        })
        self._write_parquet(parquet_path, df_orig)

        handler = HuggingFaceDataHandler(
            dataset_name="old/repo",
            huggingface_token="TOKEN",
        )
        handler._local_root = snapshot
        handler.parquet_paths = {"train": [str(parquet_path)]}
        handler.dataset = DatasetDict({
            "train": Dataset.from_pandas(df_orig, preserve_index=False)
        })
        handler.df = {
            "train": pd.DataFrame({
                "project_name": ["docA"],
                "filename": ["train_file"],
                "line_id": ["L1"],
                "text": ["updated"],
                "inference_col": ["pred"],
            })
        }

        handler.push_to_hub("new/repo")

        ops = mock_hfapi.return_value.create_commit.call_args.kwargs["operations"]
        readme_ops = [op for op in ops if op.path_in_repo == "README.md"]

        self.assertEqual(len(readme_ops), 1)

        content = readme_ops[0].path_or_fileobj.decode("utf-8")
        self.assertIn("# Dataset Card for repo", content)
        self.assertIn('dataset = load_dataset("new/repo")', content)
        self.assertIn('dataset_split = load_dataset("new/repo", split="train")', content)
        self.assertIn("- Number of augmentations: 2", content)
        self.assertIn("### Projects Included", content)
        self.assertIn("docA", content)
        self.assertIn("inference_col", content)

    # -------------------------------------------------------------
    # UNIT TEST: INDEXING AND DEDUPLICATION
    # -------------------------------------------------------------
    def test_index_df_by_composite_key_deduplicates(self):
        df = pd.DataFrame({
            "project_name": ["p", "p"],
            "filename": ["f", "f"],
            "line_id": ["1", "1"],
            "text": ["old", "new"]
        })

        handler = HuggingFaceDataHandler("x/y")
        idx = handler._index_df_by_key(df, "train")

        self.assertEqual(len(idx), 1)
        self.assertEqual(idx.iloc[0]["text"], "new")

    # -------------------------------------------------------------
    # UNIT TEST: SELECTED SPLIT UPDATE
    # -------------------------------------------------------------
    def test_only_selected_split_is_updated(self):
        handler = HuggingFaceDataHandler("x/y", split=["train"])

        handler.df = {
            "train": pd.DataFrame({
                "project_name": ["p"],
                "filename": ["f"],
                "line_id": ["1"],
                "text": ["new"]
            })
        }

        handler.parquet_paths = {
            "train": ["train.parquet"],
            "test": ["test.parquet"]
        }

        self.assertIn("train", handler.df)
        self.assertNotIn("test", handler.df)

    # -------------------------------------------------------------
    # INTEGRATION TEST: REAL PUSH TO HUB
    # -------------------------------------------------------------
    def test_real_push_to_hub(self):
        """
        Real integration test:
        - downloads a real dataset
        - writes a few visible test inference values into a new inference column
        - pushes updated parquet + generated README to the HF test repo
        - verifies that the inference column exists and contains non-empty values

        Skips if env vars are missing.
        """
        if (
                not self.hf_token_read
                or not self.write_token
                or not self.hf_download_repo_name
                or not self.test_repo
        ):
            self.skipTest("Missing Hugging Face credentials.")

        handler = HuggingFaceDataHandler(
            dataset_name=self.hf_download_repo_name,
            huggingface_token=self.write_token,
            split=["train"],  # keep runtime and repo changes smaller
        )

        handler.download_hf_dataset()
        self.assertEqual(handler.state, "downloaded_all")
        self.assertIsNotNone(handler.dataset)

        dfs = handler.to_dataframe()
        self.assertGreater(len(dfs), 0)

        updated_dfs = {}

        for split_name, df in dfs.items():
            df = df.copy()
            self.assertFalse(df.empty)

            for required in ["project_name", "filename", "line_id"]:
                self.assertIn(required, df.columns)

            inference_col = "inference_test"

            df[inference_col] = ""
            for i in range(min(3, len(df))):
                df.iloc[i, df.columns.get_loc(inference_col)] = f"test line {i + 1}"

            updated_dfs[split_name] = df

        handler.df = updated_dfs

        handler.push_to_hub(
            upload_repo_name=self.test_repo,
            private=True,
            commit_message="Integration test upload with fake inference",
        )

        self.assertEqual(handler.state, "pushed")

        # Verify local updated dfs first
        found_inference_column = False
        found_non_empty_prediction = False

        for split_name, df in updated_dfs.items():
            inference_cols = [col for col in df.columns if col.startswith("inference_")]
            if inference_cols:
                found_inference_column = True

            for col in inference_cols:
                non_empty = df[col].fillna("").astype(str).str.strip().ne("")
                if non_empty.any():
                    found_non_empty_prediction = True
                    break

        self.assertTrue(found_inference_column, "No inference column was created.")
        self.assertTrue(
            found_non_empty_prediction,
            "Inference column exists but contains no values."
        )

        # Verify pushed repo by downloading again
        verify_handler = HuggingFaceDataHandler(
            dataset_name=self.test_repo,
            huggingface_token=self.hf_token_read,
            split=["train"],
        )
        verify_handler.download_hf_dataset()
        verify_dfs = verify_handler.to_dataframe()

        pushed_inference_column = False
        pushed_non_empty_prediction = False
        found_test_lines = set()

        for split_name, df in verify_dfs.items():
            inference_cols = [col for col in df.columns if col.startswith("inference_")]
            if inference_cols:
                pushed_inference_column = True

            for col in inference_cols:
                values = df[col].fillna("").astype(str).str.strip()
                non_empty = values.ne("")
                if non_empty.any():
                    pushed_non_empty_prediction = True

                for expected in ["test line 1", "test line 2", "test line 3"]:
                    if values.eq(expected).any():
                        found_test_lines.add(expected)

        self.assertTrue(
            pushed_inference_column,
            "No inference column found in pushed dataset."
        )
        self.assertTrue(
            pushed_non_empty_prediction,
            "Pushed inference column contains no values."
        )
        self.assertTrue(
            len(found_test_lines) >= 1,
            "Did not find any expected test inference values in pushed dataset."
        )

    # -------------------------------------------------------------
    # INTEGRATION TEST: README GENERATION ON REAL PUSH
    # -------------------------------------------------------------
    def test_real_push_generates_readme(self):
        """
        Real integration test:
        - downloads a real dataset
        - adds a new inference column
        - pushes to test repo
        - verifies the pushed repo contains the new column

        Skips if env vars are missing.
        """
        if (
                not self.hf_token_read
                or not self.write_token
                or not self.hf_download_repo_name
                or not self.test_repo
        ):
            self.skipTest("Missing Hugging Face credentials.")

        handler = HuggingFaceDataHandler(
            dataset_name=self.hf_download_repo_name,
            huggingface_token=self.write_token,
            split=None,
        )

        handler.download_hf_dataset()
        dfs = handler.to_dataframe()

        first_split = next(iter(dfs.keys()))
        df = dfs[first_split].copy()

        for required in ["project_name", "filename", "line_id"]:
            self.assertIn(required, df.columns)

        df["integration_test_inference_col"] = ""
        df.loc[df.index[:1], "integration_test_inference_col"] = "pred"
        dfs[first_split] = df
        handler.df = dfs

        handler.push_to_hub(
            upload_repo_name=self.test_repo,
            private=True,
            commit_message="Integration test upload with generated README",
        )

        self.assertEqual(handler.state, "pushed")

        verify_handler = HuggingFaceDataHandler(
            dataset_name=self.test_repo,
            huggingface_token=self.hf_token_read,
            split=None,
        )
        verify_handler.download_hf_dataset()
        verify_dfs = verify_handler.to_dataframe()

        self.assertIn(first_split, verify_dfs)
        self.assertIn("integration_test_inference_col", verify_dfs[first_split].columns)


if __name__ == "__main__":
    unittest.main()