import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
import pandas as pd
from datasets import DatasetDict, Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

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

    def _unique_test_repo_name(self, suffix: str) -> str:
        if not self.test_repo:
            self.skipTest("Missing HUGGINGFACE_TEST_UPLOAD_REPO_NAME.")

        namespace = self.test_repo.rsplit("/", 1)[0]
        safe_suffix = suffix.replace("_", "-")
        return f"{namespace}/test-upload-{safe_suffix}-{uuid4().hex[:8]}"

    def _prepared_real_handler_with_inference(
        self,
        inference_col: str,
        inference_value: str,
        split: str = "train",
    ) -> HuggingFaceDataHandler:
        if not self.hf_token_read or not self.write_token or not self.hf_download_repo_name:
            self.skipTest("Missing Hugging Face credentials.")

        handler = HuggingFaceDataHandler(
            dataset_name=self.hf_download_repo_name,
            huggingface_token=self.write_token,
            split=[split],
        )

        handler.download_hf_dataset()
        self.assertEqual(handler.state, "downloaded_all")
        self.assertIsNotNone(handler.dataset)

        dfs = handler.to_dataframe()
        self.assertIn(split, dfs)

        df = dfs[split].copy()
        self.assertFalse(df.empty)

        for required in ["filename", "region_id", "line_id"]:
            self.assertIn(required, df.columns)

        df[inference_col] = ""
        df.loc[df.index[:1], inference_col] = inference_value

        dfs[split] = df
        handler.df = dfs

        return handler

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
            df_train = pd.DataFrame({
                "filename": ["f", "f"],
                "region_id": ["r", "r"],
                "line_id": ["a", "b"],
                "x": [1, 2],
            })
            df_test = pd.DataFrame({
                "filename": ["f", "f"],
                "region_id": ["r", "r"],
                "line_id": ["c", "d"],
                "x": [3, 4],
            })

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
            df_train = pd.DataFrame({
                "filename": ["f", "f"],
                "region_id": ["r", "r"],
                "line_id": ["t1", "t2"],
                "x": [1, 2],
            })
            df_test = pd.DataFrame({
                "filename": ["f", "f"],
                "region_id": ["r", "r"],
                "line_id": ["s1", "s2"],
                "x": [3, 4],
            })

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
            df_train = pd.DataFrame({
                "filename": ["f"],
                "region_id": ["r"],
                "line_id": ["t1"],
                "x": [1],
            })
            df_test = pd.DataFrame({
                "filename": ["f"],
                "region_id": ["r"],
                "line_id": ["s1"],
                "x": [2],
            })
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
    @patch("flow_inference.data_handling.HuggingFaceDataHandler._repo_exists", return_value=False)
    @patch("flow_inference.data_handling.HfApi")
    def test_push_to_hub(self, mock_hfapi, mock_repo_exists):
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
            "region_id": ["R1", "R1", "R1"],
            "line_id": ["L1", "L2", "L3"],
            "text": ["", "", ""],
        })
        df_test_orig = pd.DataFrame({
            "project_name": ["docB", "docB", "docB"],
            "filename": ["test_file", "test_file", "test_file"],
            "region_id": ["R1", "R1", "R1"],
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
                "region_id": ["R1", "R1"],
                "line_id": ["L2", "L3"],
                "text": ["hello", "world"],
                "inference_col": ["pred2", "pred3"],
            }),
            "test": pd.DataFrame({
                "project_name": ["docB"],
                "filename": ["test_file"],
                "region_id": ["R1"],
                "line_id": ["T1"],
                "text": ["test-hi"],
                "inference_col": ["predT1"],
            }),
        }

        handler.push_to_hub(
            upload_repo_name="fake/upload",
            commit_message="Unit test commit",
        )

        mock_hfapi.return_value.create_repo.assert_called_once()

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
    @patch("flow_inference.data_handling.HuggingFaceDataHandler._repo_exists", return_value=False)
    @patch("flow_inference.data_handling.HfApi")
    def test_generated_readme_is_added_to_commit(self, mock_hfapi, mock_repo_exists):
        tmp_root = Path(tempfile.mkdtemp())
        snapshot = tmp_root / "snapshot"
        (snapshot / "data/train/docA").mkdir(parents=True, exist_ok=True)

        parquet_path = snapshot / "data/train/docA/train_file.parquet"
        df_orig = pd.DataFrame({
            "project_name": ["docA"],
            "filename": ["train_file"],
            "region_id": ["R1"],
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
                "region_id": ["R1"],
                "line_id": ["L1"],
                "text": ["updated"],
                "inference_col": ["pred"],
            })
        }

        handler.push_to_hub("new/repo")

        mock_hfapi.return_value.create_repo.assert_called_once()

        ops = mock_hfapi.return_value.create_commit.call_args.kwargs["operations"]
        readme_ops = [op for op in ops if op.path_in_repo == "README.md"]

        self.assertEqual(len(readme_ops), 1)

        content = readme_ops[0].path_or_fileobj.decode("utf-8")
        self.assertIn("# Dataset Card for repo", content)
        self.assertIn('dataset = load_dataset("new/repo")', content)
        self.assertIn('dataset_split = load_dataset("new/repo", split="train")', content)
        self.assertIn("### Projects Included", content)
        self.assertIn("docA", content)
        self.assertIn("inference_col", content)

    # -------------------------------------------------------------
    # UNIT TEST: INDEXING AND DEDUPLICATION
    # -------------------------------------------------------------
    def test_index_df_by_composite_key_preserves_duplicate_rows_with_occurrence_index(self):
        df = pd.DataFrame({
            "project_name": ["p", "p"],
            "filename": ["f", "f"],
            "region_id": ["r", "r"],
            "line_id": ["1", "1"],
            "text": ["old", "new"],
        })

        idx = HuggingFaceDataHandler._index_df_by_key(df, "train")

        self.assertEqual(len(idx), 2)
        self.assertTrue(idx.index.is_unique)

        self.assertIn(("p", "f", "r", "1", 0), idx.index)
        self.assertIn(("p", "f", "r", "1", 1), idx.index)

        self.assertEqual(idx.loc[("p", "f", "r", "1", 0), "text"], "old")
        self.assertEqual(idx.loc[("p", "f", "r", "1", 1), "text"], "new")

    def test_index_df_by_key_works_without_project_name(self):
        df = pd.DataFrame({
            "filename": ["f", "f"],
            "region_id": ["r", "r"],
            "line_id": ["1", "1"],
            "text": ["old", "new"],
        })

        idx = HuggingFaceDataHandler._index_df_by_key(df, "train")

        self.assertEqual(len(idx), 2)
        self.assertTrue(idx.index.is_unique)

        self.assertIn(("f", "r", "1", 0), idx.index)
        self.assertIn(("f", "r", "1", 1), idx.index)

        self.assertEqual(idx.loc[("f", "r", "1", 0), "text"], "old")
        self.assertEqual(idx.loc[("f", "r", "1", 1), "text"], "new")

    def test_index_df_by_key_ignores_empty_project_name_column(self):
        df = pd.DataFrame({
            "project_name": ["", ""],
            "filename": ["f", "f"],
            "region_id": ["r", "r"],
            "line_id": ["1", "1"],
            "text": ["old", "new"],
        })

        idx = HuggingFaceDataHandler._index_df_by_key(df, "train")

        self.assertEqual(len(idx), 2)
        self.assertTrue(idx.index.is_unique)

        self.assertIn(("f", "r", "1", 0), idx.index)
        self.assertIn(("f", "r", "1", 1), idx.index)

    def test_index_df_by_composite_key_keeps_augmented_rows_distinct(self):
        df = pd.DataFrame({
            "project_name": ["p", "p"],
            "filename": ["f", "f"],
            "region_id": ["r", "r"],
            "line_id": ["1", "1"],
            "line_augmentation": ["original", "rotation"],
            "text": ["orig text", "aug text"],
        })

        handler = HuggingFaceDataHandler("x/y")
        idx = handler._index_df_by_key(df, "train")

        self.assertEqual(len(idx), 2)

        self.assertIn(("p", "f", "r", "1", "original", 0), idx.index)
        self.assertIn(("p", "f", "r", "1", "rotation", 0), idx.index)

        self.assertEqual(
            idx.loc[("p", "f", "r", "1", "original", 0), "text"],
            "orig text",
        )
        self.assertEqual(
            idx.loc[("p", "f", "r", "1", "rotation", 0), "text"],
            "aug text",
        )

    def test_update_parquet_file_preserves_augmented_rows(self):
        tmp_root = Path(tempfile.mkdtemp(prefix="hf_augmented_update_"))
        parquet_path = tmp_root / "data/train/docA/train_file.parquet"

        parquet_df = pd.DataFrame({
            "project_name": ["docA", "docA"],
            "filename": ["train_file", "train_file"],
            "region_id": ["R1", "R1"],
            "line_id": ["L1", "L1"],
            "line_augmentation": ["original", "rotation"],
            "text": ["", ""],
        })

        self._write_parquet(parquet_path, parquet_df)

        split_df = pd.DataFrame({
            "project_name": ["docA", "docA"],
            "filename": ["train_file", "train_file"],
            "region_id": ["R1", "R1"],
            "line_id": ["L1", "L1"],
            "line_augmentation": ["original", "rotation"],
            "text": ["original updated", ""],
            "inference_col": ["pred original", ""],
        })

        split_idx = HuggingFaceDataHandler._index_df_by_key(split_df, "train")
        HuggingFaceDataHandler._update_parquet_file(parquet_path, split_idx)

        updated = pd.read_parquet(parquet_path)

        self.assertEqual(len(updated), 2)

        original = updated[updated["line_augmentation"] == "original"].iloc[0]
        rotation = updated[updated["line_augmentation"] == "rotation"].iloc[0]

        self.assertEqual(original["text"], "original updated")
        self.assertEqual(original["inference_col"], "pred original")

        self.assertEqual(rotation["text"], "")
        self.assertEqual(rotation["inference_col"], "")

    # -------------------------------------------------------------
    # UNIT TEST: SELECTED SPLIT UPDATE
    # -------------------------------------------------------------
    def test_only_selected_split_is_updated(self):
        handler = HuggingFaceDataHandler("x/y", split=["train"])

        handler.df = {
            "train": pd.DataFrame({
                "project_name": ["p"],
                "filename": ["f"],
                "region_id": ["r"],
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
    def test_real_push_new_repo_creates_repo_and_uploads_dataset(self):
        """
        Real integration:
        - downloads source dataset
        - creates a brand-new target repo
        - uploads current parquet files + README
        - verifies pushed repo can be downloaded and contains the new inference column
        - never uploads to the source repo
        """
        target_repo = self._unique_test_repo_name("new_repo")

        handler = self._prepared_real_handler_with_inference(
            inference_col="inference_test_new_repo",
            inference_value="new repo pred",
            split="train",
        )

        self.assertNotEqual(target_repo, self.hf_download_repo_name)

        handler.push_to_hub(
            upload_repo_name=target_repo,
            private=True,
            commit_message="Integration test: new repo upload",
            upload_mode="new_repo",
        )

        self.assertEqual(handler.state, "pushed")

        verify_handler = HuggingFaceDataHandler(
            dataset_name=target_repo,
            huggingface_token=self.hf_token_read,
            split=["train"],
        )
        verify_handler.download_hf_dataset()
        verify_dfs = verify_handler.to_dataframe()

        self.assertIn("train", verify_dfs)
        self.assertIn("inference_test_new_repo", verify_dfs["train"].columns)

        values = verify_dfs["train"]["inference_test_new_repo"].fillna("").astype(str)
        self.assertTrue(values.eq("new repo pred").any())

    # -------------------------------------------------------------
    # INTEGRATION TEST: README GENERATION ON REAL PUSH
    # -------------------------------------------------------------
    def test_real_push_replace_existing_repo_replaces_contents_and_generates_readme(self):
        """
        Real integration:
        - creates a target repo with an initial upload
        - runs a second upload with upload_mode='replace'
        - verifies the replacement inference column exists
        - verifies README is generated
        - never uploads to the source repo
        """
        target_repo = self._unique_test_repo_name("replace")

        first_handler = self._prepared_real_handler_with_inference(
            inference_col="inference_replace_first",
            inference_value="first pred",
            split="train",
        )

        self.assertNotEqual(target_repo, self.hf_download_repo_name)

        first_handler.push_to_hub(
            upload_repo_name=target_repo,
            private=True,
            commit_message="Integration test: initial upload before replace",
            upload_mode="new_repo",
        )

        second_handler = self._prepared_real_handler_with_inference(
            inference_col="inference_replace_second",
            inference_value="second pred",
            split="train",
        )

        second_handler.push_to_hub(
            upload_repo_name=target_repo,
            private=True,
            commit_message="Integration test: replace upload",
            upload_mode="replace",
        )

        self.assertEqual(second_handler.state, "pushed")

        verify_handler = HuggingFaceDataHandler(
            dataset_name=target_repo,
            huggingface_token=self.hf_token_read,
            split=["train"],
        )
        verify_handler.download_hf_dataset()
        verify_dfs = verify_handler.to_dataframe()

        self.assertIn("train", verify_dfs)
        self.assertIn("inference_replace_second", verify_dfs["train"].columns)

        values = verify_dfs["train"]["inference_replace_second"].fillna("").astype(str)
        self.assertTrue(values.eq("second pred").any())

        files = HfApi().list_repo_files(
            repo_id=target_repo,
            repo_type="dataset",
            token=self.hf_token_read,
        )
        self.assertIn("README.md", files)

    def test_real_push_update_existing_repo_preserves_existing_inference_columns(self):
        """
        Real integration:
        - creates a fresh target repo
        - uploads first inference column using new_repo
        - uploads second inference column using update
        - verifies both inference columns exist afterward
        - never uploads to the source repo
        """
        target_repo = self._unique_test_repo_name("update")

        first_handler = self._prepared_real_handler_with_inference(
            inference_col="inference_update_first",
            inference_value="first update pred",
            split="train",
        )

        first_handler.push_to_hub(
            upload_repo_name=target_repo,
            private=True,
            commit_message="Integration test: initial upload before update",
            upload_mode="new_repo",
        )

        time.sleep(2)

        second_handler = self._prepared_real_handler_with_inference(
            inference_col="inference_update_second",
            inference_value="second update pred",
            split="train",
        )

        self.assertNotEqual(target_repo, self.hf_download_repo_name)

        second_handler.push_to_hub(
            upload_repo_name=target_repo,
            private=True,
            commit_message="Integration test: additive update upload",
            upload_mode="update",
        )

        self.assertEqual(second_handler.state, "pushed")

        verify_handler = HuggingFaceDataHandler(
            dataset_name=target_repo,
            huggingface_token=self.hf_token_read,
            split=["train"],
        )
        verify_handler.download_hf_dataset()
        verify_dfs = verify_handler.to_dataframe()

        df = verify_dfs["train"]

        self.assertIn("inference_update_first", df.columns)
        self.assertIn("inference_update_second", df.columns)

        first_values = df["inference_update_first"].fillna("").astype(str)
        second_values = df["inference_update_second"].fillna("").astype(str)

        self.assertTrue(first_values.eq("first update pred").any())
        self.assertTrue(second_values.eq("second update pred").any())

    def test_real_push_new_repo_refuses_existing_target_repo(self):
        """
        Real integration:
        - creates a fresh target repo
        - tries to upload to the same target again with upload_mode='new_repo'
        - expects refusal
        - never uploads to the source repo
        """
        target_repo = self._unique_test_repo_name("new_repo_refusal")

        first_handler = self._prepared_real_handler_with_inference(
            inference_col="inference_new_repo_refusal_first",
            inference_value="first pred",
            split="train",
        )

        first_handler.push_to_hub(
            upload_repo_name=target_repo,
            private=True,
            commit_message="Integration test: create repo before new_repo refusal",
            upload_mode="new_repo",
        )

        self.assertNotEqual(target_repo, self.hf_download_repo_name)

        second_handler = self._prepared_real_handler_with_inference(
            inference_col="inference_new_repo_refusal_second",
            inference_value="second pred",
            split="train",
        )

        with self.assertRaisesRegex(RuntimeError, "already exists"):
            second_handler.push_to_hub(
                upload_repo_name=target_repo,
                private=True,
                commit_message="Integration test: should refuse existing repo",
                upload_mode="new_repo",
            )

    def test_push_to_hub_refuses_source_repo_update_by_default(self):
        tmp_root = Path(tempfile.mkdtemp(prefix="hf_source_guard_"))
        local_root = tmp_root / "snapshot"
        parquet_path = local_root / "data/train/docA/train_file.parquet"

        df = pd.DataFrame({
            "project_name": ["docA"],
            "filename": ["train_file"],
            "region_id": ["R1"],
            "line_id": ["L1"],
            "text": ["hello"],
        })
        self._write_parquet(parquet_path, df)

        handler = HuggingFaceDataHandler(
            dataset_name="same/source-repo",
            huggingface_token="TOKEN",
            split=["train"],
        )
        handler._local_root = local_root
        handler.parquet_paths = {"train": [str(parquet_path)]}
        handler.dataset = DatasetDict({
            "train": Dataset.from_pandas(df, preserve_index=False),
        })
        handler.df = {"train": df.copy()}

        with self.assertRaisesRegex(RuntimeError, "Refusing to upload into the source dataset repo"):
            handler.push_to_hub(
                upload_repo_name="same/source-repo",
                upload_mode="replace",
                allow_source_repo_update=False,
            )

    @patch("flow_inference.data_handling.HfApi")
    def test_push_new_repo_refuses_existing_repo_mocked(self, mock_hfapi):
        mock_hfapi.return_value.dataset_info.return_value.sha = "existing_sha"

        tmp_root = Path(tempfile.mkdtemp(prefix="hf_new_repo_refuse_"))
        local_root = tmp_root / "snapshot"
        parquet_path = local_root / "data/train/docA/train_file.parquet"

        df = pd.DataFrame({
            "project_name": ["docA"],
            "filename": ["train_file"],
            "region_id": ["R1"],
            "line_id": ["L1"],
            "text": ["hello"],
        })
        self._write_parquet(parquet_path, df)

        handler = HuggingFaceDataHandler("source/repo", "TOKEN", split=["train"])
        handler._local_root = local_root
        handler.parquet_paths = {"train": [str(parquet_path)]}
        handler.dataset = DatasetDict({
            "train": Dataset.from_pandas(df, preserve_index=False),
        })
        handler.df = {"train": df.copy()}

        with self.assertRaisesRegex(RuntimeError, "already exists"):
            handler.push_to_hub(
                upload_repo_name="target/existing",
                upload_mode="new_repo",
            )

        mock_hfapi.return_value.create_commit.assert_not_called()

    @patch("flow_inference.data_handling.HuggingFaceDataHandler._repo_exists", return_value=False)
    @patch("flow_inference.data_handling.HfApi")
    def test_push_update_refuses_missing_target_mocked(self, mock_hfapi, mock_repo_exists):
        tmp_root = Path(tempfile.mkdtemp(prefix="hf_update_missing_"))
        local_root = tmp_root / "snapshot"
        parquet_path = local_root / "data/train/docA/train_file.parquet"

        df = pd.DataFrame({
            "project_name": ["docA"],
            "filename": ["train_file"],
            "region_id": ["R1"],
            "line_id": ["L1"],
            "text": ["hello"],
        })
        self._write_parquet(parquet_path, df)

        handler = HuggingFaceDataHandler("source/repo", "TOKEN", split=["train"])
        handler._local_root = local_root
        handler.parquet_paths = {"train": [str(parquet_path)]}
        handler.dataset = DatasetDict({
            "train": Dataset.from_pandas(df, preserve_index=False),
        })
        handler.df = {"train": df.copy()}

        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            handler.push_to_hub(
                upload_repo_name="target/missing",
                upload_mode="update",
            )

        mock_hfapi.return_value.create_commit.assert_not_called()

    def test_count_real_duplicate_lines_without_augmentation(self):
        df = pd.DataFrame({
            "project_name": ["p", "p", "p", "p"],
            "filename": ["f", "f", "f", "f"],
            "region_id": ["r", "r", "r", "r"],
            "line_id": ["1", "1", "2", "3"],
            "text": ["a", "b", "c", "d"],
        })

        counts = HuggingFaceDataHandler.count_real_duplicate_lines_by_split({
            "train": df
        })

        self.assertEqual(counts["train"]["duplicate_rows"], 2)
        self.assertEqual(counts["train"]["duplicate_groups"], 1)
        self.assertEqual(counts["train"]["duplicate_excess_rows"], 1)

    def test_count_real_duplicate_lines_with_augmentation_only_originals(self):
        df = pd.DataFrame({
            "project_name": ["p", "p", "p", "p", "p"],
            "filename": ["f", "f", "f", "f", "f"],
            "region_id": ["r", "r", "r", "r", "r"],
            "line_id": ["1", "1", "1", "2", "2"],
            "line_augmentation": [
                "original",
                "original",
                '{"rotation": 1}',
                '{"rotation": 1}',
                '{"rotation": 1}',
            ],
            "text": ["orig a", "orig b", "aug a", "aug b", "aug c"],
        })

        counts = HuggingFaceDataHandler.count_real_duplicate_lines_by_split({
            "train": df
        })

        self.assertEqual(counts["train"]["duplicate_rows"], 2)
        self.assertEqual(counts["train"]["duplicate_groups"], 1)
        self.assertEqual(counts["train"]["duplicate_excess_rows"], 1)

    def test_count_real_duplicate_lines_ignores_augmented_duplicates(self):
        df = pd.DataFrame({
            "project_name": ["p", "p", "p"],
            "filename": ["f", "f", "f"],
            "region_id": ["r", "r", "r"],
            "line_id": ["1", "1", "1"],
            "line_augmentation": [
                "original",
                '{"rotation": 1}',
                '{"rotation": 1}',
            ],
            "text": ["orig", "aug a", "aug b"],
        })

        counts = HuggingFaceDataHandler.count_real_duplicate_lines_by_split({
            "train": df
        })

        self.assertEqual(counts["train"]["duplicate_rows"], 0)
        self.assertEqual(counts["train"]["duplicate_groups"], 0)
        self.assertEqual(counts["train"]["duplicate_excess_rows"], 0)

    def test_update_parquet_file_preserves_duplicate_rows_with_same_key(self):
        tmp_root = Path(tempfile.mkdtemp(prefix="hf_duplicate_update_"))
        parquet_path = tmp_root / "data/train/docA/train_file.parquet"

        parquet_df = pd.DataFrame({
            "project_name": ["docA", "docA"],
            "filename": ["train_file", "train_file"],
            "region_id": ["R1", "R1"],
            "line_id": ["L1", "L1"],
            "text": ["old first", "old second"],
        })

        self._write_parquet(parquet_path, parquet_df)

        split_df = pd.DataFrame({
            "project_name": ["docA", "docA"],
            "filename": ["train_file", "train_file"],
            "region_id": ["R1", "R1"],
            "line_id": ["L1", "L1"],
            "text": ["new first", "new second"],
            "inference_col": ["pred first", "pred second"],
        })

        split_idx = HuggingFaceDataHandler._index_df_by_key(split_df, "train")
        HuggingFaceDataHandler._update_parquet_file(parquet_path, split_idx)

        updated = pd.read_parquet(parquet_path)

        self.assertEqual(len(updated), 2)
        self.assertEqual(updated["text"].tolist(), ["new first", "new second"])
        self.assertEqual(updated["inference_col"].tolist(), ["pred first", "pred second"])

    def test_update_parquet_file_preserves_duplicate_original_augmented_rows(self):
        tmp_root = Path(tempfile.mkdtemp(prefix="hf_duplicate_augmented_update_"))
        parquet_path = tmp_root / "data/train/docA/train_file.parquet"

        parquet_df = pd.DataFrame({
            "project_name": ["docA", "docA", "docA"],
            "filename": ["train_file", "train_file", "train_file"],
            "region_id": ["R1", "R1", "R1"],
            "line_id": ["L1", "L1", "L1"],
            "line_augmentation": ["original", "original", '{"rotation": 1}'],
            "text": ["old original first", "old original second", "old augmented"],
        })

        self._write_parquet(parquet_path, parquet_df)

        split_df = pd.DataFrame({
            "project_name": ["docA", "docA", "docA"],
            "filename": ["train_file", "train_file", "train_file"],
            "region_id": ["R1", "R1", "R1"],
            "line_id": ["L1", "L1", "L1"],
            "line_augmentation": ["original", "original", '{"rotation": 1}'],
            "text": ["new original first", "new original second", "old augmented"],
            "inference_col": ["pred first", "pred second", ""],
        })

        split_idx = HuggingFaceDataHandler._index_df_by_key(split_df, "train")
        HuggingFaceDataHandler._update_parquet_file(parquet_path, split_idx)

        updated = pd.read_parquet(parquet_path)

        self.assertEqual(len(updated), 3)

        original_rows = updated[updated["line_augmentation"] == "original"]
        augmented_rows = updated[updated["line_augmentation"] != "original"]

        self.assertEqual(
            original_rows["inference_col"].tolist(),
            ["pred first", "pred second"],
        )
        self.assertEqual(
            augmented_rows["inference_col"].tolist(),
            [""],
        )

    def test_update_parquet_file_uses_duplicate_occurrence_order(self):
        tmp_root = Path(tempfile.mkdtemp(prefix="hf_duplicate_order_update_"))
        parquet_path = tmp_root / "data/train/docA/train_file.parquet"

        parquet_df = pd.DataFrame({
            "project_name": ["docA", "docA"],
            "filename": ["train_file", "train_file"],
            "region_id": ["R1", "R1"],
            "line_id": ["L1", "L1"],
            "text": ["old first", "old second"],
        })

        self._write_parquet(parquet_path, parquet_df)

        split_df = pd.DataFrame({
            "project_name": ["docA", "docA"],
            "filename": ["train_file", "train_file"],
            "region_id": ["R1", "R1"],
            "line_id": ["L1", "L1"],
            "text": ["new first", "new second"],
            "inference_col": ["pred first", "pred second"],
        })

        split_idx = HuggingFaceDataHandler._index_df_by_key(split_df, "train")
        HuggingFaceDataHandler._update_parquet_file(parquet_path, split_idx)

        updated = pd.read_parquet(parquet_path)

        self.assertEqual(updated["text"].tolist(), ["new first", "new second"])
        self.assertEqual(updated["inference_col"].tolist(), ["pred first", "pred second"])


if __name__ == "__main__":
    unittest.main()