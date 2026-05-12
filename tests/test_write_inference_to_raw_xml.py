import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from requests import Response

from flow_inference.write_inference_to_raw_xml import InferenceToRawXMLWriter


SAMPLE_XML = """<PcGts>
    <TextRegion id="r1">
        <TextLine id="l1"/>
    </TextRegion>
</PcGts>"""


class TestInferenceToRawXMLWriter(unittest.TestCase):

    def setUp(self):
        load_dotenv()
        self.raw_repo = os.getenv("RAW_XML_REPO")
        self.inference_repo = os.getenv("INFERENCE_REPO")
        self.token = os.getenv("HUGGINGFACE_TOKEN_READ_WRITE")
        self.upload_repo = os.getenv("UPLOAD_RAW_XML_REPO")

        self.writer = InferenceToRawXMLWriter(
            raw_xml_repo=self.raw_repo,
            inference_repo=self.inference_repo,
            token=self.token,
        )
        self._tmp_dirs: list[Path] = []

    def tearDown(self):
        for path in self._tmp_dirs:
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _capture_committed_parquets(api):
        captured: dict[str, pd.DataFrame] = {}

        def _side_effect(*args, **kwargs):
            for op in kwargs["operations"]:
                if (
                        op.__class__.__name__ == "CommitOperationAdd"
                        and op.path_in_repo.endswith(".parquet")
                ):
                    captured[op.path_in_repo] = pd.read_parquet(op.path_or_fileobj)
            return unittest.mock.Mock()

        api.create_commit.side_effect = _side_effect
        return captured

    def _make_tmp_dir(self, prefix: str) -> Path:
        path = Path(tempfile.mkdtemp(prefix=prefix))
        self._tmp_dirs.append(path)
        return path

    def _unique_upload_repo_name(self, suffix: str) -> str:
        if not self.upload_repo:
            self.skipTest("Missing UPLOAD_RAW_XML_REPO.")

        namespace = self.upload_repo.rsplit("/", 1)[0]
        safe_suffix = suffix.replace("_", "-")
        return f"{namespace}/test-rawxml-{safe_suffix}-{uuid4().hex[:8]}"

    @staticmethod
    def _make_404_error() -> HfHubHTTPError:
        response = Response()
        response.status_code = 404
        return HfHubHTTPError("Not found", response=response)

    @staticmethod
    def _write_raw_parquet(
            root: Path,
            split: str = "train",
            project_name: str = "p1",
            filename: str = "a.xml",
            extra_cols: dict | None = None,
    ) -> Path:
        parquet_path = root / f"data/{split}/{project_name}/000.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "project_name": [project_name],
            "filename": [filename],
            "xml_content": [SAMPLE_XML],
        }

        if extra_cols:
            data.update(extra_cols)

        pd.DataFrame(data).to_parquet(parquet_path, index=False)
        return parquet_path

    @staticmethod
    def _write_inference_parquet(
            root: Path,
            split: str = "train",
            project_name: str = "p1",
            filename: str = "a.xml",
            prediction: str = "HELLO",
    ) -> Path:
        parquet_path = root / f"data/{split}/000.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            "project_name": [project_name],
            "filename": [filename],
            "region_id": ["r1"],
            "line_id": ["l1"],
            "line_augmentation": ["original"],
            "inference_test": [prediction],
        }).to_parquet(parquet_path, index=False)

        return parquet_path

    # --------------------------------------------------
    # UNIT TEST: BUILD LOOKUP FROM INFERENCE DF
    # --------------------------------------------------
    def test_build_lookup(self):
        df = pd.DataFrame({
            "project_name": ["p1", "p1"],
            "filename": ["a.xml", "a.xml"],
            "region_id": ["r1", "r1"],
            "line_id": ["l1", "l2"],
            "inference_test": ["A", "B"],
        })

        lookup = self.writer._build_lookup(df)

        self.assertEqual(
            lookup,
            {"p1": {"a.xml": {("r1", "l1"): "A", ("r1", "l2"): "B"}}}
        )

    def test_build_lookup_skips_augmented_rows(self):
        df = pd.DataFrame({
            "project_name": ["p1", "p1", "p1"],
            "filename": ["a.xml", "a.xml", "a.xml"],
            "region_id": ["r1", "r1", "r1"],
            "line_id": ["l1", "l1", "l2"],
            "line_augmentation": ["original", '{"erosion": 3}', "original"],
            "inference_test": ["ORIGINAL_L1", "AUGMENTED_L1", "ORIGINAL_L2"],
        })

        lookup = self.writer._build_lookup(df)

        self.assertEqual(
            lookup,
            {"p1": {"a.xml": {("r1", "l1"): "ORIGINAL_L1", ("r1", "l2"): "ORIGINAL_L2"}}}
        )

    def test_build_lookup_skips_ambiguous_duplicate_inference_keys(self):
        df = pd.DataFrame({
            "project_name": ["p1", "p1"],
            "filename": ["a.xml", "a.xml"],
            "region_id": ["r1", "r1"],
            "line_id": ["l1", "l1"],
            "line_augmentation": ["original", "original"],
            "inference_test": ["FIRST", "SECOND"],
        })

        lookup = self.writer._build_lookup(df)

        self.assertEqual(lookup, {})

    def test_build_lookup_keeps_duplicate_inference_keys_when_text_is_identical(self):
        df = pd.DataFrame({
            "project_name": ["p1", "p1"],
            "filename": ["a.xml", "a.xml"],
            "region_id": ["r1", "r1"],
            "line_id": ["l1", "l1"],
            "line_augmentation": ["original", "original"],
            "inference_test": ["SAME", "SAME"],
        })

        lookup = self.writer._build_lookup(df)

        self.assertEqual(
            lookup,
            {"p1": {"a.xml": {("r1", "l1"): "SAME"}}}
        )

    # --------------------------------------------------
    # UNIT TEST: UPDATE DF WITH INFERENCE XML
    # --------------------------------------------------
    def test_update_df(self):
        df = pd.DataFrame({
            "project_name": ["p1"],
            "filename": ["a.xml"],
            "xml_content": [SAMPLE_XML],
        })

        lookup = {"p1": {"a.xml": {("r1", "l1"): "HELLO"}}}
        new_col = "inference_xml_20260418_214612_123456_from_test_repo"

        updated, cols = self.writer._update_df(df, lookup, new_col)

        self.assertEqual(cols, [new_col])

        xml = updated.iloc[0][new_col]
        self.assertIn("HELLO", xml)
        self.assertIn("TextEquiv", xml)
        self.assertIn("Unicode", xml)

    def test_update_df_returns_no_updated_columns_when_no_line_matches(self):
        df = pd.DataFrame({
            "project_name": ["p1"],
            "filename": ["a.xml"],
            "xml_content": [SAMPLE_XML],
        })

        lookup = {"p1": {"a.xml": {("r1", "does_not_exist"): "HELLO"}}}
        new_col = "inference_xml_20260418_214612_123456_from_test_repo"

        updated, cols = self.writer._update_df(df, lookup, new_col)

        self.assertEqual(cols, [])
        self.assertIn(new_col, updated.columns)
        self.assertEqual(updated.iloc[0][new_col], "")

    # --------------------------------------------------
    # UNIT TEST: BUILD INFERENCE XML COLUMN NAME
    # --------------------------------------------------
    def test_build_inference_xml_column_name(self):
        col = self.writer._build_inference_xml_column_name()

        self.assertTrue(col.startswith("inference_xml_"))
        self.assertIn("_from_", col)
        self.assertIn(self.inference_repo.split("/")[-1].replace("-", "_").replace("/", "_"), col)

    # --------------------------------------------------
    # UNIT TEST: PROCESS AND UPLOAD
    # --------------------------------------------------
    @patch("flow_inference.write_inference_to_raw_xml.snapshot_download")
    @patch("flow_inference.write_inference_to_raw_xml.HfApi")
    def test_process_and_upload_new_repo(self, mock_api_cls, mock_snapshot):
        api = mock_api_cls.return_value

        def fake_dataset_info(repo_id=None, *args, **kwargs):
            if repo_id == self.upload_repo:
                raise self._make_404_error()

            info = unittest.mock.Mock()
            info.sha = "fake_sha"
            return info

        api.dataset_info.side_effect = fake_dataset_info
        captured_parquets = self._capture_committed_parquets(api)

        def fake_snapshot_download(*args, **kwargs):
            repo_id = kwargs["repo_id"]
            local_dir = Path(kwargs["local_dir"])

            if repo_id == self.inference_repo:
                self._write_inference_parquet(local_dir, prediction="HELLO")
            elif repo_id == self.raw_repo:
                self._write_raw_parquet(local_dir)
            else:
                raise AssertionError(f"Unexpected snapshot repo: {repo_id}")

            return str(local_dir)

        mock_snapshot.side_effect = fake_snapshot_download

        self.writer.process_and_upload(
            output_repo=self.upload_repo,
            upload_mode="new_repo",
        )

        api.create_repo.assert_called_once_with(
            repo_id=self.upload_repo,
            repo_type="dataset",
            private=True,
            exist_ok=False,
            token=self.token,
        )
        api.create_commit.assert_called_once()

        commit_ops = api.create_commit.call_args.kwargs["operations"]
        paths = [op.path_in_repo for op in commit_ops]

        self.assertIn("data/train/p1/000.parquet", paths)
        self.assertIn("README.md", paths)

        parquet_ops = [op for op in commit_ops if op.path_in_repo.endswith(".parquet")]
        self.assertEqual(len(parquet_ops), 1)

        written_df = captured_parquets["data/train/p1/000.parquet"]
        inference_xml_cols = [c for c in written_df.columns if c.startswith("inference_xml_")]

        self.assertEqual(len(inference_xml_cols), 1)

        written_xml = written_df.iloc[0][inference_xml_cols[0]]
        self.assertIn("HELLO", written_xml)
        self.assertIn("TextEquiv", written_xml)
        self.assertIn("Unicode", written_xml)

        readme_ops = [op for op in commit_ops if op.path_in_repo == "README.md"]
        self.assertEqual(len(readme_ops), 1)

        readme_text = readme_ops[0].path_or_fileobj.decode("utf-8")
        self.assertIn("# Dataset Card for", readme_text)
        self.assertIn("inference_xml_", readme_text)
        self.assertIn("xml_content", readme_text)

    @patch("flow_inference.write_inference_to_raw_xml.HfApi")
    def test_process_and_upload_new_repo_refuses_existing_target(self, mock_api_cls):
        api = mock_api_cls.return_value
        api.dataset_info.return_value.sha = "existing_sha"

        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.writer.process_and_upload(
                output_repo=self.upload_repo,
                upload_mode="new_repo",
            )

        api.create_repo.assert_not_called()
        api.create_commit.assert_not_called()

    @patch("flow_inference.write_inference_to_raw_xml.snapshot_download")
    @patch("flow_inference.write_inference_to_raw_xml.HfApi")
    def test_process_and_upload_update_preserves_existing_inference_xml_columns(
            self,
            mock_api_cls,
            mock_snapshot,
    ):
        api = mock_api_cls.return_value
        api.dataset_info.return_value.sha = "fake_sha"
        api.list_repo_files.return_value = [
            "data/train/p1/000.parquet",
            "README.md",
        ]
        captured_parquets = self._capture_committed_parquets(api)

        old_col = "inference_xml_20260101_from_old_repo"
        old_xml = "<PcGts><old>OLD XML</old></PcGts>"

        def fake_snapshot_download(*args, **kwargs):
            repo_id = kwargs["repo_id"]
            local_dir = Path(kwargs["local_dir"])

            if repo_id == self.inference_repo:
                self._write_inference_parquet(local_dir, prediction="NEW")
            elif repo_id == self.raw_repo:
                self._write_raw_parquet(local_dir)
            elif repo_id == self.upload_repo:
                self._write_raw_parquet(
                    local_dir,
                    extra_cols={old_col: [old_xml]},
                )
            else:
                raise AssertionError(f"Unexpected snapshot repo: {repo_id}")

            return str(local_dir)

        mock_snapshot.side_effect = fake_snapshot_download

        self.writer.process_and_upload(
            output_repo=self.upload_repo,
            upload_mode="update",
        )

        api.create_repo.assert_not_called()
        api.create_commit.assert_called_once()

        commit_ops = api.create_commit.call_args.kwargs["operations"]
        parquet_ops = [op for op in commit_ops if op.path_in_repo.endswith(".parquet")]
        self.assertEqual(len(parquet_ops), 1)

        written_df = captured_parquets["data/train/p1/000.parquet"]

        self.assertIn(old_col, written_df.columns)
        self.assertEqual(written_df.iloc[0][old_col], old_xml)

        inference_xml_cols = [c for c in written_df.columns if c.startswith("inference_xml_")]
        self.assertGreaterEqual(len(inference_xml_cols), 2)

        new_cols = [c for c in inference_xml_cols if c != old_col]
        self.assertEqual(len(new_cols), 1)
        self.assertIn("NEW", written_df.iloc[0][new_cols[0]])

    @patch("flow_inference.write_inference_to_raw_xml.snapshot_download")
    @patch("flow_inference.write_inference_to_raw_xml.HfApi")
    def test_process_and_upload_replace_does_not_preserve_existing_inference_xml_columns(
            self,
            mock_api_cls,
            mock_snapshot,
    ):
        api = mock_api_cls.return_value
        api.dataset_info.return_value.sha = "fake_sha"
        api.list_repo_files.return_value = [
            "data/train/p1/000.parquet",
            "data/train/old_extra.parquet",
            "README.md",
        ]
        captured_parquets = self._capture_committed_parquets(api)

        def fake_snapshot_download(*args, **kwargs):
            repo_id = kwargs["repo_id"]
            local_dir = Path(kwargs["local_dir"])

            if repo_id == self.inference_repo:
                self._write_inference_parquet(local_dir, prediction="REPLACED")
            elif repo_id == self.raw_repo:
                self._write_raw_parquet(local_dir)
            else:
                raise AssertionError(f"Unexpected snapshot repo: {repo_id}")

            return str(local_dir)

        mock_snapshot.side_effect = fake_snapshot_download

        self.writer.process_and_upload(
            output_repo=self.upload_repo,
            upload_mode="replace",
        )

        api.create_repo.assert_not_called()
        api.create_commit.assert_called_once()

        commit_ops = api.create_commit.call_args.kwargs["operations"]
        paths = [op.path_in_repo for op in commit_ops]

        self.assertIn("data/train/p1/000.parquet", paths)
        self.assertIn("README.md", paths)

        delete_ops = [
            op for op in commit_ops
            if op.__class__.__name__ == "CommitOperationDelete"
        ]
        delete_paths = [op.path_in_repo for op in delete_ops]

        self.assertIn("data/train/old_extra.parquet", delete_paths)

        written_df = captured_parquets["data/train/p1/000.parquet"]

        inference_xml_cols = [c for c in written_df.columns if c.startswith("inference_xml_")]
        self.assertEqual(len(inference_xml_cols), 1)
        self.assertIn("REPLACED", written_df.iloc[0][inference_xml_cols[0]])

    def test_process_and_upload_refuses_raw_source_repo_by_default(self):
        with self.assertRaisesRegex(RuntimeError, "source raw XML repo"):
            self.writer.process_and_upload(
                output_repo=self.raw_repo,
                upload_mode="replace",
            )

    def test_process_and_upload_refuses_inference_source_repo(self):
        with self.assertRaisesRegex(RuntimeError, "inference source repo"):
            self.writer.process_and_upload(
                output_repo=self.inference_repo,
                upload_mode="replace",
            )

    @patch("flow_inference.write_inference_to_raw_xml.snapshot_download")
    @patch("flow_inference.write_inference_to_raw_xml.HfApi")
    def test_process_and_upload_update_refuses_missing_target_repo(
        self,
        mock_api_cls,
        mock_snapshot,
    ):
        api = mock_api_cls.return_value

        def fake_dataset_info(repo_id=None, *args, **kwargs):
            if repo_id == self.upload_repo:
                raise self._make_404_error()

            info = unittest.mock.Mock()
            info.sha = "fake_sha"
            return info

        api.dataset_info.side_effect = fake_dataset_info

        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            self.writer.process_and_upload(
                output_repo=self.upload_repo,
                upload_mode="update",
            )

        api.create_commit.assert_not_called()

    def test_validate_compatible_raw_xml_base_schema_allows_extra_inference_xml_only(self):
        source_df = pd.DataFrame({
            "project_name": ["p1"],
            "filename": ["a.xml"],
            "xml_content": [SAMPLE_XML],
        })

        target_df = pd.DataFrame({
            "project_name": ["p1"],
            "filename": ["a.xml"],
            "xml_content": [SAMPLE_XML],
            "inference_xml_old": ["OLD"],
        })

        self.writer._validate_compatible_raw_xml_base_schema(
            source_df=source_df,
            target_df=target_df,
            repo_path="data/train/p1/000.parquet",
        )

    def test_validate_compatible_raw_xml_base_schema_rejects_extra_base_column(self):
        source_df = pd.DataFrame({
            "project_name": ["p1"],
            "filename": ["a.xml"],
            "xml_content": [SAMPLE_XML],
        })

        target_df = pd.DataFrame({
            "project_name": ["p1"],
            "filename": ["a.xml"],
            "xml_content": [SAMPLE_XML],
            "unexpected_base_col": ["bad"],
        })

        with self.assertRaisesRegex(RuntimeError, "base schema is incompatible"):
            self.writer._validate_compatible_raw_xml_base_schema(
                source_df=source_df,
                target_df=target_df,
                repo_path="data/train/p1/000.parquet",
            )

    # --------------------------------------------------
    # INTEGRATION TEST: PROCESS AND UPLOAD
    # --------------------------------------------------
    def test_process_and_upload_integration(self):
        """
        REAL integration test (HF Hub).
        Requires valid env vars.
        """
        if not self.raw_repo or not self.inference_repo or not self.token or not self.upload_repo:
            self.skipTest("Missing HF integration configuration")

        target_repo = self._unique_upload_repo_name("writeback")

        self.writer.process_and_upload(
            output_repo=target_repo,
            upload_mode="new_repo",
        )

        api = HfApi()
        files = api.list_repo_files(
            target_repo,
            repo_type="dataset",
            token=self.token,
        )

        self.assertTrue(any(f.endswith(".parquet") for f in files))
        self.assertIn("README.md", files)

        tmp_dir = tempfile.mkdtemp(prefix="verify_raw_xml_")
        try:
            downloaded = api.snapshot_download(
                repo_id=target_repo,
                repo_type="dataset",
                token=self.token,
                local_dir=tmp_dir,
            )

            parquet_files = list(Path(downloaded).rglob("*.parquet"))
            self.assertTrue(parquet_files, "No parquet files found after upload.")

            parquet_cache_dir = tempfile.mkdtemp(prefix="verify_raw_xml_parquet_cache_")
            self._tmp_dirs.append(Path(parquet_cache_dir))

            ds = load_dataset(
                "parquet",
                data_files=[str(p) for p in parquet_files],
                cache_dir=parquet_cache_dir,
                download_mode="force_redownload",
            )["train"]

            df = ds.to_pandas()

            inference_xml_cols = [c for c in df.columns if c.startswith("inference_xml_")]
            self.assertTrue(inference_xml_cols, "No inference_xml_ column found in uploaded parquet.")

            values = df[inference_xml_cols[0]].fillna("").astype(str).str.strip()
            self.assertTrue(values.ne("").any(), "Uploaded inference_xml_ column is empty.")

            non_empty_values = values[values.ne("")]
            self.assertTrue(
                non_empty_values.str.contains("TextEquiv", regex=False).any(),
                "Uploaded inference_xml_ column does not contain TextEquiv.",
            )
            self.assertTrue(
                non_empty_values.str.contains("Unicode", regex=False).any(),
                "Uploaded inference_xml_ column does not contain Unicode.",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @patch("flow_inference.write_inference_to_raw_xml.snapshot_download")
    @patch("flow_inference.write_inference_to_raw_xml.HfApi")
    def test_process_and_upload_handles_train_and_test_splits(
            self,
            mock_api_cls,
            mock_snapshot,
    ):
        api = mock_api_cls.return_value

        def fake_dataset_info(repo_id=None, *args, **kwargs):
            if repo_id == self.upload_repo:
                raise self._make_404_error()

            info = unittest.mock.Mock()
            info.sha = "fake_sha"
            return info

        api.dataset_info.side_effect = fake_dataset_info

        captured_parquets = self._capture_committed_parquets(api)

        def fake_snapshot_download(*args, **kwargs):
            repo_id = kwargs["repo_id"]
            local_dir = Path(kwargs["local_dir"])

            if repo_id == self.inference_repo:
                self._write_inference_parquet(
                    local_dir,
                    split="train",
                    project_name="p1",
                    filename="a.xml",
                    prediction="TRAIN_TEXT",
                )
                self._write_inference_parquet(
                    local_dir,
                    split="test",
                    project_name="p2",
                    filename="b.xml",
                    prediction="TEST_TEXT",
                )

            elif repo_id == self.raw_repo:
                self._write_raw_parquet(
                    local_dir,
                    split="train",
                    project_name="p1",
                    filename="a.xml",
                )
                self._write_raw_parquet(
                    local_dir,
                    split="test",
                    project_name="p2",
                    filename="b.xml",
                )

            else:
                raise AssertionError(f"Unexpected snapshot repo: {repo_id}")

            return str(local_dir)

        mock_snapshot.side_effect = fake_snapshot_download

        self.writer.process_and_upload(
            output_repo=self.upload_repo,
            upload_mode="new_repo",
        )

        api.create_repo.assert_called_once()
        api.create_commit.assert_called_once()

        commit_ops = api.create_commit.call_args.kwargs["operations"]
        paths = [op.path_in_repo for op in commit_ops]

        self.assertIn("data/train/p1/000.parquet", paths)
        self.assertIn("data/test/p2/000.parquet", paths)
        self.assertIn("README.md", paths)

        train_df = captured_parquets["data/train/p1/000.parquet"]
        test_df = captured_parquets["data/test/p2/000.parquet"]

        train_cols = [c for c in train_df.columns if c.startswith("inference_xml_")]
        test_cols = [c for c in test_df.columns if c.startswith("inference_xml_")]

        self.assertEqual(len(train_cols), 1)
        self.assertEqual(len(test_cols), 1)

        self.assertIn("TRAIN_TEXT", train_df.iloc[0][train_cols[0]])
        self.assertIn("TEST_TEXT", test_df.iloc[0][test_cols[0]])


if __name__ == "__main__":
    unittest.main()