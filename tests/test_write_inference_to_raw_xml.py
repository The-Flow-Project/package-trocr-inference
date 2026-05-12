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
    def test_process_and_upload(self, mock_api, mock_snapshot):
        api = mock_api.return_value

        # fake snapshot folders
        fake_raw = self._make_tmp_dir("raw_")
        fake_inf = self._make_tmp_dir("inf_")

        # create fake parquet files
        raw_parquet = fake_raw / "data/train/p1/000.parquet"
        raw_parquet.parent.mkdir(parents=True, exist_ok=True)

        inf_parquet = fake_inf / "data/train/000.parquet"
        inf_parquet.parent.mkdir(parents=True, exist_ok=True)

        # RAW XML parquet
        pd.DataFrame({
            "project_name": ["p1"],
            "filename": ["a.xml"],
            "xml_content": [SAMPLE_XML],
        }).to_parquet(raw_parquet)

        # INFERENCE parquet
        pd.DataFrame({
            "project_name": ["p1"],
            "filename": ["a.xml"],
            "region_id": ["r1"],
            "line_id": ["l1"],
            "line_augmentation": ["original"],
            "inference_test": ["HELLO"],
        }).to_parquet(inf_parquet)

        # snapshot_download is called twice: inference first, then raw
        mock_snapshot.side_effect = [str(fake_inf), str(fake_raw)]

        # dataset_info sha is accessed for both repos
        api.dataset_info.return_value.sha = "fake_sha"

        self.writer.process_and_upload(output_repo=self.upload_repo)

        api.create_repo.assert_called_once()
        api.create_commit.assert_called_once()

        commit_ops = api.create_commit.call_args.kwargs["operations"]
        self.assertTrue(len(commit_ops) >= 1)

        paths = [op.path_in_repo for op in commit_ops]
        self.assertTrue(any(p.endswith(".parquet") for p in paths))
        self.assertIn("README.md", paths)

        # Check committed parquet content
        parquet_ops = [op for op in commit_ops if op.path_in_repo.endswith(".parquet")]
        self.assertTrue(parquet_ops)

        written_df = pd.read_parquet(parquet_ops[0].path_or_fileobj)
        inference_xml_cols = [c for c in written_df.columns if c.startswith("inference_xml_")]

        self.assertTrue(inference_xml_cols)

        written_xml = written_df.iloc[0][inference_xml_cols[0]]
        self.assertIn("HELLO", written_xml)
        self.assertIn("TextEquiv", written_xml)
        self.assertIn("Unicode", written_xml)

        # Check README content
        readme_ops = [op for op in commit_ops if op.path_in_repo == "README.md"]
        self.assertEqual(len(readme_ops), 1)

        readme_text = readme_ops[0].path_or_fileobj.decode("utf-8")
        self.assertIn("# Dataset Card for", readme_text)
        self.assertIn('dataset = load_dataset("', readme_text)
        self.assertIn("### Projects Included", readme_text)
        self.assertIn("p1", readme_text)
        self.assertIn("inference_xml_", readme_text)
        self.assertIn("xml_content", readme_text)
        self.assertIn("_from_", readme_text)

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

        self.writer.process_and_upload(output_repo=target_repo)

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

            ds = load_dataset("parquet", data_files=[str(p) for p in parquet_files])["train"]
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


if __name__ == "__main__":
    unittest.main()