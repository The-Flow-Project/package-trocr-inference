"""Export OCR/HTR inference results as Voyant-compatible text archives.

This module groups line-level inference results by document, writes each document
as a plain-text file, and bundles the files into a ZIP archive that can be loaded
into Voyant Tools or similar downstream text-analysis workflows.
"""

from pathlib import Path
from typing import Dict
import zipfile
import pandas as pd
from flow_inference.data_handling import HuggingFaceDataHandler


class VoyantExporter:
    """Create Voyant-compatible ZIP archives from inference result DataFrames.

    The exporter selects the latest inference column, groups recognized text by
    document ID, optionally prefixes lines with their line IDs, and writes one
    ``.txt`` file per document into a ZIP archive.
    """
    def __init__(
        self,
        text_column_prefix: str = "inference_",
        document_id_column: str = "filename",
        line_id_column: str = "line_id",
        include_line_ids: bool = False,
    ):
        """Initialize the Voyant exporter.

        Args:
            text_column_prefix: Prefix used to identify inference text columns.
            document_id_column: Column containing the document or image identifier.
            line_id_column: Column containing the line identifier.
            include_line_ids: Whether to prefix exported text lines with line IDs.
        """
        self.text_column_prefix = text_column_prefix
        self.document_id_column = document_id_column
        self.line_id_column = line_id_column
        self.include_line_ids = include_line_ids

    # ------------------------------------------------------------
    # Export Voyant Data
    # ------------------------------------------------------------

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        split: str,
        hf_token: str | None,
        zip_path: str | Path,
        include_line_ids: bool = False,
    ) -> Path:
        """Download a Hugging Face dataset split and export it as a Voyant ZIP.

        Args:
            dataset_name: Hugging Face dataset repository ID to download.
            split: Dataset split to export.
            hf_token: Optional Hugging Face token used for private repositories.
            zip_path: Output path for the generated ZIP archive.
            include_line_ids: Whether to prefix exported text lines with line IDs.

        Returns:
            Path to the generated ZIP archive.

        Raises:
            ValueError: If the requested split is not available in the dataset.
        """
        handler = HuggingFaceDataHandler(
            dataset_name=dataset_name,
            huggingface_token=hf_token,
        )
        handler.download_hf_dataset()
        dfs = handler.to_dataframe()

        if split not in dfs:
            raise ValueError(f"Split '{split}' not found in dataset")

        exporter = cls(include_line_ids=include_line_ids)
        return exporter.export(dfs[split], zip_path)

    def export(self, df: pd.DataFrame, zip_path: str | Path) -> Path:
        """Create a Voyant-compatible ZIP archive from inference results.

        Args:
            df: DataFrame containing document IDs, line IDs, and inference text.
            zip_path: Output path for the generated ZIP archive.

        Returns:
            Path to the generated ZIP archive.

        Raises:
            ValueError: If no inference text column is available.
        """
        text_col = self._find_inference_column(df)
        documents = self._build_documents(df, text_col)
        return self._write_zip(documents, zip_path)

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _find_inference_column(self, df: pd.DataFrame) -> str:
        """Return the latest inference text column from a DataFrame."""
        inference_cols = [
            c for c in df.columns if c.startswith(self.text_column_prefix)
        ]

        if not inference_cols:
            raise ValueError("No inference column found in DataFrame")

        # select newest inference column
        return sorted(inference_cols)[-1]

    def _normalize_document_id(self, doc_id: str) -> str:
        """Return a document ID without a trailing image file extension."""
        image_extensions = {
            ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"
        }

        doc_id_lower = doc_id.lower()

        for ext in image_extensions:
            if doc_id_lower.endswith(ext):
                return doc_id[: -len(ext)]

        return doc_id

    def _build_documents(self, df: pd.DataFrame, text_col: str) -> Dict[str, str]:
        """Group line-level inference text into document-level plain text."""
        documents: Dict[str, list[str]] = {}

        df = df.sort_values(
            [self.document_id_column, self.line_id_column]
        )

        for _, row in df.iterrows():
            raw_doc_id = str(row[self.document_id_column])
            doc_id = self._normalize_document_id(raw_doc_id)
            text = str(row[text_col]).strip()

            if not text:
                continue

            documents.setdefault(doc_id, [])

            if self.include_line_ids:
                documents[doc_id].append(
                    f"[{row[self.line_id_column]}] {text}"
                )
            else:
                documents[doc_id].append(text)

        return {
            doc_id: "\n".join(lines)
            for doc_id, lines in documents.items()
        }

    def _write_zip(
        self,
        documents: Dict[str, str],
        zip_path: str | Path,
    ) -> Path:
        """Write document texts to a compressed ZIP archive.

        Args:
            documents: Mapping of document IDs to document text.
            zip_path: Output path for the ZIP archive.

        Returns:
            Path to the generated ZIP archive.
        """
        zip_path = Path(zip_path)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for doc_id, text in documents.items():
                zf.writestr(f"{doc_id}.txt", text)

        return zip_path
