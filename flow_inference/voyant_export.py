from pathlib import Path
from typing import Dict
import zipfile
import pandas as pd
from flow_inference.data_handling import HuggingFaceDataHandler


class VoyantExporter:
    """
    Export inference results to a Voyant-compatible ZIP archive.
    Each document becomes one .txt file.
    """

    def __init__(
        self,
        text_column_prefix: str = "inference_",
        document_id_column: str = "filename",
        line_id_column: str = "line_id",
        include_line_ids: bool = False,
    ):
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
        """
        Convenience entry point:
        - downloads dataset from Hugging Face
        - exports a Voyant ZIP
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
        """
        Create a Voyant-compatible ZIP archive from a DataFrame.
        """
        text_col = self._find_inference_column(df)
        documents = self._build_documents(df, text_col)
        return self._write_zip(documents, zip_path)

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _find_inference_column(self, df: pd.DataFrame) -> str:
        inference_cols = [
            c for c in df.columns if c.startswith(self.text_column_prefix)
        ]

        if not inference_cols:
            raise ValueError("No inference column found in DataFrame")

        # select newest inference column
        return sorted(inference_cols)[-1]

    def _normalize_document_id(self, doc_id: str) -> str:
        """
        Strip a trailing image file extension from a document id.
        """
        image_extensions = {
            ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"
        }

        doc_id_lower = doc_id.lower()

        for ext in image_extensions:
            if doc_id_lower.endswith(ext):
                return doc_id[: -len(ext)]

        return doc_id

    def _build_documents(self, df: pd.DataFrame, text_col: str) -> Dict[str, str]:
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
        zip_path = Path(zip_path)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for doc_id, text in documents.items():
                zf.writestr(f"{doc_id}.txt", text)

        return zip_path
