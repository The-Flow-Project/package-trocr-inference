"""Evaluate OCR/HTR inference results stored in Hugging Face datasets.

This module loads an inference result dataset, selects evaluation rows with both
ground-truth text and predictions, computes Character Error Rate (CER), uploads
evaluation artifacts, and refreshes the dataset README with evaluation metadata.
"""

import json
import re
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd
from evaluate import load
from flow_inference.data_handling import HuggingFaceDataHandler
from flow_inference.status import Status
from flow_inference.utils.logging.inference_logger import logger
from flow_inference.configure_dataset_card import HuggingFaceReadmeBuilder
from datasets import Dataset, DatasetDict, Image as DatasetImage


class Evaluation:
    """Run CER evaluation for a Hugging Face inference result dataset.

    The evaluator downloads a dataset, selects the requested split or a default
    evaluation split, finds the latest inference column, computes CER against the
    ``text`` ground-truth column, and uploads evaluation artifacts back to the
    dataset repository.
    """

    def __init__(
        self,
        evaluation_repo_name: str,
        hf_token: Optional[str],
        splits: Optional[List[str]] = None,
    ):
        """Initialize the evaluator.

        Args:
            evaluation_repo_name: Hugging Face dataset repository containing inference results.
            hf_token: Optional Hugging Face token used for private repositories.
            splits: Optional split names to evaluate. If omitted, ``test`` is preferred
                over ``train``.
        """
        self.download_repo_name = evaluation_repo_name
        self.hf_token = hf_token
        self.splits = splits
        self.statusManager = Status()

        self.data_handler = HuggingFaceDataHandler(
            dataset_name=evaluation_repo_name,
            huggingface_token=hf_token
        )

    # --------------------------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------------------------
    def load_dataset(self) -> Dict[str, pd.DataFrame]:
        self.data_handler.download_hf_dataset()

        if self.data_handler.dataset is None:
            raise RuntimeError("Dataset metadata not loaded.")

        self.data_handler.dataset = self._restore_image_feature(
            self.data_handler.dataset
        )

        dfs = self.data_handler.to_dataframe()
        return dfs

    def select_splits(self, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Select the DataFrame rows to evaluate.

        Args:
            dfs: DataFrames grouped by split name.

        Returns:
            Concatenated DataFrame for requested splits, or the default evaluation split.

        Raises:
            ValueError: If a requested split does not exist.
            RuntimeError: If no requested split is provided and neither ``test`` nor
                ``train`` is available.
        """
        if self.splits:
            chosen = []
            for split in self.splits:
                if split not in dfs:
                    raise ValueError(f"Requested split '{split}' does not exist.")
                chosen.append(dfs[split])
            return pd.concat(chosen, ignore_index=True)

        # Default behavior: prefer 'test' over 'train'
        if "test" in dfs:
            return dfs["test"]
        if "train" in dfs:
            return dfs["train"]

        # If a dataset does not contain either train or test splits
        raise RuntimeError("Dataset has neither 'train' nor 'test' split.")

    @staticmethod
    def _restore_image_feature(dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        """
        Restore Hugging Face Image feature metadata.

        Parquet image columns can be loaded as plain structs with bytes/path.
        Casting the column back to datasets.Image helps README generation keep
        the image column as an Image feature instead of a raw struct.
        """
        if isinstance(dataset, DatasetDict):
            restored = DatasetDict()

            for split_name, split_dataset in dataset.items():
                if "image" in split_dataset.column_names:
                    split_dataset = split_dataset.cast_column(
                        "image",
                        DatasetImage(decode=False),
                    )

                restored[split_name] = split_dataset

            return restored

        if "image" in dataset.column_names:
            return dataset.cast_column("image", DatasetImage(decode=False))

        return dataset

    # --------------------------------------------------------------------------
    # GROUND TRUTH EXTRACTION
    # --------------------------------------------------------------------------
    @staticmethod
    def _extract_ground_truth(df: pd.DataFrame) -> List[str]:
        """Extract ground-truth text lines from an evaluation DataFrame.

        Args:
            df: DataFrame containing a ``text`` column.

        Returns:
            Ground-truth text values as strings.

        Raises:
            ValueError: If the ``text`` column is missing.
        """
        if "text" not in df.columns:
            raise ValueError("Dataset has no 'text' column for GT.")
        return df["text"].fillna("").astype(str).tolist()

    # --------------------------------------------------------------------------
    # INFERENCE COLUMN SELECTION
    # --------------------------------------------------------------------------
    @staticmethod
    def _find_latest_inference_column(df: pd.DataFrame) -> str:
        """Find the latest plain inference column in a DataFrame.

        XML writeback columns named ``inference_xml_*`` are ignored.

        Args:
            df: DataFrame containing inference output columns.

        Returns:
            Name of the latest inference column.

        Raises:
            ValueError: If no suitable inference column exists.
        """
        cols = [
            str(col)
            for col in df.columns
            if str(col).startswith("inference_")
               and not str(col).startswith("inference_xml_")
        ]

        if not cols:
            raise ValueError("No inference column found.")

        pattern = re.compile(
            r"^inference_"
            r"(\d{8})_"  # YYYYMMDD
            r"(\d{6})_"  # HHMMSS
            r"(\d{6})_"  # microseconds
            r"model_"
        )

        timestamped_columns: list[tuple[str, str]] = []

        for col in cols:
            match = pattern.match(col)

            if match:
                sortable_timestamp = "".join(match.groups())
                timestamped_columns.append(
                    (sortable_timestamp, col)
                )

        if timestamped_columns:
            _, latest_col = max(
                timestamped_columns,
                key=lambda item: item[0],
            )
        else:
            # Compatibility fallback for older or custom inference column names.
            latest_col = max(cols)
            logger.warning(
                "No inference columns followed the expected timestamp format. "
                f"Using lexicographic fallback: {latest_col}"
            )

        logger.info(f"Using latest inference column: {latest_col}")
        return latest_col

    @staticmethod
    def _extract_hypothesis(df: pd.DataFrame, column: str) -> List[str]:
        """Extract predicted text lines from an inference column.

        Args:
            df: Evaluation DataFrame.
            column: Name of the inference column to extract.

        Returns:
            Predicted text values as strings.

        Raises:
            ValueError: If the requested column is missing.
        """
        if column not in df.columns:
            raise ValueError(f"Hypothesis column '{column}' not found.")
        return df[column].fillna("").astype(str).tolist()

    @staticmethod
    def _filter_eval_rows(
            df: pd.DataFrame,
            inference_col: str,
    ) -> pd.DataFrame:
        """Keep all rows containing both ground truth and prediction text.

        Original and augmented rows are evaluated equally. The
        ``line_augmentation`` column, when present, does not affect row selection.

        Args:
            df: DataFrame containing ground-truth and inference text.
            inference_col: Inference column to evaluate.

        Returns:
            All rows containing non-empty ground truth and prediction text.

        Raises:
            ValueError: If required columns are missing.
            RuntimeError: If no valid evaluation rows remain.
        """
        if "text" not in df.columns:
            raise ValueError("Dataset has no 'text' column for GT.")

        if inference_col not in df.columns:
            raise ValueError(
                f"Inference column '{inference_col}' not found."
            )

        gt = df["text"].fillna("").astype(str).str.strip()
        hyp = df[inference_col].fillna("").astype(str).str.strip()

        df_eval = df[(gt != "") & (hyp != "")].copy()

        logger.info(
            f"Evaluation rows: {len(df_eval)} / {len(df)} "
            f"(all rows with non-empty GT and non-empty prediction)"
        )

        if df_eval.empty:
            raise RuntimeError(
                "No rows with both ground truth and inference available "
                "for evaluation."
            )

        return df_eval

    # --------------------------------------------------------------------------
    # CER CALCULATION
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_cer(gt: List[str], hyp: List[str]) -> float:
        """Compute Character Error Rate for predictions.

        Args:
            gt: Ground-truth text lines.
            hyp: Predicted text lines.

        Returns:
            Character Error Rate as a float.
        """
        cer = load("cer")
        cer.add_batch(predictions=hyp, references=gt)
        return cer.compute()

    # --------------------------------------------------------------------------
    # OUTPUT CREATION
    # --------------------------------------------------------------------------
    def create_output_files(
            self,
            groundtruth: List[str],
            hypothesis: List[str],
            cer_score: float,
            timestamp: str,
    ) -> Dict[str, bytes]:
        """Create evaluation artifact files.

        Args:
            groundtruth: Ground-truth text lines.
            hypothesis: Predicted text lines.
            cer_score: Computed Character Error Rate.
            timestamp: Evaluation timestamp.

        Returns:
            Mapping of artifact filenames to UTF-8 encoded file content.
        """
        report = {
            "timestamp": timestamp,
            "repo_name": self.download_repo_name,
            "gt_lines": len(groundtruth),
            "hypothesis_lines": len(hypothesis),
            "eval_rows": len(groundtruth),
            "cer": cer_score,
        }

        return {
            "gt.txt": "\n".join(groundtruth).encode("utf-8"),
            "hypothesis.txt": "\n".join(hypothesis).encode("utf-8"),
            "evaluation_report.json": json.dumps(report, indent=4).encode("utf-8"),
        }

    # --------------------------------------------------------------------------
    # UPLOAD FILES
    # --------------------------------------------------------------------------
    def upload_results(self, files: Dict[str, bytes], timestamp: str) -> str:
        """Upload evaluation artifact files to the dataset repository.

        Args:
            files: Mapping of filenames to file content.
            timestamp: Timestamp used to create the evaluation output directory.

        Returns:
            Repository path where the evaluation artifacts were uploaded.
        """
        evaluation_path = f"evaluation/{timestamp}/"

        for file_name, content in files.items():
            self.data_handler.upload_file(
                repo_name=self.download_repo_name,
                target_path=f"{evaluation_path}{file_name}",
                content_bytes=content
            )

        return evaluation_path

    def upload_readme(
            self,
            dfs: Dict[str, pd.DataFrame],
            inference_col: str,
            cer_score: float,
            eval_rows: int,
            timestamp: str,
            evaluation_path: str,
    ) -> None:
        """Regenerate and upload the dataset README with evaluation metadata.

        Args:
            dfs: DataFrames grouped by split name.
            inference_col: Inference column that was evaluated.
            cer_score: Computed Character Error Rate.
            eval_rows: Number of evaluated rows.
            timestamp: Evaluation timestamp.
            evaluation_path: Repository path containing evaluation artifacts.

        Raises:
            RuntimeError: If dataset metadata or parquet path information is unavailable.
        """
        if self.data_handler.dataset is None:
            raise RuntimeError("Dataset metadata not loaded.")
        if not self.data_handler.parquet_paths:
            raise RuntimeError("Parquet paths not available.")

        evaluated_split = ",".join(self.splits) if self.splits else ("test" if "test" in dfs else "train")

        evaluation_info = {
            "timestamp": timestamp,
            "cer": cer_score,
            "eval_rows": eval_rows,
            "gt_lines": eval_rows,
            "hypothesis_lines": eval_rows,
            "evaluated_split": evaluated_split,
            "inference_column": inference_col,
            "evaluation_path": evaluation_path,
        }

        builder = HuggingFaceReadmeBuilder.from_handler(
            repo_id=self.download_repo_name,
            dataset=self.data_handler.dataset,
            dataframes=dfs,
            parquet_paths=self.data_handler.parquet_paths,
            source_repos=[self.download_repo_name],
            evaluation_info=evaluation_info,
        )

        self.data_handler.upload_file(
            repo_name=self.download_repo_name,
            target_path="README.md",
            content_bytes=builder.render().encode("utf-8"),
        )

    # --------------------------------------------------------------------------
    # MAIN EVALUATION PIPELINE
    # --------------------------------------------------------------------------
    def perform_evaluation(self) -> Dict[str, bytes]:
        """Run the full evaluation pipeline.

        The pipeline downloads the dataset, evaluates OCR/HTR inference results for all valid dataset rows,
        computes CER, uploads evaluation artifacts, refreshes the dataset README, and returns the
        generated output files.

        Returns:
            Mapping of generated evaluation artifact filenames to file content.
        """
        logger.info(f"Starting evaluation for repo: {self.download_repo_name}")

        # 1) Load dataset
        dfs = self.load_dataset()

        # 2) Select split(s)
        df = self.select_splits(dfs)

        # 3) Find inference column
        inference_col = self._find_latest_inference_column(df)

        # 4) Filter rows to only those that have BOTH GT and prediction
        df_eval = self._filter_eval_rows(df, inference_col)

        # 5) Extract ground truth
        groundtruth = self._extract_ground_truth(df_eval)

        # 6) Extract hypothesis
        hypothesis = self._extract_hypothesis(df_eval, inference_col)

        # 7) Calculate CER
        cer_score = self.compute_cer(groundtruth, hypothesis)
        logger.info(f"CER = {cer_score}")

        # 8) Create one shared timestamp
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # 9) Build output files
        files = self.create_output_files(
            groundtruth=groundtruth,
            hypothesis=hypothesis,
            cer_score=cer_score,
            timestamp=timestamp,
        )

        # 10) Upload evaluation artifacts
        evaluation_path = self.upload_results(files, timestamp.replace(":", "_"))

        # 11) Regenerate and upload README
        self.upload_readme(
            dfs=dfs,
            inference_col=inference_col,
            cer_score=cer_score,
            eval_rows=len(df_eval),
            timestamp=timestamp,
            evaluation_path=evaluation_path,
        )

        logger.info("Completed and uploaded evaluation results.")
        return files
