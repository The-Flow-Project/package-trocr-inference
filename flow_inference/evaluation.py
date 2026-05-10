import json
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd
from evaluate import load
from flow_inference.data_handling import HuggingFaceDataHandler
from flow_inference.status import Status
from flow_inference.utils.logging.inference_logger import logger
from flow_inference.configure_dataset_card import HuggingFaceReadmeBuilder


class Evaluation:
    def __init__(
        self,
        evaluation_repo_name: str,
        hf_token: Optional[str],
        splits: Optional[List[str]] = None,
    ):
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
        """Download dataset and convert to DataFrames."""
        self.data_handler.download_hf_dataset()
        dfs = self.data_handler.to_dataframe()
        return dfs

    def select_splits(self, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Select splits."""
        # requested splits
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

    # --------------------------------------------------------------------------
    # GROUND TRUTH EXTRACTION
    # --------------------------------------------------------------------------
    @staticmethod
    def _extract_ground_truth(df: pd.DataFrame) -> List[str]:
        if "text" not in df.columns:
            raise ValueError("Dataset has no 'text' column for GT.")
        return df["text"].fillna("").astype(str).tolist()

    # --------------------------------------------------------------------------
    # INFERENCE COLUMN SELECTION
    # --------------------------------------------------------------------------
    @staticmethod
    def _find_latest_inference_column(df: pd.DataFrame) -> str:
        cols = [col for col in df.columns if col.startswith("inference_")
                and not col.startswith("inference_xml_")]
        if not cols:
            raise ValueError("No inference column found.")

        # Extract timestamp from each column
        timestamps = []
        for col in cols:
            parts = col.split("_", 2)
            ts = parts[1] if len(parts) > 1 else ""
            timestamps.append((ts, col))

        # Pick column with max timestamp
        latest_timestamp, latest_col = max(timestamps, key=lambda x: x[0])

        logger.info(f"Using latest inference column: {latest_col}")
        return latest_col

    @staticmethod
    def _extract_hypothesis(df: pd.DataFrame, column: str) -> List[str]:
        if column not in df.columns:
            raise ValueError(f"Hypothesis column '{column}' not found.")
        return df[column].fillna("").astype(str).tolist()

    @staticmethod
    def _filter_eval_rows(df: pd.DataFrame, inference_col: str) -> pd.DataFrame:
        """
        Keep only rows where both GT and prediction exist.
        If line_augmentation exists, evaluate only original rows.
        """
        if "text" not in df.columns:
            raise ValueError("Dataset has no 'text' column for GT.")
        if inference_col not in df.columns:
            raise ValueError(f"Inference column '{inference_col}' not found.")

        work_df = df.copy()

        if "line_augmentation" in work_df.columns:
            aug = work_df["line_augmentation"].fillna("").astype(str).str.strip().str.lower()
            work_df = work_df[aug == "original"].copy()

        gt = work_df["text"].fillna("").astype(str).str.strip()
        hyp = work_df[inference_col].fillna("").astype(str).str.strip()

        df_eval = work_df[(gt != "") & (hyp != "")].copy()

        logger.info(
            f"Evaluation rows: {len(df_eval)} / {len(df)} "
            f"(original rows with non-empty GT + non-empty prediction)"
        )

        if len(df_eval) == 0:
            raise RuntimeError("No rows with both ground truth and inference available for evaluation.")

        return df_eval

    # --------------------------------------------------------------------------
    # CER CALCULATION
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_cer(gt: List[str], hyp: List[str]) -> float:
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
