import json
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd
from evaluate import load
from flow_inference.data_handling import HuggingFaceDataHandler
from flow_inference.status import Status
from flow_inference.utils.logging.inference_logger import logger


class Evaluation:
    def __init__(
        self,
        download_repo_name: str,
        hf_token: Optional[str],
        splits: Optional[List[str]] = None,
    ):
        self.download_repo_name = download_repo_name
        self.hf_token = hf_token
        self.splits = splits
        self.statusManager = Status()

        self.data_handler = HuggingFaceDataHandler(
            dataset_name=download_repo_name,
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
    def _extract_ground_truth(self, df: pd.DataFrame) -> List[str]:
        if "text" not in df.columns:
            raise ValueError("Dataset has no 'text' column for GT.")
        return df["text"].fillna("").astype(str).tolist()

    # --------------------------------------------------------------------------
    # INFERENCE COLUMN SELECTION
    # --------------------------------------------------------------------------
    def _find_latest_inference_column(self, df: pd.DataFrame) -> str:
        cols = [col for col in df.columns if col.startswith("inference_")]
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

    def _extract_hypothesis(self, df: pd.DataFrame, column: str) -> List[str]:
        if column not in df.columns:
            raise ValueError(f"Hypothesis column '{column}' not found.")
        return df[column].fillna("").astype(str).tolist()

    def _filter_eval_rows(self, df: pd.DataFrame, inference_col: str) -> pd.DataFrame:
        """
        Keep only rows where both GT and prediction exist.
        """
        if "text" not in df.columns:
            raise ValueError("Dataset has no 'text' column for GT.")
        if inference_col not in df.columns:
            raise ValueError(f"Inference column '{inference_col}' not found.")

        gt = df["text"].fillna("").astype(str).str.strip()
        hyp = df[inference_col].fillna("").astype(str).str.strip()

        df_eval = df[(gt != "") & (hyp != "")].copy()

        logger.info(
            f"Evaluation rows: {len(df_eval)} / {len(df)} "
            f"(non-empty GT + non-empty prediction)"
        )

        if len(df_eval) == 0:
            raise RuntimeError("No rows with both ground truth and inference available for evaluation.")

        return df_eval

    # --------------------------------------------------------------------------
    # CER CALCULATION
    # --------------------------------------------------------------------------
    def compute_cer(self, gt: List[str], hyp: List[str]) -> float:
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
        cer_score: float
    ) -> Dict[str, bytes]:

        report = {
            "timestamp": datetime.now().isoformat(),
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
    def upload_results(self, files: Dict[str, bytes]) -> None:
        for fname, content in files.items():
            self.data_handler.upload_file(
                repo_name=self.download_repo_name,
                target_path=f"evaluation/{fname}",
                content_bytes=content
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

        # 5) Extract ground truth (from filtered rows)
        groundtruth = self._extract_ground_truth(df_eval)

        # 6) Extract hypothesis (from filtered rows)
        hypothesis = self._extract_hypothesis(df_eval, inference_col)

        # 7) Calculate CER
        cer_score = self.compute_cer(groundtruth, hypothesis)
        logger.info(f"CER = {cer_score}")

        # 8) Build output files (in memory)
        files = self.create_output_files(groundtruth, hypothesis, cer_score)

        # 9) Upload results
        self.upload_results(files)

        logger.info("Completed and uploaded evaluation results.")

        return files
