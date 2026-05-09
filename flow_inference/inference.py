# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from datetime import datetime
from typing import Optional, Tuple, Dict, List, cast
from flow_inference.data_handling import HuggingFaceDataHandler
from flow_inference.image_processing import ImageHandler
from flow_inference.model_handling import ModelManager
from flow_inference.status import Status
from flow_inference.utils.logging.inference_logger import logger
from flow_inference.infer_textlines import InferenceHandler
import pandas as pd

LINE_AUGMENTATION_COLUMN = "line_augmentation"
ORIGINAL_LINE_AUGMENTATION_VALUE = "original"

# ===============================================================================
# CLASS
# ===============================================================================
class Inference:
    def __init__(self,
                 download_repo_name: str,
                 hf_token: Optional[str],
                 trocr_model="microsoft/trocr-small-handwritten",
                 target_image_size: Tuple[int, int] = None,
                 stop_on_fail: bool = False,
                 splits: Optional[List[str]] = None,
                 push_to_hub: bool = True,
                 private_repo: bool = True,
                 upload_repo_name: Optional[str] = None,
                 upload_mode: str = "new_repo",
                 allow_source_repo_update: bool = False,
                 ) -> None:

        self.download_repo_name = download_repo_name
        self.hf_token = hf_token
        self.trocr_model = trocr_model
        self.target_image_size = target_image_size
        self.stop_on_fail = stop_on_fail
        self.requested_splits = splits or ["train"]
        self.push_to_hub = push_to_hub
        self.private_repo = private_repo
        self.upload_repo_name = upload_repo_name or download_repo_name
        self.upload_mode = upload_mode
        self.allow_source_repo_update = allow_source_repo_update
        self.statusManager = Status()
        self.model_manager = ModelManager()
        self.processor = self.model_manager.load_processor(self.trocr_model)
        self.model = self.model_manager.load_model(self.trocr_model)
        self.device = self.model_manager.device

        if self.processor is None or self.model is None:
            raise RuntimeError("Failed to load TrOCR model or processor")

        logger.debug(f"Inference initialized with Hugging Face dataset: {download_repo_name}")

    @staticmethod
    def _is_original_line_augmentation_value(value) -> bool:
        if pd.isna(value):
            return False
        return str(value).strip().lower() == ORIGINAL_LINE_AUGMENTATION_VALUE

    def _filter_records_for_inference(self, records: list[dict]) -> list[dict]:
        """
        If line_augmentation is absent: keep old behavior.
        If line_augmentation is present: only infer rows where line_augmentation == 'original'.
        """
        if not records:
            return records

        if not any(LINE_AUGMENTATION_COLUMN in record for record in records):
            return records

        filtered = [
            record
            for record in records
            if self._is_original_line_augmentation_value(
                record.get(LINE_AUGMENTATION_COLUMN)
            )
        ]

        logger.info(
            f"{LINE_AUGMENTATION_COLUMN} detected: processing "
            f"{len(filtered)} original rows out of {len(records)} total rows."
        )

        return filtered

    # ===========================================================================
    # MAIN PIPELINE
    # ===========================================================================
    def perform_inference(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Perform the complete inference workflow:
        1. Download HF dataset
        2. Convert to DataFrame and extract image records
        3. Run inference on images
        4. Write results back into the DataFrame
        """
        logger.info("Starting inference pipeline for Hugging Face dataset.")

        # -------------------------------
        # STEP 1: Download & prepare dataset
        # -------------------------------
        logger.debug("Downloading dataset from Hugging Face Hub.")
        loader = HuggingFaceDataHandler(
            dataset_name=self.download_repo_name,
            huggingface_token=self.hf_token,
            split=self.requested_splits,
        )

        try:
            loader.download_hf_dataset()
            dfs = loader.to_dataframe()
            records = loader.convert_to_list_of_dicts(dfs)
        except Exception as e:
            logger.error(f"Failed to load dataset from Hugging Face: {e}")
            return None

        records_for_inference = {
            split: self._filter_records_for_inference(recs)
            for split, recs in records.items()
        }

        total_records = sum(
            len(recs)
            for split, recs in records_for_inference.items()
            if split in self.requested_splits or split == "default"
        )

        self.statusManager.initialize_status(total_records)
        logger.info(
            f"Dataset loaded successfully: {total_records} image records selected for inference."
        )

        # -------------------------------
        # STEP 2: Run inference
        # -------------------------------
        logger.debug("Running inference on image records.")

        inferred = {}

        for split, recs in records_for_inference.items():
            if split in self.requested_splits or split == "default":
                result = self.run_inference(
                    records=recs,
                    model=self.model,
                    processor=self.processor,
                    device=self.device
                )
                inferred[split] = result if result is not None else {}
            else:
                inferred[split] = {}

        # -------------------------------
        # STEP 3: Write inference results
        # -------------------------------
        logger.debug("Writing inference results to all DataFrames.")
        updated_dfs = {}

        for split, df_split in dfs.items():
            updated_dfs[split] = self.save_results(
                inferred_lines=inferred[split],
                original_df=df_split
            )

        if self.push_to_hub:
            loader.df = updated_dfs
            loader.push_to_hub(
                upload_repo_name=self.upload_repo_name,
                private=self.private_repo,
                commit_message="Add inference results",
                upload_mode=self.upload_mode,
                allow_source_repo_update=self.allow_source_repo_update,
            )

        logger.info("Inference process completed successfully.")
        logger.info(f"Total runtime: {self.statusManager.calculate_runtime()}")

        self.statusManager.summary()

        return updated_dfs

    # ===========================================================================
    # INFERENCE
    # ===========================================================================
    def run_inference(
            self,
            records: list[dict],
            model,
            processor,
            device
    ) -> dict[tuple[str, str, str], list[str]]:
        """
        Run inference on provided image records.
        """
        logger.debug(f"Running inference on {len(records)} records.")

        if not records:
            logger.info("No records selected for inference.")
            return {}

        image_handler = ImageHandler(
            processor=processor,
            target_image_size=self.target_image_size
        )

        inference_handler = InferenceHandler(
            device=device,
            model=model,
            processor=processor
        )

        inference_result = inference_handler.infer(records=records,
                                                   image_handler=image_handler)
        logger.info("Inference completed successfully.")

        # Parse results
        inferred_lines: dict[tuple[str, str, str], list[str]] = {}

        for result in inference_result:
            try:
                project, filename, line_id, inferred_text = result.split("\t", 3)
                key = (project, filename, line_id)

                inferred_lines.setdefault(key, []).append(inferred_text)

                self.statusManager.update_progress(
                    status_type="success",
                    current_item_name=f"{project}:{line_id}"
                )
            except ValueError as ve:
                logger.error(f"Malformed inference result: {result} - Error: {ve}")
                self.statusManager.update_progress(
                    status_type="failure_inference",
                    current_item_name="unknown_line_id"
                )

        return inferred_lines

    # ===========================================================================
    # SAVE RESULTS
    # ===========================================================================
    def write_inference_to_dataframe(
            self,
            inferred_lines: Dict[tuple[str, str, str], List[str]],
            original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Write inference results into a new timestamped column in the dataframe.
        """
        logger.info("Writing inference results back to DataFrame.")

        updated_df = original_df.copy()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        model_name = self.trocr_model.replace("/", "_")
        new_col = f"inference_{timestamp}_model_{model_name}"

        updated_df[new_col] = ""
        updated_df[new_col] = updated_df[new_col].astype("string")

        updated_count = 0

        for (project, filename, line_id), texts in inferred_lines.items():
            mask = cast(
                pd.Series,
                (updated_df["project_name"] == project)
                & (updated_df["filename"] == filename)
                & (updated_df["line_id"] == line_id)
            )

            if LINE_AUGMENTATION_COLUMN in updated_df.columns:
                mask = cast(
                    pd.Series,
                    mask
                    & updated_df[LINE_AUGMENTATION_COLUMN].apply(
                        self._is_original_line_augmentation_value
                    )
                )

            matching_indices = list(updated_df.index[mask])

            if not matching_indices:
                logger.warning(
                    f"No matching row found for project='{project}', filename='{filename}', line_id='{line_id}'"
                )
                continue

            if len(texts) != len(matching_indices):
                logger.warning(
                    f"Inference/writeback count mismatch for project='{project}', "
                    f"filename='{filename}', line_id='{line_id}': "
                    f"{len(texts)} predictions for {len(matching_indices)} matching rows. "
                    f"Writing up to the smaller count."
                )

            for row_index, text in zip(matching_indices, texts):
                updated_df.at[row_index, new_col] = text
                updated_count += 1

        logger.info(f"Created column '{new_col}' and updated {updated_count} rows.")
        return updated_df

    def save_results(self,
                     inferred_lines: Dict[tuple[str, str, str], List[str]],
                     original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Save inference results: add new column + update raw XML strings.
        Returns the updated DataFrame.
        """
        logger.info("Saving inference results into DataFrame and updating XML fields.")

        try:
            df_with_inference = self.write_inference_to_dataframe(inferred_lines, original_df)

            logger.info("Inference results successfully written to DataFrame.")
            return df_with_inference

        except Exception as e:
            logger.error(f"Error while saving inference results: {e}")
            raise
