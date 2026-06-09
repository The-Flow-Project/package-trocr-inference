"""Run the end-to-end TrOCR inference workflow for Hugging Face datasets.

This module downloads line-level OCR/HTR datasets, filters records for inference,
runs TrOCR prediction on image lines, writes predictions to timestamped
``inference_*`` columns, and optionally uploads the updated dataset to the
Hugging Face Hub.
"""

# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from datetime import datetime
from typing import Optional, Tuple, Dict, List, cast, Literal
from flow_inference.data_handling import HuggingFaceDataHandler
from flow_inference.image_processing import ImageHandler
from flow_inference.model_handling import ModelManager
from flow_inference.status import Status
from flow_inference.utils.logging.inference_logger import logger
from flow_inference.infer_textlines import InferenceHandler
import pandas as pd

LINE_AUGMENTATION_COLUMN = "line_augmentation"
ORIGINAL_LINE_AUGMENTATION_VALUE = "original"
REQUIRED_INFERENCE_KEY_COLUMNS = ["filename", "region_id", "line_id"]

# ===============================================================================
# CLASS
# ===============================================================================
class Inference:
    """Coordinate dataset loading, TrOCR inference, result writeback, and upload.

    The pipeline loads records from a Hugging Face dataset, runs line-level
    inference with a configured TrOCR model, writes predictions back into the
    corresponding DataFrames, and can publish the updated dataset to a target
    Hugging Face repository.
    """
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
                 upload_mode: Literal["new_repo", "replace", "update"] = "new_repo",
                 allow_source_repo_update: bool = False,
                 ) -> None:
        """Initialize the inference pipeline.

        Args:
            download_repo_name: Hugging Face dataset repository to download.
            hf_token: Optional Hugging Face token used for private repositories.
            trocr_model: Hugging Face model ID or local path for the TrOCR model.
            target_image_size: Optional target image size as ``(width, height)``.
            stop_on_fail: Whether to stop the workflow after an inference failure.
            splits: Dataset splits to process. Defaults to ``["train"]``.
            push_to_hub: Whether to upload the updated dataset after inference.
            private_repo: Whether to create the upload repository as private.
            upload_repo_name: Optional target dataset repository. Defaults to the source repository.
            upload_mode: Upload behavior. Use ``"new_repo"``, ``"replace"``, or ``"update"``.
            allow_source_repo_update: Whether uploading back into the source repository is allowed.

        Raises:
            RuntimeError: If the TrOCR model or processor cannot be loaded.
        """
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
        """Return whether a line augmentation value marks an original line."""
        if pd.isna(value):
            return False
        return str(value).strip().lower() == ORIGINAL_LINE_AUGMENTATION_VALUE

    @staticmethod
    def _validate_required_key_columns(df: pd.DataFrame, context: str) -> None:
        """Validate that a DataFrame contains the required inference key columns.

        Args:
            df: DataFrame to validate.
            context: Human-readable context used in error messages.

        Raises:
            RuntimeError: If required key columns are missing.
        """
        missing = [
            col for col in REQUIRED_INFERENCE_KEY_COLUMNS
            if col not in df.columns
        ]

        if missing:
            raise RuntimeError(
                f"Missing required inference key column(s) in {context}: {missing}. "
                "Required columns are filename, region_id, and line_id."
            )

    def _filter_records_for_inference(self, records: list[dict]) -> list[dict]:
        """Filter records to those that should receive inference.

        If no ``line_augmentation`` column is present, all records are kept. If the
        column is present, only records marked as ``original`` are processed.

        Args:
            records: Dataset records converted from a DataFrame split.

        Returns:
            Records selected for inference.
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
        """Run the complete inference workflow.

        The workflow downloads the configured dataset, converts its splits to
        DataFrames, filters line records for inference, runs TrOCR prediction, writes
        the predictions back into timestamped inference columns, and optionally uploads
        the updated dataset.

        Returns:
            Updated DataFrames grouped by split name, or ``None`` if dataset loading
            fails.
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

        if self.push_to_hub:
            loader.validate_upload_target(
                upload_repo_name=self.upload_repo_name,
                upload_mode=self.upload_mode,
                allow_source_repo_update=self.allow_source_repo_update,
            )

        try:
            loader.download_hf_dataset()
            dfs = loader.to_dataframe()
            records = loader.convert_to_list_of_dicts(dfs)
            for split, df_split in dfs.items():
                self._validate_required_key_columns(df_split, f"split '{split}'")
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

        inferred: Dict[str, Dict[tuple[str, str, str, str], List[str]]] = {}

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
    ) -> dict[tuple[str, str, str, str], list[str]]:
        """Run TrOCR inference on selected image records.

        Args:
            records: Dataset records containing image data and line metadata.
            model: Loaded TrOCR-compatible model.
            processor: TrOCR processor used for image processing and decoding.
            device: Torch device used for model inference.

        Returns:
            Mapping from ``(project_name, filename, region_id, line_id)`` to one or more
            predicted text strings.
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
        inferred_lines: dict[tuple[str, str, str, str], list[str]] = {}

        for result in inference_result:
            try:
                project, filename, region_id, line_id, inferred_text = result

                key = (project, filename, region_id, line_id)
                inferred_lines.setdefault(key, []).append(inferred_text)

                self.statusManager.update_progress(
                    status_type="success",
                    current_item_name=f"{project}:{line_id}",
                )

            except ValueError as ve:
                logger.error(f"Malformed inference result: {result} - Error: {ve}")
                self.statusManager.update_progress(
                    status_type="failure_inference",
                    current_item_name="unknown_line_id",
                )

        return inferred_lines

    # ===========================================================================
    # SAVE RESULTS
    # ===========================================================================
    def write_inference_to_dataframe(
            self,
            inferred_lines: Dict[tuple[str, str, str, str], List[str]],
            original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Write inference predictions into a timestamped DataFrame column.

        Args:
            inferred_lines: Mapping from line identity keys to predicted text strings.
            original_df: Original split DataFrame to update.

        Returns:
            Copy of the input DataFrame with a new ``inference_*`` column.
        """
        logger.info("Writing inference results back to DataFrame.")

        updated_df = original_df.copy()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        model_name = self.trocr_model.replace("/", "_")
        new_col = f"inference_{timestamp}_model_{model_name}"

        updated_df[new_col] = ""
        updated_df[new_col] = updated_df[new_col].astype("string")

        updated_count = 0

        for (project, filename, region_id, line_id), texts in inferred_lines.items():
            mask = cast(
                pd.Series,
                (updated_df["filename"].fillna("").astype(str) == str(filename))
                & (updated_df["region_id"].fillna("").astype(str) == str(region_id))
                & (updated_df["line_id"].fillna("").astype(str) == str(line_id))
            )

            if (
                    "project_name" in updated_df.columns
                    and str(project).strip()
                    and updated_df["project_name"].fillna("").astype(str).str.strip().ne("").any()
            ):
                mask = cast(
                    pd.Series,
                    mask
                    & (updated_df["project_name"].fillna("").astype(str) == str(project))
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
                    f"No matching row found for project='{project}', filename='{filename}', "
                    f"region_id='{region_id}', line_id='{line_id}'"
                )
                continue

            if len(texts) != len(matching_indices):
                logger.warning(
                    f"Inference/writeback count mismatch for project='{project}', "
                    f"filename='{filename}', region_id='{region_id}', line_id='{line_id}': "
                    f"{len(texts)} predictions for {len(matching_indices)} matching rows. "
                    f"Writing up to the smaller count."
                )

            for row_index, text in zip(matching_indices, texts):
                updated_df.at[row_index, new_col] = text
                updated_count += 1

        logger.info(f"Created column '{new_col}' and updated {updated_count} rows.")
        return updated_df

    def save_results(self,
                     inferred_lines: Dict[tuple[str, str, str, str], List[str]],
                     original_df: pd.DataFrame) -> pd.DataFrame:
        """Save inference predictions to a DataFrame.

        Args:
            inferred_lines: Mapping from line identity keys to predicted text strings.
            original_df: Original split DataFrame to update.

        Returns:
            Updated DataFrame containing a new inference result column.

        Raises:
            Exception: If writing inference results fails.
        """
        logger.info("Saving inference results into DataFrame and updating XML fields.")

        try:
            df_with_inference = self.write_inference_to_dataframe(inferred_lines, original_df)

            logger.info("Inference results successfully written to DataFrame.")
            return df_with_inference

        except Exception as e:
            logger.error(f"Error while saving inference results: {e}")
            raise
