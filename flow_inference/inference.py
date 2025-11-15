# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from flow_inference.data_handling import HuggingFaceDataHandler
from flow_inference.image_processing import ImageHandler
from flow_inference.model_handling import ModelManager
from flow_inference.status import Status
from flow_inference.utils.logging.inference_logger import logger
from flow_inference.infer_textlines import InferenceHandler
import pandas as pd

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
                 upload_repo_name: Optional[str] = None
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
        self.statusManager = Status()

        logger.debug(f"Inference initialized with Hugging Face dataset: {download_repo_name}")

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
            huggingface_token=self.hf_token
        )

        try:
            loader.download_hf_dataset()
            dfs = loader.to_dataframe()
            records = loader.convert_to_list_of_dicts(dfs)
        except Exception as e:
            logger.error(f"Failed to load dataset from Hugging Face: {e}")
            return None

        total_records = sum(len(r) for r in records.values())
        self.statusManager.initialize_status(total_records)
        logger.info(f"Dataset loaded successfully: {total_records} image records found.")

        # -------------------------------
        # STEP 2: Run inference
        # -------------------------------
        logger.debug("Running inference on image records.")

        inferred = {}

        for split, recs in records.items():
            if split in self.requested_splits or split == "default":
                result = self.run_inference(recs)
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

        for split, df in updated_dfs.items():
            for col in df.columns:
                df[col] = df[col].astype("string")

        if self.push_to_hub:
            loader.df = updated_dfs
            loader.push_to_hub(
                upload_repo_name=self.upload_repo_name,
                private=self.private_repo,
                commit_message="Add inference results"
            )

        logger.info("Inference process completed successfully.")
        logger.info(f"Total runtime: {self.statusManager.calculate_runtime()}")

        self.statusManager.summary()

        return updated_dfs

    # ===========================================================================
    # INFERENCE
    # ===========================================================================
    def run_inference(self, records: List[dict]) -> Optional[Dict[str, str]]:
        """
        Run inference on provided image records.
        """
        logger.debug(f"Running inference on {len(records)} records.")

        # Load model and processor
        model_manager = ModelManager()
        processor = model_manager.load_processor(self.trocr_model)
        model = model_manager.load_model(self.trocr_model)

        if processor is None or model is None:
            logger.error("Failed to load model or processor. Aborting.")
            return None

        image_handler = ImageHandler(
            processor=processor,
            target_image_size=self.target_image_size
        )
        inference_handler = InferenceHandler(
            device=model_manager.device,
            model=model,
            processor=processor
        )

        inference_result = inference_handler.infer(records=records,
                                                   image_handler=image_handler)
        logger.info("Inference completed successfully.")

        # Parse results
        inferred_lines = {}
        for result in inference_result:
            try:
                filename, inferred_text = result.split("\t")
                inferred_lines[filename] = inferred_text
                self.statusManager.update_progress(status_type="success", current_item_name=filename)
            except ValueError as ve:
                logger.error(f"Malformed inference result: {result} - Error: {ve}")
                self.statusManager.update_progress(status_type="failure_inference", current_item_name=filename)

        return inferred_lines

    # ===========================================================================
    # SAVE RESULTS
    # ===========================================================================
    def write_inference_to_dataframe(
            self,
            inferred_lines: Dict[str, str],
            original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Write inference results into a new timestamped column in the dataframe.
        """
        logger.info("Writing inference results back to DataFrame.")

        updated_df = original_df.copy()

        # Create unique timestamped column name
        timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
        new_col = f"inference_{timestamp}_model_{self.trocr_model}"

        updated_df[new_col] = None
        updated_df[new_col] = updated_df[new_col].astype("string").fillna("")

        updated_count = 0
        for filename, text in inferred_lines.items():
            if filename in updated_df["filename"].values:
                updated_df.loc[updated_df["filename"] == filename, new_col] = text
                updated_count += 1
            else:
                logger.warning(f"No matching filename found for {filename}")

        logger.info(f"Created column '{new_col}' and updated {updated_count} rows.")
        return updated_df

    def save_results(self, inferred_lines: Dict[str, str], original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Save inference results: add new column + update raw XML strings.
        Returns the updated DataFrame.
        """
        logger.info("Saving inference results into DataFrame and updating XML fields.")

        try:
            # write inference column
            df_with_inference = self.write_inference_to_dataframe(inferred_lines, original_df)

            logger.info("Inference results successfully written to DataFrame.")
            return df_with_inference

        except Exception as e:
            logger.error(f"Error while saving inference results: {e}")
            raise
