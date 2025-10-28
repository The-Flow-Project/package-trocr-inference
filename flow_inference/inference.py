# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from datetime import datetime
import os
from typing import Optional, Tuple, Dict, List, Callable, Coroutine, Any
from flow_inference.data_handling import HuggingFaceDataHandler
from flow_inference.image_processing import ImageHandler
from flow_inference.model_handling import ModelManager
from flow_inference.models import InferenceState, StateEnum
from flow_inference.status import Status
from flow_inference.utils.logging.inference_logger import logger
from flow_inference.infer_textlines import InferenceHandler
import pandas as pd

from flow_inference.xml_processing import XMLProcessor


# ===============================================================================
# CLASS
# ===============================================================================
class Inference:
    def __init__(self,
                 process_id,
                 hf_repo_name: str,
                 hf_token: Optional[str],
                 trocr_model="microsoft/trocr-large-handwritten",
                 target_image_size: Tuple[int, int] = None,
                 stop_on_fail: bool = False,
                 callback_inference: Callable[[dict], Coroutine[Any, Any, None]] = None,
                 **kwargs
                 ) -> None:

        self.process_id = process_id
        self.hf_repo_name = hf_repo_name
        self.hf_token = hf_token
        self.trocr_model = trocr_model
        self.target_image_size = target_image_size
        self.stop_on_fail = stop_on_fail
        self.callback = callback_inference
        self.kwargs = kwargs

        # Initialize inference state tracking
        state = InferenceState(
            process_id=self.process_id,
            hf_repo_name=self.hf_repo_name,
            trocr_model=trocr_model,
            image_size=target_image_size,
            **self.kwargs
        )
        self.progressStatus = InferenceState(**state.model_dump(by_alias=True))
        self.statusManager = Status(self.progressStatus)

        logger.debug(f"Inference initialized with Hugging Face dataset: {hf_repo_name}")

    # ===========================================================================
    # MAIN PIPELINE
    # ===========================================================================
    async def perform_inference(self) -> Optional[pd.DataFrame]:
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
            dataset_id=self.hf_repo_name,
            huggingface_token=self.hf_token
        )

        try:
            loader.download()
            df = loader.to_dataframe()
            records = loader.get_image_records()
        except Exception as e:
            logger.error(f"Failed to load dataset from Hugging Face: {e}")
            self.progressStatus = self.statusManager.update_progress(state_enum=StateEnum.FAILED)
            if self.callback:
                await self.callback(self.progressStatus.model_dump(by_alias=True))
            return None

        logger.info(f"Dataset loaded successfully: {len(records)} image records found.")
        self.progressStatus = self.statusManager.initialize_status(files_fetched=records)
        if self.callback:
            await self.callback(self.progressStatus.model_dump(by_alias=True))

        # -------------------------------
        # STEP 2: Run inference
        # -------------------------------
        logger.debug("Running inference on image records.")
        inferred_lines = await self.run_inference(records)

        if not inferred_lines:
            logger.error("Inference failed or produced no results.")
            self.progressStatus = self.statusManager.update_progress(state_enum=StateEnum.FAILED)
            if self.callback:
                await self.callback(self.progressStatus.model_dump(by_alias=True))
            return None

        # -------------------------------
        # STEP 3: Write inference results
        # -------------------------------
        logger.debug("Writing inference results to DataFrame.")
        updated_df = self.save_results(inferred_lines, df)

        self.progressStatus = self.statusManager.calculate_runtime()
        self.progressStatus = self.statusManager.update_progress(state_enum=StateEnum.DONE)

        if self.callback:
            await self.callback(self.progressStatus.model_dump(by_alias=True))

        logger.info("Inference process completed successfully.")
        return updated_df

    # ===========================================================================
    # INFERENCE
    # ===========================================================================
    async def run_inference(self, records: List[dict]) -> Optional[Dict[str, str]]:
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

        inference_result = inference_handler.infer(file_names=records, image_handler=image_handler)
        logger.info("Inference completed successfully.")

        # Parse results
        inferred_lines = {}
        for result in inference_result:
            try:
                file_name, inferred_text = result.split("\t")
                inferred_lines[file_name] = inferred_text
                self.progressStatus = self.statusManager.update_progress(status_type="success",
                                                                         current_item_name=file_name)
                if self.callback:
                    await self.callback(self.progressStatus.model_dump(by_alias=True))
            except ValueError as ve:
                logger.error(f"Malformed inference result: {result} - Error: {ve}")
                self.progressStatus = self.statusManager.update_progress(status_type="failure_inference",
                                                                         current_item_name=file_name)
                if self.callback:
                    await self.callback(self.progressStatus.model_dump(by_alias=True))

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
        new_col = f"inference_{timestamp}"

        updated_df[new_col] = None

        updated_count = 0
        for filename, text in inferred_lines.items():
            if filename in updated_df["filename"].values:
                updated_df.loc[updated_df["filename"] == filename, new_col] = text
            else:
                logger.warning(f"No matching filename found for {filename}")

        logger.info(f"Created column '{new_col}' and updated {updated_count} rows.")
        return updated_df

    def update_raw_xml_in_records(
            self,
            inferred_lines: Dict[str, str],
            original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Update the raw XML column in the DataFrame with its corresponding inference results.
        Returns the updated DataFrame.
        """
        logger.info("Updating raw XML strings in DataFrame with inference results.")

        updated_df = original_df.copy()
        updated_count = 0

        for idx, row in updated_df.iterrows():
            filename = row.get("filename")
            raw_xml = row.get("xml")

            if not raw_xml:
                logger.debug(f"No XML found for {filename}. Skipping.")
                continue

            inferred_text = inferred_lines.get(filename)
            if not inferred_text:
                continue

            try:
                xml_processor = XMLProcessor.from_string(raw_xml)
                xml_processor.insert_inferred_lines(
                    root=xml_processor.root,
                    inferred_lines={filename: inferred_text}
                )

                # Convert updated XML tree back to string
                import io
                xml_str = io.StringIO()
                xml_processor.tree.write(xml_str, encoding="unicode")
                updated_df.at[idx, "xml"] = xml_str.getvalue()
                updated_count += 1

            except Exception as e:
                logger.error(f"Failed to update XML for {filename}: {e}")

        logger.info(f"Updated XML for {updated_count} records successfully.")
        return updated_df

    def save_results(self, inferred_lines: Dict[str, str], original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Save inference results: add new column + update raw XML strings.
        Returns the updated DataFrame.
        """
        logger.info("Saving inference results into DataFrame and updating XML fields.")

        try:
            # Step 1: write inference column
            df_with_inference = self.write_inference_to_dataframe(inferred_lines, original_df)

            # Step 2: update raw XMLs
            df_final = self.update_raw_xml_in_records(inferred_lines, df_with_inference)

            logger.info("Inference results successfully written to DataFrame and XMLs updated.")
            return df_final

        except Exception as e:
            logger.error(f"Error while saving inference results: {e}")
            raise
