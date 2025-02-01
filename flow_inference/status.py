# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from datetime import datetime
from typing import List, Optional
from flow_inference.models import InferenceState, StateEnum
from flow_inference.utils.logging.inference_logger import logger


# ===============================================================================
# CLASS
# ===============================================================================
class Status:
    def __init__(self, state: InferenceState) -> None:
        """
        initialise class parameters.

        :param state: the state of the inference.
        """
        self.state = state
        logger.debug(f"Initialized Status with state: {self.state}")

    def initialize_status(self, files_fetched: List) -> InferenceState:
        """
        Initialize status.

        :param files_fetched: the list of files fetched.
        :return: the status of the inference.
        """
        logger.debug(f"Initializing status with {len(files_fetched)} fetched files.")

        self.state.files_total = len(files_fetched)
        self.state.state = StateEnum.IN_PROGRESS
        self.state.runtime = 0
        self.state.created_at = datetime.now()

        logger.debug(f"Status initialized: {self.state.files_total} files to process.")
        return InferenceState(**self.state.model_dump(by_alias=True))

    def calculate_runtime(self) -> int:
        """
        Calculate runtime.

        :return: runtime in seconds as int.
        """
        delta = datetime.now() - self.state.created_at
        runtime = int(delta.total_seconds())
        logger.debug(f"Calculated runtime: {runtime} seconds.")
        return runtime

    def calculate_processed_files(self) -> int:
        """
        Calculate the number of files processed (success, failed download, failed inference).
        :return: Total processed files.
        """
        processed_files = (self.state.files_successful
                           + self.state.files_failed_download
                           + self.state.files_failed_inference)
        logger.debug(f"Calculated processed files: {processed_files}.")
        return processed_files

    def update_progress(self, status_type: Optional[str] = None,
                        current_item_name: Optional[str] = None,
                        state_enum: Optional[StateEnum] = None) -> InferenceState:
        """
        Update progress based on file status or inference state.
        """
        logger.debug(f"Updating progress for item: {current_item_name}. Status type: {status_type}")

        if status_type:
            self._update_file_status(status_type, current_item_name)

        processed_files = self.calculate_processed_files()
        self.state.progress = int((processed_files / self.state.files_total) * 100) if self.state.files_total > 0 else 0
        self.state.runtime = self.calculate_runtime()

        if state_enum:
            self.state.state = state_enum
            logger.debug(f"State updated to: {self.state.state}")

        logger.debug(f"Progress updated: {self.state.progress}% complete.")
        return InferenceState(**self.state.model_dump(by_alias=True))

    def _update_file_status(self, status_type: str, current_item_name: str):
        """
        Helper function to update file-specific statuses.
        """
        logger.debug(f"Updating file status for {current_item_name}. Status type: {status_type}")

        if status_type == "failure_download":
            if self.state.filenames_failed_download is None:
                self.state.filenames_failed_download = []
            self.state.files_failed_download += 1
            self.state.filenames_failed_download.append(current_item_name)
            logger.warning(
                f"Download failed for file: {current_item_name}. Total failed downloads: {self.state.files_failed_download}")

        elif status_type == "failure_inference":
            if self.state.filenames_failed_inference is None:
                self.state.filenames_failed_inference = []
            self.state.files_failed_inference += 1
            self.state.filenames_failed_inference.append(current_item_name)
            logger.error(
                f"Inference failed for file: {current_item_name}. Total failed inferences: {self.state.files_failed_inference}")

        elif status_type == "success":
            if self.state.filenames_successful is None:
                self.state.filenames_successful = []
            self.state.files_successful += 1
            self.state.filenames_successful.append(current_item_name)
            logger.info(
                f"File processed successfully: {current_item_name}. Total successful files: {self.state.files_successful}")
