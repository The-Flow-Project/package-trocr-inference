"""Track and log progress for inference jobs."""
# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
from datetime import datetime
from flow_inference.utils.logging.inference_logger import logger


# ===============================================================================
# CLASS
# ===============================================================================
class Status:
    """Track file-level progress and runtime during inference.

    The status object stores counters for successful files, failed downloads,
    and failed inference attempts. It also logs progress updates and a final
    summary for long-running inference jobs.
    """
    def __init__(self) -> None:
        """Initialize empty status counters."""
        self.start_time = None
        self.total_files = 0
        self.successful = 0
        self.failed_download = 0
        self.failed_inference = 0
        logger.debug(f"Initialized Status.")

    def initialize_status(self, total_files: int):
        """Initialize counters for a new inference run.

        Args:
            total_files: Total number of files expected in the inference run.
        """
        self.start_time = datetime.now()
        self.total_files = total_files
        self.successful = 0
        self.failed_download = 0
        self.failed_inference = 0
        logger.info(f"Starting inference on {total_files} files.")

    def calculate_runtime(self) -> str:
        """Calculate the elapsed runtime since status initialization.

        Returns:
            Human-readable runtime string in seconds or minutes and seconds.

        Raises:
            RuntimeError: If the status has not been initialized yet.
        """
        if self.start_time is None:
            raise RuntimeError("Status has not been initialized.")
        delta = datetime.now() - self.start_time
        total_sec = int(delta.total_seconds())
        if total_sec < 60:
            return f"{total_sec}s"
        minutes, seconds = divmod(total_sec, 60)
        return f"{minutes}m {seconds}s"

    def calculate_processed_files(self) -> int:
        """Calculate the number of files processed so far.

        Returns:
            Number of successful, download-failed, and inference-failed files.
        """
        processed_files = self.successful + self.failed_download + self.failed_inference
        logger.debug(f"Calculated processed files: {processed_files}.")
        return processed_files

    def update_progress(self,
                        status_type: str | None = None,
                        current_item_name: str | None = None):
        """Update and log current inference progress.

        Args:
            status_type: Optional file status to register before logging progress.
            current_item_name: Optional file name associated with the status update.
        """
        if status_type and current_item_name:
            self.update_file_status(status_type, current_item_name)

        processed_files = self.calculate_processed_files()
        progress = int((processed_files / self.total_files) * 100) if self.total_files > 0 else 0
        runtime = self.calculate_runtime()

        logger.info(
            f"Progress: {progress}% ({processed_files}/{self.total_files}) "
            f"| Runtime: {runtime}"
        )

    def update_file_status(self, status_type: str, file_name: str):
        """Update counters for a single file status.

        Args:
            status_type: File status, such as ``"success"``, ``"failure_download"``,
                or ``"failure_inference"``.
            file_name: Name of the file associated with the status update.
        """
        if status_type == "failure_download":
            self.failed_download += 1
            logger.warning(
                f"Download failed for file: {file_name} "
                f"(Total failed downloads: {self.failed_download})"
            )

        elif status_type == "failure_inference":
            self.failed_inference += 1
            logger.error(
                f"Inference failed for file: {file_name} "
                f"(Total failed inferences: {self.failed_inference})"
            )

        elif status_type == "success":
            self.successful += 1
            logger.info(
                f"File processed successfully: {file_name} "
                f"(Total successful files: {self.successful})"
            )

        else:
            logger.debug(f"Unknown status type '{status_type}' for file {file_name}.")

    def summary(self):
        """Log a final summary for the inference run."""
        logger.info("Inference completed.")
        logger.info(f"Successful: {self.successful}")
        logger.info(f"Failed downloads: {self.failed_download}")
        logger.info(f"Failed inference: {self.failed_inference}")
        logger.info(f"Total runtime: {self.calculate_runtime()}")