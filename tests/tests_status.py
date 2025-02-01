import unittest
from datetime import datetime, timedelta
from flow_inference.models import InferenceState, StateEnum
from flow_inference.status import Status


class TestStatus(unittest.TestCase):
    def setUp(self):
        """Set up a fresh instance of Status for each test."""
        self.test_state = InferenceState(
            process_id="1234",
            repo_name="github-actions-test-organisation/inference_test",
            repo_folder="xml",
            files_total=0,
            files_successful=0,
            files_failed_download=0,
            files_failed_inference=0,
            filenames_failed_download=[],
            filenames_failed_inference=[],
            filenames_successful=[],
            progress=0,
            state=StateEnum.IN_PROGRESS,
            created_at=None,
            runtime=0
        )
        self.status = Status(self.test_state)

    # ==========================================================================
    # TESTS
    # ==========================================================================

    def test_initialize_status(self):
        """Test that initialize_status correctly updates state."""
        files_fetched = ["file1.jpg", "file2.png"]
        updated_state = self.status.initialize_status(files_fetched)

        self.assertEqual(updated_state.files_total, 2)
        self.assertEqual(updated_state.state, StateEnum.IN_PROGRESS)
        self.assertIsNotNone(updated_state.created_at)
        self.assertEqual(updated_state.runtime, 0)

    def test_calculate_runtime_with_valid_created_at(self):
        """Test calculate_runtime when created_at is properly set."""
        self.test_state.created_at = datetime.now() - timedelta(seconds=10)
        runtime = self.status.calculate_runtime()

        self.assertGreaterEqual(runtime, 10)

    def test_calculate_processed_files(self):
        """Test calculate_processed_files correctly sums processed files."""
        self.test_state.files_successful = 3
        self.test_state.files_failed_download = 2
        self.test_state.files_failed_inference = 1

        self.assertEqual(self.status.calculate_processed_files(), 6)

    def test_update_progress_no_files(self):
        """Test update_progress when no files are processed."""
        files_fetched = ["file1.jpg", "file2.png"]
        self.status.initialize_status(files_fetched)
        self.test_state.files_total = 0
        self.status.update_progress()

        self.assertEqual(self.test_state.progress, 0)

    def test_update_progress_partial_completion(self):
        """Test update_progress when some files are processed."""
        self.test_state.files_total = 5
        self.test_state.files_successful = 2
        self.test_state.files_failed_download = 1
        self.test_state.files_failed_inference = 0

        files_fetched = ["file1.jpg", "file2.png", "file3.jpg", "file4.png", "file5.jpg"]
        self.status.initialize_status(files_fetched)
        self.status.update_progress()
        self.assertEqual(self.test_state.progress, 60)

    def test_update_progress_full_completion(self):
        """Test update_progress when all files are processed."""
        files_fetched = ["file1.jpg", "file2.png"]
        self.status.initialize_status(files_fetched)
        self.test_state.files_total = 4
        self.test_state.files_successful = 4
        self.test_state.files_failed_download = 0
        self.test_state.files_failed_inference = 0

        self.status.update_progress()
        self.assertEqual(self.test_state.progress, 100)

    def test_update_file_status_download_failure(self):
        """Test _update_file_status for a failed download."""
        self.status._update_file_status("failure_download", "file1.jpg")

        self.assertEqual(self.test_state.files_failed_download, 1)
        self.assertIn("file1.jpg", self.test_state.filenames_failed_download)

    def test_update_file_status_inference_failure(self):
        """Test _update_file_status for a failed inference."""
        self.status._update_file_status("failure_inference", "file2.jpg")

        self.assertEqual(self.test_state.files_failed_inference, 1)
        self.assertIn("file2.jpg", self.test_state.filenames_failed_inference)

    def test_update_file_status_success(self):
        """Test _update_file_status for a successful inference."""
        self.status._update_file_status("success", "file3.jpg")

        self.assertEqual(self.test_state.files_successful, 1)
        self.assertIn("file3.jpg", self.test_state.filenames_successful)


if __name__ == '__main__':
    unittest.main()
