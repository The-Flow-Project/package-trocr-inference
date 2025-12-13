import unittest
from datetime import datetime, timedelta
from flow_inference.status import Status


class TestStatus(unittest.TestCase):
    def setUp(self):
        """Set up a fresh instance of Status for each test."""
        self.status = Status()

    # ==========================================================================
    # TESTS
    # ==========================================================================

    def test_initialize_status(self):
        """Test that initialize_status correctly updates state."""
        total_files = 2
        self.status.initialize_status(total_files)

        self.assertEqual(self.status.total_files, 2)
        self.assertEqual(self.status.successful, 0)
        self.assertEqual(self.status.failed_download, 0)
        self.assertEqual(self.status.failed_inference, 0)
        self.assertIsNotNone(self.status.start_time)

    def test_calculate_runtime_with_valid_start_time(self):
        """Test calculate_runtime when start_time is properly set."""
        self.status.start_time = datetime.now() - timedelta(seconds=10)
        runtime = self.status.calculate_runtime()

        # It should return a string like "10s" or "0m 10s"
        self.assertIsInstance(runtime, str)
        self.assertTrue(runtime.endswith("s"))

    def test_calculate_processed_files(self):
        """Test calculate_processed_files correctly sums processed files."""
        self.status.successful = 3
        self.status.failed_download = 2
        self.status.failed_inference = 1

        self.assertEqual(self.status.calculate_processed_files(), 6)

    def test_update_progress_no_files(self):
        """Test update_progress when no files are processed."""
        self.status.initialize_status(total_files=0)
        self.status.update_progress()
        # Should not crash or divide by zero
        self.assertEqual(self.status.calculate_processed_files(), 0)

    def test_update_progress_partial_completion(self):
        """Test update_progress when some files are processed."""
        self.status.initialize_status(total_files=5)
        self.status.successful = 2
        self.status.failed_download = 1
        self.status.failed_inference = 0
        self.status.update_progress()
        processed = self.status.calculate_processed_files()
        self.assertEqual(processed, 3)

    def test_update_progress_full_completion(self):
        """Test update_progress when all files are processed."""
        self.status.initialize_status(total_files=4)
        self.status.successful = 4
        self.status.failed_download = 0
        self.status.failed_inference = 0
        self.status.update_progress()
        processed = self.status.calculate_processed_files()
        self.assertEqual(processed, 4)

    def test_update_file_status_download_failure(self):
        """Test update_file_status for a failed download."""
        self.status.update_file_status("failure_download", "file1.jpg")
        self.assertEqual(self.status.failed_download, 1)

    def test_update_file_status_inference_failure(self):
        """Test update_file_status for a failed inference."""
        self.status.update_file_status("failure_inference", "file2.jpg")
        self.assertEqual(self.status.failed_inference, 1)

    def test_update_file_status_success(self):
        """Test update_file_status for a successful inference."""
        self.status.update_file_status("success", "file3.jpg")
        self.assertEqual(self.status.successful, 1)


if __name__ == '__main__':
    unittest.main()
