import glob
import os
import unittest

from flow_inference.data_handling import DataHandler


class TestDataHandler(unittest.TestCase):
    def setUp(self):
        current_dir: str = os.path.dirname(os.path.realpath(__file__))
        self.in_path_github: str = os.path.join(current_dir, "..", "test_data", "github_download")
        self.out_path_github: str = os.path.join(current_dir, "..", "test_data", "xml_with_inference")
        if not os.path.exists(self.in_path_github):
            os.makedirs(self.in_path_github)
        self.data_handler = DataHandler(self.in_path_github, self.in_path_github)

    def test_fetch_xml_files_from_github(self):
        self.data_handler.fetch_xml_files_from_github("github-actions-test-organisation/inference_test",
                                                      "xml",
                                                      self.in_path_github)

    def test_push_xml_files_to_github(self):
        xml_files = [f for f in os.listdir(self.out_path_github) if f.endswith('.xml')]
        xml_files = [os.path.join(self.out_path_github, f) for f in xml_files]
        self.data_handler.push_xml_files_to_github("github-actions-test-organisation/inference_test",
                                                   xml_files)


if __name__ == '__main__':
    unittest.main()
