import os
from typing import List

import dotenv
from flow_githubmanager.github_interaction import GitHubManager


class DataHandler:
    def __init__(self, in_path: str, out_path: str):
        dotenv.load_dotenv()
        access_token = os.getenv("GITHUB_ACCESS_TOKEN")
        self.github_manager = GitHubManager(access_token)
        self.in_path = in_path
        self.out_path = out_path

    def fetch_xml_files_from_github(self,
                                    repo_name: str,
                                    file_output_path: str):
        preprocessed_files = self.github_manager.fetch_files(repo_name,
                                                             "xml",
                                                             ".xml",
                                                             file_output_path)
        return preprocessed_files

    def push_xml_files_to_github(self,
                                 repo_name: str,
                                 file_paths: List[str]):
        self.github_manager.upload_documents(repo_name, file_paths, folder_name="inference")


if __name__ == '__main__':
    current_dir: str = os.path.dirname(os.path.realpath(__file__))
    in_path_github: str = os.path.join(current_dir, "..", "github_download")
    if not os.path.exists(in_path_github):
        os.makedirs(in_path_github)
    data_handler = DataHandler(in_path_github, in_path_github)
    data_handler.fetch_xml_files_from_github("github-actions-test-organisation/github-interaction-test",
                                             in_path_github)

