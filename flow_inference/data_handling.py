import os
from typing import List, Tuple
import dotenv
from flow_githubmanager.github_interaction import GitHubManager


class DataHandler:
    """
    Class for handling logic related to file transfers from and to GitHub repositories.
    """

    def __init__(self, download_path: str, upload_path: str):
        """
        :param download_path: the (local) path the files are taken from.
        :param upload_path: the (local) path the files are taken from when uploaded.
        """
        dotenv.load_dotenv()
        access_token = os.getenv("GITHUB_ACCESS_TOKEN")
        self.github_manager = GitHubManager(access_token)
        self.in_path = download_path
        self.out_path = upload_path

    def fetch_xml_files_from_github(self,
                                    repo_name: str,
                                    download_path: str) -> Tuple[List[str], List[str]]:
        """
        Fetches XML files from GitHub.

        :param repo_name: the name of the GitHub repository the files are fetched from.
        :param download_path: the (local) path the files are downloaded to.
        :return: Tuple consisting of list of files which were downloaded from the GitHub repository and
        list of files which couldn't be downloaded.
        """
        fetched_files = self.github_manager.fetch_files(repo_name,
                                                        "xml",
                                                        ".xml",
                                                        download_path)
        return fetched_files

    def push_xml_files_to_github(self,
                                 repo_name: str,
                                 file_paths: List[str]) -> None:
        """
        Pushes XML to a GitHub repository.

        :param repo_name: the name of the GitHub repository the files are pushed to.
        :param file_paths: the files which are uploaded to the GitHub repository (with their paths specified).
        :return: None.
        """
        self.github_manager.upload_documents(repo_name, file_paths, folder_name="inference")
