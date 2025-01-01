# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
import os
from typing import List, Tuple
import dotenv
from flow_githubmanager.github_interaction import GitHubManager
from requests.exceptions import HTTPError
from flow_inference.utils.logging.inference_logger import logger


# ===============================================================================
# CLASS
# ===============================================================================
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
        if not access_token:
            raise ValueError("GitHub access token not found in environment variables.")
        self.github_manager = GitHubManager(access_token)
        self.in_path = download_path
        self.out_path = upload_path

    def fetch_xml_files_from_github(self,
                                    repo_name: str,
                                    folder_path: str,
                                    download_path: str) -> Tuple[List[str], List[str]]:
        """
        Fetches XML files from GitHub.

        :param folder_path: the folder name in the repository.
        :param repo_name: the name of the GitHub repository the files are fetched from.
        :param download_path: the (local) path the files are downloaded to.
        :return: Tuple consisting of list of files which were downloaded from the GitHub repository and
        list of files which couldn't be downloaded.
        """
        try:
            logger.info(f"Attempting to fetch XML files from repo '{repo_name}' in folder '{folder_path}'...")
            fetched_files = self.github_manager.fetch_files(repo_name,
                                                            folder_path,
                                                            ".xml",
                                                            download_path)
            return fetched_files
        except HTTPError as e:
            logger.error(f"HTTP error occurred while fetching files from GitHub: {e}")
            raise
        except FileNotFoundError as e:
            logger.error(f"File not found error occurred while fetching files from GitHub: {e}")
            raise
        except ValueError as e:
            logger.error(f"Value error occurred during fetching files: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error occurred while fetching files from GitHub: {e}")
            raise

    def push_xml_files_to_github(self,
                                 repo_name: str,
                                 file_paths: List[str]) -> None:
        """
        Pushes XML to a GitHub repository.

        :param repo_name: the name of the GitHub repository the files are pushed to.
        :param file_paths: the files which are uploaded to the GitHub repository (with their paths specified).
        :return: None.
        """
        try:
            logger.info(f"Attempting to push {len(file_paths)} XML files to repo '{repo_name}'...")
            self.github_manager.upload_documents(repo_name, file_paths, folder_name="inference")
            logger.info(f"Successfully pushed {len(file_paths)} XML files to the GitHub repository.")
        except HTTPError as e:
            logger.error(f"HTTP error occurred while pushing files to GitHub: {e}")
            raise
        except FileNotFoundError as e:
            logger.error(f"File not found error occurred during file upload to GitHub: {e}")
            raise
        except ValueError as e:
            logger.error(f"Value error occurred while pushing files: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error occurred while pushing files to GitHub: {e}")
            raise
