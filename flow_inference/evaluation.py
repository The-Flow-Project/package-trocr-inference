import json
import os
from datetime import datetime
from typing import List, Optional, Tuple
from evaluate import load
from flow_inference.data_handling import HuggingFaceDataHandler
from flow_inference.inference import Inference
from flow_inference.utils.logging.inference_logger import logger
from flow_inference.xml_processing import XMLProcessor


class Evaluation:
    def __init__(self,
                 process_id,
                 repo_name: str,
                 model_name: str,
                 hf_url: str,
                 github_access_token: Optional[str],
                 directory: str = "data"):
        self.process_id = process_id
        self.repo_name = repo_name
        self.repo_folder = "xml"
        self.github_access_token = github_access_token
        self.modified_repo_name = repo_name.replace("/", "___")
        self.directory = directory
        self.repo_base_path = os.path.join(self.directory, self.modified_repo_name)
        self.in_path = os.path.join(self.repo_base_path, "fetched")
        self.out_path = os.path.join(self.repo_base_path, "evaluation_results")
        self.model_name = model_name
        self.hf_url = hf_url
        os.makedirs(self.out_path, exist_ok=True)

        # initialise DataHandler
        # self.data_handler = DataHandler(download_path=self.in_path, upload_path=self.out_path)

    # TODO: perform line segmentation if XML not segmented yet
    @staticmethod
    def perform_line_segmentation():
        """ Placeholder for future line segmentation logic. """
        print("Performing line segmentation (not implemented yet).")

    def _fetch_xml_files(self) -> List[str]:
        """
        Fetch XML files from GitHub.

        :return: List of fetched files.
        """
        logger.debug("Fetching XML files...")
        fetched_files, failed_files = self.data_handler.fetch_xml_files_from_github(
            repo_name=self.repo_name,
            folder_path=self.repo_folder,
            download_path=self.in_path
        )

        if not fetched_files:
            logger.error("No XML files fetched. Exiting evaluation.")
            return []

        logger.info(f"Successfully fetched {len(fetched_files)} XML files.")
        return fetched_files

    def _process_ground_truth(self, fetched_files: List[str]) -> List[str]:
        """
        Extracts ground truth text from XML files and performs segmentation if necessary.

        :param fetched_files: List of XML file paths.
        :return: List of extracted ground truth lines.
        """
        gt_lines = []

        for fetched_file in fetched_files:
            xml_processor = XMLProcessor(fetched_file)
            extracted_texts = xml_processor.extract_all_text_lines()

            if extracted_texts:
                gt_lines.extend(extracted_texts)
            else:
                text_lines_exist = bool(xml_processor.root.findall(f".//{xml_processor.xmlns}TextLine"))

                if not text_lines_exist:
                    self.perform_line_segmentation()  # Your existing segmentation placeholder

        if not gt_lines:
            logger.error("No ground truth text extracted. Exiting evaluation.")
            return []

        return gt_lines

    async def _perform_inference(self) -> List[str]:
        """
        Run model inference.

        :return: List of inferred lines.
        """
        inference = Inference(
            process_id=self.process_id,
            hf_repo_name=self.repo_name,
            hf_token=self.github_access_token,
            directory=self.directory
        )

        inferred_lines_dict = await inference.perform_inference()

        if not inferred_lines_dict:
            logger.error("Inference failed or produced no results. Exiting evaluation.")
            return []

        return list(inferred_lines_dict.values())

    @staticmethod
    def _compute_cer(gt_lines: List[str], inferred_lines: List[str]) -> float:
        """
        Compute the Character Error Rate (CER) given ground truth and inferred text.

        :param gt_lines: List of ground truth text lines.
        :param inferred_lines: List of inferred text lines.
        :return: Computed CER score.
        """
        if len(gt_lines) != len(inferred_lines):
            logger.warning("Mismatch between number of inferred lines and ground truth lines.")

        cer_metric = load('cer')
        cer_metric.add_batch(predictions=inferred_lines, references=gt_lines)
        cer_score = cer_metric.compute()

        logger.info(f"Computed CER: {cer_score}")
        return cer_score

    def _save_results(self, gt_lines: List[str], inferred_lines: List[str]) -> Tuple[str, str]:
        """
        Save the ground truth and hypothesis text files locally.

        :param gt_lines: List of ground truth text lines.
        :param inferred_lines: List of inferred text lines.
        :return: Tuple (gt_file, hypothesis_file) containing the file paths.
        """
        gt_file = os.path.join(self.out_path, "gt.txt")
        hypothesis_file = os.path.join(self.out_path, "hypothesis.txt")

        self.write_lines_to_file(gt_lines, gt_file)
        self.write_lines_to_file(inferred_lines, hypothesis_file)

        logger.info(f"Saved ground truth and hypothesis files to {self.out_path}")
        return gt_file, hypothesis_file

    @staticmethod
    def write_lines_to_file(lines: List[str], file_path: str) -> None:
        """
        Writes a list of strings to a text file, each string on a new line.

        :param lines: The list of strings to write.
        :param file_path: The path to the output text file.

        :return: None
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    @staticmethod
    def count_lines(file_path: str) -> int:
        with open(file_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _generate_evaluation_report(self,
                                    model_name,
                                    hf_url,
                                    gt_file: str,
                                    hypothesis_file: str,
                                    cer_score: float) -> str:
        """
        Generate an evaluation report using your existing logic and save it locally.

        :param gt_file: File path of the ground truth text.
        :param hypothesis_file: File path of the hypothesis text.
        :param cer_score: Computed CER score.
        :return: The saved evaluation report.
        """
        report = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "hf_url": hf_url,
            "num_lines_hypothesis": self.count_lines(hypothesis_file),
            "num_lines_gt": self.count_lines(gt_file),
            "cer_score": cer_score
        }

        report_file = os.path.join(self.out_path, "evaluation_report.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        logger.info(f"Evaluation report saved: {report_file}")
        return report_file

    def _push_saved_results(self, gt_file: str, hypothesis_file: str, report_file: str):
        """
        Push the evaluation result files to GitHub using known file paths.

        :param gt_file: Path to the saved ground truth file.
        :param hypothesis_file: Path to the saved hypothesis file.
        :param report_file: Path to the saved evaluation report.
        """
        result_files = [gt_file, hypothesis_file, report_file]
        try:
            logger.info(f"Pushing {len(result_files)} evaluation result files to GitHub...")
            self.data_handler.push_xml_files_to_github(self.repo_name, result_files)
            logger.info("Successfully pushed evaluation results to GitHub.")
        except Exception as e:
            logger.error(f"Failed to push evaluation results to GitHub: {e}")

    async def perform_evaluation(self):
        """
        Perform the evaluation process:
        - Fetch XML files
        - Extract ground truth text
        - Run inference
        - Compute Character Error Rate (CER)
        - Save results
        - Generate and save evaluation report
        - Push results to GitHub repository
        """
        logger.info(f"Starting evaluation for repository {self.repo_name}")

        # Step 1: Fetch XML files
        fetched_files = self._fetch_xml_files()
        if not fetched_files:
            return
        logger.info(f"Successfully fetched {len(fetched_files)} XML files.")

        # Step 2: Extract ground truth text from XML files
        gt_lines = self._process_ground_truth(fetched_files)
        if not gt_lines:
            return

        # Step 3: Perform inference
        inferred_lines = await self._perform_inference()
        if not inferred_lines:
            return

        # Step 4: Compute CER
        cer_score = self._compute_cer(gt_lines, inferred_lines)

        # Step 5: Save ground truth and hypothesis files locally
        gt_file, hypothesis_file = self._save_results(gt_lines, inferred_lines)

        # Step 6: Generate and save evaluation report
        report_file = self._generate_evaluation_report(model_name=self.model_name,
                                                       hf_url=self.hf_url,
                                                       gt_file=gt_file,
                                                       hypothesis_file=hypothesis_file,
                                                       cer_score=cer_score)

        # Step 7: Push results to GitHub
        self._push_saved_results(gt_file, hypothesis_file, report_file)

        logger.info("Evaluation process completed.")
