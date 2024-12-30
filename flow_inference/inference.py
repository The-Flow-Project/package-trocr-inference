import glob
import os
from typing import Optional, Tuple, Dict, List

from flow_inference.data_handling import DataHandler
from flow_inference.image_processing import ImageHandler
from flow_inference.model_handling import ModelManager
from flow_inference.utils.logging.inference_logger import logger
from flow_inference.infer_textlines import InferenceHandler
from flow_inference.xml_processing import XMLProcessor


class Inference:
    def __init__(self,
                 repo_name: str,
                 repo_folder: str,
                 github_access_token: Optional[str],
                 directory: str = "tmp",
                 in_path: str = "",
                 out_path: str = "inference_results",
                 trocr_model="microsoft/trocr-large-handwritten",
                 trocr_processor="microsoft/trocr-large-handwritten",
                 use_cuda: bool = True,
                 do_resize: bool = False,
                 aspect_ratio_resize: bool = False,
                 output_txt: bool = True,
                 output_xml: bool = True,
                 image_size: Tuple[int, int] = (384, 384),
                 preprocessing_uri: Optional[str] = None,
                 **kwargs
                 ) -> None:
        self.repo_name = repo_name
        self.repo_folder = repo_folder
        self.github_access_token = github_access_token
        self.directory = directory
        self.in_path = in_path
        self.out_path = out_path
        self.trocr_model = trocr_model
        self.trocr_processor = trocr_processor
        self.use_cuda = use_cuda
        self.do_resize = do_resize
        self.aspect_ratio_resize = aspect_ratio_resize
        self.output_txt = output_txt
        self.output_xml = output_xml
        self.image_size = image_size
        self.preprocessing_uri = preprocessing_uri
        self.data_handler = DataHandler(download_path=in_path, upload_path=out_path)

    def perform_inference(self) -> None:
        """
        Perform inference on a list of XML files.
        :return: None
        """
        image_files = self.get_image_files()
        fetched_xml_files, failed_files = self.fetch_xml_files()

        if failed_files:
            logger.warning(f"Failed to fetch the following files: {failed_files}")
        if not fetched_xml_files:
            logger.error("No XML files fetched. Exiting inference.")
            return

        inferred_lines = self.run_inference(image_files)
        if not inferred_lines:
            logger.error("Inference failed or produced no results.")
            return

        self.save_results(inferred_lines, fetched_xml_files)

    def run_inference(self, image_files: List[str]) -> Optional[Dict[str, str]]:
        """
        Run inference on the provided image files.
        """
        model_manager = ModelManager(self.use_cuda)

        processor = model_manager.load_processor(self.trocr_processor)
        if processor is None:
            logger.error("Processor loading failed. Aborting inference.")
            return None

        model = model_manager.load_model(self.trocr_model)
        if model is None:
            logger.error("Model loading failed. Aborting inference.")
            return None

        logger.info("Inference can now proceed with the loaded processor and model.")

        image_handler = ImageHandler(
            processor=processor,
            image_size=self.image_size,
            do_resize=self.do_resize,
            aspect_ratio_resize=self.aspect_ratio_resize
        )
        inference_handler = InferenceHandler(
            device=model_manager.device,
            model=model,
            processor=processor
        )

        inference_result = inference_handler.infer(
            file_names=image_files,
            image_handler=image_handler
        )

        # Process inference results into a dictionary
        inferred_lines = {}
        for result in inference_result:
            try:
                file_name, inferred_text = result.split("\t")
                inferred_lines[file_name] = inferred_text
            except ValueError:
                logger.error(f"Malformed inference result: {result}")
        return inferred_lines

    def get_image_files(self) -> List[str]:
        """
        Get image files from the local directory or URI.
        """
        image_files = []
        if self.preprocessing_uri is None:
            modified_repo_path = self.repo_name.replace("/", "_")
            preprocessing_path = os.path.join(self.in_path, self.directory, "preprocessed", modified_repo_path)
            for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff']:
                image_files.extend(glob.glob(f'{preprocessing_path}/*.{ext}'))
                image_files.extend(glob.glob(f'{preprocessing_path}/*.{ext.upper()}'))
            if not image_files:
                logger.warning(f"No image files found in {preprocessing_path}.")
        else:
            # TODO: Implement URI handling
            logger.warning("URI handling for preprocessing is not implemented yet.")
        return image_files

    def fetch_xml_files(self) -> Tuple[List[str], List[str]]:
        """
        Fetch XML files from GitHub repository.
        """
        modified_repo_path = self.repo_name.replace("/", "_")
        try:
            fetched_files, failed_files = self.data_handler.fetch_xml_files_from_github(
                repo_name=self.repo_name,
                folder_path=self.repo_folder,
                download_path=os.path.join(self.in_path, self.directory, modified_repo_path)
            )
            logger.info(f"Successfully fetched XML files: {fetched_files}")
            return fetched_files, failed_files
        except Exception as e:
            logger.error(f"Error fetching XML files from GitHub: {e}")
            return [], []

    def push_to_github(self, updated_files: List[str]) -> None:
        """
        Push the updated XML files to GitHub after saving them locally.
        """
        if not updated_files:
            logger.warning("No updated files to push to GitHub.")
            return

        try:
            data_handler = DataHandler(download_path=self.in_path, upload_path=self.out_path)
            data_handler.push_xml_files_to_github(self.repo_name, updated_files)
            logger.info(f"Successfully pushed updated XML files to GitHub.")
        except Exception as e:
            logger.error(f"Failed to push updated XML files to GitHub: {e}")

    def save_results(self, inferred_lines: Dict[str, str], fetched_xml_files: List[str]) -> None:
        """
        Save the inference results to text and XML files.
        """
        if self.output_txt:
            self.write_to_text_files(inferred_lines)
        if self.output_xml:
            self.write_to_xml_files(inferred_lines, fetched_xml_files)

    def write_to_text_files(self, inferred_lines: Dict[str, str]) -> None:
        """
        Write inferred lines to text files.
        """
        from collections import defaultdict

        # Group lines by document name
        grouped_lines = defaultdict(list)
        for file_name, inferred_text in inferred_lines.items():
            # Extract document name by splitting before the first '.'
            doc_name = file_name.split('.', 1)[0]
            grouped_lines[doc_name].append(f"{file_name} {inferred_text}")

        # Write grouped lines to text files
        for doc_name, lines in grouped_lines.items():
            txt_file_path = os.path.join(self.out_path, f"{doc_name}.txt")
            try:
                with open(txt_file_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))  # Write all lines for this document
                logger.info(f"Saved inferred text to {txt_file_path}")
            except IOError as e:
                logger.error(f"Failed to write text file {txt_file_path}: {e}")

    def write_to_xml_files(self, inferred_lines: Dict[str, str], fetched_xml_files: List[str]) -> None:
        """
        Write inferred lines to XML files.
        """
        for xml_file in fetched_xml_files:
            try:
                logger.info(f"Processing XML file: {xml_file}")
                xml_processor = XMLProcessor(xml_file)
                xml_processor.insert_inferred_lines(
                    root=xml_processor.root,
                    inferred_lines=inferred_lines
                )
                output_path = os.path.join(self.out_path, os.path.basename(xml_file))
                xml_processor.save_xml(tree=xml_processor.tree, output_path=output_path)
                logger.info(f"Updated XML saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to update XML file {xml_file}: {e}")
