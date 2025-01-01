# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
import glob
import os
from typing import Optional, Tuple, Dict, List
from flow_inference.data_handling import DataHandler
from flow_inference.image_processing import ImageHandler
from flow_inference.model_handling import ModelManager
from flow_inference.utils.logging.inference_logger import logger
from flow_inference.infer_textlines import InferenceHandler
from flow_inference.xml_processing import XMLProcessor
from collections import defaultdict


# ===============================================================================
# CLASS
# ===============================================================================
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

        # Initialize attributes
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

        # initialise DataHandler
        self.data_handler = DataHandler(download_path=in_path, upload_path=out_path)

        logger.debug(f"Inference class initialized with repo_name={repo_name}, directory={directory}, "
                     f"in_path={in_path}, out_path={out_path}")

    def perform_inference(self) -> None:
        """
        Perform inference.

        :return: None
        """
        logger.info("Starting the inference process.")

        # Step 1: Fetch image files
        logger.debug("Fetching image files.")
        image_files = self.get_image_files()
        fetched_xml_files, failed_files = self.fetch_xml_files()

        # Step 2: Fetch XML files
        logger.debug("Fetching XML files.")
        if failed_files:
            logger.warning(f"Failed to fetch the following files: {failed_files}")
        if not fetched_xml_files:
            logger.error("No XML files fetched. Exiting inference.")
            raise FileNotFoundError("No XML files fetched for inference.")

        # Step 3: Run inference
        logger.debug("Running inference on image files.")
        inferred_lines = self.run_inference(image_files)
        if not inferred_lines:
            logger.error("Inference failed or produced no results.")
            raise RuntimeError("Inference produced no results.")

        # Step 4: Save results
        logger.debug("Saving inference results.")
        self.save_results(inferred_lines, fetched_xml_files)
        logger.info("Inference process completed successfully.")

    def run_inference(self, image_files: List[str]) -> Optional[Dict[str, str]]:
        """
        Run inference on the provided image files.

        :param image_files: List of image lines to run inference on.
        :return: Dictionary of inference results (with line numbers as keys).
        """
        logger.debug(f"Starting run_inference with {len(image_files)} image files.")

        # Load model and processor
        model_manager = ModelManager(self.use_cuda)
        logger.debug("ModelManager initialized.")

        processor = model_manager.load_processor(self.trocr_processor)
        if processor is None:
            logger.error(f"Processor loading failed for {self.trocr_processor}. Aborting inference.")
            raise ImportError("Failed to load the processor.")

        model = model_manager.load_model(self.trocr_model)
        if model is None:
            logger.error(f"Model loading failed for {self.trocr_model}. Aborting inference.")
            raise ImportError("Failed to load the model.")

        logger.info(f"Model and processor loaded successfully for {self.trocr_model}.")

        # Run inference
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

        logger.debug("Starting inference process.")
        inference_result = inference_handler.infer(file_names=image_files, image_handler=image_handler)
        logger.info("Inference completed successfully.")

        # Parse results
        inferred_lines = {}
        for result in inference_result:
            try:
                file_name, inferred_text = result.split("\t")
                inferred_lines[file_name] = inferred_text
            except ValueError as ve:
                logger.error(f"Malformed inference result: {result} - Error: {ve}")
                raise ValueError(f"Inference result format error: {result}")

        return inferred_lines

    def get_image_files(self) -> List[str]:
        """
        Get image files from the local directory or URI.

        :return: List of image files.
        """
        image_files = []
        if self.preprocessing_uri is None:
            modified_repo_path = self.repo_name.replace("/", "_")
            preprocessing_path = os.path.join(self.in_path, self.directory, "preprocessed", modified_repo_path)
            logger.debug(f"Looking for image files in {preprocessing_path}")

            for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff']:
                image_files.extend(glob.glob(f'{preprocessing_path}/*.{ext}'))
                image_files.extend(glob.glob(f'{preprocessing_path}/*.{ext.upper()}'))
            if not image_files:
                logger.warning(f"No image files found in {preprocessing_path}.")
                raise FileNotFoundError(f"No image files found in {preprocessing_path}.")
        else:
            # TODO: Implement URI handling
            logger.warning("URI handling for preprocessing is not implemented yet.")

        logger.debug(f"Found {len(image_files)} image files.")
        return image_files

    def fetch_xml_files(self) -> Tuple[List[str], List[str]]:
        """
        Fetch XML files from GitHub repository.

        :return: List of fetched XML files and list of files which couldn't be downloaded.
        """
        try:
            modified_repo_path = self.repo_name.replace("/", "_")
            logger.debug(f"Fetching XML files from GitHub for {self.repo_name}")

            fetched_files, failed_files = self.data_handler.fetch_xml_files_from_github(
                repo_name=self.repo_name,
                folder_path=self.repo_folder,
                download_path=os.path.join(self.in_path, self.directory, modified_repo_path)
            )

            logger.info(f"Fetched {len(fetched_files)} XML files.")
            if failed_files:
                logger.warning(f"Failed to fetch {len(failed_files)} XML files: {failed_files}")

            return fetched_files, failed_files
        except ConnectionError as e:
            logger.error(f"Network issue while fetching XML files: {e}")
            return [], []
        except FileNotFoundError as e:
            logger.error(f"Repository folder not found: {e}")
            return [], []
        except Exception as e:
            logger.error(f"Unexpected error while fetching XML files: {e}")
            return [], []

    # TODO: push files with inference results back to GitHub?
    def push_to_github(self, updated_files: List[str]) -> None:
        """
        Push the updated XML files to GitHub after saving them locally.

        :param: updated_files: XML files containing inference results.
        :return: None.
        """
        if not updated_files:
            logger.warning("No updated files to push to GitHub.")
            return

        try:
            data_handler = DataHandler(download_path=self.in_path, upload_path=self.out_path)
            data_handler.push_xml_files_to_github(self.repo_name, updated_files)
            logger.info(f"Successfully pushed updated XML files to GitHub.")
        except ConnectionError as e:
            logger.error(f"Network issue during push to GitHub: {e}")
        except PermissionError as e:
            logger.error(f"Permission denied when accessing repository: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during push to GitHub: {e}")
            raise

    def save_results(self, inferred_lines: Dict[str, str], fetched_xml_files: List[str]) -> None:
        """
            Save the inference results to text and XML files.

            :param inferred_lines: dict with inference results (key is line number).
            :param fetched_xml_files: the XML files in which the inference results are saved.
            :return: None.
            """
        try:
            if self.output_txt:
                logger.debug("Writing inference results to text files.")
                self.write_to_text_files(inferred_lines)
        except FileNotFoundError as e:
            logger.error(f"File path for saving text files not found. Error: {e}")
        except PermissionError as e:
            logger.error(f"Permission denied for writing text files. Error: {e}")
        except IOError as e:
            logger.error(f"IOError occurred while writing text files. Error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving text results. Error: {e}")

        try:
            if self.output_xml:
                logger.debug("Writing inference results to XML files.")
                self.write_to_xml_files(inferred_lines, fetched_xml_files)
        except FileNotFoundError as e:
            logger.error(f"XML file path for saving not found. Error: {e}")
        except PermissionError as e:
            logger.error(f"Permission denied for writing XML files. Error: {e}")
        except IOError as e:
            logger.error(f"IOError occurred while writing XML files. Error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving XML results. Error: {e}")

    def write_to_text_files(self, inferred_lines: Dict[str, str]) -> None:
        """
        Write inferred lines to text files.

        :param inferred_lines: dict with inference results (key is line number).
        :return: None.
        """
        grouped_lines = defaultdict(list)
        for file_name, inferred_text in inferred_lines.items():
            doc_name = file_name.split('.', 1)[0]
            grouped_lines[doc_name].append(f"{file_name} {inferred_text}")

        for doc_name, lines in grouped_lines.items():
            txt_file_path = os.path.join(self.out_path, f"{doc_name}.txt")
            logger.debug(f"Writing {len(lines)} lines to text file: {txt_file_path}")
            try:
                with open(txt_file_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))
                logger.info(f"Saved inferred text to {txt_file_path}")
            except FileNotFoundError as e:
                logger.error(f"File path not found: {txt_file_path}. Error: {e}")
            except PermissionError as e:
                logger.error(f"Permission denied for file: {txt_file_path}. Error: {e}")
            except IOError as e:
                logger.error(f"IOError occurred while writing to file {txt_file_path}: {e}")

    def write_to_xml_files(self, inferred_lines: Dict[str, str], fetched_xml_files: List[str]) -> None:
        """
        Write inferred lines to XML files.

        :param inferred_lines: dict with inference results (key is line number).
        :param fetched_xml_files: list of fetched XML files (from URI or GitHub).
        :return: None.
        """
        for xml_file in fetched_xml_files:
            try:
                logger.debug(f"Processing XML file: {xml_file}")
                xml_processor = XMLProcessor(xml_file)

                # Insert inferred lines into the XML
                xml_processor.insert_inferred_lines(
                    root=xml_processor.root,
                    inferred_lines=inferred_lines
                )

                # Save updated XML
                output_path = os.path.join(self.out_path, os.path.basename(xml_file))
                logger.debug(f"Saving updated XML to: {output_path}")
                xml_processor.save_xml(tree=xml_processor.tree, output_path=output_path)
                logger.info(f"Updated XML saved to: {output_path}")
            except FileNotFoundError as e:
                logger.error(f"XML file not found: {xml_file}. Error: {e}")
            except PermissionError as e:
                logger.error(f"Permission denied while accessing XML file: {xml_file}. Error: {e}")
            except IOError as e:
                logger.error(f"IOError occurred while processing XML file: {xml_file}. Error: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing XML file: {xml_file}. Error: {e}")
