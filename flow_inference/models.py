from datetime import datetime
import enum
from typing import Optional, Tuple, List

from pydantic import BaseModel, Field


class StateEnum(enum.Enum):
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    DONE = "done"


class InferenceState(BaseModel):
    process_id: str = Field(alias="process_id",
                            description="The uniqueid of the inference status.",
                            title="ID")
    repo_name: str = Field(
        alias="repo_name",
        description="Name of the GitHub-repository.",
        title="Repository-Name",
        examples=["your_github_name/your_repo_name"],
        frozen=True,
    )
    repo_folder: str = Field(
        alias="repo_folder",
        description="Folder in the repository the files are fetched from.",
        title="Repository-Folder",
        examples=["xml", "page"],
    )
    directory: Optional[str] = Field(
        default="data",
        alias="directory",
        description="Directory to save the files temporarily to.",
        title="Directory",
        examples=["data"],
    )
    in_path: Optional[str] = Field(
        default="preprocessed",
        alias="in_path",
        description="Path to save the preprocessed files.",
        title="In-Path",
        examples=["fetched"],
    )
    out_path: Optional[str] = Field(
        default="inference_results",
        alias="out_path",
        description="Path to save the inference results.",
        title="Out-Path",
        examples=["inference_results"],
    )
    trocr_model: Optional[str] = Field(
        default="microsoft/trocr-large-handwritten",
        alias="trocr_model",
        description="TrOCR model used in inference process.",
        title="TrOCR_Model",
    )
    trocr_processor: Optional[str] = Field(
        default="microsoft/trocr-large-handwritten",
        alias="trocr_processor",
        description="TrOCR processor used in inference process.",
        title="TrOCR_Processor",
    )
    use_cuda: Optional[bool] = Field(
        default=True,
        alias="use_cuda",
        description="Whether to use cuda.",
        title="Use-Cuda",
    )
    do_resize: Optional[bool] = Field(
        default=True,
        alias="do_resize",
        description="Whether to resize image.",
        title="Do-Resize",
    )
    aspect_ratio_resize: Optional[bool] = Field(
        default=True,
        alias="aspect_ratio_resize",
        description="Whether to use aspect ratio to resize image.",
        title="Aspect-Ratio-Resize",
    )
    output_txt: Optional[bool] = Field(
        default=True,
        alias="output_txt",
        description="Whether to create an .txt file for inference results.",
        title="Output-txt",
    )
    output_xml: Optional[bool] = Field(
        default=True,
        alias="output_xml",
        description="Whether to save inference results in XML file.",
        title="Output-XML",
    )
    image_size: Optional[Tuple[int, int]] = Field(
        default=(384, 384),
        alias="image_size",
        description="size of image to be processed.",
        title="Image-Size",
    )
    preprocesing_uri: Optional[str] = Field(
        default="",
        alias="preprocesing_uri",
        description="URI to download preprocessing files from.",
        title="Preprocessing-URI",
    )
    progress: int = Field(alias="progress",
                          description="The progress of the inference process.",
                          title="Progress",
                          default=0)
    state: StateEnum = Field(alias="state",
                             description="The state of the inference process.",
                             title="State",
                             default=StateEnum.IN_PROGRESS)
    files_successful: Optional[int] = Field(alias="files_successful",
                                            description="The amount of successfully inferred files.",
                                            title="Files-Successful",
                                            default=0)
    files_failed_inference: Optional[int] = Field(alias="files_failed_inference",
                                                  description="The amount of files that failed inference.",
                                                  title="Files-Failed-Inference",
                                                  default=0)
    files_failed_download: Optional[int] = Field(alias="files_failed_download",
                                                 description="The amount of files that failed downloading.",
                                                 title="Files-Failed-Download",
                                                 default=0)
    files_total: Optional[int] = Field(alias="files_total",
                                       description="The total amount of files.",
                                       title="Files-Total",
                                       default=0)
    filenames_successful: Optional[List] = Field(alias="filenames_successful",
                                                 description="The names of the successfully processed files.",
                                                 title="Filenames-Successful",
                                                 default=[])
    filenames_failed_inference: Optional[List] = Field(alias="filenames_failed_inference",
                                                       description="The names of the files that failed processing.",
                                                       title="Filenames-Failed-Inference",
                                                       default=[])
    filenames_failed_download: Optional[List] = Field(alias="filenames_failed_download",
                                                      description="The names of the files that failed downloading.",
                                                      title="Filenames-Failed-Download",
                                                      default=[])
    runtime: Optional[int] = Field(alias="runtime",
                                   description="The runtime of the preprocess status.",
                                   title="Runtime",
                                   default=0)
    created_at: Optional[datetime] = Field(alias="created_at",
                                           description="the start time of the process.",
                                           title="Created_at",
                                           default_factory=datetime.now)
