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
    directory: Optional[str] = Field(
        default="data",
        alias="directory",
        description="Directory to save the files temporarily to.",
        title="Directory",
        examples=["data"],
    )
    trocr_model: Optional[str] = Field(
        default="microsoft/trocr-large-handwritten",
        alias="trocr_model",
        description="TrOCR model used in inference process.",
        title="TrOCR_Model",
    )
    target_image_size: Optional[Tuple[int, int]] = Field(
        default=(384, 384),
        alias="target_image_size",
        description="target size of image.",
        title="Target-Image-Size",
    )
    abbrev: bool = Field(default=False,
                         alias="abbrev",
                         description="Whether to expand abbreviations in text.",
                         title="Abbrev")
    crop: bool = Field(default=False,
                       alias="crop",
                       description="Whether to crop images to their linemask.",
                       title="Crop")
    stop_on_fail: bool = Field(default=True,
                               alias="stop_on_fail",
                               description="Whether to stop processing on failure.",
                               title="Stop-On-Fail")
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
