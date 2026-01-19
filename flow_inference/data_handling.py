# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Set, Union
import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from datasets.exceptions import DatasetNotFoundError
from flow_inference.utils.logging.inference_logger import logger
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub._commit_api import CommitOperationAdd


# ===============================================================================
# CLASS
# ===============================================================================
class HuggingFaceDataHandler:
    """
    Download and convert Hugging Face datasets.

    Supports structure:
      data/train/<doc_folder>/...*.parquet
      data/test/<doc_folder>/...*.parquet

    Also supports:
      - train-only repos
      - test-only repos
      - default repos (no data/train or data/test; parquet somewhere under data/**)
    """

    # --------------------------------------------------------------------------
    # INIT
    # --------------------------------------------------------------------------
    def __init__(
        self,
        dataset_name: str,
        huggingface_token: str | None = None,
        split: Union[str, Split, Iterable[str], None] = None,
        cache_dir: Optional[str] = None,
        revision: str = "main",
    ):
        self.dataset_name = dataset_name
        self.huggingface_token = huggingface_token
        self.cache_dir = cache_dir
        self.revision = revision

        self.auto_split = split is None
        self.requested_splits: Set[str] = self._normalize_splits(split)

        self.dataset: DatasetDict | None = None
        self.df: Optional[Dict[str, pd.DataFrame]] = None
        self.state: str = "initialized"

        self.parquet_paths: dict[str, list[str]] = {}
        self._local_root: Path | None = None
        self._resolved_sha: str | None = None

    # ==========================================================================
    # INTERNAL HELPERS
    # ==========================================================================
    @staticmethod
    def _normalize_splits(split) -> Set[str]:
        if split is None:
            return set()
        if isinstance(split, (list, tuple, set)):
            return {str(s).lower() for s in split}
        return {str(split).lower()}

    @staticmethod
    def _exists_any(paths: List[Path]) -> bool:
        return len(paths) > 0

    # --------------------------------------------------------------------------
    # SNAPSHOT DOWNLOAD HELPERS
    # --------------------------------------------------------------------------
    def _build_allow_patterns(self) -> list[str]:
        if self.auto_split:
            return ["data/**/*.parquet"]

        if self.requested_splits == {"default"}:
            return ["data/**/*.parquet"]

        patterns = []
        if "train" in self.requested_splits:
            patterns.append("data/train/**/*.parquet")
        if "test" in self.requested_splits:
            patterns.append("data/test/**/*.parquet")

        return patterns or ["data/**/*.parquet"]

    def _download_snapshot(self, allow_patterns: list[str]) -> Path:
        if not self.cache_dir:
            self.cache_dir = tempfile.mkdtemp(prefix="hf_ds_")

        local_root = Path(self.cache_dir) / self._resolved_sha / "snapshot"
        local_root.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=self.dataset_name,
            repo_type="dataset",
            revision=self._resolved_sha,
            token=self.huggingface_token,
            local_dir=str(local_root),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )

        return local_root

    # --------------------------------------------------------------------------
    # PARQUET DISCOVERY / SPLIT SELECTION
    # --------------------------------------------------------------------------
    def _discover_parquet_files(self, local_root: Path) -> dict[str, list[Path]]:
        return {
            "train": list((local_root / "data" / "train").rglob("*.parquet")),
            "test": list((local_root / "data" / "test").rglob("*.parquet")),
            "default": list((local_root / "data").rglob("*.parquet")),
        }

    def _select_parquet_paths(self, found: dict[str, list[Path]]) -> dict[str, list[Path]]:
        parquet_paths = {}

        if self.auto_split:
            if found["train"]:
                parquet_paths["train"] = found["train"]
            if found["test"]:
                parquet_paths["test"] = found["test"]
            if not parquet_paths:
                if not found["default"]:
                    raise RuntimeError("No parquet files found under data/**")
                parquet_paths["default"] = found["default"]
            return parquet_paths

        if "default" in self.requested_splits:
            if not found["default"]:
                raise RuntimeError("Requested split 'default' but no parquet files found")
            return {"default": found["default"]}

        if "train" in self.requested_splits:
            if not found["train"]:
                raise RuntimeError("Requested split 'train' but no parquet files found")
            parquet_paths["train"] = found["train"]

        if "test" in self.requested_splits:
            if not found["test"]:
                raise RuntimeError("Requested split 'test' but no parquet files found")
            parquet_paths["test"] = found["test"]

        return parquet_paths

    # ==========================================================================
    # DOWNLOAD DATASETS (MAIN)
    # ==========================================================================
    def download_hf_dataset(self) -> None:
        mode = "AUTO" if self.auto_split else "EXPLICIT"
        logger.info(
            f"Downloading dataset: {self.dataset_name} | mode={mode} | requested={self.requested_splits or 'AUTO'}"
        )

        try:
            api = HfApi()
            info = api.dataset_info(
                repo_id=self.dataset_name,
                revision=self.revision,
                token=self.huggingface_token,
            )
            self._resolved_sha = info.sha
            logger.info(f"Resolved dataset revision: {self._resolved_sha} (requested: {self.revision})")

            allow_patterns = self._build_allow_patterns()
            self._local_root = self._download_snapshot(allow_patterns)

            found = self._discover_parquet_files(self._local_root)
            parquet_paths = self._select_parquet_paths(found)

            data_files = {k: [str(p) for p in v] for k, v in parquet_paths.items()}

            hf_dataset = load_dataset("parquet", data_files=data_files)

            self.parquet_paths = {k: [str(p) for p in v] for k, v in parquet_paths.items()}
            self.dataset = hf_dataset if isinstance(hf_dataset, DatasetDict) else DatasetDict({"default": hf_dataset})
            self.state = "downloaded_all"

            logger.info(
                f"Loaded splits={list(self.dataset.keys())} | parquet_files="
                f"{ {k: len(v) for k, v in self.parquet_paths.items()} }"
            )

        except DatasetNotFoundError:
            self.state = "failed"
            logger.error(f"Dataset not found: '{self.dataset_name}'")
            raise
        except Exception:
            self.state = "failed"
            logger.exception("Failed to download dataset")
            raise

    # ==========================================================================
    # CONVERSION
    # ==========================================================================
    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call download_hf_dataset() first.")

        dfs = {}
        for split_name in self.dataset.keys():
            logger.info(f"Converting split '{split_name}' to DataFrame...")
            dfs[split_name] = self.dataset[split_name].to_pandas()

        self.df = dfs
        self.state = "converted"
        return dfs

    def convert_to_list_of_dicts(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        return {split: df.to_dict(orient="records") for split, df in dfs.items()}

    def convert_df_into_hf_dataset(self) -> Dataset:
        if self.df is None:
            raise RuntimeError("DataFrame not available. Call to_dataframe() first.")
        return Dataset.from_pandas(next(iter(self.df.values())), preserve_index=False)

    # ==========================================================================
    # PUSH TO HUB HELPERS
    # ==========================================================================
    def _index_df_by_key(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        KEY = ["project_name", "filename", "line_id"]

        for col in KEY:
            if col not in df.columns:
                raise RuntimeError(f"Column '{col}' missing in split '{split}'")

        idx = df.set_index(KEY)

        if not idx.index.is_unique:
            logger.warning(f"{split}: duplicate keys detected — keeping last")
            idx = idx[~idx.index.duplicated(keep="last")]

        return idx

    def _update_parquet_file(self, local_path: Path, split_df: pd.DataFrame) -> Path:
        KEY = ["project_name", "filename", "line_id"]

        parquet_df = pd.read_parquet(local_path)
        parquet_idx = parquet_df.set_index(KEY)

        if not parquet_idx.index.is_unique:
            parquet_idx = parquet_idx[~parquet_idx.index.duplicated(keep="last")]

        common = parquet_idx.index.intersection(split_df.index)
        if len(common) == 0:
            return local_path

        for col in split_df.columns:
            if col not in parquet_idx.columns:
                parquet_idx[col] = pd.NA

        parquet_idx.loc[common, split_df.columns] = split_df.loc[common].to_numpy()

        updated = parquet_idx.reset_index()
        updated.to_parquet(local_path, index=False)
        return local_path

    def _make_commit_op(self, local_path: Path) -> CommitOperationAdd:
        hf_path = str(local_path.relative_to(self._local_root)).replace("\\", "/")
        return CommitOperationAdd(path_in_repo=hf_path, path_or_fileobj=str(local_path))

    # ==========================================================================
    # PUSH UPDATED DATASET
    # ==========================================================================
    def push_to_hub(
        self,
        upload_repo_name: str,
        private: bool = True,
        commit_message: str = "Upload updated dataset",
    ):
        if self.df is None:
            raise RuntimeError("No DataFrames stored. Call to_dataframe() first.")
        if not self.parquet_paths:
            raise RuntimeError("No parquet file map available")
        if self._local_root is None:
            raise RuntimeError("Missing local snapshot root.")

        api = HfApi()
        operations: list[CommitOperationAdd] = []

        api.create_repo(
            repo_id=upload_repo_name,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=self.huggingface_token,
        )

        df_by_split_idx = {
            split: self._index_df_by_key(df, split)
            for split, df in self.df.items()
        }

        for split, parquet_files in self.parquet_paths.items():
            if split not in df_by_split_idx:
                continue

            for parquet_path in parquet_files:
                local_path = Path(parquet_path)
                self._update_parquet_file(local_path, df_by_split_idx[split])
                operations.append(self._make_commit_op(local_path))

        api.create_commit(
            repo_id=upload_repo_name,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message,
            token=self.huggingface_token,
        )

        logger.info(f"Uploaded updated parquet files to HF Hub: {upload_repo_name}")
        self.state = "pushed"

    # ==========================================================================
    # SINGLE FILE UPLOAD
    # ==========================================================================
    def upload_file(self, repo_name: str, target_path: str, content_bytes: bytes):
        api = HfApi()
        api.upload_file(
            path_or_fileobj=content_bytes,
            path_in_repo=target_path,
            repo_id=repo_name,
            repo_type="dataset",
            token=self.huggingface_token,
        )
