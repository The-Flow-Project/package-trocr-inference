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

        # AUTO vs EXPLICIT split handling
        self.auto_split = split is None
        self.requested_splits: Set[str] = self._normalize_splits(split)

        # Loaded objects
        self.dataset: DatasetDict | None = None
        self.df: Optional[Dict[str, pd.DataFrame]] = None
        self.state: str = "initialized"

        # Local parquet file paths per split (absolute paths)
        self.parquet_paths: dict[str, list[str]] = {}

        # Local snapshot root that mirrors HF repo paths
        self._local_root: Path | None = None

        # Resolved commit SHA
        self._resolved_sha: str | None = None

    # ---------------------------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------------------------
    @staticmethod
    def _normalize_splits(split) -> Set[str]:
        """
        Normalize split input into a set[str].
        Accepted:
          - None (AUTO mode)
          - "train" / "test" / "default"
          - ["train", "test"] etc
        """
        if split is None:
            return set()
        if isinstance(split, (list, tuple, set)):
            return {str(s).lower() for s in split}
        return {str(split).lower()}

    @staticmethod
    def _exists_any(paths: List[Path]) -> bool:
        return len(paths) > 0

    # ---------------------------------------------------------------------------
    # DOWNLOAD DATASETS
    # ---------------------------------------------------------------------------
    def download_hf_dataset(self) -> None:
        """
        Downloads parquet files using snapshot_download so repo structure is preserved locally.
        Then loads the parquet files into a DatasetDict using datasets.load_dataset("parquet", data_files=...).

        AUTO mode (split=None):
          - if data/train exists -> load train
          - if data/test exists  -> load test
          - if neither exists    -> load default from data/**

        EXPLICIT mode (split=["train"], ["test"], ["train","test"], ["default"]):
          - only load requested splits
          - error if requested split is missing
        """
        mode = "AUTO" if self.auto_split else "EXPLICIT"
        logger.info(f"Downloading dataset: {self.dataset_name} | mode={mode} | requested={self.requested_splits or 'AUTO'}")

        try:
            api = HfApi()
            info = api.dataset_info(
                repo_id=self.dataset_name,
                revision=self.revision,
                token=self.huggingface_token,
            )
            resolved_sha = info.sha
            self._resolved_sha = resolved_sha
            logger.info(f"Resolved dataset revision: {resolved_sha} (requested: {self.revision})")

            if not self.cache_dir:
                self.cache_dir = tempfile.mkdtemp(prefix="hf_ds_")

            local_root = Path(self.cache_dir) / resolved_sha / "snapshot"
            local_root.mkdir(parents=True, exist_ok=True)

            # In AUTO mode, download all parquet under data/**
            # In EXPLICIT mode, download only requested splits (unless default requested)
            if self.auto_split:
                allow_patterns = ["data/**/*.parquet"]
            else:
                if self.requested_splits == {"default"}:
                    allow_patterns = ["data/**/*.parquet"]
                else:
                    allow_patterns = []
                    if "train" in self.requested_splits:
                        allow_patterns.append("data/train/**/*.parquet")
                    if "test" in self.requested_splits:
                        allow_patterns.append("data/test/**/*.parquet")
                    if not allow_patterns:
                        # if user passed something odd, still try all parquet
                        allow_patterns = ["data/**/*.parquet"]

            snapshot_download(
                repo_id=self.dataset_name,
                repo_type="dataset",
                revision=resolved_sha,
                token=self.huggingface_token,
                local_dir=str(local_root),
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
            )

            self._local_root = local_root

            # Discover downloaded parquet files locally
            train_files = list((local_root / "data" / "train").rglob("*.parquet"))
            test_files = list((local_root / "data" / "test").rglob("*.parquet"))
            default_files = list((local_root / "data").rglob("*.parquet"))

            parquet_paths: dict[str, list[Path]] = {}

            if self.auto_split:
                # AUTO: include splits that exist, else default
                if self._exists_any(train_files):
                    parquet_paths["train"] = train_files
                if self._exists_any(test_files):
                    parquet_paths["test"] = test_files
                if not parquet_paths:
                    if not self._exists_any(default_files):
                        raise RuntimeError("No parquet files found under data/**")
                    parquet_paths["default"] = default_files

            else:
                # EXPLICIT: requested splits and error if missing
                if "default" in self.requested_splits:
                    if not self._exists_any(default_files):
                        raise RuntimeError("Requested split 'default' but no parquet files found under data/**")
                    parquet_paths["default"] = default_files
                else:
                    if "train" in self.requested_splits:
                        if not self._exists_any(train_files):
                            raise RuntimeError("Requested split 'train' but no data/train parquet files found")
                        parquet_paths["train"] = train_files
                    if "test" in self.requested_splits:
                        if not self._exists_any(test_files):
                            raise RuntimeError("Requested split 'test' but no data/test parquet files found")
                        parquet_paths["test"] = test_files

            # Build datasets "data_files" as explicit file lists
            data_files = {k: [str(p) for p in v] for k, v in parquet_paths.items()}

            hf_dataset = load_dataset("parquet", data_files=data_files)

            self.parquet_paths = {k: [str(p) for p in v] for k, v in parquet_paths.items()}
            self.dataset = hf_dataset if isinstance(hf_dataset, DatasetDict) else DatasetDict({"default": hf_dataset})
            self.state = "downloaded_all"

            logger.info(f"Loaded splits={list(self.dataset.keys())} | parquet_files={ {k: len(v) for k,v in self.parquet_paths.items()} }")

        except DatasetNotFoundError:
            self.state = "failed"
            logger.error(f"Dataset not found: '{self.dataset_name}'")
            raise
        except Exception:
            self.state = "failed"
            logger.exception("Failed to download dataset")
            raise

    # ---------------------------------------------------------------------------
    # CONVERSION
    # ---------------------------------------------------------------------------
    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """Convert loaded HF DatasetDict splits to pandas DataFrames."""
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call download_hf_dataset() first.")

        dfs: Dict[str, pd.DataFrame] = {}
        for split_name in self.dataset.keys():
            logger.info(f"Converting split '{split_name}' to DataFrame...")
            dfs[split_name] = self.dataset[split_name].to_pandas()

        self.df = dfs
        self.state = "converted"
        return dfs

    def convert_to_list_of_dicts(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        """Convert all DataFrames into list-of-dicts."""
        recs: Dict[str, List[Dict]] = {}
        for split_name, df in dfs.items():
            recs[split_name] = df.to_dict(orient="records")
        return recs

    def convert_df_into_hf_dataset(self) -> Dataset:
        """Convert internal df to a HF Dataset (single split)."""
        if self.df is None:
            raise RuntimeError("DataFrame not available. Call to_dataframe() first.")
        only_df = next(iter(self.df.values()))
        return Dataset.from_pandas(only_df, preserve_index=False)

    # ---------------------------------------------------------------------------
    # PUSH UPDATED DATASET TO HUGGING FACE HUB
    # ---------------------------------------------------------------------------
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

        # Ensure repo exists & is private if created
        api.create_repo(
            repo_id=upload_repo_name,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=self.huggingface_token,
        )

        KEY = ["project_name", "filename", "line_id"]

        # ------------------------------------------------------------------
        # Build per-split indexed DataFrames
        # ------------------------------------------------------------------
        df_by_split_idx = {}

        for split, df in self.df.items():
            for col in KEY:
                if col not in df.columns:
                    raise RuntimeError(f"Column '{col}' missing in split '{split}'")

            idx = df.set_index(KEY)

            # Keep last inference if duplicates exist
            if not idx.index.is_unique:
                logger.warning(f"{split}: duplicate (project,filename,line_id) detected — keeping last")
                idx = idx[~idx.index.duplicated(keep="last")]

            df_by_split_idx[split] = idx

        # ------------------------------------------------------------------
        # Update each parquet file
        # ------------------------------------------------------------------
        for split, parquet_files in self.parquet_paths.items():
            if split not in df_by_split_idx:
                continue

            split_df = df_by_split_idx[split]

            for parquet_path in parquet_files:
                local_path = Path(parquet_path)
                hf_path = str(local_path.relative_to(self._local_root)).replace("\\", "/")

                parquet_df = pd.read_parquet(local_path)

                for col in KEY:
                    if col not in parquet_df.columns:
                        logger.warning(f"{hf_path}: missing {col}, skipping")
                        continue

                parquet_idx = parquet_df.set_index(KEY)

                if not parquet_idx.index.is_unique:
                    parquet_idx = parquet_idx[~parquet_idx.index.duplicated(keep="last")]

                common = parquet_idx.index.intersection(split_df.index)

                if len(common) > 0:
                    cols_to_write = list(split_df.columns)

                    for col in cols_to_write:
                        if col not in parquet_idx.columns:
                            parquet_idx[col] = pd.NA

                    parquet_idx.loc[common, cols_to_write] = (
                        split_df.loc[common, cols_to_write].to_numpy()
                    )

                updated = parquet_idx.reset_index()
                updated.to_parquet(local_path, index=False)

                operations.append(
                    CommitOperationAdd(
                        path_in_repo=hf_path,
                        path_or_fileobj=str(local_path),
                    )
                )

        api.create_commit(
            repo_id=upload_repo_name,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message,
            token=self.huggingface_token,
        )

        logger.info(f"Uploaded updated parquet files with preserved structure to HF Hub: {upload_repo_name}")
        self.state = "pushed"

    def upload_file(self, repo_name: str, target_path: str, content_bytes: bytes):
        api = HfApi()
        api.upload_file(
            path_or_fileobj=content_bytes,
            path_in_repo=target_path,
            repo_id=repo_name,
            repo_type="dataset",
            token=self.huggingface_token,
        )
