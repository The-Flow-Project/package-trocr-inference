# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Set, Union
import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from datasets.exceptions import DatasetNotFoundError
from flow_inference.configure_dataset_card import HuggingFaceReadmeBuilder
from flow_inference.utils.logging.inference_logger import logger
from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

LINE_AUGMENTATION_COLUMN = "line_augmentation"
ORIGINAL_LINE_AUGMENTATION_VALUE = "original"
_UPDATE_OCCURRENCE_COLUMN = "__update_occurrence"


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
        self.real_duplicate_counts: Dict[str, Dict[str, int]] = {}

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

    @staticmethod
    def _key_columns_for_df(df: pd.DataFrame) -> list[str]:
        key = ["project_name", "filename", "line_id"]

        if LINE_AUGMENTATION_COLUMN in df.columns:
            key.append(LINE_AUGMENTATION_COLUMN)

        return key

    @classmethod
    def _add_update_occurrence_column(
        cls,
        df: pd.DataFrame,
        key: list[str],
    ) -> pd.DataFrame:
        """
        Add a temporary occurrence counter per update key.

        This lets us preserve duplicate physical rows instead of collapsing them.
        The temporary column is removed before writing parquet.
        """
        df_with_occurrence = df.copy()
        df_with_occurrence[_UPDATE_OCCURRENCE_COLUMN] = (
            df_with_occurrence.groupby(key, dropna=False).cumcount()
        )
        return df_with_occurrence

    @classmethod
    def _update_index_columns_for_df(cls, df: pd.DataFrame) -> list[str]:
        key = cls._key_columns_for_df(df)
        return key + [_UPDATE_OCCURRENCE_COLUMN]

    @staticmethod
    def _is_original_line_augmentation_value(value) -> bool:
        if pd.isna(value):
            return False
        return str(value).strip().lower() == ORIGINAL_LINE_AUGMENTATION_VALUE

    @staticmethod
    def _real_duplicate_key_columns() -> list[str]:
        return ["project_name", "filename", "line_id"]

    @classmethod
    def _df_for_real_duplicate_check(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the rows that should be considered for real duplicate-line counting.

        If line_augmentation is absent:
            consider all rows.

        If line_augmentation is present:
            consider only rows where line_augmentation == 'original'.

        Augmented rows are deliberately ignored because duplicate augmented rows can
        occur naturally when random transformations produce identical metadata.
        """
        if LINE_AUGMENTATION_COLUMN not in df.columns:
            return df

        return df[
            df[LINE_AUGMENTATION_COLUMN].apply(
                cls._is_original_line_augmentation_value
            )
        ]

    @classmethod
    def count_real_duplicate_lines(cls, df: pd.DataFrame) -> int:
        """
        Count rows that participate in real duplicate line keys.

        The real duplicate key is always:
            project_name + filename + line_id

        For augmented datasets, only original rows are considered.
        """
        key = cls._real_duplicate_key_columns()

        for col in key:
            if col not in df.columns:
                raise RuntimeError(f"Column '{col}' missing for duplicate-line check")

        check_df = cls._df_for_real_duplicate_check(df)

        return int(check_df.duplicated(key, keep=False).sum())

    @classmethod
    def count_real_duplicate_line_groups(cls, df: pd.DataFrame) -> int:
        """
        Count duplicate key groups, not rows.

        Example:
            3 rows with the same project_name + filename + line_id count as:
                duplicate rows: 3
                duplicate groups: 1
        """
        key = cls._real_duplicate_key_columns()

        for col in key:
            if col not in df.columns:
                raise RuntimeError(f"Column '{col}' missing for duplicate-line check")

        check_df = cls._df_for_real_duplicate_check(df)

        group_sizes = check_df.groupby(key, dropna=False).size()
        return int((group_sizes > 1).sum())

    @classmethod
    def count_real_duplicate_excess_lines(cls, df: pd.DataFrame) -> int:
        """
        Count only duplicate rows beyond the first occurrence.

        Example:
            3 rows with the same project_name + filename + line_id count as:
                duplicate rows: 3
                duplicate groups: 1
                duplicate excess rows: 2
        """
        key = cls._real_duplicate_key_columns()

        for col in key:
            if col not in df.columns:
                raise RuntimeError(f"Column '{col}' missing for duplicate-line check")

        check_df = cls._df_for_real_duplicate_check(df)

        group_sizes = check_df.groupby(key, dropna=False).size()
        return int((group_sizes[group_sizes > 1] - 1).sum())

    @classmethod
    def count_real_duplicate_lines_by_split(
        cls,
        dfs: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[str, int]]:
        """
        Count real duplicate lines per split.

        Returned values:
            duplicate_rows:
                Number of rows participating in duplicate keys.

            duplicate_groups:
                Number of distinct duplicate keys.

            duplicate_excess_rows:
                Number of duplicate rows beyond the first occurrence.
        """
        return {
            split: {
                "duplicate_rows": cls.count_real_duplicate_lines(df),
                "duplicate_groups": cls.count_real_duplicate_line_groups(df),
                "duplicate_excess_rows": cls.count_real_duplicate_excess_lines(df),
            }
            for split, df in dfs.items()
        }

    # --------------------------------------------------------------------------
    # DOWNLOAD HELPERS
    # --------------------------------------------------------------------------
    def _build_allow_patterns(self) -> list[str]:
        patterns: list[str] = []

        if self.auto_split:
            patterns.append("data/**/*.parquet")
            return patterns

        if self.requested_splits == {"default"}:
            patterns.append("data/**/*.parquet")
            return patterns

        if "train" in self.requested_splits:
            patterns.append("data/train/**/*.parquet")
        if "test" in self.requested_splits:
            patterns.append("data/test/**/*.parquet")

        return patterns

    def _download_snapshot(self, allow_patterns: list[str]) -> Path:
        if not self.cache_dir:
            self.cache_dir = tempfile.mkdtemp(prefix="hf_ds_")

        if self._resolved_sha is None:
            raise RuntimeError("Missing resolved dataset revision SHA.")

        local_root = Path(self.cache_dir) / self._resolved_sha / "snapshot"
        local_root.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=self.dataset_name,
            repo_type="dataset",
            revision=self._resolved_sha,
            token=self.huggingface_token,
            local_dir=str(local_root),
            allow_patterns=allow_patterns,
        )

        return local_root

    # --------------------------------------------------------------------------
    # SPLIT SELECTION
    # --------------------------------------------------------------------------
    @staticmethod
    def _discover_parquet_files(local_root: Path) -> dict[str, list[Path]]:
        return {
            "train": list((local_root / "data" / "train").rglob("*.parquet")),
            "test": list((local_root / "data" / "test").rglob("*.parquet")),
            "default": list((local_root / "data").rglob("*.parquet")),
        }

    def _select_parquet_paths(self, found: dict[str, list[Path]]) -> dict[str, list[Path]]:
        parquet_paths: dict[str, list[Path]] = {}

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
    # DOWNLOAD HUGGING FACE DATASETS
    # ==========================================================================
    def download_hf_dataset(self) -> None:
        mode = "AUTO" if self.auto_split else "EXPLICIT"
        logger.info(
            f"Downloading dataset: {self.dataset_name} | mode={mode} | "
            f"requested={self.requested_splits or 'AUTO'}"
        )

        try:
            api = HfApi()
            info = api.dataset_info(
                repo_id=self.dataset_name,
                revision=self.revision,
                token=self.huggingface_token,
            )
            self._resolved_sha = info.sha
            logger.info(
                f"Resolved dataset revision: {self._resolved_sha} "
                f"(requested: {self.revision})"
            )

            allow_patterns = self._build_allow_patterns()
            self._local_root = self._download_snapshot(allow_patterns)

            found = self._discover_parquet_files(self._local_root)
            parquet_paths = self._select_parquet_paths(found)

            data_files = {k: [str(p) for p in v] for k, v in parquet_paths.items()}
            hf_dataset = load_dataset("parquet", data_files=data_files)

            self.parquet_paths = {k: [str(p) for p in v] for k, v in parquet_paths.items()}
            self.dataset = (
                hf_dataset
                if isinstance(hf_dataset, DatasetDict)
                else DatasetDict({"default": hf_dataset})
            )
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

        dfs: Dict[str, pd.DataFrame] = {}
        for split_name in self.dataset.keys():
            logger.info(f"Converting split '{split_name}' to DataFrame...")
            dfs[split_name] = self.dataset[split_name].to_pandas()

        self.df = dfs
        self.real_duplicate_counts = self.count_real_duplicate_lines_by_split(dfs)

        logger.info(
            f"Real duplicate line counts by split: {self.real_duplicate_counts}"
        )

        self.state = "converted"
        return dfs

    @staticmethod
    def convert_to_list_of_dicts(dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        return {split: df.to_dict(orient="records") for split, df in dfs.items()}

    def convert_df_into_hf_dataset(self) -> Dataset:
        if self.df is None:
            raise RuntimeError("DataFrame not available. Call to_dataframe() first.")
        return Dataset.from_pandas(next(iter(self.df.values())), preserve_index=False)

    # ==========================================================================
    # PUSH TO HUGGING FACE HELPERS
    # ==========================================================================
    @classmethod
    def _index_df_by_key(cls, df: pd.DataFrame, split: str) -> pd.DataFrame:
        key = cls._key_columns_for_df(df)

        for col in key:
            if col not in df.columns:
                raise RuntimeError(f"Column '{col}' missing in split '{split}'")

        duplicate_rows = int(df.duplicated(key, keep=False).sum())
        if duplicate_rows:
            logger.warning(
                f"{split}: duplicate update keys detected for {key}: "
                f"{duplicate_rows} rows. Preserving duplicates with occurrence index."
            )

        df_with_occurrence = cls._add_update_occurrence_column(df, key)
        index_cols = key + [_UPDATE_OCCURRENCE_COLUMN]

        idx = df_with_occurrence.set_index(index_cols)

        if not idx.index.is_unique:
            raise RuntimeError(
                f"{split}: update index is still not unique after adding occurrence column. "
                f"Index columns: {index_cols}"
            )

        return idx

    @classmethod
    def _update_parquet_file(cls, local_path: Path, split_df: pd.DataFrame) -> Path:
        parquet_df = pd.read_parquet(local_path)

        key = cls._key_columns_for_df(parquet_df)

        for col in key:
            if col not in parquet_df.columns:
                raise RuntimeError(f"Column '{col}' missing in parquet file '{local_path}'")

        duplicate_rows = int(parquet_df.duplicated(key, keep=False).sum())
        if duplicate_rows:
            logger.warning(
                f"{local_path}: duplicate update keys detected for {key}: "
                f"{duplicate_rows} rows. Preserving duplicates with occurrence index."
            )

        parquet_with_occurrence = cls._add_update_occurrence_column(parquet_df, key)
        index_cols = key + [_UPDATE_OCCURRENCE_COLUMN]
        parquet_idx = parquet_with_occurrence.set_index(index_cols)

        if not parquet_idx.index.is_unique:
            raise RuntimeError(
                f"{local_path}: parquet update index is still not unique after adding "
                f"occurrence column. Index columns: {index_cols}"
            )

        common = parquet_idx.index.intersection(split_df.index)
        if len(common) == 0:
            return local_path

        for col in split_df.columns:
            if col == _UPDATE_OCCURRENCE_COLUMN:
                continue

            if col not in parquet_idx.columns:
                parquet_idx[col] = pd.NA

            parquet_idx.loc[common, col] = split_df.loc[common, col]

        updated = parquet_idx.reset_index()

        if _UPDATE_OCCURRENCE_COLUMN in updated.columns:
            updated = updated.drop(columns=[_UPDATE_OCCURRENCE_COLUMN])

        updated.to_parquet(local_path, index=False)
        return local_path

    def _make_commit_op(self, local_path: Path) -> CommitOperationAdd:
        if self._local_root is None:
            raise RuntimeError("Missing local snapshot root.")

        hf_path = str(local_path.relative_to(self._local_root)).replace("\\", "/")
        return CommitOperationAdd(path_in_repo=hf_path, path_or_fileobj=str(local_path))

    def _add_generated_readme_commit_op(
        self,
        target_repo: str,
    ) -> CommitOperationAdd:
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded.")
        if self.df is None:
            raise RuntimeError("Updated DataFrames not available.")

        builder = HuggingFaceReadmeBuilder.from_handler(
            repo_id=target_repo,
            dataset=self.dataset,
            dataframes=self.df,
            parquet_paths=self.parquet_paths,
            source_repos=[self.dataset_name],
        )

        text = builder.render()

        return CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=text.encode("utf-8"),
        )

    # ==========================================================================
    # PUSH UPDATED DATASET
    # ==========================================================================
    def push_to_hub(
        self,
        upload_repo_name: str,
        private: bool = True,
        commit_message: str = "Upload updated dataset",
    ) -> None:
        if self.df is None:
            raise RuntimeError("No DataFrames stored. Call to_dataframe() first.")
        if not self.parquet_paths:
            raise RuntimeError("No parquet file map available.")
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

        for split, df in self.df.items():
            key = self._key_columns_for_df(df)
            dupes = df[df.duplicated(key, keep=False)].sort_values(key)

            if dupes.empty:
                logger.debug(f"{split}: no duplicate update keys for {key}.")
            else:
                logger.warning(
                    f"{split}: duplicate update keys detected for {key}: "
                    f"{len(dupes)} rows. They will be preserved with occurrence indexing."
                )
                logger.debug(
                    "\n%s",
                    dupes[key].head(100).to_string(index=False),
                )

        # ------------------------------------------------------------------
        # 1) Index DataFrames by composite key
        # ------------------------------------------------------------------
        df_by_split_idx = {
            split: self._index_df_by_key(df, split)
            for split, df in self.df.items()
        }

        # ------------------------------------------------------------------
        # 2) Update parquet files + add commit operations
        # ------------------------------------------------------------------
        for split, parquet_files in self.parquet_paths.items():
            if split not in df_by_split_idx:
                continue

            for parquet_path in parquet_files:
                local_path = Path(parquet_path)
                self._update_parquet_file(local_path, df_by_split_idx[split])
                operations.append(self._make_commit_op(local_path))

        # ------------------------------------------------------------------
        # 3) Add README to same commit
        # ------------------------------------------------------------------
        operations.append(
            self._add_generated_readme_commit_op(
                target_repo=upload_repo_name,
            )
        )

        # ------------------------------------------------------------------
        # 4) Single commit (parquet + README)
        # ------------------------------------------------------------------
        api.create_commit(
            repo_id=upload_repo_name,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message,
            token=self.huggingface_token,
        )

        logger.info(f"Uploaded updated parquet files + README to HF Hub: {upload_repo_name}")
        self.state = "pushed"

    # ==========================================================================
    # SINGLE FILE UPLOAD
    # ==========================================================================
    def upload_file(self, repo_name: str, target_path: str, content_bytes: bytes) -> None:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=content_bytes,
            path_in_repo=target_path,
            repo_id=repo_name,
            repo_type="dataset",
            token=self.huggingface_token,
        )