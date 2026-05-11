# ===============================================================================
# IMPORT STATEMENTS
# ===============================================================================
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Set, Union, Literal
import pandas as pd
from datasets import Dataset, DatasetDict, Split, load_dataset
from datasets.exceptions import DatasetNotFoundError
from flow_inference.configure_dataset_card import HuggingFaceReadmeBuilder
from flow_inference.utils.logging.inference_logger import logger
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

LINE_AUGMENTATION_COLUMN = "line_augmentation"
ORIGINAL_LINE_AUGMENTATION_VALUE = "original"
_UPDATE_OCCURRENCE_COLUMN = "__update_occurrence"
UploadMode = Literal["new_repo", "replace", "update"]


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

    @classmethod
    def _normalize_inference_columns_across_splits(
            cls,
            dfs: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Ensure every split DataFrame has the same inference_* columns.

        Hugging Face can fail when parquet files in the same dataset load have
        different schemas. This keeps additive inference columns safe across splits
        and across multiple update runs.
        """
        all_inference_columns: list[str] = []

        for df in dfs.values():
            for col in df.columns:
                col_str = str(col)
                if cls._is_inference_column(col_str) and col_str not in all_inference_columns:
                    all_inference_columns.append(col_str)

        normalized: Dict[str, pd.DataFrame] = {}

        for split, df in dfs.items():
            normalized_df = df.copy()

            for col in all_inference_columns:
                if col not in normalized_df.columns:
                    normalized_df[col] = ""

            normalized[split] = normalized_df

        return normalized

    @staticmethod
    def _exists_any(paths: List[Path]) -> bool:
        return len(paths) > 0

    @staticmethod
    def _has_usable_project_name(df: pd.DataFrame) -> bool:
        if "project_name" not in df.columns:
            return False

        return df["project_name"].fillna("").astype(str).str.strip().ne("").any()

    @classmethod
    def _key_columns_for_df(cls, df: pd.DataFrame) -> list[str]:
        key = []

        if cls._has_usable_project_name(df):
            key.append("project_name")

        key.extend(["filename", "region_id", "line_id"])

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

    @staticmethod
    def _is_original_line_augmentation_value(value) -> bool:
        if pd.isna(value):
            return False
        return str(value).strip().lower() == ORIGINAL_LINE_AUGMENTATION_VALUE

    @classmethod
    def _real_duplicate_key_columns(cls, df: pd.DataFrame) -> list[str]:
        key = []

        if cls._has_usable_project_name(df):
            key.append("project_name")

        key.extend(["filename", "region_id", "line_id"])
        return key

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

        The real duplicate key is:
            filename + region_id + line_id
        plus project_name when project_name is present and non-empty.

        For augmented datasets, only original rows are considered.
        """
        key = cls._real_duplicate_key_columns(df)

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
        key = cls._real_duplicate_key_columns(df)

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
            3 rows with the same project_name + filename + region_id + line_id count as:
                duplicate rows: 3
                duplicate groups: 1
                duplicate excess rows: 2
        """
        key = cls._real_duplicate_key_columns(df)

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

    @staticmethod
    def _is_inference_column(column: str) -> bool:
        return column.startswith("inference_")

    @classmethod
    def _validate_required_key_columns(cls, df: pd.DataFrame, context: str) -> None:
        missing = [
            col for col in ["filename", "region_id", "line_id"]
            if col not in df.columns
        ]

        if missing:
            raise RuntimeError(
                f"Missing required key column(s) in {context}: {missing}. "
                "Required columns are filename, region_id, and line_id."
            )

    @staticmethod
    def _repo_exists(api: HfApi, repo_id: str, token: Optional[str]) -> bool:
        try:
            api.dataset_info(
                repo_id=repo_id,
                token=token,
            )
            return True
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return False
            raise

    @staticmethod
    def _list_repo_files_if_exists(
        api: HfApi,
        repo_id: str,
        token: Optional[str],
    ) -> list[str]:
        if not HuggingFaceDataHandler._repo_exists(api, repo_id, token):
            return []

        return api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    def _repo_path_for_local_path(self, local_path: Path) -> str:
        if self._local_root is None:
            raise RuntimeError("Missing local snapshot root.")

        return str(local_path.relative_to(self._local_root)).replace("\\", "/")

    def _current_parquet_repo_paths(self) -> set[str]:
        paths: set[str] = set()

        for parquet_files in self.parquet_paths.values():
            for parquet_path in parquet_files:
                paths.add(self._repo_path_for_local_path(Path(parquet_path)))

        return paths

    @classmethod
    def _base_columns(cls, df: pd.DataFrame) -> list[str]:
        return [
            col
            for col in df.columns
            if not cls._is_inference_column(str(col))
        ]

    @classmethod
    def _validate_compatible_base_schema(
        cls,
        current_df: pd.DataFrame,
        target_df: pd.DataFrame,
        target_path: str,
    ) -> None:
        current_base = cls._base_columns(current_df)
        target_base = cls._base_columns(target_df)

        if set(current_base) != set(target_base):
            missing_in_target = sorted(set(current_base) - set(target_base))
            extra_in_target = sorted(set(target_base) - set(current_base))

            raise RuntimeError(
                "Refusing update: target parquet schema is incompatible with "
                f"the current dataset for '{target_path}'.\n"
                f"Missing in target: {missing_in_target}\n"
                f"Extra in target: {extra_in_target}\n"
                "Use upload_mode='replace' if you intentionally want to replace "
                "the target dataset structure."
            )

    def _merge_existing_target_inference_columns(
            self,
            target_repo: str,
            target_files: list[str],
    ) -> None:
        """
        For upload_mode='update':

        - Validate target parquet file layout exactly matches this run.
        - Validate target base schema matches current source schema.
        - Validate each matching parquet file has the same rows.
        - Copy existing target inference_* columns into self.df.
        - Keep newly generated inference columns already present in self.df.
        """
        if self.df is None:
            raise RuntimeError("No DataFrames stored. Call to_dataframe() first.")

        self._validate_update_target_files(target_files)

        target_root = self._download_target_snapshot_for_update(target_repo)

        for split, current_df in list(self.df.items()):
            # Full split-level dataframe. This is where we merge old target
            # inference columns into the newly generated dataframe.
            current_idx = self._index_df_by_key(current_df, split)

            for current_parquet_path in self.parquet_paths.get(split, []):
                current_local_path = Path(current_parquet_path)
                repo_path = self._repo_path_for_local_path(current_local_path)
                target_local_path = target_root / repo_path

                if not target_local_path.exists():
                    raise RuntimeError(
                        "Refusing update: target parquet file is missing after layout "
                        f"validation: '{repo_path}'."
                    )

                current_file_df = pd.read_parquet(current_local_path)
                target_file_df = pd.read_parquet(target_local_path)

                self._validate_compatible_base_schema(
                    current_df=current_file_df,
                    target_df=target_file_df,
                    target_path=repo_path,
                )

                current_file_idx = self._index_df_by_key(current_file_df, split)
                target_file_idx = self._index_df_by_key(target_file_df, split)

                missing_target_rows = target_file_idx.index.difference(current_file_idx.index)
                if len(missing_target_rows) > 0:
                    raise RuntimeError(
                        "Refusing update: target contains rows that are not present "
                        f"in the current dataset for '{repo_path}'. "
                        "Use upload_mode='replace' if this dataset structure change "
                        "is intentional."
                    )

                missing_current_rows = current_file_idx.index.difference(target_file_idx.index)
                if len(missing_current_rows) > 0:
                    raise RuntimeError(
                        "Refusing update: current dataset contains rows that are not present "
                        f"in the target dataset for '{repo_path}'. "
                        "Use upload_mode='replace' if this dataset structure change is intentional."
                    )

                target_inference_cols = [
                    col
                    for col in target_file_idx.columns
                    if self._is_inference_column(str(col))
                ]

                for col in target_inference_cols:
                    if col not in current_idx.columns:
                        current_idx[col] = pd.NA

                    # Important: assign only rows from this file into the full split index.
                    current_idx.loc[target_file_idx.index, col] = target_file_idx[col]

            merged = current_idx.reset_index()

            if _UPDATE_OCCURRENCE_COLUMN in merged.columns:
                merged = merged.drop(columns=[_UPDATE_OCCURRENCE_COLUMN])

            self.df[split] = merged

    def _build_replace_delete_operations(
        self,
        target_files: list[str],
        paths_that_will_be_added: set[str],
    ) -> list[CommitOperationDelete]:
        operations: list[CommitOperationDelete] = []

        for path in target_files:
            should_delete = path == "README.md" or path.startswith("data/")
            will_be_readded = path in paths_that_will_be_added

            if should_delete and not will_be_readded:
                operations.append(CommitOperationDelete(path_in_repo=path))

        return operations

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
    # UPDATE HELPERS
    # --------------------------------------------------------------------------

    def _validate_source_repo_update_allowed(
        self,
        upload_repo_name: str,
        allow_source_repo_update: bool,
    ) -> None:
        if upload_repo_name == self.dataset_name and not allow_source_repo_update:
            raise RuntimeError(
                "Refusing to upload into the source dataset repo. "
                f"download_repo_name and upload_repo_name are both '{self.dataset_name}'. "
                "Pass allow_source_repo_update=True only if this is intentional."
            )

    def _build_add_current_dataset_operations(
        self,
        target_repo: str,
    ) -> list[CommitOperationAdd]:
        operations: list[CommitOperationAdd] = []

        for parquet_files in self.parquet_paths.values():
            for parquet_path in parquet_files:
                operations.append(self._make_commit_op(Path(parquet_path)))

        operations.append(
            self._add_generated_readme_commit_op(
                target_repo=target_repo,
            )
        )

        return operations

    def _validate_update_target_files(
        self,
        target_files: list[str],
    ) -> None:
        current_parquet_paths = self._current_parquet_repo_paths()

        target_parquet_paths = {
            path
            for path in target_files
            if path.startswith("data/") and path.endswith(".parquet")
        }

        extra_target_parquets = target_parquet_paths - current_parquet_paths
        missing_target_parquets = current_parquet_paths - target_parquet_paths

        if extra_target_parquets or missing_target_parquets:
            extra_formatted = "\n".join(sorted(extra_target_parquets)) or "(none)"
            missing_formatted = "\n".join(sorted(missing_target_parquets)) or "(none)"

            raise RuntimeError(
                "Refusing update: target repo parquet layout does not exactly match "
                "the current run. Use upload_mode='replace' if this dataset structure "
                "change is intentional.\n"
                f"Extra target parquet files:\n{extra_formatted}\n"
                f"Missing target parquet files:\n{missing_formatted}"
            )

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

    def _download_target_snapshot_for_update(
        self,
        target_repo: str,
    ) -> Path:
        target_cache_dir = tempfile.mkdtemp(prefix="hf_target_ds_")
        target_root = Path(target_cache_dir) / "snapshot"
        target_root.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=target_repo,
            repo_type="dataset",
            token=self.huggingface_token,
            local_dir=str(target_root),
            allow_patterns=["data/**/*.parquet"],
        )

        return target_root

    # ==========================================================================
    # CONVERSION
    # ==========================================================================
    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call download_hf_dataset() first.")

        dfs: Dict[str, pd.DataFrame] = {}
        for split_name in self.dataset.keys():
            logger.info(f"Converting split '{split_name}' to DataFrame...")
            df = self.dataset[split_name].to_pandas()
            self._validate_required_key_columns(df, f"split '{split_name}'")
            dfs[split_name] = df

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
        hf_path = self._repo_path_for_local_path(local_path)
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
            duplicate_info=self.count_real_duplicate_lines_by_split(self.df),
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
            upload_mode: UploadMode = "new_repo",
            allow_source_repo_update: bool = False,
    ) -> None:
        if self.df is None:
            raise RuntimeError("No DataFrames stored. Call to_dataframe() first.")
        if not self.parquet_paths:
            raise RuntimeError("No parquet file map available.")
        if self._local_root is None:
            raise RuntimeError("Missing local snapshot root.")

        if upload_mode not in {"new_repo", "replace", "update"}:
            raise ValueError(
                "upload_mode must be one of: 'new_repo', 'replace', 'update'"
            )

        self._validate_source_repo_update_allowed(
            upload_repo_name=upload_repo_name,
            allow_source_repo_update=allow_source_repo_update,
        )

        api = HfApi()

        target_exists = self._repo_exists(
            api=api,
            repo_id=upload_repo_name,
            token=self.huggingface_token,
        )

        if target_exists and upload_mode == "new_repo":
            raise RuntimeError(
                f"Target dataset repo '{upload_repo_name}' already exists. "
                "Use upload_mode='update' to add inference columns to an existing compatible repo, "
                "or upload_mode='replace' to replace the target dataset contents."
            )

        if not target_exists and upload_mode == "update":
            raise RuntimeError(
                f"Target dataset repo '{upload_repo_name}' does not exist. "
                "Use upload_mode='new_repo' to create it."
            )

        if not target_exists:
            api.create_repo(
                repo_id=upload_repo_name,
                repo_type="dataset",
                private=private,
                exist_ok=False,
                token=self.huggingface_token,
            )
            target_files: list[str] = []
            logger.info(f"Created new HF dataset repo: {upload_repo_name}")
        else:
            target_files = api.list_repo_files(
                repo_id=upload_repo_name,
                repo_type="dataset",
                token=self.huggingface_token,
            )

        if upload_mode == "update" and target_exists:
            self._merge_existing_target_inference_columns(
                target_repo=upload_repo_name,
                target_files=target_files,
            )

        self.df = self._normalize_inference_columns_across_splits(self.df)

        operations: list[CommitOperationAdd | CommitOperationDelete] = []

        paths_that_will_be_added = self._current_parquet_repo_paths()
        paths_that_will_be_added.add("README.md")

        if upload_mode == "replace" and target_exists:
            operations.extend(
                self._build_replace_delete_operations(
                    target_files=target_files,
                    paths_that_will_be_added=paths_that_will_be_added,
                )
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

        operations.append(
            self._add_generated_readme_commit_op(
                target_repo=upload_repo_name,
            )
        )

        api.create_commit(
            repo_id=upload_repo_name,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message,
            token=self.huggingface_token,
        )

        logger.info(
            f"Uploaded dataset to HF Hub: {upload_repo_name} "
            f"(upload_mode={upload_mode})"
        )
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