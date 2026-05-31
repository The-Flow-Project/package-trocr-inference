"""Write line-level inference results back into raw XML datasets.

This module combines an inference result repository with a raw XML repository,
inserts recognized text into matching PAGE XML ``TextLine`` and ``TextRegion``
elements, stores the updated XML in new ``inference_xml_*`` columns, and uploads
the resulting dataset to the Hugging Face Hub.
"""

import tempfile
from pathlib import Path
from typing import cast, Any, Literal
import shutil
import pandas as pd
from huggingface_hub import (
    HfApi,
    snapshot_download,
    CommitOperationAdd,
    CommitOperationDelete,
)
from huggingface_hub.utils import HfHubHTTPError
from datasets import load_dataset, DatasetDict

from flow_inference.configure_dataset_card import HuggingFaceReadmeBuilder
from flow_inference.utils.logging.inference_logger import logger
from flow_inference.xml_processing import XMLProcessor

LINE_AUGMENTATION_COLUMN = "line_augmentation"
ORIGINAL_LINE_AUGMENTATION_VALUE = "original"
UploadMode = Literal["new_repo", "replace", "update"]


class InferenceToRawXMLWriter:
    """Write inference output into raw XML records and upload the result.

    The writer downloads a raw XML dataset and an inference result dataset from
    the Hugging Face Hub, builds a lookup from line identifiers to inferred text,
    inserts the text into matching XML records, and writes the updated data to a
    new or existing target dataset repository.
    """
    def __init__(self,
                 raw_xml_repo: str,
                 inference_repo: str,
                 token: str,
                 allow_source_repo_update: bool = False):
        """Initialize the raw XML writeback pipeline.

        Args:
            raw_xml_repo: Source Hugging Face dataset repository containing raw XML records.
            inference_repo: Source Hugging Face dataset repository containing inference results.
            token: Hugging Face token used for downloading and uploading datasets.
            allow_source_repo_update: Whether writing back into the raw XML source repository is allowed.
        """
        self.raw_xml_repo = raw_xml_repo
        self.inference_repo = inference_repo
        self.token = token
        self.allow_source_repo_update = allow_source_repo_update

    # ------------------------------------------------------------
    # REPO / DOWNLOAD HELPERS
    # ------------------------------------------------------------
    @staticmethod
    def _repo_exists(api: HfApi, repo_id: str, token: str | None) -> bool:
        """Return whether a Hugging Face dataset repository exists."""
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

    def _fresh_snapshot_download(
        self,
        repo_id: str,
        revision: str,
        prefix: str,
    ) -> Path:
        """Download a fresh dataset snapshot into a temporary directory."""
        local_dir = tempfile.mkdtemp(prefix=prefix)

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            token=self.token,
            local_dir=local_dir,
            force_download=True,
            allow_patterns=["data/**/*.parquet"],
        )

        return Path(local_dir)

    @staticmethod
    def _load_parquet_dataset_fresh(data_files):
        """Load parquet files as a Hugging Face dataset without reusing cached data."""
        return load_dataset(
            "parquet",
            data_files=data_files,
            cache_dir=tempfile.mkdtemp(prefix="hf_parquet_cache_"),
            download_mode="force_redownload",
        )

    @classmethod
    def _parquet_paths_by_split(cls, root: Path, parquet_paths: list[Path]) -> dict[str, list[str]]:
        """Group local parquet paths by dataset split name."""
        result: dict[str, list[str]] = {}

        for parquet_path in parquet_paths:
            repo_path = str(parquet_path.relative_to(root)).replace("\\", "/")
            split_name = cls._split_name_for_repo_path(repo_path)
            result.setdefault(split_name, []).append(str(parquet_path))

        return result

    @staticmethod
    def _split_name_for_repo_path(repo_path: str) -> str:
        """Infer the dataset split name from a repository-relative parquet path."""
        parts = Path(repo_path).parts

        if len(parts) >= 3 and parts[0] == "data" and parts[1] in {"train", "test"}:
            return parts[1]

        return "default"

    @staticmethod
    def _inference_xml_columns(df: pd.DataFrame) -> list[str]:
        """Return columns that contain XML writeback output."""
        return [
            col
            for col in df.columns
            if str(col).startswith("inference_xml_")
        ]

    @staticmethod
    def _build_replace_delete_operations(
        target_files: list[str],
        paths_that_will_be_added: set[str],
    ) -> list[CommitOperationDelete]:
        """Build delete operations for files removed during replace-mode upload."""
        operations: list[CommitOperationDelete] = []

        for path in target_files:
            should_delete = path == "README.md" or path.startswith("data/")
            will_be_readded = path in paths_that_will_be_added

            if should_delete and not will_be_readded:
                operations.append(CommitOperationDelete(path_in_repo=path))

        return operations

    @staticmethod
    def _base_columns_for_raw_xml_update(df: pd.DataFrame) -> list[str]:
        """Return non-writeback columns used for raw XML schema compatibility checks."""
        return [
            col
            for col in df.columns
            if not str(col).startswith("inference_xml_")
        ]

    @classmethod
    def _validate_compatible_raw_xml_base_schema(
            cls,
            source_df: pd.DataFrame,
            target_df: pd.DataFrame,
            repo_path: str,
    ) -> None:
        """Validate that raw XML source and target schemas are update-compatible."""
        source_base = set(cls._base_columns_for_raw_xml_update(source_df))
        target_base = set(cls._base_columns_for_raw_xml_update(target_df))

        if source_base != target_base:
            missing_in_target = sorted(source_base - target_base)
            extra_in_target = sorted(target_base - source_base)

            raise RuntimeError(
                "Refusing update: target raw XML base schema is incompatible with "
                f"the source raw XML file '{repo_path}'.\n"
                f"Missing in target: {missing_in_target}\n"
                f"Extra in target: {extra_in_target}\n"
                "Only extra inference_xml_* columns are allowed in update mode. "
                "Use upload_mode='replace' if this source/target mismatch is intentional."
            )

    def _merge_existing_target_inference_xml_columns(
            self,
            target_root: Path,
            raw_root: Path,
            raw_parquet_paths: list[Path],
    ) -> None:
        """Preserve existing target ``inference_xml_*`` columns during update uploads.

        Start from the fresh raw XML source files, then copy existing target
        ``inference_xml_*`` columns into them. This allows the source repo to lack
        previous writeback columns while the target repo preserves them.
        """
        for raw_parquet_path in raw_parquet_paths:
            rel_path = raw_parquet_path.relative_to(raw_root)
            repo_path = str(rel_path).replace("\\", "/")
            target_parquet_path = target_root / rel_path

            if not target_parquet_path.exists():
                raise RuntimeError(
                    "Refusing update: target repo parquet layout does not match "
                    f"the current raw XML source. Missing target file: '{repo_path}'. "
                    "Use upload_mode='replace' if this structure change is intentional."
                )

            source_df = pd.read_parquet(raw_parquet_path)
            target_df = pd.read_parquet(target_parquet_path)

            self._validate_compatible_raw_xml_base_schema(
                source_df=source_df,
                target_df=target_df,
                repo_path=repo_path,
            )

            if len(source_df) != len(target_df):
                raise RuntimeError(
                    "Refusing update: target parquet row count differs from current "
                    f"raw XML source for '{repo_path}'. "
                    f"Source rows: {len(source_df)}, target rows: {len(target_df)}. "
                    "Use upload_mode='replace' if this structure change is intentional."
                )

            target_inference_xml_cols = self._inference_xml_columns(target_df)

            for col in target_inference_xml_cols:
                if col in source_df.columns:
                    continue

                source_df[col] = target_df[col].values

            source_df.to_parquet(raw_parquet_path, index=False)

    # ------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------
    def process_and_upload(
            self,
            output_repo: str | None = None,
            upload_mode: UploadMode = "new_repo",
            private: bool = True,
    ) -> dict:
        """Run the raw XML writeback pipeline and upload the result.

        The pipeline downloads the inference and raw XML source repositories, builds
        a lookup from the latest inference column, writes inferred text into matching
        XML records, stores the updated XML in a new ``inference_xml_*`` column, and
        uploads the modified dataset to the target Hugging Face repository.

        Args:
            output_repo: Optional target repository. If omitted, the raw XML source
                repository is used as the target.
            upload_mode: Upload behavior. Use ``"new_repo"``, ``"replace"``, or ``"update"``.
            private: Whether to create the target repository as private when it does not exist.

        Returns:
            Summary dictionary containing the number of updated records.

        Raises:
            RuntimeError: If repositories, parquet layouts, schemas, or XML writeback
                inputs are incompatible with the selected upload mode.
            ValueError: If ``upload_mode`` is not supported.
        """
        logger.info("Starting raw XML writeback pipeline.")

        api = HfApi()
        temp_roots: list[Path] = []

        try:
            target_repo = output_repo or self.raw_xml_repo

            logger.info(
                f"Raw XML writeback configuration: "
                f"raw_xml_repo='{self.raw_xml_repo}', "
                f"inference_repo='{self.inference_repo}', "
                f"target_repo='{target_repo}', "
                f"upload_mode='{upload_mode}', "
                f"private={private}, "
                f"allow_source_repo_update={self.allow_source_repo_update}"
            )

            if upload_mode not in {"new_repo", "replace", "update"}:
                raise ValueError(
                    "upload_mode must be one of: 'new_repo', 'replace', 'update'"
                )

            # --------------------------------------------------
            # 0. Safety checks
            # --------------------------------------------------
            logger.info("Running raw XML writeback safety checks.")

            if target_repo == self.raw_xml_repo and not self.allow_source_repo_update:
                raise RuntimeError(
                    "Refusing to upload into the source raw XML repo. "
                    "Pass allow_source_repo_update=True only if this is intentional."
                )

            if target_repo == self.inference_repo:
                raise RuntimeError(
                    "Refusing to upload into the inference source repo."
                )

            target_exists = self._repo_exists(
                api=api,
                repo_id=target_repo,
                token=self.token,
            )

            logger.info(
                f"Target repository check complete: "
                f"target_repo='{target_repo}', exists={target_exists}"
            )

            if upload_mode == "new_repo" and target_exists:
                raise RuntimeError(
                    f"Target dataset repo '{target_repo}' already exists. "
                    "Use upload_mode='update' to preserve existing inference_xml columns, "
                    "or upload_mode='replace' to replace target contents."
                )

            if upload_mode == "update" and not target_exists:
                raise RuntimeError(
                    f"Target dataset repo '{target_repo}' does not exist. "
                    "Use upload_mode='new_repo' to create it."
                )

            # --------------------------------------------------
            # 1. Fresh-load inference source repo
            # --------------------------------------------------
            logger.info(f"Downloading inference source repo: {self.inference_repo}")

            inf_info = api.dataset_info(self.inference_repo, token=self.token)

            logger.info(f"Resolved inference repo revision: {inf_info.sha}")

            inf_root = self._fresh_snapshot_download(
                repo_id=self.inference_repo,
                revision=inf_info.sha,
                prefix="fresh_inference_repo_",
            )
            temp_roots.append(inf_root)

            inf_parquets = sorted(inf_root.rglob("*.parquet"))

            if not inf_parquets:
                raise RuntimeError(
                    f"No parquet files found in inference repo '{self.inference_repo}'."
                )

            logger.info(
                f"Downloaded inference repo successfully: "
                f"{len(inf_parquets)} parquet file(s) found."
            )

            inf_paths_by_split = self._parquet_paths_by_split(
                root=inf_root,
                parquet_paths=inf_parquets,
            )

            logger.info(
                f"Loading inference parquet dataset with splits={list(inf_paths_by_split.keys())}."
            )

            inf_dataset = self._load_parquet_dataset_fresh(
                data_files=inf_paths_by_split,
            )

            if not isinstance(inf_dataset, DatasetDict):
                inf_dataset = DatasetDict({"default": inf_dataset})

            inf_df = pd.concat(
                [split_ds.to_pandas() for split_ds in inf_dataset.values()],
                ignore_index=True,
            )

            logger.info(
                f"Inference dataset loaded successfully: {len(inf_df)} row(s)."
            )

            logger.info("Building XML writeback lookup from latest inference_* column.")

            lookup = self._build_lookup(inf_df)

            logger.info(
                f"XML writeback lookup built: {len(lookup)} project group(s)."
            )

            if not lookup:
                logger.warning(
                    "Inference lookup is empty. No raw XML lines will be updated."
                )

            # --------------------------------------------------
            # 2. Fresh-load raw XML source repo
            # --------------------------------------------------
            logger.info(f"Downloading raw XML source repo: {self.raw_xml_repo}")

            raw_info = api.dataset_info(self.raw_xml_repo, token=self.token)

            logger.info(f"Resolved raw XML repo revision: {raw_info.sha}")

            raw_root = self._fresh_snapshot_download(
                repo_id=self.raw_xml_repo,
                revision=raw_info.sha,
                prefix="fresh_raw_xml_repo_",
            )
            temp_roots.append(raw_root)

            raw_parquets = sorted(raw_root.rglob("*.parquet"))

            if not raw_parquets:
                raise RuntimeError(
                    f"No parquet files found in raw XML repo '{self.raw_xml_repo}'."
                )

            logger.info(
                f"Downloaded raw XML repo successfully: "
                f"{len(raw_parquets)} parquet file(s) found."
            )

            # --------------------------------------------------
            # 3. Add new inference_xml column to raw XML parquet files
            # --------------------------------------------------
            new_xml_column = self._build_inference_xml_column_name()

            logger.info(
                f"Writing inferred text into new raw XML column: {new_xml_column}"
            )

            parquet_repo_paths: list[str] = []
            updated_records = 0

            for parquet_path in raw_parquets:
                rel_path = str(parquet_path.relative_to(raw_root)).replace("\\", "/")

                logger.info(f"Updating raw XML parquet file: {rel_path}")

                df = pd.read_parquet(parquet_path)

                df, updated_count = self._update_df(df, lookup, new_xml_column)
                updated_records += updated_count

                df.to_parquet(parquet_path, index=False)

                logger.info(
                    f"Finished raw XML parquet file: {rel_path} "
                    f"({updated_count} record(s) updated out of {len(df)} row(s))."
                )

                parquet_repo_paths.append(rel_path)

            logger.info(
                f"Raw XML parquet update step completed: "
                f"{updated_records} record(s) updated across "
                f"{len(parquet_repo_paths)} parquet file(s)."
            )

            # --------------------------------------------------
            # 4. If update mode: fresh-download target and preserve old XML columns
            # --------------------------------------------------
            if upload_mode == "update":
                logger.info(
                    f"Upload mode is 'update'. Downloading target repo to preserve "
                    f"existing inference_xml_* columns: {target_repo}"
                )

                target_info = api.dataset_info(target_repo, token=self.token)

                logger.info(f"Resolved target repo revision: {target_info.sha}")

                target_root = self._fresh_snapshot_download(
                    repo_id=target_repo,
                    revision=target_info.sha,
                    prefix="fresh_target_repo_",
                )
                temp_roots.append(target_root)

                target_parquet_paths = {
                    str(p.relative_to(target_root)).replace("\\", "/")
                    for p in target_root.rglob("*.parquet")
                }

                current_parquet_paths = set(parquet_repo_paths)

                extra_target_parquets = target_parquet_paths - current_parquet_paths
                missing_target_parquets = current_parquet_paths - target_parquet_paths

                if extra_target_parquets or missing_target_parquets:
                    extra_formatted = "\n".join(sorted(extra_target_parquets)) or "(none)"
                    missing_formatted = "\n".join(sorted(missing_target_parquets)) or "(none)"

                    raise RuntimeError(
                        "Refusing update: target repo parquet layout does not exactly "
                        "match the current raw XML source. Use upload_mode='replace' "
                        "if this dataset structure change is intentional.\n"
                        f"Extra target parquet files:\n{extra_formatted}\n"
                        f"Missing target parquet files:\n{missing_formatted}"
                    )

                self._merge_existing_target_inference_xml_columns(
                    target_root=target_root,
                    raw_root=raw_root,
                    raw_parquet_paths=raw_parquets,
                )

                logger.info("Existing target inference_xml_* columns preserved successfully.")

            # --------------------------------------------------
            # 5. Build final split DataFrames and parquet map after optional merge
            # --------------------------------------------------
            logger.info("Building final split DataFrames and parquet path map.")

            parquet_paths_by_split = self._parquet_paths_by_split(
                root=raw_root,
                parquet_paths=raw_parquets,
            )

            updated_frames_by_split: dict[str, list[pd.DataFrame]] = {}

            for parquet_path in raw_parquets:
                rel_path = str(parquet_path.relative_to(raw_root)).replace("\\", "/")
                split_name = self._split_name_for_repo_path(rel_path)
                df = pd.read_parquet(parquet_path)
                updated_frames_by_split.setdefault(split_name, []).append(df)

            combined_dfs = {
                split_name: pd.concat(frames, ignore_index=True)
                for split_name, frames in updated_frames_by_split.items()
            }

            raw_dataset = self._load_parquet_dataset_fresh(
                data_files=parquet_paths_by_split,
            )

            if not isinstance(raw_dataset, DatasetDict):
                raw_dataset = DatasetDict({"default": raw_dataset})

            logger.info(
                f"Final raw XML dataset prepared with splits={list(combined_dfs.keys())}."
            )

            # --------------------------------------------------
            # 6. Build commit operations
            # --------------------------------------------------
            logger.info("Building Hugging Face commit operations.")

            operations: list[CommitOperationAdd | CommitOperationDelete] = []

            paths_that_will_be_added = set(parquet_repo_paths)
            paths_that_will_be_added.add("README.md")

            if target_exists and upload_mode == "replace":
                logger.info(
                    f"Upload mode is 'replace'. Listing existing target files in {target_repo}."
                )

                target_files = api.list_repo_files(
                    repo_id=target_repo,
                    repo_type="dataset",
                    token=self.token,
                )

                operations.extend(
                    self._build_replace_delete_operations(
                        target_files=target_files,
                        paths_that_will_be_added=paths_that_will_be_added,
                    )
                )

            for parquet_path in raw_parquets:
                rel_path = str(parquet_path.relative_to(raw_root)).replace("\\", "/")

                operations.append(
                    CommitOperationAdd(
                        path_in_repo=rel_path,
                        path_or_fileobj=str(parquet_path),
                    )
                )

            logger.info(
                f"Prepared {len(operations)} commit operation(s) before README handling."
            )

            # --------------------------------------------------
            # 7. README handling
            # --------------------------------------------------
            logger.info("Building dataset README.")

            builder = HuggingFaceReadmeBuilder(
                repo_id=target_repo,
                dataset=raw_dataset,
                dataframes=combined_dfs,
                parquet_paths=parquet_paths_by_split,
                source_repos=[self.raw_xml_repo, self.inference_repo],
                description_text=(
                    f"This dataset is derived from "
                    f"[{self.raw_xml_repo}](https://huggingface.co/datasets/{self.raw_xml_repo}) "
                    f"and enriched with raw XML inference derived from "
                    f"[{self.inference_repo}](https://huggingface.co/datasets/{self.inference_repo})."
                ),
                tags=["xml", "pagexml", "inference", "htr"],
            )

            operations.append(
                CommitOperationAdd(
                    path_in_repo="README.md",
                    path_or_fileobj=builder.render().encode("utf-8"),
                )
            )

            logger.info(
                f"Prepared {len(operations)} total commit operation(s), including README."
            )

            # --------------------------------------------------
            # 8. Create repo if needed
            # --------------------------------------------------
            if not target_exists:
                logger.info(
                    f"Creating target dataset repository: {target_repo} "
                    f"(private={private})"
                )

                api.create_repo(
                    repo_id=target_repo,
                    repo_type="dataset",
                    private=private,
                    exist_ok=False,
                    token=self.token,
                )

            # --------------------------------------------------
            # 9. Single commit
            # --------------------------------------------------
            logger.info(f"Uploading raw XML writeback dataset to HF Hub: {target_repo}")

            api.create_commit(
                repo_id=target_repo,
                repo_type="dataset",
                operations=operations,
                commit_message="Write inference into raw XML.",
                token=self.token,
            )

            logger.info(
                f"Uploaded raw XML writeback dataset to HF Hub: {target_repo} "
                f"(upload_mode={upload_mode})"
            )

            logger.info(
                f"Raw XML writeback completed successfully: "
                f"updated_records={updated_records}, target_repo='{target_repo}'"
            )

            return {
                "updated_records": updated_records,
            }

        finally:
            for temp_root in temp_roots:
                logger.debug(f"Cleaning temporary directory: {temp_root}")
                shutil.rmtree(temp_root, ignore_errors=True)

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------
    @staticmethod
    def _build_inference_xml_column_name() -> str:
        """Build a timestamped column name for XML writeback output."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        return f"inference_xml_{timestamp}"

    @staticmethod
    def _is_original_line_augmentation_value(value) -> bool:
        """Return whether a line augmentation value marks an original line."""
        if pd.isna(value):
            return False
        return str(value).strip().lower() == ORIGINAL_LINE_AUGMENTATION_VALUE

    @staticmethod
    def _get_latest_inference_column(df: pd.DataFrame) -> str:
        """Return the latest non-XML inference text column from a DataFrame."""
        inference_cols = [
            c for c in df.columns
            if c.startswith("inference_") and not c.startswith("inference_xml_")
        ]

        if not inference_cols:
            raise RuntimeError("No inference column found in inference dataset.")

        return sorted(inference_cols)[-1]

    @classmethod
    def _build_lookup(cls, df: pd.DataFrame) -> dict:
        """Build a lookup from document and line identifiers to inferred text."""
        inference_col = cls._get_latest_inference_column(df)

        required_cols = ["filename", "region_id", "line_id", inference_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required columns for XML writeback: {missing}")

        work_df = df.copy()

        has_project_name = (
                "project_name" in work_df.columns
                and work_df["project_name"].fillna("").astype(str).str.strip().ne("").any()
        )

        if not has_project_name:
            work_df["project_name"] = ""
            logger.warning(
                "No usable project_name column found for XML writeback lookup. "
                "Using filename + region_id + line_id only."
            )

        if LINE_AUGMENTATION_COLUMN in work_df.columns:
            work_df = work_df[
                work_df[LINE_AUGMENTATION_COLUMN].apply(
                    cls._is_original_line_augmentation_value
                )
            ]

        lookup: dict[str, dict[str, dict[tuple[str, str], str]]] = {}

        key_cols = ["project_name", "filename", "region_id", "line_id"]

        for raw_key_values, group in work_df.groupby(key_cols, dropna=False):
            key_values = cast(tuple[Any, Any, Any, Any], raw_key_values)

            project_name, filename, region_id, line_id = [
                "" if pd.isna(v) else str(v).strip()
                for v in key_values
            ]

            texts = (
                group[inference_col]
                .dropna()
                .astype(str)
                .map(str.strip)
            )

            unique_texts = sorted({text for text in texts if text != ""})

            if len(unique_texts) == 0:
                continue

            if len(unique_texts) > 1:
                logger.warning(
                    f"Ambiguous inference texts for "
                    f"project='{project_name}', filename='{filename}', "
                    f"region_id='{region_id}', line_id='{line_id}': {unique_texts}. "
                    f"Skipping XML writeback for this line."
                )
                continue

            lookup.setdefault(project_name, {}).setdefault(filename, {})[
                (region_id, line_id)
            ] = unique_texts[0]

        return lookup

    @staticmethod
    def _update_df(df: pd.DataFrame, lookup: dict, new_col: str):
        """Write inferred XML strings into a DataFrame.

        Args:
            df: Raw XML DataFrame containing ``filename`` and ``xml_content`` columns.
            lookup: Nested lookup of project name, filename, and line identifiers to text.
            new_col: Name of the column that should receive updated XML strings.

        Returns:
            Updated DataFrame and number of rows where XML content changed.
        """
        df[new_col] = ""

        updated_count = 0

        for i, r in df.iterrows():
            project_name = ""
            if "project_name" in df.columns and pd.notna(r.get("project_name")):
                project_name = str(r.get("project_name", "")).strip()

            filename = str(r["filename"]).strip()

            filename_lookup = None

            if project_name and project_name in lookup and filename in lookup[project_name]:
                filename_lookup = lookup[project_name][filename]

            elif "" in lookup and filename in lookup[""]:
                filename_lookup = lookup[""][filename]

            if filename_lookup is None:
                continue

            xp = XMLProcessor.from_string(r["xml_content"])

            changed_count = xp.insert_inferred_lines(xp.root, filename_lookup)

            if changed_count == 0:
                continue

            df.at[i, new_col] = xp.tree_to_string()
            updated_count += 1

        return df, updated_count
