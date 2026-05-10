from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, snapshot_download, CommitOperationAdd
from datasets import load_dataset, DatasetDict
from flow_inference.xml_processing import XMLProcessor
from flow_inference.configure_dataset_card import HuggingFaceReadmeBuilder
from flow_inference.utils.logging.inference_logger import logger

LINE_AUGMENTATION_COLUMN = "line_augmentation"
ORIGINAL_LINE_AUGMENTATION_VALUE = "original"


class InferenceToRawXMLWriter:
    def __init__(self,
                 raw_xml_repo: str,
                 inference_repo: str,
                 token: str,
                 allow_source_repo_update: bool = False):
        self.raw_xml_repo = raw_xml_repo
        self.inference_repo = inference_repo
        self.token = token
        self.allow_source_repo_update = allow_source_repo_update

    # ------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------
    def process_and_upload(self, output_repo: str | None = None):
        api = HfApi()

        target_repo = output_repo or self.raw_xml_repo

        if target_repo == self.raw_xml_repo and not self.allow_source_repo_update:
            raise RuntimeError("Refusing to upload into the source raw XML repo")

        if target_repo == self.inference_repo:
            raise RuntimeError("Refusing to upload into the inference source repo")

        # --------------------------------------------------
        # 1. Load inference
        # --------------------------------------------------
        inf_info = api.dataset_info(self.inference_repo, token=self.token)
        inf_root = snapshot_download(
            self.inference_repo,
            repo_type="dataset",
            revision=inf_info.sha,
            token=self.token,
        )

        inf_parquets = list(Path(inf_root).rglob("*.parquet"))
        inf_ds = load_dataset("parquet", data_files=[str(p) for p in inf_parquets])["train"]
        lookup = self._build_lookup(inf_ds.to_pandas())

        # --------------------------------------------------
        # 2. Snapshot raw XML repo
        # --------------------------------------------------
        raw_info = api.dataset_info(self.raw_xml_repo, token=self.token)
        raw_root = snapshot_download(
            self.raw_xml_repo,
            repo_type="dataset",
            revision=raw_info.sha,
            token=self.token,
        )

        raw_root = Path(raw_root)
        raw_parquets = list(raw_root.rglob("*.parquet"))

        ops: list[CommitOperationAdd] = []

        # --------------------------------------------------
        # 3. Update parquet files in place
        # --------------------------------------------------
        updated_frames_by_split: dict[str, list[pd.DataFrame]] = {}
        parquet_paths_by_split: dict[str, list[str]] = {}
        new_xml_column = self._build_inference_xml_column_name()

        for parquet_path in raw_parquets:
            df = pd.read_parquet(parquet_path)

            df, _ = self._update_df(df, lookup, new_xml_column)
            df.to_parquet(parquet_path, index=False)

            rel_path = parquet_path.relative_to(raw_root)
            parts = rel_path.parts

            if len(parts) >= 3 and parts[0] == "data" and parts[1] in {"train", "test"}:
                split_name = parts[1]
            else:
                split_name = "default"

            updated_frames_by_split.setdefault(split_name, []).append(df)
            parquet_paths_by_split.setdefault(split_name, []).append(str(parquet_path))

            ops.append(
                CommitOperationAdd(
                    path_in_repo=str(rel_path).replace("\\", "/"),
                    path_or_fileobj=str(parquet_path),
                )
            )

        # --------------------------------------------------
        # 4. README handling (generate fresh)
        # --------------------------------------------------
        combined_dfs = {
            split_name: pd.concat(frames, ignore_index=True)
            for split_name, frames in updated_frames_by_split.items()
        }

        raw_dataset = load_dataset(
            "parquet",
            data_files=parquet_paths_by_split,
        )

        if not isinstance(raw_dataset, DatasetDict):
            raw_dataset = DatasetDict({"default": raw_dataset})

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

        ops.append(
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=builder.render().encode("utf-8"),
            )
        )

        # --------------------------------------------------
        # 5. Ensure repo exists
        # --------------------------------------------------
        api.create_repo(
            repo_id=target_repo,
            repo_type="dataset",
            private=True,
            exist_ok=True,
            token=self.token,
        )

        # --------------------------------------------------
        # 6. Single commit
        # --------------------------------------------------
        api.create_commit(
            repo_id=target_repo,
            repo_type="dataset",
            operations=ops,
            commit_message="Write inference into raw XML.",
            token=self.token,
        )

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------
    def _build_inference_xml_column_name(self) -> str:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
        repo_name = self.inference_repo.split("/")[-1].replace("-", "_").replace("/", "_")
        return f"inference_xml_{timestamp}_from_{repo_name}"

    @staticmethod
    def _is_original_line_augmentation_value(value) -> bool:
        if pd.isna(value):
            return False
        return str(value).strip().lower() == ORIGINAL_LINE_AUGMENTATION_VALUE

    @staticmethod
    def _get_latest_inference_column(df: pd.DataFrame) -> str:
        inference_cols = [
            c for c in df.columns
            if c.startswith("inference_") and not c.startswith("inference_xml_")
        ]

        if not inference_cols:
            raise RuntimeError("No inference column found in inference dataset.")

        return sorted(inference_cols)[-1]

    @classmethod
    def _build_lookup(cls, df: pd.DataFrame) -> dict:
        inference_col = cls._get_latest_inference_column(df)

        required_cols = ["project_name", "filename", "line_id", inference_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required columns for XML writeback: {missing}")

        work_df = df.copy()

        if LINE_AUGMENTATION_COLUMN in work_df.columns:
            work_df = work_df[
                work_df[LINE_AUGMENTATION_COLUMN].apply(
                    cls._is_original_line_augmentation_value
                )
            ]

        lookup: dict[str, dict[str, dict[str, str]]] = {}

        key_cols = ["project_name", "filename", "line_id"]

        for key_values, group in work_df.groupby(key_cols, dropna=False):
            project_name, filename, line_id = [str(v) for v in key_values]

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
                    f"{project_name}/{filename}/{line_id}: {unique_texts}. "
                    f"Skipping XML writeback for this line."
                )
                continue

            lookup.setdefault(project_name, {}).setdefault(filename, {})[line_id] = unique_texts[0]

        return lookup

    @staticmethod
    def _update_df(df: pd.DataFrame, lookup: dict, new_col: str):
        df[new_col] = ""

        updated = False

        for i, r in df.iterrows():
            proj = str(r["project_name"])
            fn = str(r["filename"])

            if proj not in lookup or fn not in lookup[proj]:
                continue

            xp = XMLProcessor.from_string(r["xml_content"])

            changed_count = xp.insert_inferred_lines(xp.root, lookup[proj][fn])

            if changed_count == 0:
                continue

            df.at[i, new_col] = xp.tree_to_string()
            updated = True

        return df, ([new_col] if updated else [])
