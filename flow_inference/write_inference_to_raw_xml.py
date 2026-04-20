from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, snapshot_download, CommitOperationAdd
from datasets import load_dataset, DatasetDict
from flow_inference.xml_processing import XMLProcessor
from flow_inference.configure_dataset_card import HuggingFaceReadmeBuilder


class InferenceToRawXMLWriter:
    def __init__(self, raw_xml_repo: str, inference_repo: str, token: str):
        self.raw_xml_repo = raw_xml_repo
        self.inference_repo = inference_repo
        self.token = token

    # ------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------
    def process_and_upload(self, output_repo: str | None = None):
        api = HfApi()

        target_repo = output_repo or self.raw_xml_repo

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
    def _build_lookup(df: pd.DataFrame) -> dict:
        inference_col = [c for c in df.columns if c.startswith("inference_")][0]

        lookup = {}
        for _, r in df.iterrows():
            lookup.setdefault(r["project_name"], {}) \
                  .setdefault(r["filename"], {})[r["line_id"]] = r[inference_col]
        return lookup

    @staticmethod
    def _update_df(df: pd.DataFrame, lookup: dict, new_col: str):
        df[new_col] = ""

        updated = False

        for i, r in df.iterrows():
            proj = r["project_name"]
            fn = r["filename"]

            if proj not in lookup or fn not in lookup[proj]:
                continue

            xp = XMLProcessor.from_string(r["xml_content"])
            xp.insert_inferred_lines(xp.root, lookup[proj][fn])
            df.at[i, new_col] = xp.tree_to_string()
            updated = True

        return df, ([new_col] if updated else [])
