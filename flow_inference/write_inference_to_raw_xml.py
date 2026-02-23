import tempfile
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub._commit_api import CommitOperationAdd
from datasets import load_dataset
from huggingface_hub.errors import HfHubHTTPError
from flow_inference.xml_processing import XMLProcessor
from flow_inference.configure_dataset_card import HuggingFaceReadmeEditor


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
        new_feature_cols = set()

        for parquet_path in raw_parquets:
            df = pd.read_parquet(parquet_path)

            df, created_cols = self._update_df(df, lookup)
            new_feature_cols |= set(created_cols)

            df.to_parquet(parquet_path, index=False)

            ops.append(
                CommitOperationAdd(
                    path_in_repo=str(parquet_path.relative_to(raw_root)),
                    path_or_fileobj=str(parquet_path),
                )
            )

        # --------------------------------------------------
        # 4. README handling (copy + modify)
        # --------------------------------------------------
        readme_path = raw_root / "README.md"
        if readme_path.exists():
            text = readme_path.read_text(encoding="utf-8")

            edited = (
                HuggingFaceReadmeEditor(text)
                .add_features(list(new_feature_cols))
                .rewrite_title(target_repo)
                .rewrite_usage_repo_ids(target_repo)
                .replace_repo_name(self.raw_xml_repo, target_repo)
                .render()
            )

            ops.append(
                CommitOperationAdd(
                    path_in_repo="README.md",
                    path_or_fileobj=edited.encode("utf-8"),
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
    def _build_lookup(self, df: pd.DataFrame) -> dict:
        inference_col = [c for c in df.columns if c.startswith("inference_")][0]

        lookup = {}
        for _, r in df.iterrows():
            lookup.setdefault(r["project_name"], {}) \
                  .setdefault(r["filename"], {})[r["line_id"]] = r[inference_col]
        return lookup

    def _update_df(self, df: pd.DataFrame, lookup: dict):
        new_col = f"inference_xml_{pd.Timestamp.now().strftime('%Y%m%d')}"
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
