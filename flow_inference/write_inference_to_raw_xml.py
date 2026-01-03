import tempfile

import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from flow_inference.xml_processing import XMLProcessor


class InferenceToRawXMLWriter:
    def __init__(self,
                 raw_xml_repo: str,
                 inference_repo: str,
                 token: str):
        self.raw_xml_repo = raw_xml_repo
        self.inference_repo = inference_repo
        self.token = token
        self.raw_dataset = None
        self.inference_dataset = None

    # ------------------------------------------------------------
    # WRITE INFERENCE TO RAW XML PIPELINE
    # ------------------------------------------------------------
    def process_and_upload(self, output_repo: str = None):
        """
        1. Load datasets
        2. Detect inference column automatically
        3. Build per-line inference lookup
        4. Update XML
        5. Upload updated dataset
        """

        # Step 1: load datasets
        self.load_datasets()

        # Step 2: detect inference column + build lookup
        lookup = self.build_inference_lookup()

        # Step 3: update raw XML column
        updated_df = self.update_raw_xml_dataset(lookup)

        # Step 4: upload dataset with modified XML to Hugging Face Hub
        target_repo = output_repo if output_repo else self.raw_xml_repo
        self.upload_updated_dataset(updated_df, target_repo)

    # ------------------------------------------------------------
    # HELPER METHODS
    # ------------------------------------------------------------
    def _extract_default(self, ds):
        """Return a dataset regardless of split structure."""
        if isinstance(ds, Dataset):
            return ds
        return ds[next(iter(ds.keys()))]

    def load_datasets(self):
        api = HfApi()

        raw_info = api.dataset_info(self.raw_xml_repo, token=self.token)
        inf_info = api.dataset_info(self.inference_repo, token=self.token)

        raw_cache = tempfile.mkdtemp(prefix="hf_raw_")
        inf_cache = tempfile.mkdtemp(prefix="hf_inf_")

        raw = load_dataset(
            self.raw_xml_repo,
            token=self.token,
            revision=raw_info.sha,
            cache_dir=raw_cache,
        )

        inf = load_dataset(
            self.inference_repo,
            token=self.token,
            revision=inf_info.sha,
            cache_dir=inf_cache,
        )

        self.raw_dataset = self._extract_default(raw)
        self.inference_dataset = self._extract_default(inf)

    def detect_inference_column(self):
        """Automatically find the column starting with 'inference_'."""
        cols = self.inference_dataset.column_names
        infer_cols = [c for c in cols if c.startswith("inference_")]

        if not infer_cols:
            raise ValueError(
                f"No inference column found in inference dataset. "
                f"Expected columns starting with 'inference_'. Found: {cols}"
            )
        if len(infer_cols) > 1:
            return infer_cols[0]

        return infer_cols[0]

    def build_inference_lookup(self):
        """Uses auto-detected inference column."""
        inference_column = self.detect_inference_column()

        df = self.inference_dataset.to_pandas()
        lookup = {}

        for _, row in df.iterrows():
            filename = row["filename"]
            line_id = row["line_id"]
            text = row[inference_column]

            lookup.setdefault(filename, {})[line_id] = text

        return lookup

    def update_raw_xml_dataset(self, lookup: dict):
        df = self.raw_dataset.to_pandas()
        new_col = f"inference_xml_{pd.Timestamp.now().strftime('%Y%m%d')}"
        df[new_col] = ""

        updated = 0

        for idx, row in df.iterrows():
            filename = row["filename"]
            raw_xml = row["xml"]

            if filename not in lookup:
                continue

            xp = XMLProcessor.from_string(raw_xml)
            xp.insert_inferred_lines(xp.root, lookup[filename])
            updated_xml = xp.tree_to_string()

            df.at[idx, new_col] = updated_xml
            updated += 1

        print(f"Updated {updated} XML records.")
        return df

    def upload_updated_dataset(self, df, repo_id: str):
        api = HfApi()

        try:
            api.repo_info(repo_id, repo_type="dataset", token=self.token)
        except HfHubHTTPError:
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=True,
                token=self.token
            )

        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(repo_id, token=self.token)
