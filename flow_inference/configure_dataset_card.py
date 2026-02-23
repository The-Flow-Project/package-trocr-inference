from __future__ import annotations
import yaml
import re
from typing import Tuple


class HuggingFaceReadmeEditor:
    def __init__(self, readme_text: str):
        self.readme_text = readme_text

    # --------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------
    def _split_frontmatter(self) -> Tuple[dict, str]:
        """
        Splits README into (metadata_dict, body_text)

        Supports:
        - proper frontmatter
        - broken YAML (best effort)
        - no YAML at all
        """
        text = self.readme_text.lstrip()

        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                yaml_text = parts[1]
                body = parts[2]
            else:
                yaml_text = ""
                body = text
        else:
            # No frontmatter → assume everything is body
            return {}, self.readme_text

        try:
            data = yaml.safe_load(yaml_text) or {}
        except Exception:
            data = {}

        return data, body.lstrip("\n")

    def _build_frontmatter(self, metadata: dict, body: str) -> str:
        yaml_text = yaml.safe_dump(
            metadata,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ).strip()

        return f"---\n{yaml_text}\n---\n\n{body.lstrip()}"

    # --------------------------------------------------
    # METADATA
    # --------------------------------------------------
    def add_features(self, new_features: list[str]) -> "HuggingFaceReadmeEditor":
        metadata, body = self._split_frontmatter()

        dataset_info = metadata.setdefault("dataset_info", {})
        features = dataset_info.setdefault("features", [])

        existing = {f.get("name") for f in features if isinstance(f, dict)}

        for f in new_features:
            if f not in existing:
                features.append({"name": f, "dtype": "string"})

        self.readme_text = self._build_frontmatter(metadata, body)
        return self

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------
    def rewrite_title(self, target_repo: str) -> "HuggingFaceReadmeEditor":
        short = target_repo.split("/")[-1]
        self.readme_text = re.sub(
            r'(?m)^(#\s*Dataset Card for\s+).*$',
            rf'\1{short}',
            self.readme_text
        )
        return self

    # --------------------------------------------------
    # USAGE SNIPPETS
    # --------------------------------------------------
    def rewrite_usage_repo_ids(self, target_repo: str) -> "HuggingFaceReadmeEditor":
        pattern = r'(load_dataset\(\s*[\'"])([^\'"]+)([\'"])'
        self.readme_text = re.sub(pattern, rf"\1{target_repo}\3", self.readme_text)
        return self

    # --------------------------------------------------
    # GENERIC REPLACE
    # --------------------------------------------------
    def replace_repo_name(self, old: str, new: str) -> "HuggingFaceReadmeEditor":
        self.readme_text = self.readme_text.replace(old, new)
        return self

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------
    def render(self) -> str:
        return self.readme_text
