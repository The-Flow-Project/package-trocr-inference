from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd
import yaml
from datasets import DatasetDict, Features, Sequence, Value, Image as DatasetImage


@dataclass
class ReadmeStats:
    splits_info: dict[str, int]
    split_bytes: dict[str, int]
    total_samples: int
    total_bytes: int
    projects: list[str]
    features: Features


class HuggingFaceReadmeBuilder:
    def __init__(
            self,
            repo_id: str,
            dataset: DatasetDict,
            dataframes: dict[str, pd.DataFrame],
            parquet_paths: dict[str, list[str]],
            source_repos: list[str] | None = None,
            description_text: str | None = None,
            evaluation_info: dict[str, Any] | None = None,
            tags: list[str] | None = None,
            license_name: str = "mit",
    ):
        self.repo_id = repo_id
        self.dataset = dataset
        self.dataframes = dataframes
        self.parquet_paths = parquet_paths
        self.source_repos = source_repos or []
        self.description_text = description_text
        self.evaluation_info = evaluation_info
        self.tags = tags or ["image-to-text", "htr", "trocr", "inference", "pagexml"]
        self.license_name = license_name

    @classmethod
    def from_handler(
            cls,
            repo_id: str,
            dataset: DatasetDict,
            dataframes: dict[str, pd.DataFrame],
            parquet_paths: dict[str, list[str]],
            source_repos: list[str] | None = None,
            evaluation_info: dict[str, Any] | None = None,
    ) -> "HuggingFaceReadmeBuilder":
        return cls(
            repo_id=repo_id,
            dataset=dataset,
            dataframes=dataframes,
            parquet_paths=parquet_paths,
            source_repos=source_repos,
            evaluation_info=evaluation_info,
        )

    def _get_splits_info(self) -> dict[str, int]:
        return {
            split_name: len(df)
            for split_name, df in self.dataframes.items()
        }

    def _get_split_bytes(self) -> dict[str, int]:
        split_bytes: dict[str, int] = {}

        for split_name in self.dataframes.keys():
            files = self.parquet_paths.get(split_name, [])
            split_bytes[split_name] = sum(
                Path(p).stat().st_size
                for p in files
                if Path(p).exists()
            )

        return split_bytes

    def _get_projects(self) -> list[str]:
        projects = set()

        for df in self.dataframes.values():
            if "project_name" in df.columns:
                values = df["project_name"].dropna().astype(str)
                projects.update(v for v in values if v.strip())

        return sorted(projects)

    def _get_features(self) -> Features:
        merged_features = Features({})

        for split_name in self.dataset.keys():
            for feature_name, feature in self.dataset[split_name].features.items():
                if feature_name not in merged_features:
                    merged_features[feature_name] = feature

        extra_columns = set()
        for df in self.dataframes.values():
            extra_columns.update(df.columns)

        for col in sorted(extra_columns):
            if col not in merged_features:
                merged_features[col] = Value("string")

        return merged_features

    def _build_stats(self) -> ReadmeStats:
        splits_info = self._get_splits_info()
        split_bytes = self._get_split_bytes()
        total_samples = sum(splits_info.values())
        total_bytes = sum(split_bytes.values())
        projects = self._get_projects()
        features = self._get_features()

        return ReadmeStats(
            splits_info=splits_info,
            split_bytes=split_bytes,
            total_samples=total_samples,
            total_bytes=total_bytes,
            projects=projects,
            features=features,
        )

    def _build_description(self) -> str:
        if self.description_text:
            return self.description_text

        if self.source_repos:
            links = ", ".join(
                f"[{repo}](https://huggingface.co/datasets/{repo})"
                for repo in self.source_repos
            )
            return (
                f"This dataset is derived from {links} "
                f"and has been enriched with inference results."
            )

        return "This dataset contains inference results."

    def _render_evaluation_section(self) -> list[str]:
        if not self.evaluation_info:
            return []

        lines = [
            "## Evaluation Results",
            "",
            "This repository includes evaluation artifacts for the latest inference output.",
            "",
        ]

        if "timestamp" in self.evaluation_info:
            lines.append(f"- **Timestamp**: {self.evaluation_info['timestamp']}")
        if "cer" in self.evaluation_info:
            lines.append(f"- **CER**: {self.evaluation_info['cer']}")
        if "eval_rows" in self.evaluation_info:
            lines.append(f"- **Evaluated rows**: {self.evaluation_info['eval_rows']}")
        if "evaluated_split" in self.evaluation_info:
            lines.append(f"- **Evaluated split**: {self.evaluation_info['evaluated_split']}")
        if "inference_column" in self.evaluation_info:
            lines.append(f"- **Inference column**: `{self.evaluation_info['inference_column']}`")

        lines.extend([
            "",
            "### Evaluation Files",
            "",
            "```text",
            self.evaluation_info.get("evaluation_path", "evaluation/"),
            "├── gt.txt",
            "├── hypothesis.txt",
            "└── evaluation_report.json",
            "```",
            "",
        ])

        return lines

    def _generate_frontmatter_dict(self, stats: ReadmeStats) -> dict[str, Any]:
        return {
            "dataset_info": {
                "config_name": "default",
                "features": self._features_to_yaml_objects(stats.features),
                "splits": [
                    {
                        "name": split_name,
                        "num_examples": stats.splits_info[split_name],
                        "num_bytes": stats.split_bytes.get(split_name, 0),
                    }
                    for split_name in stats.splits_info
                ],
                "download_size": stats.total_bytes,
                "dataset_size": stats.total_bytes,
            },
            "configs": [
                {
                    "config_name": "default",
                    "data_files": [
                        {
                            "split": split_name,
                            "path": (
                                "data/**/*.parquet"
                                if split_name == "default"
                                else f"data/{split_name}/**/*.parquet"
                            ),
                        }
                        for split_name in stats.splits_info
                    ],
                }
            ],
            "tags": self.tags,
            "license": self.license_name,
        }

    def _render_frontmatter(self, stats: ReadmeStats) -> str:
        data = self._generate_frontmatter_dict(stats)
        return yaml.safe_dump(
            data,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ).strip()

    def _features_to_yaml_objects(self, features: Features) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for name, feature in features.items():
            result.append({
                "name": name,
                "dtype": self._feature_dtype_to_object(feature),
            })

        return result

    def _feature_dtype_to_object(self, feature: Any) -> Any:
        if isinstance(feature, Value):
            return feature.dtype

        if isinstance(feature, DatasetImage):
            if hasattr(feature, "decode") and feature.decode is False:
                return {"image": {"decode": False}}
            return {"image": {}}

        if isinstance(feature, Sequence):
            inner = getattr(feature, "feature", None)
            if inner is None:
                return "list"
            return {"sequence": self._feature_dtype_to_object(inner)}

        if isinstance(feature, dict):
            return {
                "struct": [
                    {
                        "name": sub_name,
                        "dtype": self._feature_dtype_to_object(sub_feature),
                    }
                    for sub_name, sub_feature in feature.items()
                ]
            }

        return str(feature)

    def _render_body(self, stats: ReadmeStats) -> str:
        repo_short = self.repo_id.split("/")[-1]
        approx_mb = stats.total_bytes / (1024 * 1024) if stats.total_bytes else 0.0

        lines = [
            f"# Dataset Card for {repo_short}",
            "",
            self._build_description(),
            "",
            "## Dataset Summary",
            "",
            f"This dataset contains {stats.total_samples:,} samples across {len(stats.splits_info)} split(s).",
            "",
            "### Projects Included",
            "",
            ", ".join(stats.projects) if stats.projects else "No project metadata available.",
            "",
        ]

        lines.extend(self._render_evaluation_section())

        lines.extend([
            "## Dataset Structure",
            "",
            "### Data Splits",
            "",
        ])

        for split_name, count in stats.splits_info.items():
            lines.append(f"- **{split_name}**: {count:,} samples")

        lines.extend([
            "",
            "### Dataset Size",
            "",
            f"- Approximate total size: {approx_mb:,.2f} MB",
            f"- Total samples: {stats.total_samples:,}",
        ])

        lines.extend([
            "",
            "### Features",
            "",
        ])

        for feature_name, feature_type in stats.features.items():
            lines.append(f"- **{feature_name}**: `{feature_type}`")

        lines.extend([
            "",
            "## Data Organization",
            "",
            "Data is organized as parquet shards by split and project:",
            "",
            "```",
            "data/",
            "├── <split>/",
            "│   └── <project_name>/",
            "│       └── <timestamp>-<shard>.parquet",
            "```",
            "",
            "The HuggingFace Hub automatically merges all parquet files when loading the dataset.",
            "",
            "## Usage",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
            "# Load entire dataset",
            f'dataset = load_dataset("{self.repo_id}")',
            "",
            "# Load specific split",
        ])

        first_split = next(iter(stats.splits_info.keys()), "train")
        lines.append(f'dataset_split = load_dataset("{self.repo_id}", split="{first_split}")')

        lines.extend([
            "```",
            "",
        ])

        return "\n".join(lines)

    def render(self) -> str:
        stats = self._build_stats()
        frontmatter = self._render_frontmatter(stats)
        body = self._render_body(stats)
        return f"---\n{frontmatter}\n---\n\n{body}"