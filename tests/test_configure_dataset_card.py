import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset, DatasetDict, Features, Sequence, Value, Image as DatasetImage

from flow_inference.configure_dataset_card import (
    HuggingFaceReadmeBuilder,
    ReadmeStats,
)


class TestHuggingFaceReadmeBuilder(unittest.TestCase):
    # -------------------------------------------------------------
    # UNIT TEST: MAKE DATASET DICT
    # -------------------------------------------------------------
    @staticmethod
    def _make_dataset_dict() -> DatasetDict:
        features = Features(
            {
                "image": DatasetImage(decode=False),
                "text": Value("string"),
                "line_id": Value("string"),
                "line_reading_order": Value("int64"),
                "line_coords": Sequence(Sequence(Value("int64"))),
                "filename": Value("string"),
                "project_name": Value("string"),
            }
        )

        train = Dataset.from_dict(
            {
                "image": [{"path": "a.png", "bytes": None}, {"path": "b.png", "bytes": None}],
                "text": ["foo", "bar"],
                "line_id": ["l1", "l2"],
                "line_reading_order": [1, 2],
                "line_coords": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                "filename": ["file1.xml", "file2.xml"],
                "project_name": ["proj_b", "proj_a"],
            },
            features=features,
        )

        test = Dataset.from_dict(
            {
                "image": [{"path": "c.png", "bytes": None}],
                "text": ["baz"],
                "line_id": ["l3"],
                "line_reading_order": [3],
                "line_coords": [[[9, 10], [11, 12]]],
                "filename": ["file3.xml"],
                "project_name": ["proj_c"],
            },
            features=features,
        )

        return DatasetDict({"train": train, "test": test})

    # -------------------------------------------------------------
    # UNIT TEST: MAKE DATAFRAMES
    # -------------------------------------------------------------
    @staticmethod
    def _make_dataframes() -> dict[str, pd.DataFrame]:
        return {
            "train": pd.DataFrame(
                {
                    "image": ["img1", "img2"],
                    "text": ["foo", "bar"],
                    "line_id": ["l1", "l2"],
                    "line_reading_order": [1, 2],
                    "line_coords": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                    "filename": ["file1.xml", "file2.xml"],
                    "project_name": ["proj_b", "proj_a"],
                    "inference_2024_model_x": ["pred1", "pred2"],
                }
            ),
            "test": pd.DataFrame(
                {
                    "image": ["img3"],
                    "text": ["baz"],
                    "line_id": ["l3"],
                    "line_reading_order": [3],
                    "line_coords": [[[9, 10], [11, 12]]],
                    "filename": ["file3.xml"],
                    "project_name": ["proj_c"],
                    "inference_2024_model_x": ["pred3"],
                }
            ),
        }

    # -------------------------------------------------------------
    # UNIT TEST: MAKE PARQUET PATHS
    # -------------------------------------------------------------
    @staticmethod
    def _make_parquet_paths(tmp_path: Path) -> dict[str, list[str]]:
        train_file = tmp_path / "train.parquet"
        test_file = tmp_path / "test.parquet"
        train_file.write_bytes(b"a" * 10)
        test_file.write_bytes(b"b" * 20)

        return {
            "train": [str(train_file)],
            "test": [str(test_file)],
        }

    def _make_builder(
        self,
        tmp_path: Path,
    ) -> HuggingFaceReadmeBuilder:
        return HuggingFaceReadmeBuilder(
            repo_id="user/my-dataset",
            dataset=self._make_dataset_dict(),
            dataframes=self._make_dataframes(),
            parquet_paths=self._make_parquet_paths(tmp_path),
        )

    # -------------------------------------------------------------
    # UNIT TEST: FROM HANDLER
    # -------------------------------------------------------------
    def test_from_handler_builds_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            builder = HuggingFaceReadmeBuilder.from_handler(
                repo_id="user/my-dataset",
                dataset=self._make_dataset_dict(),
                dataframes=self._make_dataframes(),
                parquet_paths=self._make_parquet_paths(tmp_path)
            )

            self.assertIsInstance(builder, HuggingFaceReadmeBuilder)
            self.assertEqual(builder.repo_id, "user/my-dataset")

    # -------------------------------------------------------------
    # UNIT TEST: GET SPLITS INFO
    # -------------------------------------------------------------
    def test_get_splits_info_returns_dataframe_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(Path(tmp_dir))
            self.assertEqual(builder._get_splits_info(), {"train": 2, "test": 1})

    # -------------------------------------------------------------
    # UNIT TEST: GET SPLIT BYTES EXISTING FILES
    # -------------------------------------------------------------
    def test_get_split_bytes_sums_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(Path(tmp_dir))
            self.assertEqual(builder._get_split_bytes(), {"train": 10, "test": 20})

    # -------------------------------------------------------------
    # UNIT TEST: GET SPLIT BYTES MISSING FILES
    # -------------------------------------------------------------
    def test_get_split_bytes_ignores_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            parquet_paths = self._make_parquet_paths(tmp_path)
            parquet_paths["train"].append(str(tmp_path / "missing.parquet"))

            builder = HuggingFaceReadmeBuilder(
                repo_id="user/my-dataset",
                dataset=self._make_dataset_dict(),
                dataframes=self._make_dataframes(),
                parquet_paths=parquet_paths,
            )

            self.assertEqual(builder._get_split_bytes(), {"train": 10, "test": 20})

    # -------------------------------------------------------------
    # UNIT TEST: GET PROJECTS
    # -------------------------------------------------------------
    def test_get_projects_returns_sorted_unique_nonempty_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dfs = self._make_dataframes()
            dfs["test"] = pd.concat(
                [
                    dfs["test"],
                    pd.DataFrame(
                        {
                            "image": ["img4"],
                            "text": ["qux"],
                            "line_id": ["l4"],
                            "line_reading_order": [4],
                            "line_coords": [[[1, 1], [2, 2]]],
                            "filename": ["file4.xml"],
                            "project_name": ["proj_a"],
                            "inference_2024_model_x": ["pred4"],
                        }
                    ),
                ],
                ignore_index=True,
            )

            builder = HuggingFaceReadmeBuilder(
                repo_id="user/my-dataset",
                dataset=self._make_dataset_dict(),
                dataframes=dfs,
                parquet_paths=self._make_parquet_paths(Path(tmp_dir)),
            )

            self.assertEqual(builder._get_projects(), ["proj_a", "proj_b", "proj_c"])

    # -------------------------------------------------------------
    # UNIT TEST: GET FEATURES
    # -------------------------------------------------------------
    def test_get_features_preserves_existing_and_adds_new_columns_as_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(Path(tmp_dir))
            features = builder._get_features()

            self.assertIn("image", features)
            self.assertIn("text", features)
            self.assertIn("line_id", features)
            self.assertIn("inference_2024_model_x", features)
            self.assertIsInstance(features["inference_2024_model_x"], Value)
            self.assertEqual(features["inference_2024_model_x"].dtype, "string")

    # -------------------------------------------------------------
    # UNIT TEST: GET STATS
    # -------------------------------------------------------------
    def test_build_stats_aggregates_expected_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(Path(tmp_dir))
            stats = builder._build_stats()

            self.assertIsInstance(stats, ReadmeStats)
            self.assertEqual(stats.splits_info, {"train": 2, "test": 1})
            self.assertEqual(stats.split_bytes, {"train": 10, "test": 20})
            self.assertEqual(stats.total_samples, 3)
            self.assertEqual(stats.total_bytes, 30)
            self.assertEqual(stats.projects, ["proj_a", "proj_b", "proj_c"])
            self.assertIn("inference_2024_model_x", stats.features)

    # -------------------------------------------------------------
    # UNIT TEST: FEATURE DTYPE TO OBJECT - BASIC CASE
    # -------------------------------------------------------------
    def test_feature_dtype_to_object_handles_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(Path(tmp_dir))
            self.assertEqual(builder._feature_dtype_to_object(Value("string")), "string")

    # -------------------------------------------------------------
    # UNIT TEST: FEATURE DTYPE TO OBJECT - IMAGE DECODE FALSE
    # -------------------------------------------------------------
    def test_feature_dtype_to_object_handles_image_decode_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(Path(tmp_dir))
            self.assertEqual(
                builder._feature_dtype_to_object(DatasetImage(decode=False)),
                {"image": {"decode": False}},
            )

    # -------------------------------------------------------------
    # UNIT TEST: FEATURE DTYPE TO OBJECT - NESTED SEQUENCE
    # -------------------------------------------------------------
    def test_feature_dtype_to_object_handles_nested_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(Path(tmp_dir))
            result = builder._feature_dtype_to_object(Sequence(Sequence(Value("int64"))))
            self.assertEqual(result, {"sequence": {"sequence": "int64"}})

    # -------------------------------------------------------------
    # UNIT TEST: FEATURES TO YAML OBJECTS
    # -------------------------------------------------------------
    def test_features_to_yaml_objects_contains_expected_feature_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(Path(tmp_dir))
            features = builder._get_features()

            yaml_objects = builder._features_to_yaml_objects(features)
            names = [item["name"] for item in yaml_objects]

            self.assertIn("image", names)
            self.assertIn("text", names)
            self.assertIn("inference_2024_model_x", names)

    # -------------------------------------------------------------
    # UNIT TEST: GENERATE FRONTMATTER DICT
    # -------------------------------------------------------------
    def test_generate_frontmatter_dict_contains_expected_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(Path(tmp_dir))
            stats = builder._build_stats()
            frontmatter = builder._generate_frontmatter_dict(stats)

            self.assertEqual(frontmatter["dataset_info"]["config_name"], "default")
            self.assertEqual(frontmatter["dataset_info"]["download_size"], 30)
            self.assertEqual(frontmatter["dataset_info"]["dataset_size"], 30)
            self.assertEqual(frontmatter["license"], "mit")
            self.assertEqual(
                frontmatter["tags"],
                ["image-to-text", "htr", "trocr", "inference", "pagexml"],
            )

            data_files = frontmatter["configs"][0]["data_files"]
            self.assertIn({"split": "train", "path": "data/train/**/*.parquet"}, data_files)
            self.assertIn({"split": "test", "path": "data/test/**/*.parquet"}, data_files)

    # -------------------------------------------------------------
    # UNIT TEST: GENERATE FRONTMATTER DICT DEFAULT SPLIT PATH
    # -------------------------------------------------------------
    def test_generate_frontmatter_dict_uses_default_path_for_default_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset = DatasetDict({"default": self._make_dataset_dict()["train"]})
            dfs = {"default": self._make_dataframes()["train"].copy()}

            default_file = tmp_path / "default.parquet"
            default_file.write_bytes(b"x" * 5)

            builder = HuggingFaceReadmeBuilder(
                repo_id="user/default-dataset",
                dataset=dataset,
                dataframes=dfs,
                parquet_paths={"default": [str(default_file)]},
            )

            stats = builder._build_stats()
            frontmatter = builder._generate_frontmatter_dict(stats)

            self.assertEqual(
                frontmatter["configs"][0]["data_files"],
                [{"split": "default", "path": "data/**/*.parquet"}],
            )

    # -------------------------------------------------------------
    # UNIT TEST: RENDER FRONTMATTER IS VALID YAML
    # -------------------------------------------------------------
    def test_render_frontmatter_is_valid_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(Path(tmp_dir))
            stats = builder._build_stats()

            frontmatter_text = builder._render_frontmatter(stats)
            parsed = yaml.safe_load(frontmatter_text)

            self.assertEqual(parsed["dataset_info"]["download_size"], 30)
            self.assertEqual(parsed["license"], "mit")
            self.assertEqual(parsed["configs"][0]["config_name"], "default")

    # -------------------------------------------------------------
    # UNIT TEST: RENDER BODY CONTAINS EXPECTED SECTIONS
    # -------------------------------------------------------------
    def test_render_body_contains_expected_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(Path(tmp_dir))
            stats = builder._build_stats()
            body = builder._render_body(stats)

            self.assertIn("# Dataset Card for my-dataset", body)
            self.assertIn("## Dataset Summary", body)
            self.assertIn("This dataset contains 3 samples across 2 split(s).", body)
            self.assertIn("- **train**: 2 samples", body)
            self.assertIn("- **test**: 1 samples", body)
            self.assertIn("### Features", body)
            self.assertIn('dataset = load_dataset("user/my-dataset")', body)
            self.assertIn(
                'dataset_split = load_dataset("user/my-dataset", split="train")',
                body,
            )
            self.assertIn("proj_a, proj_b, proj_c", body)

    # -------------------------------------------------------------
    # UNIT TEST: RENDER RETURNS FULL README WITH FRONTMATTER AND BODY
    # -------------------------------------------------------------
    def test_render_returns_full_readme_with_frontmatter_and_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = self._make_builder(Path(tmp_dir))
            readme = builder.render()

            self.assertTrue(readme.startswith("---\n"))
            self.assertIn("\n---\n\n# Dataset Card for my-dataset", readme)
            self.assertIn("dataset_info:", readme)
            self.assertIn("configs:", readme)
            self.assertIn("license: mit", readme)
            self.assertIn("### Projects Included", readme)
            self.assertIn("proj_a, proj_b, proj_c", readme)


if __name__ == "__main__":
    unittest.main()