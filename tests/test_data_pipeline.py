from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from minova_ai.data_pipeline import (
    _split_groups,
    detect_language,
    normalize_record,
    prepare_dataset,
)
from minova_ai.pipeline import run_pipeline
from minova_ai.schema import Example, Provenance


def record(number: int, code: str = "print('ok')", language: str = "python") -> dict[str, object]:
    return {
        "id": str(number),
        "task_type": "code_generation",
        "language": language,
        "prompt": "Write a small program.",
        "completion": code,
        "source_name": f"pilot-{number}",
        "source_uri": f"synthetic://pilot/{number}",
        "license": "CC0-1.0",
    }


class DataPipelineTests(unittest.TestCase):
    def test_training_pipeline_prepares_data_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "processed"
            prepared_report = {"accepted": 2, "train": 1, "validation": 1}
            events: list[str] = []
            training_module = types.ModuleType("minova_ai.train")
            setattr(training_module, "main", lambda: events.append("train"))
            with (
                patch("minova_ai.pipeline.prepare_dataset", return_value=prepared_report) as prepare,
                patch.dict(sys.modules, {"minova_ai.train": training_module}),
                patch.dict(os.environ, {}, clear=True),
            ):
                prepare.side_effect = lambda *args: (events.append("prepare"), prepared_report)[1]
                report = run_pipeline(
                    Path(directory) / "raw",
                    output_dir,
                    0.2,
                    {"MIT"},
                    {"python"},
                    "Minova",
                    "internal://minova/dataset",
                    "Minova-Proprietary",
                )
                self.assertEqual(report, prepared_report)
                self.assertEqual(events, ["prepare", "train"])
                self.assertEqual(os.environ["TRAIN_FILE"], str(output_dir / "train.jsonl"))
                self.assertEqual(os.environ["VALIDATION_FILE"], str(output_dir / "validation.jsonl"))
            prepare.assert_called_once_with(
                Path(directory) / "raw",
                output_dir,
                0.2,
                {"MIT"},
                {"python"},
                "Minova",
                "internal://minova/dataset",
                "Minova-Proprietary",
            )

    def test_training_pipeline_does_not_train_if_data_preparation_fails(self) -> None:
        with patch("minova_ai.pipeline.prepare_dataset", side_effect=ValueError("invalid data")):
            with self.assertRaisesRegex(ValueError, "invalid data"):
                run_pipeline(Path("raw"), Path("processed"), 0.2, {"MIT"})

    def test_detects_supported_language_from_fence_and_extension(self) -> None:
        self.assertEqual(detect_language("```py\nprint('ok')\n```"), "python")
        self.assertEqual(detect_language("anything", "hello.rs"), "rust")

    def test_normalization_rejects_secret(self) -> None:
        value = record(1, "api_key = 'abcdefghijk123456'")
        with self.assertRaisesRegex(ValueError, "sensitive data"):
            normalize_record(value, "pilot.jsonl", 1)

    def test_normalization_checks_python_syntax(self) -> None:
        with self.assertRaisesRegex(ValueError, "syntax check failed"):
            normalize_record(record(1, "def broken(:\n  pass"), "pilot.py", 1)

    def test_pipeline_accepts_json_lines_and_python_task_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_file = root / "datasets.json"
            rows = [
                {
                    "instruction": "Write a Python function to add two numbers.",
                    "input": "",
                    "output": "Step 3 - Write the code\ndef add(a, b):\n    return a + b\n\nStep 4 - Test",
                    "language": "python",
                    "task": "generate",
                },
                {
                    "instruction": "Fix this Python function.",
                    "input": "",
                    "output": "Step 2 - Apply the fix\ndef safe_add(a, b):\n    return a + b\n\nStep 3 - Test",
                    "language": "python",
                    "task": "fix",
                },
                {
                    "instruction": "Write a JavaScript addition function.",
                    "input": "",
                    "output": "const add = (a, b) => a + b;",
                    "language": "javascript",
                    "task": "generate",
                },
            ]
            input_file.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = prepare_dataset(
                input_file,
                root / "processed",
                allowed_licenses={"Minova-Proprietary"},
                languages={"python"},
                source_name_default="Minova Technologies Enterprise",
                source_uri_default="internal://minova-technologies-enterprise/datasets.json",
                license_default="Minova-Proprietary",
            )

            self.assertEqual(report["accepted"], 2)
            self.assertEqual(report["filtered"], 1)
            self.assertEqual(report["train"], 1)
            self.assertEqual(report["validation"], 1)
            self.assertEqual(report["split_strategy"], "record_level_single_source")
            self.assertEqual(report["rejected"], 0)
            train_records = [
                json.loads(line)
                for line in (root / "processed" / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            validation_records = [
                json.loads(line)
                for line in (root / "processed" / "validation.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {item["task_type"] for item in train_records + validation_records},
                {"code_generation", "debugging"},
            )

    def test_repository_groups_do_not_leak_between_splits(self) -> None:
        examples = [
            Example(str(i), "code_generation", "python", "p", "print(1)", Provenance("s", f"uri:{i}", "MIT", "same"))
            for i in range(3)
        ]
        examples[-1] = Example(
            "2", "code_generation", "python", "p", "print(1)", Provenance("s", "uri:2", "MIT", "other")
        )
        train, validation = _split_groups(examples, 0.5)
        train_groups = {item.provenance.repository for item in train}
        validation_groups = {item.provenance.repository for item in validation}
        self.assertFalse(train_groups & validation_groups)

    def test_prepare_removes_duplicates_and_rejects_unapproved_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_file = root / "raw.jsonl"
            rows = [
                record(1),
                record(1),
                {**record(2, "print('different')"), "license": "NOASSERTION"},
                record(3, "print('different')"),
            ]
            input_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = prepare_dataset(input_file, root / "processed")
            self.assertEqual(result["duplicates_removed"], 1)
            self.assertEqual(result["rejected"], 1)
            self.assertEqual(result["accepted"], 2)
            self.assertTrue((root / "processed" / "validation.jsonl").is_file())

    def test_invalid_json_lines_are_rejected_without_storing_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_file = root / "raw.jsonl"
            first = record(1)
            second = record(2, "print('second')")
            input_file.write_text(
                json.dumps(first) + "\n{not json}\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            result = prepare_dataset(input_file, root / "processed")
            rejected = (root / "processed" / "rejected.jsonl").read_text(encoding="utf-8")
            self.assertEqual(result["rejected"], 1)
            self.assertNotIn("not json", rejected)
            self.assertIn("valid JSON object", rejected)

    def test_test_spec_values_must_be_strings(self) -> None:
        value = record(1)
        value["tests"] = {"stdin": 7, "expected_stdout": "49"}
        with self.assertRaisesRegex(ValueError, "must be strings"):
            normalize_record(value, "pilot.jsonl", 1)

    def test_csv_style_json_tests_are_normalized(self) -> None:
        value = record(1)
        value["tests"] = json.dumps({"stdin": "7", "expected_stdout": "49"})
        example = normalize_record(value, "pilot.csv", 1)
        self.assertEqual(example.tests, {"stdin": "7", "expected_stdout": "49"})


if __name__ == "__main__":
    unittest.main()
