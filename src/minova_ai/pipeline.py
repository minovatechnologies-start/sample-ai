from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from minova_ai.data_pipeline import DEFAULT_LICENSES, prepare_dataset


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    validation_ratio: float,
    allowed_licenses: set[str],
    languages: set[str] | None = None,
    source_name: str = "",
    source_uri: str = "",
    license_name: str = "",
) -> dict[str, object]:
    report = prepare_dataset(
        input_path,
        output_dir,
        validation_ratio,
        allowed_licenses,
        languages,
        source_name,
        source_uri,
        license_name,
    )
    print(json.dumps(report, indent=2))

    os.environ["TRAIN_FILE"] = str(output_dir / "train.jsonl")
    os.environ["VALIDATION_FILE"] = str(output_dir / "validation.jsonl")

    from minova_ai.train import main as train

    train()
    return report


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Prepare approved local data and fine-tune the configured model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
        help="Local JSONL, JSON, or CSV file/directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")),
    )
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument(
        "--languages",
        default=os.getenv("TRAIN_LANGUAGES", "python"),
        help="Comma-separated languages to include (defaults to Python only)",
    )
    parser.add_argument(
        "--source-name",
        default=os.getenv("DATASET_SOURCE_NAME", ""),
        help="Provenance name for records that do not declare one",
    )
    parser.add_argument(
        "--source-uri",
        default=os.getenv("DATASET_SOURCE_URI", ""),
        help="Internal/source URI for records that do not declare one",
    )
    parser.add_argument(
        "--license",
        dest="license_name",
        default=os.getenv("DATASET_LICENSE", ""),
        help="Dataset license/proprietary terms for records that do not declare one",
    )
    parser.add_argument(
        "--allowed-licenses",
        default=",".join(sorted(DEFAULT_LICENSES)),
        help="Comma-separated SPDX license IDs approved for your use case",
    )
    args = parser.parse_args()
    allowed_licenses = {item.strip() for item in args.allowed_licenses.split(",") if item.strip()}
    if args.license_name:
        allowed_licenses.add(args.license_name)
    languages = {item.strip() for item in args.languages.split(",") if item.strip()} or None
    run_pipeline(
        args.input,
        args.output_dir,
        args.validation_ratio,
        allowed_licenses,
        languages,
        args.source_name,
        args.source_uri,
        args.license_name,
    )


if __name__ == "__main__":
    main()
