"""Pipeline to run all data preprocessing steps.

Usage:
    python -m src.jaguar_reidentification.pipelines.data_preprocessing --input_path=data/raw/17_11_2025 --intermediate_path=data/intermediate/test
"""

import argparse
import logging
from pathlib import Path

from src.jaguar_reidentification.data_preprocessing.clean_labels import run as clean_labels
from src.jaguar_reidentification.data_preprocessing.preprocess_videos import run as preprocess_videos


def run(input_path: Path, intermediate_path: Path, generate_reports: bool = True) -> None:
    cleaned_labels_csv = intermediate_path / "cleaned_labels.csv"

    # Step 1: Clean labels
    clean_labels(
        input_dir=input_path,
        output_csv=cleaned_labels_csv,
        generate_report=generate_reports,
    )

    # Step 2: Preprocess videos
    preprocessed_labels = intermediate_path / "preprocessed_labels.csv"
    preprocess_videos(
        input_csv=cleaned_labels_csv,
        output_csv=preprocessed_labels,
        generate_report=generate_reports,
    )


def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing Pipeline")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the raw input data.",
    )
    parser.add_argument(
        "--intermediate_path",
        type=str,
        required=True,
        help="Path to save intermediate processed data.",
    )
    parser.add_argument(
        "--no_reports",
        action="store_true",
        help="Disable generation of processing reports.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    input_path = Path(args.input_path)
    intermediate_path = Path(args.intermediate_path)
    run(input_path, intermediate_path, generate_reports=not args.no_reports)


if __name__ == "__main__":
    raise SystemExit(main())
