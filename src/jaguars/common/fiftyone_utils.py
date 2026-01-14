from typing import Any

import fiftyone as fo
from pathlib import Path

from jaguars.common.logging_utils import setup_logger

logger = setup_logger("common.fiftyone_utils")


def get_or_create_dataset(name: str, persistent: bool = True, overwrite: bool = False) -> Any:
    """Gets an existing FiftyOne dataset or creates a new one.

    Args:
        name: Name of the dataset
        persistent: Whether the dataset should be persistent
        overwrite: If True, deletes existing dataset with same name

    Returns:
        The FiftyOne dataset
    """
    if overwrite and name in fo.list_datasets():
        logger.info("Deleting existing dataset: %s", name)
        fo.delete_dataset(name)

    if name in fo.list_datasets():
        logger.info("Loading existing dataset: %s", name)
        dataset = fo.load_dataset(name)
    else:
        logger.info("Creating new dataset: %s", name)
        dataset = fo.Dataset(name=name, persistent=persistent)

    return dataset


def ensure_dataset_exists(name: str) -> Any:
    """Ensures a dataset exists, raising an error if it doesn't."""
    if name not in fo.list_datasets():
        raise ValueError(f"Dataset '{name}' does not exist. Please run ingestion first.")
    return fo.load_dataset(name)


def export_dataset(dataset: fo.Dataset, export_dir: Path) -> None:
    """Exports the given FiftyOne dataset to its configured export directory."""
    logger.info("Exporting dataset '%s' to %s", dataset.name, export_dir)
    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.FiftyOneDataset,
        overwrite=True,
    )


def delete_dataset_if_exists(name: str) -> None:
    """Deletes a dataset if it exists."""
    if name in fo.list_datasets():
        fo.delete_dataset(name)
        logger.info("Deleted dataset: %s", name)
