import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

# FiftyOne establishes its database connection at import time. Point both its
# config file and embedded MongoDB at a one-process temporary directory before
# any test module can import FiftyOne.
_ISOLATED_ROOT = Path(tempfile.mkdtemp(prefix="jaguars-fiftyone-integration-"))
os.environ["FIFTYONE_CONFIG_PATH"] = str(_ISOLATED_ROOT / "config.json")
os.environ["FIFTYONE_DATABASE_DIR"] = str(_ISOLATED_ROOT / "database")
os.environ["FIFTYONE_DATABASE_NAME"] = "jaguars_task4_integration"
os.environ["FIFTYONE_DATASET_ZOO_DIR"] = str(_ISOLATED_ROOT / "dataset-zoo")
os.environ["FIFTYONE_MODEL_ZOO_DIR"] = str(_ISOLATED_ROOT / "model-zoo")
os.environ["FIFTYONE_PLUGINS_DIR"] = str(_ISOLATED_ROOT / "plugins")
os.environ["MPLCONFIGDIR"] = str(_ISOLATED_ROOT / "matplotlib")

import fiftyone as fo  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_fiftyone_database() -> None:
    assert Path(fo.config.database_dir).is_relative_to(_ISOLATED_ROOT)
    assert fo.config.database_name == "jaguars_task4_integration"
    assert fo.config.database_uri is None


@pytest.fixture
def dataset_names() -> Iterator[tuple[str, str]]:
    # The database is isolated, but unique names make accidental cross-test
    # interaction impossible and make collision tests explicit.
    suffix = uuid4().hex
    final_name = f"task4-final-{suffix}"
    temporary_name = f"task4-temporary-{suffix}"
    yield final_name, temporary_name

    for name in (temporary_name, final_name):
        if fo.dataset_exists(name):
            fo.delete_dataset(name)
