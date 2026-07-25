import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

# FiftyOne establishes its database connection at import time. Point both its
# config file and embedded MongoDB at a one-process temporary directory before
# any test module can import FiftyOne.
_MANAGED_ENV_KEYS = (
    "FIFTYONE_CONFIG_PATH",
    "FIFTYONE_DATABASE_DIR",
    "FIFTYONE_DATABASE_NAME",
    "FIFTYONE_DATASET_ZOO_DIR",
    "FIFTYONE_MODEL_ZOO_DIR",
    "FIFTYONE_PLUGINS_DIR",
    "FIFTYONE_PRIVATE_DATABASE_PORT",
    "MPLCONFIGDIR",
)
_ORIGINAL_ENV = {key: os.environ.get(key) for key in _MANAGED_ENV_KEYS}
_ISOLATED_ROOT = Path(tempfile.mkdtemp(prefix="jaguars-fiftyone-integration-"))
os.environ["FIFTYONE_CONFIG_PATH"] = str(_ISOLATED_ROOT / "config.json")
os.environ["FIFTYONE_DATABASE_DIR"] = str(_ISOLATED_ROOT / "database")
os.environ["FIFTYONE_DATABASE_NAME"] = "jaguars_task4_integration"
os.environ["FIFTYONE_DATASET_ZOO_DIR"] = str(_ISOLATED_ROOT / "dataset-zoo")
os.environ["FIFTYONE_MODEL_ZOO_DIR"] = str(_ISOLATED_ROOT / "model-zoo")
os.environ["FIFTYONE_PLUGINS_DIR"] = str(_ISOLATED_ROOT / "plugins")
os.environ["MPLCONFIGDIR"] = str(_ISOLATED_ROOT / "matplotlib")
os.environ.pop("FIFTYONE_PRIVATE_DATABASE_PORT", None)

import fiftyone as fo  # noqa: E402
import fiftyone.core.odm.database as food  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def isolated_fiftyone_database() -> Iterator[None]:
    assert Path(fo.config.database_dir).is_relative_to(_ISOLATED_ROOT)
    assert fo.config.database_name == "jaguars_task4_integration"
    assert fo.config.database_uri is None
    yield

    for dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    food._disconnect()
    database_service = food._db_service
    if database_service is not None:
        child = database_service.child
        database_service.stop()
        food._db_service = None
        if child is not None:
            child.wait(timeout=10)

    shutil.rmtree(_ISOLATED_ROOT)
    for key, original_value in _ORIGINAL_ENV.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    for key, original_value in _ORIGINAL_ENV.items():
        assert os.environ.get(key) == original_value
    assert not _ISOLATED_ROOT.exists()


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
