import shutil
import tempfile
import unittest
from pathlib import Path

from jaguars.common.io_utils import ensure_dir, validate_dir_exists, validate_file_exists
from jaguars.common.logging_utils import setup_logger


class TestCommonUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_file = self.test_dir / "test.txt"
        self.test_file.touch()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_logging_utils(self) -> None:
        log_file = self.test_dir / "test.log"
        logger = setup_logger("test_logger", log_file=log_file)
        logger.info("Test message")

        self.assertTrue(log_file.exists())
        with open(log_file) as f:
            content = f.read()
            self.assertIn("Test message", content)

        # Close all handlers to release the log file on Windows
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    def test_io_utils(self) -> None:
        # Test file validation
        self.assertEqual(validate_file_exists(self.test_file), self.test_file)
        with self.assertRaises(FileNotFoundError):
            validate_file_exists(self.test_dir / "nonexistent.txt")

        # Test directory validation
        self.assertEqual(validate_dir_exists(self.test_dir), self.test_dir)
        with self.assertRaises(FileNotFoundError):
            validate_dir_exists(self.test_dir / "nonexistent_dir")

        # Test ensure dir
        new_dir = self.test_dir / "new_dir"
        ensure_dir(new_dir)
        self.assertTrue(new_dir.exists())


if __name__ == "__main__":
    unittest.main()
