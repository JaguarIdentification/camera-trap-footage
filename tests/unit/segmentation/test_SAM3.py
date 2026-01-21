"""Tests for SAM3 segmentation module."""
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fiftyone as fo

from jaguars.segmentation.SAM3 import run_processing


class TestSAM3(unittest.TestCase):
    """Test suite for SAM3 segmentation."""

    def setUp(self) -> None:
        """Setup mocks."""
        self.dataset_name = "test_sam3_dataset"
        
        # Mock fiftyone.list_datasets to include our test dataset
        self.list_datasets_patcher = patch("fiftyone.list_datasets")
        self.mock_list_datasets = self.list_datasets_patcher.start()
        self.mock_list_datasets.return_value = [self.dataset_name]
        
        # Mock get_or_create_dataset
        self.get_dataset_patcher = patch("jaguars.segmentation.SAM3.get_or_create_dataset")
        self.mock_get_dataset = self.get_dataset_patcher.start()
        
        # Create a mock dataset
        self.mock_dataset = MagicMock(spec=fo.Dataset)
        self.mock_dataset.name = self.dataset_name
        self.mock_dataset.__len__.return_value = 10
        self.mock_get_dataset.return_value = self.mock_dataset
        
        # Mock the view returned by select_group_slices
        self.mock_view = MagicMock()
        self.mock_view.__len__.return_value = 5 # 5 images
        self.mock_dataset.select_group_slices.return_value = self.mock_view
        
        # Mock apply_model on the view
        self.mock_view.apply_model = MagicMock()
        self.mock_view.save = MagicMock()
        
        # Mock filter_labels for summary
        self.mock_detections = MagicMock()
        self.mock_detections.__len__.return_value = 2
        self.mock_view.filter_labels.return_value = self.mock_detections
        
        # Mock foz
        self.foz_register_patcher = patch("fiftyone.zoo.register_zoo_model_source")
        self.mock_foz_register = self.foz_register_patcher.start()
        
        self.foz_load_patcher = patch("fiftyone.zoo.load_zoo_model")
        self.mock_foz_load = self.foz_load_patcher.start()
        self.mock_model = MagicMock()
        self.mock_foz_load.return_value = self.mock_model

        # Mock pytorch checks (used in device detection)
        self.torch_patcher = patch("jaguars.segmentation.SAM3.torch")  # We need to mock torch imported inside the function or module
        # Since torch is imported inside run_processing if device is None, we need to be careful.
        # However, checking the code, it does `import torch` inside the function.
        # Patching sys.modules is a way, but let's try assuming it picks a device or we pass explicit device.
        # Passing explicit device avoids torch import logic in test if we want.

    def tearDown(self) -> None:
        """Stop patchers."""
        self.list_datasets_patcher.stop()
        self.get_dataset_patcher.stop()
        self.foz_register_patcher.stop()
        self.foz_load_patcher.stop()

    def test_dry_run(self) -> None:
        """Test dry run mode."""
        result = run_processing(
            dataset_name=self.dataset_name,
            dry_run=True,
            verbose=True
        )
        self.assertIsNone(result)
        self.mock_foz_load.assert_not_called()

    def test_run_processing_grouped(self) -> None:
        """Test processing on a grouped dataset."""
        # Setup dataset as grouped
        self.mock_dataset.group_field = "group"
        
        run_processing(
            dataset_name=self.dataset_name,
            prompt="jaguar",
            threshold=0.6,
            device="cpu", # Avoid torch import check
            verbose=True
        )
        
        # Verify dataset load
        self.mock_get_dataset.assert_called_with(self.dataset_name)
        
        # Verify slice selection
        self.mock_dataset.select_group_slices.assert_called_with("image")
        
        # Verify model load
        self.mock_foz_register.assert_called()
        self.mock_foz_load.assert_called_with("facebook/sam3", device="cpu")
        
        # Verify model config
        self.assertEqual(self.mock_model.prompt, "jaguar")
        self.assertEqual(self.mock_model.threshold, 0.6)
        
        # Verify apply_model
        self.mock_view.apply_model.assert_called()
        self.mock_view.save.assert_called()

    def test_run_processing_flat(self) -> None:
        """Test processing on a flat dataset."""
        # Setup dataset as flat (no group_field)
        self.mock_dataset.group_field = None
        
        # Update view mock to match dataset (dataset becomes the view)
        self.mock_dataset.apply_model = MagicMock()
        self.mock_dataset.save = MagicMock()
        self.mock_dataset.filter_labels.return_value = self.mock_detections
        
        run_processing(
            dataset_name=self.dataset_name,
            device="cpu",
            verbose=True
        )
        
        # Verify NO slice selection
        self.mock_dataset.select_group_slices.assert_not_called()
        
        # Verify apply_model on dataset
        self.mock_dataset.apply_model.assert_called()

    def test_summary_generation(self) -> None:
        """Test that summary file is generated."""
        self.mock_dataset.group_field = "group"
        
        with patch("builtins.open", new_callable=MagicMock) as mock_open:
            with patch("json.dump") as mock_json_dump:
                run_processing(
                    dataset_name=self.dataset_name,
                    summary_location=Path("test_summary.json"),
                    device="cpu"
                )
                
                mock_open.assert_called_with(Path("test_summary.json"), "w")
                mock_json_dump.assert_called()
                
                # Check call args of json.dump
                args, _ = mock_json_dump.call_args
                data = args[0]
                self.assertEqual(data["dataset_name"], self.dataset_name)
                # We mocked 5 images, 2 detections
                self.assertEqual(data["total_samples"], 5)
                self.assertEqual(data["samples_with_detections"], 2)


if __name__ == "__main__":
    unittest.main()
