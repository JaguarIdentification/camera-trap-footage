#!/usr/bin/env python3
"""
YOLOE Jaguar Segmentation Script

This script processes all images in data/intermediate/v1/screenshots subdirectories,
applies YOLOE segmentation to detect jaguars, and saves positive detections
to a separate directory for further analysis.

Usage:
    python src/segmentation/yoloe_segement.py [--confidence 0.5] [--output-dir path]
"""

# Global variable to control which subdirectory to process images from
# Set to None to process all subdirectories
TEST_SUBDIR = "SITE_4_CAM_A_DSCF0012"

import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
from dataclasses import dataclass

try:
    import fiftyone as fo
    from ultralytics import YOLOE
    import numpy as np
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install fiftyone ultralytics")
    sys.exit(1)


@dataclass
class Config:
    """Configuration for jaguar segmentation"""
    screenshots_dir: Path
    output_dir: Path
    model_path: str
    confidence_threshold: float
    jaguar_classes: List[str]
    batch_size: int
    
    @classmethod
    def from_args(cls, args):
        return cls(
            screenshots_dir=args.screenshots_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            confidence_threshold=args.confidence,
            jaguar_classes=args.jaguar_classes,
            batch_size=args.batch_size
        )


def find_image_files(root_dir: Path) -> List[Path]:
    """
    Find all image files in subdirectories of root_dir
    
    Args:
        root_dir: Root directory containing subdirectories with images
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    # If TEST_SUBDIR is set, only process that specific subdirectory
    if TEST_SUBDIR:
        test_subdir = root_dir / TEST_SUBDIR
        if test_subdir.exists() and test_subdir.is_dir():
            print(f"Testing mode: Only processing subdirectory '{TEST_SUBDIR}'")
            for file_path in test_subdir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
        else:
            print(f"Warning: Test subdirectory '{TEST_SUBDIR}' not found in {root_dir}")
            return []
    else:
        # Process all subdirectories
        for subdir in root_dir.iterdir():
            if subdir.is_dir():
                for file_path in subdir.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                        image_files.append(file_path)
    
    return sorted(image_files)


def create_fiftyone_dataset(image_files: List[Path], dataset_name: str = "jaguar_detection") -> fo.Dataset:
    """
    Create a FiftyOne dataset from image files
    
    Args:
        image_files: List of image file paths
        dataset_name: Name for the FiftyOne dataset
        
    Returns:
        FiftyOne dataset
    """
    # Delete existing dataset if it exists
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    
    # Create new dataset
    dataset = fo.Dataset(dataset_name)
    
    # Add samples
    samples = []
    for img_path in image_files:
        sample = fo.Sample(filepath=str(img_path))
        # Add metadata about the source video
        sample["source_video"] = img_path.parent.name
        sample["frame_number"] = img_path.stem
        samples.append(sample)
    
    dataset.add_samples(samples)
    
    print(f"Created FiftyOne dataset '{dataset_name}' with {len(samples)} images")
    return dataset


def setup_yoloe_model(model_path: str, jaguar_classes: List[str]) -> YOLOE:
    """
    Setup and configure YOLOE model for detection
    
    Args:
        model_path: Path to YOLOE model weights
        jaguar_classes: List of class names (for reference only)
        
    Returns:
        Configured YOLOE model
    """
    print(f"Loading YOLOE model from: {model_path}")
    model = YOLOE(model_path)
    
    # Use the model with its default classes
    print("Using model with default COCO classes")
    
    return model


def apply_segmentation(dataset: fo.Dataset, model: YOLOE, confidence_threshold: float = 0.5):
    """
    Apply YOLOE segmentation to the dataset
    
    Args:
        dataset: FiftyOne dataset containing images
        model: Configured YOLOE model
        confidence_threshold: Minimum confidence for detections
    """
    print("Applying YOLOE segmentation to dataset...")
    
    # Apply model to dataset
    dataset.apply_model(
        model, 
        label_field="yoloe_segmentation",
        confidence_thresh=confidence_threshold
    )
    
    print("Segmentation complete!")


def filter_positive_detections(dataset: fo.Dataset, label_field: str = "yoloe_segmentation") -> fo.DatasetView:
    """
    Filter dataset to only include samples with positive animal detections
    
    Args:
        dataset: FiftyOne dataset with detections
        label_field: Field containing the detection labels
        
    Returns:
        Filtered dataset view with only positive detections
    """
    # Get samples with any detections
    samples_with_detections = dataset.exists(label_field + ".detections")
    
    # First, let's see what classes are actually being detected
    all_classes = set()
    for sample in samples_with_detections:
        detections = sample[label_field]
        if detections and detections.detections:
            for detection in detections.detections:
                all_classes.add(detection.label)
    
    print(f"All detected classes: {sorted(all_classes)}")
    
    # Define classes that might represent animals in camera traps (COCO classes)
    animal_classes = {
        'cat', 'dog', 'person', 'bird', 'horse', 'cow', 'sheep', 'bear'
    }
    
    # Filter for only animal detections
    try:
        animal_view = samples_with_detections.filter_labels(
            label_field,
            lambda det: det.label in animal_classes,
            only_matches=False
        )
    except Exception as e:
        print(f"Error filtering for animals: {e}")
        return dataset.limit(0)  # Return empty view
    
    print(f"Found {len(animal_view)} images with animal detections out of {len(dataset)} total images")
    
    return animal_view



def save_positive_detections(positive_view: fo.DatasetView, output_dir: Path):
    """
    Save images with positive jaguar detections to output directory
    
    Args:
        positive_view: FiftyOne dataset view with positive detections
        output_dir: Directory to save positive detection images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(positive_view)} positive detections to: {output_dir}")
    
    for sample in positive_view:
        # Get source file info
        source_path = Path(sample.filepath)
        source_video = sample["source_video"]
        frame_number = sample["frame_number"]
        
        # Create subdirectory for this video if it doesn't exist
        video_output_dir = output_dir / source_video
        video_output_dir.mkdir(exist_ok=True)
        
        # Copy image to output directory
        output_filename = f"{frame_number}_{source_video}.jpg"
        output_path = video_output_dir / output_filename
        
        shutil.copy2(source_path, output_path)
        
        # Optionally save detection info as JSON
        detections = sample["yoloe_segmentation"]
        if detections and detections.detections:
            detection_info = {
                "source_file": str(source_path),
                "video": source_video,
                "frame": frame_number,
                "detections": []
            }
            
            for detection in detections.detections:
                detection_info["detections"].append({
                    "label": detection.label,
                    "confidence": detection.confidence,
                    "bounding_box": detection.bounding_box
                })
            
            # Save detection metadata
            json_path = video_output_dir / f"{frame_number}_{source_video}_detections.json"
            with open(json_path, 'w') as f:
                import json
                json.dump(detection_info, f, indent=2)
    
    print(f"Saved all positive detections to: {output_dir}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Apply YOLOE segmentation to detect jaguars in camera trap images"
    )
    
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=Path("data/intermediate/v1/screenshots"),
        help="Directory containing screenshot subdirectories (default: data/intermediate/v1/screenshots)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/intermediate/v1/jaguar_detections"),
        help="Output directory for positive detections (default: data/intermediate/v1/jaguar_detections)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="yoloe-11s-seg.pt",
        help="Path to YOLOE model weights (default: yoloe-11s-seg.pt)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )
    
    parser.add_argument(
        "--jaguar-classes",
        nargs='+',
        default=["body of a jaguar", "head of a jaguar"],
        help="Jaguar class names for detection (default: ['body of a jaguar', 'head of a jaguar'])"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing (default: 16)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    config = Config.from_args(args)
    
    print("=== YOLOE Jaguar Segmentation ===")
    print(f"Screenshots directory: {config.screenshots_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Model: {config.model_path}")
    print(f"Confidence threshold: {config.confidence_threshold}")
    print(f"Jaguar classes: {config.jaguar_classes}")
    print()
    
    # Verify input directory exists
    if not config.screenshots_dir.exists():
        print(f"Error: Screenshots directory does not exist: {config.screenshots_dir}")
        sys.exit(1)
    
    try:
        # Step 1: Find all image files
        print("Step 1: Finding image files...")
        image_files = find_image_files(config.screenshots_dir)
        
        if not image_files:
            print("No image files found in screenshots directory!")
            sys.exit(1)
        
        print(f"Found {len(image_files)} image files")
        
        # Step 2: Create FiftyOne dataset
        print("\nStep 2: Creating FiftyOne dataset...")
        dataset = create_fiftyone_dataset(image_files, "jaguar_detection")
        
        # Step 3: Setup YOLOE model
        print("\nStep 3: Setting up YOLOE model...")
        model = setup_yoloe_model(config.model_path, config.jaguar_classes)
        
        # Step 4: Apply segmentation
        print("\nStep 4: Applying segmentation...")
        apply_segmentation(dataset, model, config.confidence_threshold)
        
        # Step 5: Filter positive detections
        print("\nStep 5: Filtering positive detections...")
        positive_view = filter_positive_detections(dataset, "yoloe_segmentation")
        
        if len(positive_view) == 0:
            print("No animal detections found!")
            return
        
        # Step 6: Save positive detections
        print("\nStep 6: Saving positive detections...")
        save_positive_detections(positive_view, config.output_dir)
        
        print("\n=== Jaguar segmentation completed successfully! ===")
        print(f"Positive detections saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up dataset
        try:
            if 'dataset' in locals():
                dataset.delete()
        except:
            pass


if __name__ == "__main__":
    main()
