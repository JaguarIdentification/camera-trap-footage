"""Create or launch FiftyOne dataset from CSV file.

Usage: 
    python src/jaguar_reidentification/fiftyone/create_or_launch.py --csv path/to/labels.csv \
        --dataset_path path/to/dataset/files --name my_dataset [--create]
"""

import fiftyone as fo
import argparse
import pandas as pd
from pathlib import Path
from typing import Any


def create_persistent_dataset(df: pd.DataFrame, dataset_name: str, dataset_path: Path) -> fo.Dataset:
    try:
        dataset = fo.Dataset(dataset_name)
    except ValueError as e:
        raise RuntimeError(f"Dataset '{dataset_name}' already exists. Use --create to replace it.") from e

    dataset.persistent = True

    for _, row in df.iterrows():
        data = row.to_dict()
        filepath = dataset_path / row["FILE PATH"]

        # Convert pandas NA values to None for FiftyOne compatibility
        cleaned_data: dict[str, Any] = {}
        for k, v in data.items():
            # Skip the FILE PATH field as it's already used for filepath
            if k == "FILE PATH":
                continue

            # Convert pandas NA/NaN to None or empty string for string fields
            cleaned_data[k] = v if pd.notna(v) else None

        # Create bounding box for cropped region if crop info exists
        detections = []
        if all(pd.notna(row.get(k)) for k in ["CROP LEFT", "CROP TOP", "CROP RIGHT", "CROP BOTTOM", "IMAGE WIDTH", "IMAGE HEIGHT"]):
            left = int(row["CROP LEFT"])
            top = int(row["CROP TOP"])
            right = int(row["CROP RIGHT"])
            bottom = int(row["CROP BOTTOM"])
            width = int(row["IMAGE WIDTH"])
            height = int(row["IMAGE HEIGHT"])
            
            # Only add bounding box if there's actual cropping
            if left > 0 or top > 0 or right < width or bottom < height:
                # Convert to normalized coordinates [0, 1]
                # FiftyOne uses [x, y, width, height] in relative coordinates
                rel_x = left / width
                rel_y = top / height
                rel_width = (right - left) / width
                rel_height = (bottom - top) / height
                
                detections.append(
                    fo.Detection(
                        label="cropped_region",
                        bounding_box=[rel_x, rel_y, rel_width, rel_height],
                    )
                )

        sample = fo.Sample(
            filepath=filepath,
            label=row["JAGUAR ID"] if pd.notna(row["JAGUAR ID"]) else None,
            **cleaned_data,
        )
        
        if detections:
            sample["crop_box"] = fo.Detections(detections=detections)
        
        dataset.add_sample(sample)

    dataset.compute_metadata()

    return dataset


def load_datasets_from_csv(csv_path: Path, dataset_path: Path, dataset_name: str) -> list[fo.Dataset]:
    df = pd.read_csv(csv_path)

    if "FILE TYPE" not in df.columns or df["FILE TYPE"].nunique() == 1:
        return [create_persistent_dataset(df, dataset_name, dataset_path)]
    else:
        print("Splitting dataset by FILE TYPE because the dataset contains both VIDEO and IMAGE files...")

        df_video = df[df["FILE TYPE"] == "VIDEO"]
        df_image = df[df["FILE TYPE"] == "IMAGE"]

        if len(df_video) > len(df_image):
            df_main = df_video
            df_other = df_image
            other_dataset_name = dataset_name + "_videos"
        else:
            df_main = df_image
            df_other = df_video
            other_dataset_name = dataset_name + "_images"

        return [
            create_persistent_dataset(df_main, dataset_name, dataset_path),
            create_persistent_dataset(df_other, other_dataset_name, dataset_path),
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="Force recreation of dataset")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--name", type=str, default="my_dataset")
    args = parser.parse_args()

    names = [args.name, args.name + "_images", args.name + "_videos"]

    # ------------------
    # CASE 1: force recreate (--create)
    # ------------------
    if args.create:
        print(f"Recreating dataset '{args.name}'...")
        for name in names:
            if fo.dataset_exists(name):
                fo.delete_dataset(name)
        datasets = load_datasets_from_csv(Path(args.csv), Path(args.dataset_path), args.name)
        print("Dataset(s) created.")

    # ------------------
    # CASE 2: dataset exists → load it
    # ------------------
    elif any(fo.dataset_exists(name) for name in names):
        print(f"Loading existing datasets '{args.name}'")

        datasets = [fo.load_dataset(name) for name in names if fo.dataset_exists(name)]

    # ------------------
    # CASE 3: dataset doesn't exist → create it
    # ------------------
    else:
        print(f"Dataset '{args.name}' not found. Creating it...")
        datasets = load_datasets_from_csv(Path(args.csv), Path(args.dataset_path), args.name)

    session = fo.launch_app(dataset=datasets[0])
    session.wait()


if __name__ == "__main__":
    main()
