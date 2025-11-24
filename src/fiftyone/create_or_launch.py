"""Create or launch FiftyOne dataset from CSV file.

Usage: 
    python src/fiftyone/create_or_launch.py --csv path/to/labels.csv \
        --dataset_path path/to/dataset/files --name my_dataset [--create]
"""

import fiftyone as fo
import argparse
import pandas as pd
from pathlib import Path

def create_persistent_dataset(df: pd.DataFrame, dataset_name: str, dataset_path: Path) -> fo.Dataset:
    try:
        dataset = fo.Dataset(dataset_name)
    except ValueError as e:
        raise RuntimeError(f"Dataset '{dataset_name}' already exists. Use --create to replace it.") from e
    
    dataset.persistent = True
    
    for _, row in df.iterrows():
        data = row.to_dict()
        filepath = dataset_path / row["FILE PATH"]

        dataset.add_sample(
            fo.Sample(
                filepath=filepath,
                label=row["JAGUAR ID"],
                **{k: v for k, v in data.items() if k != "FILE PATH"},
            )
        )

    dataset.compute_metadata()

    return dataset

def load_datasets_from_csv(csv_path: Path, dataset_path: Path, dataset_name: str) -> list[fo.Dataset]:
    df = pd.read_csv(csv_path)

    if "FILE TYPE" not in df.columns:
        return [create_persistent_dataset(df, dataset_name, dataset_path)]
    else:
        print("Splitting dataset by FILE TYPE because the dataset contains both VIDEO and IMAGE files...")
        
        df_video = df[df["FILE TYPE"] == "VIDEO"]
        df_image = df[df["FILE TYPE"] == "IMAGE"]

        if (len(df_video) < len(df_image)):
            df_main = df_video
            df_other = df_image
            other_dataset_name = dataset_name + "_videos"
        else:
            df_main = df_video
            df_other = df_image
            other_dataset_name = dataset_name + "_images"
            
        
        return [
            create_persistent_dataset(df_main, dataset_name, dataset_path),
            create_persistent_dataset(df_other, other_dataset_name, dataset_path),
        ]    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true",
                        help="Force recreation of dataset")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--dataset_path",  type=str, required=True)
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
        datasets = load_datasets_from_csv(args.csv, args.name)

    session = fo.launch_app(dataset=datasets[0])
    session.wait()

if __name__ == "__main__":
    main()