from pathlib import Path
from typing import Optional, Union

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def load_grounding_dino_model(
    name: str = "zero-shot-detection-transformer-torch",
    name_or_path: str = "IDEA-Research/grounding-dino-base",
    device: str = "cpu",
    text_prompt: str = "jaguar",
) -> fo.Model:
    return foz.load_zoo_model(
        name, 
        name_or_path=name_or_path, 
        device=device,
        classes=[text_prompt],  # Use classes instead of text_prompt
    )


def detect_jaguars_in_dataset(
    dataset: fo.Dataset,
    model: Optional[fo.Model] = None,
    text_prompt: str = "jaguar",
    confidence_threshold: float = 0.2,
    label_field: str = "detections",
    progress: bool = True,
) -> fo.Dataset:
    if model is None:
        model = load_grounding_dino_model(text_prompt=text_prompt)
    
    dataset.apply_model(
        model,
        label_field=label_field,
        confidence_thresh=confidence_threshold,
        progress=progress,
    )
    return dataset


def filter_high_confidence_detections(
    dataset: fo.Dataset,
    label_field: str = "detections",
    min_confidence: float = 0.5,
) -> fo.DatasetView:
    return dataset.filter_labels(label_field, F("confidence") >= min_confidence)


def get_samples_with_detections(
    dataset: fo.Dataset,
    label_field: str = "detections",
) -> fo.DatasetView:
    return dataset.match(F(f"{label_field}.detections").length() > 0)


def export_detections(
    dataset: Union[fo.Dataset, fo.DatasetView],
    export_dir: Union[str, Path],
    export_format: str = "FiftyOneDataset",
    export_media: bool = True,
) -> None:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    dataset.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.FiftyOneDataset,
        export_media=export_media,
    )


def create_dataset_from_directory(
    image_dir: Union[str, Path],
    dataset_name: str,
    recursive: bool = True,
    overwrite: bool = False,
) -> fo.Dataset:
    if overwrite and fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    
    if fo.dataset_exists(dataset_name):
        print(f"Loading existing dataset: {dataset_name}")
        return fo.load_dataset(dataset_name)
    
    dataset = fo.Dataset.from_dir(
        dataset_dir=str(image_dir),
        dataset_type=fo.types.ImageDirectory,
        name=dataset_name,
        recursive=recursive,
    )
    dataset.persistent = True
    return dataset


def run_detection_pipeline(
    image_dir: Union[str, Path],
    dataset_name: str,
    text_prompt: str = "jaguar",
    confidence_threshold: float = 0.2,
    min_confidence_filter: Optional[float] = None,
    export_dir: Optional[Union[str, Path]] = None,
    overwrite_dataset: bool = False,
) -> fo.Dataset:
    print(f"Creating dataset from: {image_dir}")
    dataset = create_dataset_from_directory(
        image_dir=image_dir,
        dataset_name=dataset_name,
        overwrite=overwrite_dataset,
    )
    print(f"Dataset has {len(dataset)} samples")
    
    print("Loading Grounding-DINO model...")
    model = load_grounding_dino_model()
    
    print(f"Running detection with prompt: '{text_prompt}'")
    dataset = detect_jaguars_in_dataset(
        dataset=dataset,
        model=model,
        text_prompt=text_prompt,
        confidence_threshold=confidence_threshold,
    )
    
    result = dataset
    if min_confidence_filter is not None:
        print(f"Filtering detections with confidence >= {min_confidence_filter}")
        result = filter_high_confidence_detections(
            dataset=dataset,
            min_confidence=min_confidence_filter,
        )
    
    if export_dir is not None:
        print(f"Exporting to: {export_dir}")
        export_detections(result, export_dir)
    
    return result


def detect_jaguars(
    image_dir: Union[str, Path],
    dataset_name: str = "jaguar_detections",
) -> fo.Dataset:
    return run_detection_pipeline(
        image_dir=image_dir,
        dataset_name=dataset_name,
    )


def select_best_detection_body(
    dataset: fo.Dataset,
    raw_bboxes_field_name: str = "raw_bboxes_body",
    output_field: str = "bboxes_body",
) -> None:
    for sample in dataset:
        if raw_bboxes_field_name not in sample or not sample[raw_bboxes_field_name]:
            continue
        raw_bboxes = sample[raw_bboxes_field_name]
        if raw_bboxes.detections:
            best_detection = max(
                raw_bboxes.detections,
                key=lambda d: d.bounding_box[2] * d.bounding_box[3],
            )
            sample[output_field] = fo.Detections(detections=[best_detection])
            sample.save()


def select_best_detection_head(
    dataset: fo.Dataset,
    raw_bboxes_field_name: str = "raw_bboxes_head",
    output_field: str = "bboxes_head",
) -> None:
    for sample in dataset:
        if raw_bboxes_field_name not in sample or not sample[raw_bboxes_field_name]:
            continue
        raw_bboxes = sample[raw_bboxes_field_name]
        if raw_bboxes.detections:
            best_detection = max(
                raw_bboxes.detections,
                key=lambda d: d.confidence,
            )
            sample[output_field] = fo.Detections(detections=[best_detection])
            sample.save()


def select_best_detection(
    dataset: fo.Dataset,
    detection_type: str = "body",
    raw_bboxes_field_name: str = None,
    cleanup: bool = True,
) -> None:
    if raw_bboxes_field_name is None:
        raw_bboxes_field_name = f"raw_bboxes_{detection_type}"
    
    if detection_type == "body":
        select_best_detection_body(dataset, raw_bboxes_field_name)
    else:
        select_best_detection_head(dataset, raw_bboxes_field_name)
    
    if cleanup and raw_bboxes_field_name in dataset.get_field_schema():
        dataset.delete_sample_field(raw_bboxes_field_name)
