"""Check the current state of the JID FiftyOne dataset.

This script inspects the dataset to determine which ingestion steps
have been completed and what still needs to be done.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import fiftyone as fo
from jaguars.common.config import JID_MASTER_DATASET

def check_dataset_state(dataset_name: str = JID_MASTER_DATASET) -> dict:
    """Check which steps have been completed in the dataset."""
    
    if not fo.dataset_exists(dataset_name):
        print(f"❌ Dataset '{dataset_name}' does not exist")
        return {"exists": False}
    
    print(f"✓ Dataset '{dataset_name}' exists\n")
    
    dataset = fo.load_dataset(dataset_name)
    state = {"exists": True}
    
    # Basic stats
    print(f"{'='*70}")
    print("DATASET OVERVIEW")
    print(f"{'='*70}")
    print(f"Total samples: {len(dataset)}")
    print(f"Group field: {dataset.group_field}")
    
    if dataset.group_field:
        images_view = dataset.select_group_slices("image")
        videos_view = dataset.select_group_slices("video")
        print(f"Image samples: {len(images_view)}")
        print(f"Video samples: {len(videos_view)}")
        state["images_count"] = len(images_view)
        state["videos_count"] = len(videos_view)
    else:
        images_view = dataset
        state["images_count"] = len(dataset)
    
    # Check fields
    fields = dataset.get_field_schema()
    print(f"\n{'='*70}")
    print("FIELDS")
    print(f"{'='*70}")
    for field_name, field_type in fields.items():
        print(f"  {field_name}: {field_type}")
    
    # Check tags
    tags = dataset.distinct("tags")
    print(f"\n{'='*70}")
    print("TAGS")
    print(f"{'='*70}")
    if tags:
        for tag in sorted(tags):
            count = len(dataset.match_tags(tag))
            print(f"  {tag}: {count} samples")
    else:
        print("  No tags found")
    
    # Check ingestion steps completion
    print(f"\n{'='*70}")
    print("INGESTION STEPS STATUS")
    print(f"{'='*70}")
    
    # Step 1: CSV/PPTX Loading
    has_jaguar_id = "jaguar_id" in fields or "ground_truth" in fields
    has_pptx = "pptx_detections" in fields
    print(f"✓ CSV Labels Loaded: {'YES' if has_jaguar_id else 'NO'}")
    if has_jaguar_id:
        id_field = "jaguar_id" if "jaguar_id" in fields else "ground_truth"
        num_with_ids = len(dataset.exists(id_field))
        print(f"  Samples with IDs: {num_with_ids}")
    print(f"✓ PPTX Loaded: {'YES' if has_pptx else 'NO'}")
    if has_pptx:
        num_with_pptx = len(dataset.exists("pptx_detections"))
        print(f"  Samples with PPTX detections: {num_with_pptx}")
    state["csv_loaded"] = has_jaguar_id
    state["pptx_loaded"] = has_pptx
    
    # Step 2: Video Sampling
    if dataset.group_field:
        sampled_images = sum(1 for s in images_view if s.source_type == "video_frame")
        print(f"✓ Video Frames Sampled: {sampled_images} samples")
        state["frames_sampled"] = sampled_images
    
    # Step 3: Segmentation
    segmentation_fields = [f for f in fields if "segmentation" in f.lower() or f == "detections"]
    has_segmentation = len(segmentation_fields) > 0
    print(f"✓ Segmentation: {'YES' if has_segmentation else 'NO'}")
    if has_segmentation:
        print(f"  Fields: {segmentation_fields}")
        for field in segmentation_fields:
            count = len(images_view.exists(field))
            print(f"  Samples with {field}: {count}")
            state[f"segmentation_{field}"] = count
    
    # Step 4: Embeddings
    # Check for sample-level embeddings (full images)
    sample_embedding_fields = [f for f in fields if f.endswith("embeddings") or f == "embedding" or f.startswith("embeddings_")]
    has_sample_embeddings = len(sample_embedding_fields) > 0
    
    # Check for detection-level embeddings (segmented patches)
    detection_embedding_info = {}
    for seg_field in segmentation_fields:
        try:
            # Get a sample with this segmentation field
            sample_with_seg = images_view.exists(seg_field).first()
            if sample_with_seg and sample_with_seg[seg_field]:
                detections = sample_with_seg[seg_field].detections
                if detections and len(detections) > 0:
                    # Check first detection for embedding fields
                    det = detections[0]
                    det_emb_fields = [k for k in det.field_names if "embedding" in k.lower()]
                    if det_emb_fields:
                        # Count samples where detections have embeddings
                        count = 0
                        for sample in images_view.exists(seg_field):
                            if sample[seg_field] and sample[seg_field].detections:
                                for d in sample[seg_field].detections:
                                    if any(d.has_field(ef) and d[ef] is not None for ef in det_emb_fields):
                                        count += 1
                                        break
                        detection_embedding_info[seg_field] = {
                            "fields": det_emb_fields,
                            "samples_with_embeddings": count
                        }
        except Exception:
            pass
    
    has_detection_embeddings = len(detection_embedding_info) > 0
    has_embeddings = has_sample_embeddings or has_detection_embeddings
    
    print(f"✓ Embeddings Computed: {'YES' if has_embeddings else 'NO'}")
    
    if has_sample_embeddings:
        print(f"  Sample-level embeddings (full images):")
        for emb_field in sample_embedding_fields:
            count = len(images_view.exists(emb_field))
            print(f"    {emb_field}: {count} samples")
            state[f"embeddings_{emb_field}"] = count
    
    if has_detection_embeddings:
        print(f"  Detection-level embeddings (segmented patches):")
        for seg_field, info in detection_embedding_info.items():
            print(f"    {seg_field} detections with {info['fields']}: {info['samples_with_embeddings']} samples")
            state[f"detection_embeddings_{seg_field}"] = info['samples_with_embeddings']
    
    # Step 5: Splits
    split_fields = [f for f in fields if "split" in f.lower()]
    has_splits = len(split_fields) > 0
    print(f"✓ Train/Val/Test Splits: {'YES' if has_splits else 'NO'}")
    if has_splits:
        print(f"  Fields: {split_fields}")
        for field in split_fields:
            splits = dataset.distinct(field)
            print(f"  {field}: {splits}")
            for split_val in splits:
                if split_val:
                    count = len(dataset.match(fo.ViewField(field) == split_val))
                    print(f"    {split_val}: {count} samples")
        state["splits"] = split_fields
    
    # Step 6: Deduplication
    has_dedup = "is_duplicate" in fields
    print(f"✓ Deduplication: {'YES' if has_dedup else 'NO'}")
    if has_dedup:
        duplicates = len(dataset.match(fo.ViewField("is_duplicate") == True))
        print(f"  Duplicates flagged: {duplicates}")
        state["duplicates_flagged"] = duplicates
    
    # Brain results
    brain_keys = dataset.list_brain_runs()
    if brain_keys:
        print(f"\n{'='*70}")
        print("BRAIN RESULTS")
        print(f"{'='*70}")
        for key in brain_keys:
            print(f"  {key}")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    recommendations = []
    if not has_jaguar_id:
        recommendations.append("⚠ Run Step 1a: CSV Labels Ingestion")
    if not has_pptx:
        recommendations.append("⚠ Run Step 1b: PPTX Ingestion (optional)")
    if dataset.group_field and state.get("frames_sampled", 0) == 0:
        recommendations.append("⚠ Run Step 2: Video Frame Sampling")
    if not has_segmentation:
        recommendations.append("⚠ Run Step 3: Segmentation")
    if not has_embeddings:
        recommendations.append("⚠ Run Step 4: Compute Embeddings")
    if not has_splits:
        recommendations.append("⚠ Run Step 5: Create Splits")
    if not has_dedup:
        recommendations.append("⚠ Run Step 6: Deduplication")
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("✅ All ingestion steps appear complete!")
    
    return state


if __name__ == "__main__":
    check_dataset_state()
