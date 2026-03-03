"""Evaluation module for jaguar re-identification.

This module follows the architecture pattern:
- validate_resources()
- write_summary()
- run_processing()
- main()

Provides metrics:
- Mean Average Precision (mAP)
- Cumulative Matching Characteristics (CMC)
- Cosine similarity analysis
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from jaguars.common.logging_utils import setup_logger
from jaguars.reidentification.backbone import get_backbone
from jaguars.reidentification.config import ReidentificationConfig, get_default_config, load_config_from_dict
from jaguars.reidentification.data_loader import load_dataset
from jaguars.reidentification.model import ArcFaceModel, build_model

MODULE_NAME = "reidentification.evaluation"
logger = setup_logger(MODULE_NAME)


def compute_comprehensive_metrics(
    model: ArcFaceModel | None,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    train_labels: np.ndarray | None,
    all_labels: np.ndarray | None,
    label_encoder: LabelEncoder,
    device: str | None = None,
    max_cmc_rank: int = 50,
    min_total_samples_for_filtered: int | None = 9,
    use_embeddings_directly: bool = False,
) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics efficiently.

    Computes similarity matrix once, then derives:
    - Sample-level mAP (micro-average)
    - Identity-balanced mAP (macro-average across identities)
    - CMC@k metrics (k=1,5,10,20)
    - Filtered mAP/CMC@k for identities with at least N total samples across train/val/test

    Args:
        model: Trained model (can be None if use_embeddings_directly=True)
        val_embeddings: Validation embeddings (num_samples, embedding_dim)
        val_labels: Validation labels (num_samples,)
        train_labels: Training labels, or None
        all_labels: Labels across all splits (train/val/test), or None
        label_encoder: Label encoder
        device: Device to run on
        max_cmc_rank: Maximum rank for CMC curve
        min_total_samples_for_filtered: Minimum total identity support across all splits
        use_embeddings_directly: If True, use val_embeddings directly without passing through model

    Returns:
        Dictionary with all metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_embeddings_directly:
        # Use embeddings directly (for baselines with pre-computed embeddings)
        finetuned_emb = val_embeddings
    else:
        # Pass through model (for trained models)
        if model is None:
            raise ValueError("model must be provided when use_embeddings_directly=False")
        model.eval()
        with torch.no_grad():
            # Get fine-tuned embeddings
            val_tensor = torch.FloatTensor(val_embeddings).to(device)
            finetuned_emb = model.get_embeddings(val_tensor).cpu().numpy()

    # Compute cosine similarity matrix ONCE
    sim_matrix = cosine_similarity(finetuned_emb)
    np.fill_diagonal(sim_matrix, -1)  # Exclude self-similarity

    def _compute_map_and_cmc(
        labels: np.ndarray,
        matrix: np.ndarray,
        eval_indices: np.ndarray,
    ) -> tuple[float, float, dict[str, float], int, int]:
        """Compute sample mAP, identity-balanced mAP, and CMC@k for selected indices."""
        if len(eval_indices) == 0:
            zero_cmc = {"cmc@1": 0.0, "cmc@5": 0.0, "cmc@10": 0.0, "cmc@20": 0.0}
            return 0.0, 0.0, zero_cmc, 0, 0

        query_aps: dict[int, tuple[int, float]] = {}
        cmc_hits = np.zeros(max_cmc_rank)
        num_queries = 0

        for query_idx in eval_indices:
            query_label = labels[query_idx]

            gallery_indices = eval_indices[eval_indices != query_idx]
            if len(gallery_indices) == 0:
                continue

            similarities = matrix[query_idx, gallery_indices]
            matches = (labels[gallery_indices] == query_label).astype(int)

            sorted_indices = np.argsort(-similarities)
            sorted_matches = matches[sorted_indices]

            n_positives = sorted_matches.sum()
            if n_positives == 0:
                continue

            cumsum = np.cumsum(sorted_matches)
            precision_at_k = cumsum / np.arange(1, len(sorted_matches) + 1)
            ap = float(np.sum(precision_at_k * sorted_matches) / n_positives)
            query_aps[int(query_idx)] = (int(query_label), ap)

            first_match_idx = np.where(sorted_matches)[0]
            if len(first_match_idx) > 0:
                first_match_rank = int(first_match_idx[0])
                if first_match_rank < max_cmc_rank:
                    cmc_hits[first_match_rank:] += 1
            num_queries += 1

        if num_queries == 0 or not query_aps:
            zero_cmc = {"cmc@1": 0.0, "cmc@5": 0.0, "cmc@10": 0.0, "cmc@20": 0.0}
            return 0.0, 0.0, zero_cmc, 0, 0

        sample_map = float(np.mean([ap for _, ap in query_aps.values()]))

        identity_aps: dict[int, list[float]] = {}
        for _query_idx, (label, ap) in query_aps.items():
            if label not in identity_aps:
                identity_aps[label] = []
            identity_aps[label].append(ap)

        identity_balanced_map = float(np.mean([np.mean(aps) for aps in identity_aps.values()]))
        cmc_curve = cmc_hits / num_queries
        cmc_at_k = {
            "cmc@1": float(cmc_curve[0]) if len(cmc_curve) > 0 else 0.0,
            "cmc@5": float(cmc_curve[4]) if len(cmc_curve) > 4 else 0.0,
            "cmc@10": float(cmc_curve[9]) if len(cmc_curve) > 9 else 0.0,
            "cmc@20": float(cmc_curve[19]) if len(cmc_curve) > 19 else 0.0,
        }
        return sample_map, identity_balanced_map, cmc_at_k, num_queries, len(identity_aps)

    eval_all = np.arange(len(val_labels))
    sample_map, identity_balanced_map, cmc_at_k, num_queries, num_identities = _compute_map_and_cmc(
        labels=val_labels,
        matrix=sim_matrix,
        eval_indices=eval_all,
    )

    filtered_metrics: dict[str, Any] = {}
    if min_total_samples_for_filtered is not None:
        total_labels = all_labels
        if total_labels is None:
            if train_labels is not None:
                total_labels = np.concatenate([train_labels, val_labels])
            else:
                total_labels = val_labels

        unique_labels, label_counts = np.unique(total_labels, return_counts=True)
        eligible_labels = set(unique_labels[label_counts >= min_total_samples_for_filtered].tolist())

        eval_filtered = np.array([idx for idx, lab in enumerate(val_labels) if lab in eligible_labels], dtype=int)
        filtered_map, _filtered_ib_map_unused, filtered_cmc, filtered_queries, filtered_identities = _compute_map_and_cmc(
            labels=val_labels,
            matrix=sim_matrix,
            eval_indices=eval_filtered,
        )

        suffix = f"min_total_{min_total_samples_for_filtered}"
        filtered_metrics = {
            f"map_{suffix}": filtered_map,
            f"cmc_{suffix}@1": filtered_cmc["cmc@1"],
            f"cmc_{suffix}@5": filtered_cmc["cmc@5"],
            f"cmc_{suffix}@10": filtered_cmc["cmc@10"],
            f"cmc_{suffix}@20": filtered_cmc["cmc@20"],
            f"num_queries_{suffix}": filtered_queries,
            f"num_identities_{suffix}": filtered_identities,
            "min_total_samples_threshold": min_total_samples_for_filtered,
        }

    # Prepare results
    results = {
        "map": sample_map,
        "identity_balanced_map": identity_balanced_map,
        **cmc_at_k,
        **filtered_metrics,
        "num_queries": num_queries,
        "num_identities": num_identities,
    }

    return results


def compute_validation_map(
    model: ArcFaceModel, val_embeddings: np.ndarray, val_labels: np.ndarray, label_encoder: LabelEncoder, device: str | None = None
) -> float:
    """Compute identity-balanced mean Average Precision on validation set.

    DEPRECATED: Use compute_comprehensive_metrics() for more detailed metrics.
    Kept for backward compatibility.

    Args:
        model: Trained model
        val_embeddings: Validation embeddings (num_samples, embedding_dim)
        val_labels: Validation labels (num_samples,)
        label_encoder: Label encoder
        device: Device to run on

    Returns:
        Identity-balanced mean average precision
    """
    metrics = compute_comprehensive_metrics(
        model=model,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        train_labels=None,
        all_labels=val_labels,
        label_encoder=label_encoder,
        device=device,
        max_cmc_rank=50,
    )
    return metrics["identity_balanced_map"]


def compute_cmc(embeddings: np.ndarray, labels: np.ndarray, top_k: list[int] | None = None) -> dict[int, float]:
    """Compute Cumulative Matching Characteristics.

    Args:
        embeddings: Embeddings (num_samples, embedding_dim)
        labels: Labels (num_samples,)
        top_k: List of k values to compute

    Returns:
        Dictionary mapping k -> accuracy at rank k
    """
    if top_k is None:
        top_k = [1, 5, 10, 20]

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -np.inf)  # Exclude self with very negative value

    # For each query, get ranking
    cmc_scores = {k: 0.0 for k in top_k}
    num_queries = len(labels)

    for query_idx in range(num_queries):
        query_label = labels[query_idx]

        # Get similarities
        similarities = sim_matrix[query_idx]

        # Sort by similarity
        sorted_indices = np.argsort(-similarities)
        sorted_labels = labels[sorted_indices]

        # Find first correct match
        matches = sorted_labels == query_label
        if not matches.any():
            continue

        first_match_idx = np.where(matches)[0][0]

        # Update CMC scores
        for k in top_k:
            if first_match_idx < k:
                cmc_scores[k] += 1

    # Normalize
    cmc_scores = {k: v / num_queries for k, v in cmc_scores.items()}

    return cmc_scores


def validate_resources(config: ReidentificationConfig, model_path: Path | None = None) -> None:
    """Validate that all required resources exist.

    Args:
        config: Configuration
        model_path: Path to model checkpoint

    Raises:
        ValueError: If resources are missing
    """
    # Check dataset
    if config.dataset.source == "disk" and config.dataset.data_dir is None:
        raise ValueError("data_dir must be specified for disk-based datasets")
    elif config.dataset.source == "huggingface" and config.dataset.hf_repo is None:
        raise ValueError("hf_repo must be specified for HuggingFace datasets")
    elif config.dataset.source == "fiftyone":
        try:
            import fiftyone as fo

            if not fo.dataset_exists(config.dataset.fo_dataset_name):
                raise ValueError(f"FiftyOne dataset '{config.dataset.fo_dataset_name}' does not exist")
        except ImportError as exc:
            raise ImportError("fiftyone library required for FiftyOne datasets") from exc

    # Check model checkpoint
    if model_path is not None and not model_path.exists():
        raise ValueError(f"Model checkpoint not found: {model_path}")

    logger.info("Resource validation passed")


def write_summary(summary_data: dict[str, Any] | None, summary_location: Path, to_wandb: bool = False) -> None:
    """Write evaluation summary.

    Args:
        summary_data: Summary dictionary
        summary_location: Path to save summary
        to_wandb: Whether to log to wandb
    """
    if summary_location:
        summary_location.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_location, "w") as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"Summary written to {summary_location}")

    if to_wandb:
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(summary_data)
        except ImportError:
            logger.warning("wandb not installed")


def run_processing(
    config: ReidentificationConfig | None = None,
    config_dict: dict[str, Any] | None = None,
    model_path: Path | None = None,
    summary_location: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    baseline_mode: str | None = None,
) -> dict[str, Any]:
    """Core evaluation logic.

    Supports three modes:
    1. Standard: Evaluate a trained model on test embeddings
    2. random_baseline: Evaluate random embeddings (sanity check)
    3. backbone_only: Evaluate pre-computed backbone embeddings without fine-tuning

    Args:
        config: Configuration object
        config_dict: Configuration dictionary (alternative)
        model_path: Path to model checkpoint (required for standard mode)
        summary_location: Path to save summary
        dry_run: Only validate
        verbose: Verbose logging
        baseline_mode: Optional baseline mode ("random_baseline" or "backbone_only")

    Returns:
        Dictionary with evaluation results
    """
    logger_instance = setup_logger(MODULE_NAME, level=logging.DEBUG if verbose else logging.INFO)

    # Load configuration
    if config is None:
        if config_dict is not None:
            config = load_config_from_dict(config_dict)
        else:
            config = get_default_config()

    if dry_run:
        config.dry_run = True

    logger_instance.info("Starting evaluation...")
    logger_instance.info(f"Dataset source: {config.dataset.source}")

    # Validate resources
    validate_resources(config, model_path)

    if dry_run:
        logger_instance.info("DRY RUN: Would evaluate model")
        return {"status": "dry_run"}

    # Initialize wandb
    if config.wandb.enabled:
        try:
            import wandb

            wandb.init(
                entity=config.wandb.entity,
                project=config.wandb.project,
                name=config.wandb.run_name,
                tags=config.wandb.tags + ["evaluation"],
                notes=config.wandb.notes,
            )
        except ImportError:
            logger_instance.warning("wandb not installed")
            config.wandb.enabled = False

    # Set device
    device = config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger_instance.warning("CUDA not available, using CPU")
        device = "cpu"

    # Load dataset
    logger_instance.info("Loading dataset...")
    dataset_metadata = load_dataset(config.dataset)

    # Get test split
    test_data = dataset_metadata.get_split(config.dataset.test_split)
    
    # Get train split for stratified metrics
    train_data = dataset_metadata.get_split(config.dataset.train_split)

    logger_instance.info(f"Train set: {len(train_data)} samples, {train_data.num_classes} classes")
    logger_instance.info(f"Test set: {len(test_data)} samples, {test_data.num_classes} classes")

    # Handle baselines
    if baseline_mode == "random_baseline":
        logger_instance.info("Running random baseline evaluation (no training)")
        
        # Create random embeddings (no need for actual input_size or model)
        np.random.seed(42)
        test_embeddings = np.random.randn(len(test_data), config.model.embedding_dim).astype(np.float32)
        # Normalize
        test_embeddings = test_embeddings / (np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-12)
        
        train_embeddings = np.random.randn(len(train_data), config.model.embedding_dim).astype(np.float32)
        train_embeddings = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-12)
    
    elif baseline_mode == "backbone_only":
        logger_instance.info("Running backbone-only evaluation (no fine-tuning)")

        # Reuse cached embeddings when available; fall back to backbone extraction otherwise
        backbone = None

        if test_data.embeddings is not None:
            logger_instance.info("Using pre-computed test embeddings")
            test_embeddings = test_data.embeddings
        else:
            logger_instance.info("Extracting test embeddings with backbone...")
            backbone = get_backbone(config.backbone, device)
            backbone.eval()
            test_embeddings = backbone.extract_embeddings(test_data.image_paths, device, desc="Test embeddings")

        if train_data.embeddings is not None:
            logger_instance.info("Using pre-computed train embeddings")
            train_embeddings = train_data.embeddings
        else:
            logger_instance.info("Extracting train embeddings with backbone...")
            if backbone is None:
                backbone = get_backbone(config.backbone, device)
                backbone.eval()
            train_embeddings = backbone.extract_embeddings(train_data.image_paths, device, desc="Train embeddings")
    
    else:
        # Standard evaluation: extract or load embeddings
        if test_data.embeddings is None:
            logger_instance.info("Extracting embeddings...")
            backbone = get_backbone(config.backbone, device)
            backbone.eval()
            test_embeddings = backbone.extract_embeddings(test_data.image_paths, device, desc="Test embeddings")
        else:
            logger_instance.info("Using pre-computed test embeddings")
            test_embeddings = test_data.embeddings
        
        if train_data.embeddings is None:
            logger_instance.info("Extracting train embeddings...")
            backbone = get_backbone(config.backbone, device)
            backbone.eval()
            train_embeddings = backbone.extract_embeddings(train_data.image_paths, device, desc="Train embeddings")
        else:
            logger_instance.info("Using pre-computed train embeddings")
            train_embeddings = train_data.embeddings

    # Load model (only for standard mode, not for baselines)
    model = None
    use_embeddings_directly = baseline_mode in ("random_baseline", "backbone_only")
    
    if not use_embeddings_directly:
        # Standard mode: build and load trained model
        input_dim = test_embeddings.shape[1]
        model = build_model(input_dim, dataset_metadata.num_classes, config.model)
        model.to(device)

        if model_path is not None:
            logger_instance.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger_instance.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

        model.eval()

    # Compute metrics
    results: dict[str, Any] = {}

    if config.evaluation.compute_map:
        logger_instance.info("Computing comprehensive metrics...")
        comprehensive_metrics = compute_comprehensive_metrics(
            model=model,
            val_embeddings=test_embeddings,
            val_labels=test_data.labels_encoded,
            train_labels=train_data.labels_encoded,
            all_labels=dataset_metadata.labels_encoded,
            label_encoder=dataset_metadata.label_encoder,
            device=device,
            max_cmc_rank=50,
            min_total_samples_for_filtered=9,
            use_embeddings_directly=use_embeddings_directly,
        )
        
        # Log key metrics
        logger_instance.info(f"Sample-level mAP: {comprehensive_metrics['map']:.4f}")
        logger_instance.info(f"Identity-balanced mAP: {comprehensive_metrics['identity_balanced_map']:.4f}")
        if "map_min_total_9" in comprehensive_metrics:
            logger_instance.info(
                f"Filtered mAP (>=9 total samples): {comprehensive_metrics['map_min_total_9']:.4f} "
                f"({comprehensive_metrics['num_identities_min_total_9']}/{comprehensive_metrics['num_identities']} identities)"
            )
            logger_instance.info(
                f"Filtered CMC@1: {comprehensive_metrics['cmc_min_total_9@1']:.4f}, "
                f"CMC@5: {comprehensive_metrics['cmc_min_total_9@5']:.4f}"
            )
        logger_instance.info(f"CMC@1: {comprehensive_metrics['cmc@1']:.4f}")
        logger_instance.info(f"CMC@5: {comprehensive_metrics['cmc@5']:.4f}")
        
        # Add all comprehensive metrics to results
        results.update(comprehensive_metrics)

    # Note: CMC is now included in comprehensive_metrics, but we keep this for backward compatibility
    # if compute_map is False
    if config.evaluation.compute_cmc and not config.evaluation.compute_map:
        logger_instance.info("Computing CMC...")
        # Get fine-tuned embeddings
        if use_embeddings_directly:
            finetuned_emb = test_embeddings
        else:
            with torch.no_grad():
                test_tensor = torch.FloatTensor(test_embeddings).to(device)
                finetuned_emb = model.get_embeddings(test_tensor).cpu().numpy()

        cmc_scores = compute_cmc(finetuned_emb, test_data.labels_encoded, config.evaluation.cmc_top_k)
        results["cmc"] = cmc_scores

        for k, score in cmc_scores.items():
            logger_instance.info(f"CMC@{k}: {score:.4f}")

    # Save embeddings/predictions if requested
    if config.evaluation.save_embeddings or config.evaluation.save_predictions:
        output_dir = config.evaluation.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.evaluation.save_embeddings:
            if use_embeddings_directly:
                finetuned_emb = test_embeddings
            else:
                with torch.no_grad():
                    test_tensor = torch.FloatTensor(test_embeddings).to(device)
                    finetuned_emb = model.get_embeddings(test_tensor).cpu().numpy()

            emb_path = output_dir / "test_embeddings.npy"
            np.save(emb_path, finetuned_emb)
            logger_instance.info(f"Saved embeddings to {emb_path}")

        if config.evaluation.save_predictions:
            pred_path = output_dir / "test_predictions.json"
            predictions = {
                "image_paths": test_data.image_paths,
                "true_labels": test_data.labels,
                "label_encoder": {"classes": dataset_metadata.label_encoder.classes_.tolist()},
            }
            with open(pred_path, "w") as f:
                json.dump(predictions, f, indent=2)
            logger_instance.info(f"Saved predictions to {pred_path}")

    # Add to FiftyOne if requested
    if config.evaluation.add_to_fiftyone and config.evaluation.fo_dataset_name:
        logger_instance.info("Adding results to FiftyOne...")
        try:
            import fiftyone as fo

            # Get fine-tuned embeddings
            if use_embeddings_directly:
                finetuned_emb = test_embeddings
            else:
                with torch.no_grad():
                    test_tensor = torch.FloatTensor(test_embeddings).to(device)
                    finetuned_emb = model.get_embeddings(test_tensor).cpu().numpy()

            # Load dataset
            fo_dataset = fo.load_dataset(config.evaluation.fo_dataset_name)

            # Add embeddings to samples
            for img_path, embedding in zip(test_data.image_paths, finetuned_emb, strict=True):
                # Find matching sample
                sample = fo_dataset.match(fo.ViewField("filepath") == img_path).first()
                if sample is not None:
                    sample[config.evaluation.fo_embeddings_field] = embedding.tolist()
                    sample.save()

            logger_instance.info(f"Added embeddings to FiftyOne dataset: {config.evaluation.fo_dataset_name}")

        except ImportError:
            logger_instance.warning("fiftyone not installed, skipping FiftyOne export")
        except Exception as e:
            logger_instance.error(f"Error adding to FiftyOne: {e}")

    # Prepare final results
    final_results = {
        "status": "completed",
        "num_test_samples": len(test_data),
        "num_classes": test_data.num_classes,
        **results,
    }

    # Write summary
    if summary_location is not None or config.wandb.enabled:
        write_summary(final_results, summary_location, config.wandb.enabled)

    # Close wandb
    if config.wandb.enabled:
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass

    logger_instance.info("Evaluation completed!")

    return final_results


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate jaguar re-identification model", formatter_class=argparse.RawDescriptionHelpFormatter)

    # Model args
    parser.add_argument("--model-path", type=Path, required=True, help="Path to model checkpoint")

    # Dataset args
    parser.add_argument("--dataset-source", type=str, choices=["disk", "huggingface", "fiftyone"], default="fiftyone")
    parser.add_argument("--data-dir", type=Path, help="Data directory")
    parser.add_argument("--hf-repo", type=str, help="HuggingFace repo")
    parser.add_argument("--fo-dataset", type=str, default="JID_Master_Dataset")

    # Evaluation args
    parser.add_argument("--compute-map", action="store_true", default=True)
    parser.add_argument("--compute-cmc", action="store_true", default=True)
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")

    # FiftyOne args
    parser.add_argument("--add-to-fiftyone", action="store_true")
    parser.add_argument("--fo-output-dataset", type=str, help="FiftyOne dataset to add results to")

    # Wandb args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="camerate-trap-reidentificationentification")
    parser.add_argument("--wandb-entity", type=str)

    # Output args
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/reidentification"))
    parser.add_argument("--summary-location", type=Path)

    # Runtime args
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Build configuration
    config = get_default_config()

    # Dataset config
    config.dataset.source = args.dataset_source
    if args.data_dir:
        config.dataset.data_dir = args.data_dir
    if args.hf_repo:
        config.dataset.hf_repo = args.hf_repo
    config.dataset.fo_dataset_name = args.fo_dataset

    # Evaluation config
    config.evaluation.compute_map = args.compute_map
    config.evaluation.compute_cmc = args.compute_cmc
    config.evaluation.save_embeddings = args.save_embeddings
    config.evaluation.save_predictions = args.save_predictions
    config.evaluation.output_dir = args.output_dir
    config.evaluation.add_to_fiftyone = args.add_to_fiftyone
    if args.fo_output_dataset:
        config.evaluation.fo_dataset_name = args.fo_output_dataset

    # Wandb config
    config.wandb.enabled = args.wandb
    config.wandb.project = args.wandb_project
    if args.wandb_entity:
        config.wandb.entity = args.wandb_entity

    # Runtime
    config.dry_run = args.dry_run
    config.verbose = args.verbose

    try:
        run_processing(
            config=config,
            model_path=args.model_path,
            summary_location=args.summary_location,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
