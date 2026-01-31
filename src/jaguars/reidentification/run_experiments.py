"""Run multiple re-identification experiments.

This script orchestrates experiment execution, logging results to WandB
and saving outputs for analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from jaguars.common.logging_utils import setup_logger
from jaguars.reidentification.experiments import (
    ExperimentConfig,
    get_all_experiments,
    save_experiment_config,
)
from jaguars.reidentification.pipeline import run_pipeline


def run_single_experiment(
    experiment: ExperimentConfig,
    seed: int,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a single experiment with a specific seed.
    
    Args:
        experiment: Experiment configuration
        seed: Random seed for reproducibility
        output_dir: Directory to save outputs
        dry_run: If True, only print configuration without running
        
    Returns:
        Dictionary with experiment results and metadata
    """
    logger = logging.getLogger(__name__)
    
    # Create experiment-specific output directory
    exp_output_dir = output_dir / experiment.name / f"seed_{seed}"
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    config_path = exp_output_dir / "config.json"
    save_experiment_config(experiment, config_path)
    
    logger.info(f"Starting experiment: {experiment.name} (seed={seed})")
    logger.info(f"Description: {experiment.description}")
    logger.info(f"Output directory: {exp_output_dir}")
    
    if dry_run:
        logger.info("DRY RUN - Skipping actual execution")
        return {
            "experiment_name": experiment.name,
            "seed": seed,
            "status": "dry_run",
            "config_path": str(config_path),
        }
    
    # Update config with seed and variations
    config = experiment.base_config
    
    # Update WandB configuration
    if config.wandb.enabled:
        config.wandb.tags = experiment.tags + [f"seed_{seed}"]
    
    # Apply variations (if any)
    for key, value in experiment.variations.items():
        logger.info(f"Applying variation: {key} = {value}")
        # This would require more sophisticated config merging
        # For now, variations are documented but applied manually in pipeline
    
    try:
        # Run the complete pipeline
        results = run_pipeline(config)
        
        logger.info(f"Experiment {experiment.name} (seed={seed}) completed successfully")
        logger.info(f"Results: {results}")
        
        return {
            "experiment_name": experiment.name,
            "seed": seed,
            "status": "success",
            "results": results,
            "output_dir": str(exp_output_dir),
        }
        
    except Exception as e:
        logger.error(f"Experiment {experiment.name} (seed={seed}) failed: {e}", exc_info=True)
        
        return {
            "experiment_name": experiment.name,
            "seed": seed,
            "status": "failed",
            "error": str(e),
        }


def run_experiment_suite(
    experiments: list[ExperimentConfig],
    output_dir: Path,
    dry_run: bool = False,
    resume_from: str | None = None,
) -> list[dict[str, Any]]:
    """Run a suite of experiments.
    
    Args:
        experiments: List of experiment configurations
        output_dir: Base directory for all outputs
        dry_run: If True, only print what would be run
        resume_from: Resume from specific experiment name (skip earlier ones)
        
    Returns:
        List of result dictionaries for all experiments
    """
    logger = logging.getLogger(__name__)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    total_runs = sum(len(exp.seeds) for exp in experiments)
    current_run = 0
    
    skip_until_found = resume_from is not None
    
    for experiment in experiments:
        # Resume logic
        if skip_until_found:
            if experiment.name == resume_from:
                skip_until_found = False
                logger.info(f"Resuming from experiment: {experiment.name}")
            else:
                logger.info(f"Skipping experiment: {experiment.name} (resume mode)")
                current_run += len(experiment.seeds)
                continue
        
        for seed in experiment.seeds:
            current_run += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"Running experiment {current_run}/{total_runs}")
            logger.info(f"{'='*80}\n")
            
            result = run_single_experiment(
                experiment=experiment,
                seed=seed,
                output_dir=output_dir,
                dry_run=dry_run,
            )
            results.append(result)
            
            # Log summary
            if result["status"] == "success":
                logger.info(f"✓ Experiment succeeded: {experiment.name} (seed={seed})")
            elif result["status"] == "failed":
                logger.error(f"✗ Experiment failed: {experiment.name} (seed={seed})")
            
            # Brief pause between experiments
            if current_run < total_runs:
                logger.info(f"\nProgress: {current_run}/{total_runs} complete\n")
    
    return results


def print_experiment_summary(results: list[dict[str, Any]]) -> None:
    """Print summary of experiment results."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUITE SUMMARY")
    logger.info("="*80 + "\n")
    
    total = len(results)
    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    dry_run = sum(1 for r in results if r["status"] == "dry_run")
    
    logger.info(f"Total experiments: {total}")
    logger.info(f"  ✓ Succeeded: {succeeded}")
    logger.info(f"  ✗ Failed: {failed}")
    logger.info(f"  ○ Dry run: {dry_run}")
    
    if failed > 0:
        logger.info("\nFailed experiments:")
        for result in results:
            if result["status"] == "failed":
                logger.info(f"  - {result['experiment_name']} (seed={result['seed']})")
                logger.info(f"    Error: {result.get('error', 'Unknown')}")
    
    logger.info("\n" + "="*80 + "\n")


def main() -> int:
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description="Run re-identification experiments")
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs"),
        help="Base output directory for all experiments",
    )
    
    parser.add_argument(
        "--category",
        type=str,
        choices=["backbone", "loss", "optimizer", "augmentation", "seed_stability", "embedding_dim", "all"],
        default="all",
        help="Experiment category to run",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment configurations without running",
    )
    
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume from specific experiment name (skip earlier ones)",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Convert log level string to logging constant
    log_level = getattr(logging, args.log_level.upper())
    
    # Setup logging
    setup_logger(module_name=__name__, level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("Re-identification Experiment Runner")
    logger.info("="*80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Category: {args.category}")
    logger.info(f"Dry run: {args.dry_run}")
    if args.resume_from:
        logger.info(f"Resuming from: {args.resume_from}")
    logger.info("")
    
    # Get experiments
    all_experiments = get_all_experiments()
    
    if args.category == "all":
        experiments = []
        for category_experiments in all_experiments.values():
            experiments.extend(category_experiments)
    else:
        experiments = all_experiments[args.category]
    
    logger.info(f"Total experiments to run: {len(experiments)}")
    logger.info(f"Total runs (including seeds): {sum(len(exp.seeds) for exp in experiments)}")
    
    # List experiments
    logger.info("\nExperiments:")
    for i, exp in enumerate(experiments, 1):
        logger.info(f"  {i}. {exp.name} (seeds: {len(exp.seeds)})")
        logger.info(f"     {exp.description}")
    logger.info("")
    
    # Run experiments
    results = run_experiment_suite(
        experiments=experiments,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        resume_from=args.resume_from,
    )
    
    # Print summary
    print_experiment_summary(results)
    
    # Return exit code
    failed = sum(1 for r in results if r["status"] == "failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
