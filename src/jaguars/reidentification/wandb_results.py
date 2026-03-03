"""Utilities for loading experiment metrics from Weights & Biases.

This module provides helpers to retrieve the most recent W&B run per
experiment by filtering on tags, making notebooks robust to interrupted runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import re
from typing import Any, Protocol


class WandbConfigLike(Protocol):
    """Protocol for the subset of wandb config needed by these helpers."""

    tags: list[str]
    run_name: str | None


class BaseConfigLike(Protocol):
    """Protocol for the subset of experiment config needed by these helpers."""

    wandb: WandbConfigLike


class ExperimentLike(Protocol):
    """Protocol for experiment objects used in notebooks."""

    name: str
    base_config: BaseConfigLike


METRIC_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "map": ("map", "val/map", "validation/map", "eval/map", "metrics/map"),
    "cmc@1": ("cmc@1", "val/cmc@1", "validation/cmc@1", "eval/cmc@1", "metrics/cmc@1"),
    "cmc@5": ("cmc@5", "val/cmc@5", "validation/cmc@5", "eval/cmc@5", "metrics/cmc@5"),
    "map_min_total_9": (
        "map_min_total_9",
        "val/map_min_total_9",
        "validation/map_min_total_9",
        "eval/map_min_total_9",
        "metrics/map_min_total_9",
    ),
    "identity_balanced_map": (
        "identity_balanced_map",
        "val/identity_balanced_map",
        "validation/identity_balanced_map",
        "eval/identity_balanced_map",
    ),
    "closed_set_map": ("closed_set_map", "val/closed_set_map", "validation/closed_set_map", "eval/closed_set_map"),
    "validation/map": ("validation/map", "val/map", "map", "eval/map", "metrics/map"),
}


def _parse_created_at(value: str | None) -> datetime:
    """Parse a W&B created_at string into a sortable datetime.

    Args:
        value: W&B created_at timestamp string.

    Returns:
        Parsed datetime, or datetime.min when parsing fails.
    """
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def _to_float_or_original(value: Any) -> float | Any:
    """Convert numeric-like values to float and keep others unchanged."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _extract_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Extract canonical metric keys from a W&B run summary.

    Args:
        summary: W&B run summary mapping.

    Returns:
        Dictionary with canonical metric keys where values were found.
    """
    extracted: dict[str, Any] = {}
    for canonical_key, aliases in METRIC_KEY_ALIASES.items():
        for alias in aliases:
            if alias in summary:
                extracted[canonical_key] = _to_float_or_original(summary[alias])
                break
    return extracted


def _normalize_name(value: str | None) -> str:
    """Normalize a run or experiment name for robust comparisons."""
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _candidate_expected_names(experiment: ExperimentLike) -> list[str]:
    """Build candidate strings that may match the W&B run name."""
    expected = experiment.base_config.wandb.run_name or ""
    names = [expected, experiment.name]

    if "__" in expected:
        names.append(expected.split("__", 1)[1])

    if experiment.name.startswith("baseline_"):
        names.append(experiment.name.replace("baseline_", "", 1))
    if experiment.name.startswith("loss_"):
        names.append(experiment.name.replace("loss_", "", 1))
    if experiment.name.startswith("backbone_"):
        names.append(experiment.name.replace("backbone_", "", 1))

    unique_names: list[str] = []
    for name in names:
        if name and name not in unique_names:
            unique_names.append(name)
    return unique_names


def _filter_by_name(runs: list[Any], experiment: ExperimentLike) -> list[Any]:
    """Filter runs using normalized name matching against experiment hints."""
    candidates = _candidate_expected_names(experiment)
    if not candidates:
        return runs

    normalized_candidates = [_normalize_name(candidate) for candidate in candidates if candidate]
    matched: list[Any] = []
    for run in runs:
        run_name_normalized = _normalize_name(getattr(run, "name", ""))
        if any(
            candidate and (candidate in run_name_normalized or run_name_normalized in candidate)
            for candidate in normalized_candidates
        ):
            matched.append(run)
    return matched


def _make_filters(required_tags: list[str]) -> dict[str, Any]:
    """Create W&B run filters for a list of required tags."""
    if not required_tags:
        return {}
    filter_conditions = [{"tags": {"$in": [tag]}} for tag in required_tags]
    return {"$and": filter_conditions}


def _relaxed_tags(required_tags: list[str]) -> list[str]:
    """Drop dynamic tags that often vary across notebook revisions."""
    return [
        tag
        for tag in required_tags
        if not (
            tag.startswith("run_batch:")
            or tag.startswith("dataset:")
            or tag.startswith("source:")
            or tag.startswith("backbone:")
        )
    ]


def fetch_latest_metrics_for_experiments(
    experiments: Iterable[ExperimentLike],
    entity: str,
    project: str,
    *,
    additional_tags: Iterable[str] | None = None,
    prefer_run_name_match: bool = True,
    max_runs_per_query: int = 200,
) -> dict[str, dict[str, Any]]:
    """Fetch latest metrics for each experiment from W&B.

    For each experiment, the function filters runs by the experiment's tags
    (plus optional additional tags), then selects the most recent run.

    Args:
        experiments: Experiments to resolve.
        entity: W&B entity name.
        project: W&B project name.
        additional_tags: Extra tags to require for every experiment.
        prefer_run_name_match: Prefer runs whose name matches configured run_name.
        max_runs_per_query: Maximum number of runs fetched per experiment.

    Returns:
        Mapping experiment_name -> metrics dict (or error dict).
    """
    try:
        import wandb
    except ImportError:
        return {
            experiment.name: {"error": "wandb package not installed"}
            for experiment in experiments
        }

    api = wandb.Api()
    path = f"{entity}/{project}"
    common_tags = list(additional_tags or [])

    resolved: dict[str, dict[str, Any]] = {}
    for experiment in experiments:
        expected_run_name = experiment.base_config.wandb.run_name
        required_tags = list(dict.fromkeys([*experiment.base_config.wandb.tags, *common_tags]))

        runs = list(api.runs(path=path, filters=_make_filters(required_tags), per_page=max_runs_per_query))
        if not runs:
            relaxed_tags = _relaxed_tags(required_tags)
            runs = list(api.runs(path=path, filters=_make_filters(relaxed_tags), per_page=max_runs_per_query))

        if not runs:
            resolved[experiment.name] = {"error": f"No W&B run found for tags: {required_tags}"}
            continue

        runs_sorted = sorted(runs, key=lambda run: _parse_created_at(run.created_at), reverse=True)
        name_matched_runs = _filter_by_name(runs_sorted, experiment)
        if name_matched_runs:
            runs_sorted = name_matched_runs

        selected_run = None
        if prefer_run_name_match and expected_run_name:
            run_name_matches = [run for run in runs_sorted if getattr(run, "name", None) == expected_run_name]
            if run_name_matches:
                selected_run = run_name_matches[0]
            else:
                contains_matches = [
                    run
                    for run in runs_sorted
                    if expected_run_name in (getattr(run, "name", "") or "")
                ]
                if contains_matches:
                    selected_run = contains_matches[0]

        if selected_run is None:
            selected_run = runs_sorted[0]

        metrics = _extract_metrics(selected_run.summary)
        resolved[experiment.name] = {
            **metrics,
            "_wandb_run_id": selected_run.id,
            "_wandb_run_name": selected_run.name,
            "_wandb_created_at": selected_run.created_at,
            "_wandb_state": selected_run.state,
        }

        if not metrics:
            resolved[experiment.name]["error"] = "No supported metrics found in selected W&B run summary"

    return resolved