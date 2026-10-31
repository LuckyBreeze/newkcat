"""Paired, row-preserving prediction diagnostics on an existing sample manifest.

This module never trains, selects checkpoints, reweights labels, or removes rows.
All models are grouped by the *same manifest* status and length bins. Optional
confidence intervals resample whole manifest clusters, retaining paired errors.
"""

from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np

from ..metrics import regression_metrics


_LABEL_RTOL = 2 * np.finfo(np.float32).eps
_LABEL_ATOL = 2 * np.finfo(np.float32).eps
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def _csv(path: Path) -> tuple[list[dict], dict]:
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)) or any(not f for f in fields):
        raise ValueError(f"Missing, empty, or duplicate CSV headers: {path}")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row width: {path}")
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                  "rows": len(rows), "columns": fields}


def _finite(value: str, name: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid numeric {name}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {name}")
    return result


def _indexed(rows: list[dict], name: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id.strip():
            raise ValueError(f"Missing sample_id in {name}")
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {name}")
        result[sample_id] = row
    return result


def _split(value: str) -> str:
    aliases = {"valid": "val", "validation": "val"}
    result = aliases.get(value, value)
    if result not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {value!r}")
    return result


def _column(fields: list[str], explicit: str | None, aliases: tuple[str, ...],
            kind: str) -> str:
    if explicit is not None:
        if explicit not in fields:
            raise ValueError(f"Missing explicit {kind} column {explicit!r}")
        return explicit
    matches = [name for name in aliases if name in fields]
    if len(matches) != 1:
        raise ValueError(f"Expected one unambiguous {kind} column from {aliases}, got {matches}")
    return matches[0]


def _length(text: str | None) -> int | None:
    return len(text) + 2 if text else None


def _length_bin(length: int | None) -> str:
    if length is None:
        return "missing"
    lower = 1
    for upper in (128, 256, 512, 1024, 2048):
        if length <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return ">2048"


def _row_diagnostics(row: dict, group_column: str | None) -> dict:
    status = row.get("reaction_status", "").strip() or "missing"
    sr = row.get("sr_smiles", "")
    reaction = row.get("reaction_smiles", "")
    if status == "ok":
        selected, representation = sr, "sr_smiles"
    elif status in {"fallback_reaction", "reaction_only"}:
        selected, representation = reaction, "reaction_smiles"
    else:
        selected, representation = "", "unavailable"
    spans = None
    if row.get("edit_spans", ""):
        try:
            spans = json.loads(row["edit_spans"])
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid edit_spans for {row['sample_id']}") from error
        previous = 0
        if not isinstance(spans, list):
            raise ValueError(f"edit_spans must be a list for {row['sample_id']}")
        for span in spans:
            if (not isinstance(span, list) or len(span) != 2
                    or any(type(x) is not int for x in span)
                    or not previous <= span[0] < span[1] <= len(sr)):
                raise ValueError(f"Invalid edit span for {row['sample_id']}")
            previous = span[1]
    changed = sum(end - start for start, end in spans) if spans is not None else None
    return {
        "reaction_status": status,
        "cluster_id": row.get(group_column, "") if group_column else "",
        "reaction_tokens": _length(reaction),
        "sr_tokens": _length(sr),
        "selected_input_tokens": _length(selected),
        "selected_input_format": representation if selected else "unavailable",
        "dynamic_fields": len(spans) if spans is not None else None,
        "dynamic_characters": changed,
        "dynamic_character_fraction": changed / len(sr) if sr and changed is not None else None,
    }


def _groups(diagnostics: list[dict]) -> list[tuple[str, str, np.ndarray]]:
    result = [("overall", "all", np.arange(len(diagnostics)))]
    group_values = {"reaction_status": [row["reaction_status"] for row in diagnostics]}
    for key in ("reaction_tokens", "selected_input_tokens"):
        group_values[key] = [_length_bin(row[key]) for row in diagnostics]
    for grouping, values in group_values.items():
        for value in sorted(set(values)):
            result.append((grouping, value, np.flatnonzero(np.asarray(values) == value)))
    return result


def _cluster_ci(target: np.ndarray, a: np.ndarray, b: np.ndarray,
                clusters: np.ndarray, repetitions: int, seed: int) -> dict:
    result = {"bootstrap_repetitions": repetitions,
              "bootstrap_status": "disabled" if repetitions == 0 else "insufficient_clusters",
              "delta_mse_ci95_low": None, "delta_mse_ci95_high": None,
              "delta_rmse_ci95_low": None, "delta_rmse_ci95_high": None}
    if repetitions == 0:
        return result
    _, inverse = np.unique(clusters, return_inverse=True)
    counts = np.bincount(inverse)
    if len(counts) < 2:
        return result
    with np.errstate(over="raise", invalid="raise"):
        sums_a = np.bincount(inverse, weights=(a - target) ** 2)
        sums_b = np.bincount(inverse, weights=(b - target) ** 2)
    rng = np.random.default_rng(seed)
    delta_mse, delta_rmse = [], []
    # Bound temporary draw arrays even when clusters or requested repetitions are large.
    chunk_size = max(1, min(128, 1_000_000 // len(counts)))
    for offset in range(0, repetitions, chunk_size):
        draws = rng.integers(0, len(counts), size=(min(chunk_size, repetitions-offset), len(counts)))
        denominator = counts[draws].sum(axis=1)
        mse_a = sums_a[draws].sum(axis=1) / denominator
        mse_b = sums_b[draws].sum(axis=1) / denominator
        delta_mse.extend((mse_a-mse_b).tolist())
        delta_rmse.extend((np.sqrt(mse_a)-np.sqrt(mse_b)).tolist())
    low, high = np.quantile(delta_mse, [0.025, 0.975])
    rmse_low, rmse_high = np.quantile(delta_rmse, [0.025, 0.975])
    result.update(bootstrap_status="cluster_resampling", delta_mse_ci95_low=float(low),
                  delta_mse_ci95_high=float(high), delta_rmse_ci95_low=float(rmse_low),
                  delta_rmse_ci95_high=float(rmse_high))
    return result


def _check_output(output: Path, inputs: list[Path]) -> None:
    if any(output == source or output in source.parents for source in inputs):
        raise ValueError("output_dir cannot be an input file or an ancestor of an input file")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output_dir must be absent or empty; audit never overwrites results")


def _write_csv(path: Path, rows: list[dict], empty_fields: list[str]) -> None:
    fields = list(rows[0]) if rows else empty_fields
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_predictions(
    samples_path: str | Path,
    predictions: Mapping[str, str | Path] | None,
    split: str,
    output_dir: str | Path,
    *,
    group_column: str | None = "group_id",
    bootstrap_repetitions: int = 0,
    bootstrap_seed: int = 668,
    sample_target_column: str = "log10_kcat",
    prediction_target_column: str | None = None,
    prediction_column: str | None = None,
) -> dict:
    """Audit exact prediction/manifest joins, optionally without prediction files.

    Prediction CSVs must contain exactly the selected split's IDs and finite
    labels matching the manifest within two float32 epsilons (absolute/relative).
    The manifest's float64 labels are used for every model's metrics. Auto aliases
    are target/y_true and prediction/y_pred; ambiguous headers require an explicit
    column. Bootstrap defaults to disabled and requires complete cluster IDs when
    enabled. All groups are descriptive, never used to refit or select models.
    """
    if type(bootstrap_repetitions) is not int or bootstrap_repetitions < 0:
        raise ValueError("bootstrap_repetitions must be a nonnegative integer")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if predictions is not None and not isinstance(predictions, Mapping):
        raise ValueError("predictions must map model names to CSV paths")
    prediction_paths = {}
    for name, value in (predictions or {}).items():
        if not isinstance(name, str) or not _MODEL_NAME.fullmatch(name):
            raise ValueError(f"Invalid model name: {name!r}")
        prediction_paths[name] = Path(value).resolve()
    samples_path, output = Path(samples_path).resolve(), Path(output_dir).resolve()
    input_paths = [samples_path, *prediction_paths.values()]
    _check_output(output, input_paths)
    split = _split(split)
    samples, sample_source = _csv(samples_path)
    _indexed(samples, "manifest")  # Includes other splits: IDs may not be reused.
    if sample_target_column not in sample_source["columns"]:
        raise ValueError(f"Missing sample target column {sample_target_column!r}")
    selected = [row for row in samples if _split(row.get("split", "")) == split]
    if not selected:
        raise ValueError(f"No manifest records in split {split!r}")
    sample_ids = [row["sample_id"] for row in selected]
    targets = np.asarray([_finite(row[sample_target_column], "manifest target")
                          for row in selected], dtype=np.float64)
    diagnostics = [_row_diagnostics(row, group_column) for row in selected]
    clusters = np.asarray([row["cluster_id"] for row in diagnostics], dtype=str)
    if bootstrap_repetitions and (not group_column or any(not x.strip() for x in clusters)):
        raise ValueError("Cluster bootstrap requires a complete, nonempty group_column")
    group_indices = _groups(diagnostics)
    sources, aligned, label_checks = {}, {}, {}
    for model_name, path in prediction_paths.items():
        prediction_rows, source = _csv(path)
        by_id = _indexed(prediction_rows, model_name)
        missing = set(sample_ids) - by_id.keys()
        extra = by_id.keys() - set(sample_ids)
        if missing or extra:
            raise ValueError(f"Prediction ID set mismatch for {model_name}: "
                             f"missing={len(missing)} {sorted(missing)[:5]}, "
                             f"extra={len(extra)} {sorted(extra)[:5]}; no rows may be dropped")
        target_key = _column(source["columns"], prediction_target_column,
                             ("target", "y_true"), "prediction target")
        prediction_key = _column(source["columns"], prediction_column,
                                 ("prediction", "y_pred"), "prediction")
        provided = np.asarray([_finite(by_id[sid][target_key], f"{model_name} target")
                               for sid in sample_ids], dtype=np.float64)
        same = np.isclose(provided, targets, rtol=_LABEL_RTOL, atol=_LABEL_ATOL)
        if not same.all():
            bad = [sample_ids[index] for index in np.flatnonzero(~same)[:5]]
            raise ValueError(f"Prediction label mismatch for {model_name}: {bad}")
        aligned[model_name] = np.asarray([
            _finite(by_id[sid][prediction_key], f"{model_name} prediction") for sid in sample_ids
        ], dtype=np.float64)
        sources[model_name] = source
        label_checks[model_name] = {"matched_rows": len(sample_ids),
                                   "max_absolute_label_difference": float(np.max(abs(provided-targets)))}
    coverage, model_metrics, paired = [], [], []
    for grouping, group, indices in group_indices:
        y, cluster = targets[indices], clusters[indices]
        n_clusters = len({x for x in cluster if x.strip()})
        common = {"grouping": grouping, "group": group, "n": len(indices)}
        coverage.append({**common, "fraction_of_split": len(indices)/len(selected),
                         "label_mean": float(y.mean()), "label_sd": float(y.std()),
                         "label_min": float(y.min()), "label_median": float(np.median(y)),
                         "label_max": float(y.max()), "n_clusters": n_clusters,
                         "missing_cluster_rows": sum(not x.strip() for x in cluster)})
        for name, prediction in aligned.items():
            model_metrics.append({"model": name, **common,
                                  **regression_metrics(y, prediction[indices]),
                                  "prediction_mean": float(prediction[indices].mean()),
                                  "prediction_sd": float(prediction[indices].std()),
                                  "mean_residual": float((prediction[indices]-y).mean())})
        for name_a, name_b in itertools.combinations(aligned, 2):
            a, b = aligned[name_a][indices], aligned[name_b][indices]
            metrics_a, metrics_b = regression_metrics(y, a), regression_metrics(y, b)
            delta = metrics_a["mse"] - metrics_b["mse"]
            paired.append({"model_a": name_a, "model_b": name_b, **common,
                           "n_clusters": n_clusters, "delta_mse": delta,
                           "delta_rmse": metrics_a["rmse"]-metrics_b["rmse"],
                           "delta_mae": metrics_a["mae"]-metrics_b["mae"],
                           "weighted_delta_mse": delta*len(indices)/len(selected),
                           **_cluster_ci(y, a, b, cluster, bootstrap_repetitions, bootstrap_seed)})
    joined = [{"sample_id": sid, "split": split, "target": float(targets[i]),
               **diagnostics[i], **{f"prediction__{name}": float(values[i])
                                    for name, values in aligned.items()}}
              for i, sid in enumerate(sample_ids)]
    report = {
        "schema_version": 1, "split": split, "n": len(selected), "rows_dropped": 0,
        "models": list(aligned), "target_source": "manifest",
        "sample_target_column": sample_target_column,
        "label_tolerance": {"rtol": float(_LABEL_RTOL), "atol": float(_LABEL_ATOL)},
        "label_checks": label_checks,
        "sources": {"manifest": sample_source, "predictions": sources},
        "group_column": group_column, "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": bootstrap_seed, "confidence_level": 0.95,
        "coverage": coverage, "model_metrics": model_metrics, "paired_differences": paired,
        "notes": [
            "Groups come from the same manifest for every model; input status in prediction files is ignored.",
            "Token lengths are stored-string character counts plus BOS/EOS under the fixed character tokenizer.",
            "Selected-input lengths use sr text for status ok and full reaction text for reaction fallback.",
            "Paired differences are model_a minus model_b; negative error differences favor model_a.",
            "Status weighted_delta_mse values sum to the overall delta_mse; do not average group RMSEs.",
            "Bootstrap resamples whole clusters within each reported group and preserves row-paired errors.",
            "Diagnostics are descriptive; no labels, splits, model parameters, or sample weights are changed.",
        ],
        "outputs": {name: str(output / name) for name in (
            "audit.json", "group_coverage.csv", "model_metrics.csv", "paired_differences.csv",
            "joined_predictions.csv")},
    }
    # Validate JSON before creating the destination; invalid inputs leave no partial output.
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _check_output(output, input_paths)
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(encoded, encoding="utf-8")
    _write_csv(output / "group_coverage.csv", coverage, ["grouping", "group", "n"])
    _write_csv(output / "model_metrics.csv", model_metrics, ["model", "grouping", "group", "n"])
    _write_csv(output / "paired_differences.csv", paired,
               ["model_a", "model_b", "grouping", "group", "n", "delta_mse"])
    _write_csv(output / "joined_predictions.csv", joined, ["sample_id", "split", "target"])
    return report
