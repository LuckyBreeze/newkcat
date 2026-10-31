"""Import official CataPro CSV folds and existing global feature caches.

Run with ``python -m newkcat.import_catapro --help``. This is a format adapter,
not a feature extractor or a new homology-clustering procedure.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold(value: str) -> str:
    value = str(value).strip()
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", value):
        return str(int(value.split(".")[0]))
    if not value:
        raise ValueError("Fold assignments must not be empty")
    return value


def _read_samples(path: Path, test_fold: str, val_fold: str, id_column: str) -> list[dict]:
    test_fold, val_fold = _fold(test_fold), _fold(val_fold)
    if test_fold == val_fold:
        raise ValueError("Validation and test folds must differ")
    samples, seen = [], set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {id_column, "Sequence", "Smiles", "kcat(s^-1)", "fold"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CataPro CSV must contain {sorted(required)}; got {reader.fieldnames}")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CataPro CSV has duplicate column names")
        for line, record in enumerate(reader, 2):
            if None in record or any(record[key] is None for key in required):
                raise ValueError(f"Inconsistent CSV column count at line {line}")
            sample_id = record[id_column].strip()
            if not sample_id or sample_id in seen:
                raise ValueError(f"Empty or duplicate sample ID at CSV line {line}")
            seen.add(sample_id)
            try:
                kcat = float(record["kcat(s^-1)"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid kcat at CSV line {line}") from error
            if not math.isfinite(kcat) or kcat <= 0:
                raise ValueError(f"kcat must be positive and finite at CSV line {line}")
            fold = _fold(record["fold"])
            split = "test" if fold == test_fold else "val" if fold == val_fold else "train"
            samples.append({
                "sample_id": sample_id, "sequence": record["Sequence"].strip(),
                "smiles": record["Smiles"].strip(), "log10_kcat": math.log10(kcat),
                "split": split, "ec": record.get("EC", "").strip(), "source_fold": fold,
            })
    counts = {split: sum(row["split"] == split for row in samples)
              for split in ("train", "val", "test")}
    if counts["train"] < 2 or counts["val"] < 1 or counts["test"] < 1:
        raise ValueError(f"Need at least 2 train, 1 val, and 1 test samples; found {counts}")
    return samples


def _read_features(path: Path, *, trust_pickle: bool) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if not {"sample_ids", "protein", "molt5", "maccs"}.issubset(archive.files):
                raise ValueError("NPZ needs sample_ids, protein, molt5, maccs")
            ids = archive["sample_ids"]
            if ids.ndim != 1 or ids.dtype.kind != "U":
                raise ValueError("NPZ sample_ids must be a 1D Unicode array")
            parts = []
            for key, width in (("protein", 1024), ("molt5", 768), ("maccs", 167)):
                part = archive[key]
                if part.shape != (len(ids), width):
                    raise ValueError(f"{key} must have shape [{len(ids)}, {width}]")
                if part.dtype.kind not in "fiu":
                    raise ValueError(f"{key} must contain numeric features")
                parts.append(part)
            matrix = np.concatenate(parts, axis=1)
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        if not trust_pickle:
            raise ValueError("Pickle can execute code: use --trust-pickle only for your trusted cache")
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError("Install the importer extra: pip install -e '.[catapro]'") from error
        frame = pd.read_pickle(path)
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("Official feature pickle must contain a pandas DataFrame")
        if isinstance(frame.index, pd.MultiIndex):
            raise ValueError("Feature index must consist of sample IDs, not a MultiIndex")
        ids = np.asarray([str(value) for value in frame.index], dtype=str)
        if frame.shape[1] != 1959:
            raise ValueError("Official feature table must contain exactly 1959 ordered columns")
        # The upstream DataFrame is created with columns 0..1958. Match these
        # semantic positions explicitly; width alone cannot identify towers.
        labels = [str(column) for column in frame.columns]
        expected_labels = [str(index) for index in range(1959)]
        if len(set(labels)) != 1959 or set(labels) != set(expected_labels):
            raise ValueError("Official feature columns must be unique labels 0..1958")
        column_positions = {label: index for index, label in enumerate(labels)}
        frame = frame.iloc[:, [column_positions[label] for label in expected_labels]]
        if any(dtype.kind not in "fiu" for dtype in frame.dtypes):
            raise ValueError("All feature columns must be numeric")
        matrix = frame.to_numpy()
    else:
        raise ValueError("Feature input must be the official .pkl table or newkcat .npz")
    if len(ids) != len(set(ids.tolist())) or any(not str(value).strip() for value in ids):
        raise ValueError("Feature sample IDs must be nonempty and unique")
    # Check after conversion too: a finite float64 can overflow float32.
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature table contains nonfinite values or float32 overflow")
    maccs = matrix[:, 1792:]
    if not np.isin(maccs, [0, 1]).all() or np.any(maccs[:, 0]):
        raise ValueError("MACCS must be 167 binary values with reserved bit 0 equal to zero")
    return ids, matrix


def import_catapro(
    data_path: str | Path, features_path: str | Path, output_dir: str | Path,
    *, test_fold: str, val_fold: str, id_column: str = "",
    trust_pickle: bool = False, overwrite: bool = False,
) -> dict:
    data_path, features_path, output = Path(data_path), Path(features_path), Path(output_dir)
    names = ("samples.csv", "features.npz", "import_manifest.json")
    if not overwrite and any((output / name).exists() for name in names):
        raise FileExistsError(f"Output files exist in {output}; use --overwrite to replace them")
    samples = _read_samples(data_path, test_fold, val_fold, id_column)
    ids, matrix = _read_features(features_path, trust_pickle=trust_pickle)
    lookup = {sample_id: i for i, sample_id in enumerate(ids.tolist())}
    missing = [sample["sample_id"] for sample in samples if sample["sample_id"] not in lookup]
    if missing:
        raise ValueError(f"Feature cache lacks {len(missing)} sample IDs, e.g. {missing[:5]}")
    order = [lookup[sample["sample_id"]] for sample in samples]
    selected = matrix[order]
    manifest = {
        "schema": "newkcat.catapro-import.v1", "test_fold": _fold(test_fold),
        "val_fold": _fold(val_fold), "original_folds_preserved": True,
        "new_homology_clustering_performed": False,
        "target": "log10(kcat / s^-1)",
        "counts": {split: sum(row["split"] == split for row in samples)
                   for split in ("train", "val", "test")},
        "inputs": {
            "data": {"path": str(data_path.resolve()), "sha256": _sha256(data_path)},
            "features": {"path": str(features_path.resolve()), "sha256": _sha256(features_path)},
        },
        "feature_order": {"protein": [0, 1024], "molt5": [1024, 1792], "maccs": [1792, 1959]},
        "pickle_columns_aligned_by_label": features_path.suffix.lower() in {".pkl", ".pickle"},
    }
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".import-", dir=output) as temporary:
        temp = Path(temporary)
        with (temp / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        np.savez_compressed(
            temp / "features.npz", sample_ids=np.asarray([row["sample_id"] for row in samples]),
            protein=selected[:, :1024], molt5=selected[:, 1024:1792], maccs=selected[:, 1792:],
        )
        # Perform actual dataset checks before publishing either file. This includes
        # exact sequence leakage, chemistry, feature alignment, and target shape.
        from .data import KcatDataset
        KcatDataset(temp / "samples.csv", temp / "features.npz")
        (temp / "import_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for name in names:
            (temp / name).replace(output / name)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-fold", required=True)
    parser.add_argument("--val-fold", required=True)
    parser.add_argument("--id-column", default="", help="Default: official unnamed CSV index column")
    parser.add_argument("--trust-pickle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = import_catapro(
        args.data, args.features, args.output, test_fold=args.test_fold, val_fold=args.val_fold,
        id_column=args.id_column, trust_pickle=args.trust_pickle, overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
