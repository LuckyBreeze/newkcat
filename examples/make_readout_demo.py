"""Create SYNTHETIC, non-biological fixtures for the new readout CLI.

All labels and protein features are artificial. No pretrained weights or private
datasets are downloaded. Run from a fresh output directory after installing -e .
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from newkcat.sequence.prepare import prepare_sequences


def make_demo(output: Path) -> dict:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(668)
    records = []
    for index in range(9):
        chain = "C" * (1 + index % 4)
        if index == 0:
            chain = "*" + chain
        records.append({
            "sample_id": f"synthetic_{index}",
            "sequence": "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 30 + index)),
            "smiles": chain + "Br",
            "reaction_smiles": chain + "Br.[OH-]>>" + chain + "O.[Br-]",
            "log10_kcat": float(rng.normal()),
            "split": "train" if index < 5 else "val" if index < 7 else "test",
            "group_id": f"synthetic_cluster_{index}",
        })
    source = output / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    prepare_sequences(source, output_dir=output / "data", reaction_only=True,
                      max_length=96, on_error="error")
    order = rng.permutation(len(records))
    features = rng.normal(size=(len(records), 1024)).astype(np.float32)
    np.savez_compressed(output / "features.npz",
                        sample_ids=np.asarray([r["sample_id"] for r in records])[order],
                        protein=features[order])
    models = {
        "r0": {},
        "protein1024": {"protein_readout": "full"},
        "mean_max": {"readout": "mean_max"},
        "queries": {"readout": "queries"},
        "queries_full": {"readout": "queries", "protein_readout": "full",
                         "chemical_dim": None},
        "conditioned": {"readout": "queries", "protein_readout": "full",
                        "chemical_dim": None, "condition_queries": True},
        "fusion": {"readout": "queries", "protein_readout": "full",
                   "chemical_dim": None, "fusion": "transformer"},
        "flatten": {"readout": "flatten", "protein_readout": "full",
                    "chemical_dim": None},
    }
    for name, changes in models.items():
        model = {"chemical_encoder": "reaction_smiles", "d_model": 16, "nhead": 4,
                 "num_layers": 1, "dim_feedforward": 32, "head_dim": 32,
                 "chemical_dim": 32, "max_length": 96, "dropout": 0.0,
                 "num_queries": 3, **changes}
        config = {
            "data": {"samples": "data/samples.csv", "features": "features.npz",
                     "missing_reaction": "error"},
            "model": model,
            "training": {"seed": 668, "batch_size": 1, "grad_accum_steps": 2,
                         "epochs": 2, "patience": 2, "lr": 0.001,
                         "num_threads": 1, "device": "cpu", "deterministic": True},
            "output_dir": f"models/{name}",
        }
        (output / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False),
                                            encoding="utf-8")
    info = {"synthetic": True, "n": 9, "counts": {"train": 5, "val": 2, "test": 2},
            "models": list(models), "output": str(output),
            "warning": "Random synthetic targets/features: pipeline checks only, not kcat results."}
    (output / "DEMO_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/readout_demo"))
    args = parser.parse_args()
    print(json.dumps(make_demo(args.output), indent=2))
