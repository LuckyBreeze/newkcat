"""Create synthetic fixtures for the complete reaction-sequence CLI workflow.

Run ``python examples/make_sequence_demo.py`` after installing ``.[sequence]``.
The generated labels, protein vectors, MolT5 vectors, and MACCS-shaped bits are
arbitrary synthetic data. No generated metric measures biological performance.
The mapped substitutions are balanced representation examples, not measurements
or assertions about enzyme-catalysed reactions. Existing files are never replaced.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from newkcat.sequence.chemistry import normalize_reaction, prepare_reaction
from newkcat.sequence.prepare import prepare_sequences
from newkcat.sequence.runner import load_config, validate


SEED = 668
MAX_LENGTH = 256


def _substitution(carbons: int, leaving: str) -> tuple[str, str]:
    """An atom-, hydrogen-, and charge-balanced halide/hydroxide substitution."""
    chain = "[CH3:1]" + "".join(f"[CH2:{i}]" for i in range(2, carbons + 1))
    halide, oxygen = carbons + 1, carbons + 2
    reaction = (
        f"{chain}[{leaving}:{halide}].[OH-:{oxygen}]"
        f">>{chain}[OH:{oxygen}].[{leaving}-:{halide}]"
    )
    return "C" * carbons + leaving, reaction


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_demo(output: str | Path = "runs/sequence_demo") -> dict:
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Demo output must be an empty directory: {output}")

    reactions = [
        _substitution(1, "Br"),
        _substitution(2, "Br"),
        _substitution(3, "Br"),
        ("*CBr", "[*:1][CH2:2][Br:3].[OH-:4]>>[*:1][CH2:2][OH:4].[Br-:3]"),
        _substitution(1, "Cl"),
        _substitution(2, "Cl"),
        _substitution(1, "I"),
        _substitution(2, "I"),
    ]
    # These extra reactions are independent of every downstream split. The MLM
    # corpora therefore exercise transfer without reusing held-out test chemistry.
    corpus_reactions = {
        "train": [_substitution(n, halide)[1]
                  for n, halide in ((3, "Cl"), (4, "Cl"), (4, "Br"), (5, "Br"))],
        "val": [_substitution(n, halide)[1] for n, halide in ((5, "Cl"), (6, "Br"))],
    }
    # Check the optional converter before writing any files, rather than silently
    # generating a fallback dataset when the dependency is unavailable.
    corpora = {split: [prepare_reaction(text) for text in values]
               for split, values in corpus_reactions.items()}
    downstream_reactions = {normalize_reaction(reaction) for _, reaction in reactions}
    for representation in ("reaction_smiles", "sr_smiles"):
        train_texts = {item[representation] for item in corpora["train"]}
        val_texts = {item[representation] for item in corpora["val"]}
        if train_texts & val_texts:
            raise ValueError(f"Synthetic MLM {representation} train/val overlap")
    if downstream_reactions & {
        item["reaction_smiles"] for values in corpora.values() for item in values
    }:
        raise ValueError("Synthetic MLM corpus overlaps downstream chemistry")

    rng = np.random.default_rng(SEED)
    alphabet = np.asarray(list("ACDEFGHIKLMNPQRSTVWY"))
    labels = ("-1.20", "0.25", "0.60", "1.50", "-0.40", "2.00", "-0.10", "1.10")
    records = []
    for i, (substrate, reaction) in enumerate(reactions):
        records.append({
            "sample_id": f"synthetic_sequence_{i:03d}",
            "sequence": "M" + "".join(rng.choice(alphabet, size=48 + i).tolist()),
            "smiles": substrate, "log10_kcat": labels[i],
            "split": "train" if i < 4 else "val" if i < 6 else "test",
            "group_id": f"synthetic_group_{i:03d}",
            "reaction_smiles": normalize_reaction(reaction), "mapped_rxn": reaction,
        })
    # Different lengths guarantee exact sequence separation; these random groups
    # are not a substitute for real homology-clustered experimental splits.
    n = len(records)
    protein = rng.normal(0, 0.25, size=(n, 1024)).astype(np.float32)
    molt5 = rng.normal(0, 0.25, size=(n, 768)).astype(np.float32)
    maccs = rng.integers(0, 2, size=(n, 167), dtype=np.int8)
    maccs[:, 0] = 0

    output.mkdir(parents=True, exist_ok=True)
    source = output / "source_samples.csv"
    _write_csv(source, records)
    report = prepare_sequences(source, output_dir=output / "data",
                               max_length=MAX_LENGTH, on_error="error")
    # Deliberate reordering checks that the pipeline joins caches by sample ID.
    order = rng.permutation(n)
    np.savez_compressed(
        output / "features.npz",
        sample_ids=np.asarray([row["sample_id"] for row in records], dtype="U")[order],
        protein=protein[order], molt5=molt5[order], maccs=maccs[order],
    )

    arms = {
        "baseline": ("catapro_features", "mean"),
        "substrate": ("substrate_smiles", "mean"),
        "reaction": ("reaction_smiles", "mean"),
        "sr": ("sr_smiles", "mean"),
        "sr_local": ("sr_smiles", "edit_conditioned"),
        "sr_pretrained": ("sr_smiles", "mean"),
    }
    for name, (encoder, pooling) in arms.items():
        config = {
            "data": {"samples": "data/samples.csv", "features": "features.npz",
                     "missing_reaction": "error"},
            "model": {"chemical_encoder": encoder, "pooling": pooling,
                      "d_model": 32, "nhead": 4, "num_layers": 1,
                      "dim_feedforward": 64, "head_dim": 32,
                      "max_length": MAX_LENGTH, "dropout": 0.0},
            "training": {"seed": SEED, "batch_size": 2, "epochs": 2,
                         "patience": 2, "lr": 0.001, "num_threads": 1,
                         "device": "cpu", "deterministic": True},
            "output_dir": f"models/{name}",
        }
        if name == "sr_pretrained":
            config.update(encoder_checkpoint="mlm_output/best_encoder.pt", freeze_encoder=True)
        path = output / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        # Validation checks all generated rows and cache alignment, but does not
        # load the future pretraining checkpoint or fit any model.
        validate(load_config(path))

    corpus_directory = output / "mlm"
    corpus_directory.mkdir()
    corpus_paths = {}
    for split, values in corpora.items():
        for name, representation in (("sr", "sr_smiles"), ("reaction", "reaction_smiles")):
            path = corpus_directory / f"{split}_{name}.txt"
            path.write_text("".join(item[representation] + "\n" for item in values),
                            encoding="utf-8")
            corpus_paths[f"{split}_{name}"] = str(path.relative_to(output))

    result = {
        "synthetic": True, "seed": SEED, "output": str(output),
        "counts": {"train": 4, "val": 2, "test": 2},
        "prepared_rows": report["n"], "reaction_failures": report["failed"],
        "wildcard_samples": 1, "model_configs": [f"{name}.yaml" for name in arms],
        "mlm_corpora": corpus_paths,
        "mlm_counts": {split: len(values) for split, values in corpora.items()},
        "mlm_downstream_reaction_overlap": 0,
        "warning": "SYNTHETIC pipeline smoke only: arbitrary labels and random feature caches. "
                   "Metrics cannot support claims about kcat prediction or biological activity. "
                   "A wildcard is retained as an unknown atom; its chemistry is not recovered.",
    }
    (output / "DEMO_INFO.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/sequence_demo"))
    args = parser.parse_args()
    try:
        result = make_demo(args.output)
    except (ValueError, FileExistsError, ImportError) as error:
        parser.exit(2, f"make_sequence_demo: {error}\n")
    print(json.dumps(result, indent=2))
