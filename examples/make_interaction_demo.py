"""Create tiny SYNTHETIC caches for CLI integration testing, not model evaluation.

Usage: python examples/make_interaction_demo.py --output /tmp/interaction-demo
The production cache command always uses pretrained encoders, not these fixtures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import yaml

from newkcat.interaction.data import create_cache


class _DemoProtein:
    metadata = {"kind": "synthetic_integration_fixture", "hidden_size": 32}

    def encode(self, sequence):
        seed = int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return {"residue_states": rng.standard_normal((len(sequence), 32)).astype(np.float32),
                "metadata": self.metadata}


class _DemoReaction:
    metadata = {"kind": "synthetic_integration_fixture", "hidden_size": 16}

    def encode(self, text):
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        atoms = np.array([c.isalpha() or c == "*" for c in text], dtype=bool)
        roles = np.array([0 if i < text.index(">") else 2 for i in range(len(text))])
        roles[np.array([c == ">" for c in text])] = 3
        pairs = [(i, i) for i, active in enumerate(atoms) if active]
        types = [1] * len(pairs)
        for i in range(len(text) - 1):
            if atoms[i] and atoms[i + 1] and roles[i] == roles[i + 1]:
                pairs.extend([(i, i + 1), (i + 1, i)])
                types.extend([2, 2])
        return {
            "reaction_states": rng.standard_normal((len(text), 16)).astype(np.float32),
            "roles": roles.astype(np.int64), "atom_mask": atoms, "topology_mask": atoms,
            "edge_index": np.array(pairs, dtype=np.int64).T,
            "edge_types": np.array(types, dtype=np.int64), "metadata": self.metadata,
        }


def make_demo(output: str | Path):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    residues = "ACDEFGHIKLMN"
    for i, amino_acid in enumerate(residues):
        rows.append({
            "sample_id": f"synthetic-{i:02}", "sequence": f"MA{amino_acid}KAAAK",
            "smiles": "CCO", "reaction_smiles": "CCO>>CC=O" if i % 2 else "CC*>>C=C*",
            "log10_kcat": (i % 5 - 2) * 0.3,
            "split": "train" if i < 6 else "val" if i < 9 else "test",
        })
    samples = output / "samples.csv"
    with samples.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    create_cache(samples, output / "cache", _DemoProtein(), _DemoReaction())
    config = {
        "data": {"samples": "samples.csv", "cache": "cache"},
        "model": {"protein_dim": 32, "reaction_dim": 16, "d_model": 16,
                  "nhead": 4, "num_layers": 1, "dim_feedforward": 32,
                  "head_hidden_dim": 32, "head_bottleneck_dim": 16, "dropout": 0.0},
        "training": {"seed": 668, "batch_size": 1, "grad_accum_steps": 4,
                     "epochs": 2, "patience": 2, "lr": 0.001, "device": "cpu"},
        "output_dir": "train_run",
    }
    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return output / "config.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(make_demo(parser.parse_args().output))
