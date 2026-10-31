"""Synthetic full-row experiment, including explicit-H sr conversion failures.

Every arm retains the same eight sample IDs, targets and splits. These arbitrary
labels and random feature vectors demonstrate software behaviour, not accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from make_sequence_demo import make_demo
from newkcat.sequence.prepare import prepare_sequences


def make_full_row_demo(output: Path) -> dict:
    output = output.resolve()
    make_demo(output)
    with (output / "source_samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # The net reaction is atom- and charge-balanced. The isolated proton is
    # explicit on the left and becomes an atom H count on the right; standard
    # sr conversion cannot merge these unequal explicit-node inventories.
    for index in (0, 4, 6):
        rows[index].update(
            smiles="OO", reaction_smiles="OO.[Cl-].[H+]>>O.OCl",
            mapped_rxn="[OH:1][OH:2].[Cl-:3].[H+]>>[OH2:1].[OH:2][Cl:3]",
        )
    source = output / "full_source_samples.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired = prepare_sequences(source, output_dir=output / "full_data", max_length=256,
                                on_error="reaction")
    independent = prepare_sequences(source, output_dir=output / "reaction_only_data",
                                     max_length=256, reaction_only=True)
    assert paired["n"] == independent["n"] == 8
    assert paired["fallback_reaction_rows"] == 3
    for name, encoder, pooling, policy, dataset in (
        ("full_reaction", "reaction_smiles", "mean", "error", "full_data"),
        ("full_sr", "sr_smiles", "mean", "reaction", "full_data"),
        ("full_sr_local", "sr_smiles", "edit_conditioned", "reaction", "full_data"),
        ("reaction_only", "reaction_smiles", "mean", "error", "reaction_only_data"),
    ):
        config = yaml.safe_load((output / "sr.yaml").read_text())
        config["data"].update(samples=f"{dataset}/samples.csv", missing_reaction=policy)
        config["model"].update(chemical_encoder=encoder, pooling=pooling)
        config["output_dir"] = f"models/{name}"
        (output / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    result = {"synthetic": True, "n": 8, "sr_rows": 5, "full_reaction_fallback_rows": 3,
              "rows_dropped": 0, "output": str(output)}
    (output / "FULL_ROW_DEMO_INFO.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/sequence_full_demo"))
    args = parser.parse_args()
    print(json.dumps(make_full_row_demo(args.output), indent=2))
