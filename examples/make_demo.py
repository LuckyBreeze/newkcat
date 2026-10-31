"""Create small, explicitly synthetic fixtures for checking the complete pipeline.

No feature or target generated here is a biological measurement or a PLM output.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys


def make_demo(output: Path, seed: int = 17, overwrite: bool = False) -> None:
    paths = [output / name for name in ("samples.csv", "features.npz", "centers.jsonl")]
    if any(path.exists() for path in paths) and not overwrite:
        raise FileExistsError(f"Demo files already exist in {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    smiles_pool = [
        "CCO", "CC(=O)O", "CC(N)C(=O)O", "c1ccccc1O", "CCOC(=O)C",
        "OCC(O)CO", "CC(=O)NC", "CNC(=O)C", "CC(O)C", "OC1CCCCC1",
        "C[C@H](O)C(=O)O", "NCC(=O)O",
    ]
    n = 48
    protein = rng.normal(0, 0.25, (n, 1024)).astype(np.float32)
    molt5 = rng.normal(0, 0.25, (n, 768)).astype(np.float32)
    maccs, records, centers = [], [], []
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    for i in range(n):
        smiles = smiles_pool[i % len(smiles_pool)]
        mol = Chem.MolFromSmiles(smiles)
        fingerprint = np.asarray(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
        maccs.append(fingerprint)
        split = "train" if i < 32 else "val" if i < 40 else "test"
        sample_id = f"synthetic_{i:03d}"
        sequence = "".join(rng.choice(alphabet, size=80).tolist())
        target = float(0.5 + 0.4 * protein[i, 0] - 0.2 * molt5[i, 0]
                       + 0.01 * mol.GetNumAtoms())
        records.append({
            "sample_id": sample_id, "sequence": sequence, "smiles": smiles,
            "log10_kcat": target, "split": split, "ec": f"1.{i % 3 + 1}.1.1",
            "group_id": f"synthetic_group_{i:03d}",
        })
        if i % 5 == 0:
            continue  # Deliberately exercise the missing-center fallback.
        scores = {}
        for atom in mol.GetAtoms():
            number = 1001 + 13 * atom.GetIdx()
            atom.SetAtomMapNum(number)
            scores[str(number)] = 0.8 if atom.GetAtomicNum() in (7, 8) else 0.1
        centers.append({
            "sample_id": sample_id,
            "mapped_smiles": Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True),
            "scores": scores, "kind": "predicted",
            "source": "synthetic-demo-not-biological", "uses_ec": False,
        })
    with paths[0].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # IDs, not row order, define alignment. Deliberately permute the feature table.
    order = rng.permutation(n)
    np.savez_compressed(
        paths[1], sample_ids=np.asarray([r["sample_id"] for r in records])[order],
        protein=protein[order], molt5=molt5[order],
        maccs=np.asarray(maccs, dtype=np.float32)[order],
    )
    with paths[2].open("w", encoding="utf-8") as handle:
        for record in centers:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Created SYNTHETIC pipeline fixtures in {output.resolve()}: 32 train / 8 val / 8 test")
    print("These random vectors and labels cannot validate kcat prediction performance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    make_demo(args.output, args.seed, args.overwrite)
