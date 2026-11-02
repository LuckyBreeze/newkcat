# Random-set optimization: AdamW + Warmup/Cosine + Xavier + Dropout

**Parent branch:** `exp/pfam-random-set`  
**New branch:** `exp/pfam-random-set-warmcos-xavier-b8`  
Original `main` and `exp/pfam-random-set` are unchanged. The token processing and original CataPro BN, MolT5 and MACCS feature paths are preserved.

## Configuration (single experiment, not a grid search)

| Setting | New experiment |
|---|---|
| Source / Pfam | random embedding, deduplicated alphabetical Pfam ID set, no PE |
| Batch size | 8 (actual forward batch, BatchNorm enabled) |
| Learning rate | 5e-5 |
| Optimizer | AdamW |
| Weight decay | 1e-3 on trainable 2-D+ matrices; biases and norm scalars excluded |
| Transformer dropout | 0.1 |
| NEW Pfam FC dropout | 0.1, after Flatten -> Linear -> LayerNorm -> ReLU |
| Initialization | Xavier uniform for Pfam embedding, projection, Transformer matrices, Pfam FC, and newly sized CataPro fusion Linear; all new biases zero |
| Preserved initialization | Existing CataPro BatchNorm parameters/stats and original prediction head are left unchanged |
| Warmup/Cosine | 3 warmup epochs; per-optimizer-step cosine to 10% of base LR at planned epoch 50 |
| Epoch ceiling | 50; early stopping patience=30 (may stop before cosine reaches minimum) |
| Checkpoint selection | Best **complete-validation** RMSE (equivalent to max full-validation R² for fixed labels) |
| Default seed | 668 (reconfigure for seed 42/1024/65535) |

**Important:** This changes several controls at once, so it is a *combined training recipe*, not evidence identifying which individual change improves generalization. Compare on the same cluster split, seed and model input data. A separate factorial ablation is required for isolated effects. An official CataPro checkpoint can optionally be loaded via `--init-catapro`; that changes initialization and is a separate experiment, not used automatically. If loaded, it overwrites the new fusion weights' original-input columns and zeroes added Pfam columns as defined by the existing transfer method.

## Data needed

Follow the original four-way experiment's `docs/PFAM_FUSION.md`. Provide real `train.csv`, `valid.csv`, `test.csv`, `pfam_hits.csv` and `features.npz` aligned through `row_uid`. The code never generates fake Pfam annotations or ProtT5/MolT5 embeddings.

## Commands

```bash
python -m pip install -e '.[catapro,test]'
python -m newkcat.pfam_fusion.cli audit --config configs/pfam_experiment.yaml
python -m newkcat.pfam_fusion.cli train \
  --config configs/pfam_experiment.yaml \
  --output runs/random_set_optim_cluster40_seed668
python -m newkcat.pfam_fusion.cli evaluate \
  --config configs/pfam_experiment.yaml \
  --checkpoint runs/random_set_optim_cluster40_seed668/best.pt \
  --output runs/random_set_optim_cluster40_seed668_test
```

To repeat with a new seed, change `training.seed` and the output folder. Never reuse an existing output folder.

## Loss logging and fair comparison

Every epoch logs one JSON row to STDOUT, appends to `history.csv` (flushed so intermediate results survive preemption), and includes the full history in `report.json`.

- `train_online_loss`: example-weighted train-mode MSE accumulated during optimizer updates, with dropout and per-batch BN. This is **not** directly comparable with validation eval-mode loss.
- `train_loss`: full training-set **eval-mode MSE at the end of the epoch** using the final weights, no dropout, moving BN statistics.
- `val_loss`: full validation-set **eval-mode MSE**, exactly the square of `val_rmse`.
- `val_r2`, `val_pcc`, `val_mae`, `val_rmse`: calculated on the entire validation set, **not** averaged per batch.
- `lr_first` and `lr_last`: exact learning rates used for optimizer steps in this epoch, not a scheduler preview.
- `best_epoch_so_far`, `early_stop_counter`: checkpoint selection tracking.

Other artifacts: `best.pt` (checkpoint at validation optimum), `best_valid_predictions.csv` (row_uid/target/prediction at optimum), `report.json`. The separate explicit test command emits `predictions.csv` and `metrics.json` without refitting or selecting on test.

### Scientific caveats

- AdamW weight decay is decoupled and **not identical** to adding an L2 loss penalty in Adam.
- With batch=8, BN still uses relatively small batch statistics. Comparing train and validation *eval-mode* losses reduces one diagnostic confound, but cannot alone prove overfitting.
- Validation metric selection is full-dataset; do not use batchwise R².
- Parameter changes are all specified in the YAML. Never claim performance improvement until real data runs have completed.
- `Nmax` and Pfam vocab continue to be fitted from training only; unchanged sample/split processing.
