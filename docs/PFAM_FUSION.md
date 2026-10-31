# Pfam–CataPro: 2 × 2 reproducible experiment

This is an **isolated addition** under `src/newkcat/pfam_fusion`, not a modification to the existing graph-center CataPro model. Nothing in `main` is overwritten. Each branch contains its own selected config at `configs/pfam_experiment.yaml`.

## Factorial design

| Embedding source | Domain treatment | Branch |
|---|---|---|
| Random trainable 100-D | Unique Pfam IDs, sorted lexically, no positional encoding | `exp/pfam-random-set` |
| Random trainable 100-D | Ordered by start coordinate, repeated domains retained, trainable absolute positional encoding | `exp/pfam-random-ordered` |
| DeepBGC 100-D pretrained initialization, **fine-tuned** | Unique sorted Pfam IDs, no positional encoding | `exp/pfam-deepbgc-set` |
| DeepBGC 100-D pretrained initialization, **fine-tuned** | Ordered, repeated domains retained, positional encoding | `exp/pfam-deepbgc-ordered` |

The Pfam branch is `[B,Nmax]` -> `Embedding(100)` -> `Linear(100,256)` -> `1 × pre-LN Transformer Encoder (4 heads; FFN=512)` -> mask invalid query outputs -> `Flatten[Nmax*256]` -> `Linear(256) + LN + ReLU`. The output is `[B,256]`. The CataPro baseline follows the existing repository's `ProtT5(1024) -> BN`, `MolT5(768) -> BN`, `MACCS(167) raw`; concatenate `1024+768+167+256=2215`, then existing CataPro decoder `Linear(2215,256)->BN->Dropout->ReLU->Linear(256,1)`.

This is a **two-factor comparison**, not an isolated positional encoding experiment, because deduplication/ordering also differs. Follow-up positional-only ablation needs identical tokens with PE on/off.

## Real data prerequisites (do not fabricate)

User-supplied `train.csv` (19,194), `valid.csv` (2,698), `test.csv` (2,698) have `row_uid,sequence,log10kcat_max,split,cluster_id`. They **do not** include Pfam IDs nor cached global embeddings. Provide:

1. `data/pfam/pfam_hits.csv` with columns `row_uid,pfam_id,start,end`. Each row is one annotated domain occurrence. Coordinates are 1-based, start<=end. The same Pfam ID at different coordinates is kept in ordered mode. Only exact duplicate hit rows are deduplicated. No hits -> all-PAD zero Pfam branch (logged). Ordered mode requires coordinates. Verify Pfam 38.2 IDs against the older DeepBGC Pfam31 vectors by accession, not name guesses.
2. `data/pfam/features.npz` with Unicode `sample_ids` matching `row_uid` and arrays `protein[N,1024]`, `molt5[N,768]`, `maccs[N,167]`. Caches must be extracted from the actual baseline CataPro features and aligned by IDs; never infer features from the provided CSV.
3. For DeepBGC variants only, official **100-D** `pfam2vec.csv.gz` from [Merck/deepbgc v0.1.0](https://github.com/Merck/deepbgc/releases/download/v0.1.0/pfam2vec.csv.gz). DeepBGC loads `pfam_id` as index. This table is **not** the newer Pfam annotation database.

## Launch

```bash
pip install -e '.[catapro]'
# Download with curl/wget yourself and verify source; do not commit large data:
curl -fL https://github.com/Merck/deepbgc/releases/download/v0.1.0/pfam2vec.csv.gz -o data/pfam/pfam2vec.csv.gz
# Copy the supplied CSVs to the config locations, create REAL hits and CataPro cache
python -m newkcat.pfam_fusion.cli audit --config configs/pfam_experiment.yaml
python -m newkcat.pfam_fusion.cli train --config configs/pfam_experiment.yaml --output runs/pfam_experiment_seed17
python -m newkcat.pfam_fusion.cli evaluate --config configs/pfam_experiment.yaml --checkpoint runs/pfam_experiment_seed17/best.pt --output runs/pfam_experiment_seed17_test
```

Use `--init-catapro /path/to/official_state_dict.pt` optionally for a **shape-checked** transfer via existing `NewKcat.load_catapro_weights`: the original global feature columns are copied and the added 256 columns initialize to zero. No claim is made that this has been tested on the real split.

## Important invariants

* Training-only vocabulary: PAD=0, UNK=1, unseen valid/test IDs become UNK; a train-embedding mean substitutes for an otherwise untrained UNK vector. In DeepBGC mode, missing pretrained Pfams are randomly initialized and logged through coverage audits, not discarded.
* Nmax is derived **only from training split**, using 99th percentile of domain occurrences capped at 32; fixed across validation/test. Extra domains truncate with audit counts. Same fixed Nmax used across 4 variants.
* Self-attention masks PAD keys; PAD query outputs are zeroed **again after the encoder**, before flatten. All-PAD input returns exactly zero Pfam output.
* No Pfam target labels are invented; no test loss used for training or early stopping. Best validation checkpoint is reloaded for test.
* Test is held out until explicit `evaluate`. Each new output folder is refuse-to-overwrite.
* `audit` reads train/valid/test **only for descriptive quality checks**. This should not be used to tune hyperparameters from test statistics.
* This repo does not provide Pfam scanning itself; the fact that a sequence has Pfam annotation is not proof it is covered by the older DeepBGC vector lookup.

## Caveats

The set mode's Transformer without positional coding is permutation equivariant, but **flatten is not permutation invariant**. We sort unique IDs to ensure deterministic input slots; strict permutation invariance would require a set pooling operator. The pretrained DeepBGC branch as implemented **fine-tunes** the pretrained weights; it is not frozen. All results require controlled training and multiple seeds.

