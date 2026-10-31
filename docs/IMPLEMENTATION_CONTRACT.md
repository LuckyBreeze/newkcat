# Implementation and API contract

This is a research implementation, not a claim of improved kcat accuracy.
Model variants share data, splits, global feature normalization and training protocol.
The first implemented version uses cached global protein features; residue-level
cross-attention, learned reaction generation, and 3D coordinates are not prerequisites.

## Data

`samples.csv`: `sample_id,sequence,smiles,log10_kcat,split` (split=train/val/test).
Optional `ec`, `group_id` (all records from a protein cluster/parent stay together).
IDs are unique; one sequence and one sequence-SMILES pair cannot cross splits.
Missing targets are allowed for inference only. Never invent activity labels.

`features.npz` (allow_pickle=False): Unicode `sample_ids`, float `protein` [N,1024],
`molt5` [N,768], `maccs` [N,167]. Join by IDs, never by file row order.
MACCS uses 167 RDKit bits, including bit 0. No learned feature standardizer sees test.

`centers.jsonl`: each record has `sample_id`, `mapped_smiles`, `scores` (JSON object
atom-map-number -> score), `kind` (`predicted` or `reference`), `source` (nonempty),
`uses_ec` (bool). Mapped SMILES must have unique positive map IDs on every atom and
must describe the same stereochemical molecule as samples.csv after maps are removed.
Score keys must match the full set of atom maps; values finite and in [0,1].
No automatic ambiguous isomorphism matching of external atom-index arrays.
Missing records become an explicitly missing center; invalid records fail loudly.
Reference centers are an oracle diagnostic and require explicit permission in config.
EC-informed center runs require an EC-enabled model and an equal-information baseline.

## Shared API

`types.GraphBatch` defines all tensors and padded-index conventions.

`chem.py`: `ATOM_DIM`, `BOND_DIM`, `featurize_smiles(smiles)` returning a record with
atom_features, edge_index, edge_features, distances, atom_maps. Chemical identities
ignore mapping numbers but preserve charge/isotope/stereo. Features should use
absolute CIP labels, not atom-order-dependent stereochemical tags.

`data.py`: `KcatDataset(samples_path, features_path, centers_path=None,
splits=None, ec_vocab=None, require_targets=True)`; `fit_ec_vocab(records)`;
`collate_graphs(items) -> GraphBatch`; `.records` list of dictionaries; `.atom_dim`,
`.bond_dim`; `.center_uses_ec`; `.has_reference_centers`; `.ec_vocab`.
Records not selected by split are not featurized. Full split metadata is validated.
Features can contain extra rows but every selected record must match exactly once.

`model.py`: `ModelConfig` dataclass with mode baseline/global/center/shell/conditioned/
learned, atom_dim, bond_dim, hidden_dim=128, num_layers=3, local_dim=64, dropout=0.0,
use_ec2=False, ec_vocab_size=1, ec_dim=16. `NewKcat(config)`; `forward(batch,
return_diagnostics=False)` returns [B] predictions or (predictions, diagnostics).
`load_catapro_weights(state_dict)` copies official key-compatible BN/decoder/out
weights and zeros new decoder input columns; eval predictions then equal baseline.

`position.py`: center-relative distance profiles with five bins: 0,1,2,>=3,
disconnected. Normalization is over candidate-center mass, not nearest-center
probability. Exact zero scores and missing records deactivate center-dependent
branches *after* all biased layers/normalization. Center mass statistics are scores,
not calibrated reliability. Padded nodes/edges must never influence outputs.

## Model semantics

Original 1024 ProtT5 -> BN and 768 MolT5 -> BN; 167 MACCS direct; concat ->
Linear(1959 [+64] [+ec_dim],256) -> BN -> Dropout -> ReLU -> Linear(256,1).
No raw protein-vector-plus-attention-vector residual. Five-bin graph-local branch
uses LayerNorm, not batch-dependent normalization. Global/learned modes ignore
centers completely; center mode only weighted-pools; shell mode uses positional
features plus ordered shell slots; conditioned mode additionally weights slots
with protein context. Learned mode uses the same number of learned atom queries
without center labels as a capacity control. Share modules across graph variants
where possible. No scalar output head may produce accidental [B,B] losses.

## Training/evaluation

Train and validate only in fit; explicit separate evaluate command for test.
Early stop by validation RMSE, reload best checkpoint. Target is log10(s^-1).
No refitting or threshold selection on test. Checkpoint includes ModelConfig,
EC vocabulary, config, data hashes and split IDs. Use torch.load(weights_only=True)
for own primitive/tensor checkpoints. Reject overwrite by default.
Train batches must have >=2 examples for authentic CataPro BN; merge a final singleton
into the preceding batch rather than silently drop records. Validation uses eval().
Metrics include RMSE/MAE/R2/Pearson/Spearman; undefined correlations become null.
Seeded runs and explicit CPU/CUDA selection; no background large GPU jobs.

## Required checks

Graph atom permutation, swapped undirected endpoints, padding/batch invariance in eval,
disconnected distance bins, all-zero/missing centers, score scaling vs absolute mass,
bad scores/maps/feature IDs, stereo mismatch, split/sequence/group leakage, no EC
vocab fitting on validation/test, no test access during training, prediction shape,
finite backward gradients, reference information flags, original checkpoint transfer,
tiny synthetic CLI smoke run (explicitly not biological evidence).
