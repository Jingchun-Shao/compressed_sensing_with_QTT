# Code organization

The active MATLAB code retains the structure used to produce the reported
results:

- `src/tensor`: tensor contraction and TT-SVD utilities.
- `src/recovery`: alternating linear scheme recovery routines.
- `experiments/rip/paper_sliding_window`: empirical RIP scaling studies.
- `experiments/rip/baselines`: the starting Fourier/random-TT comparison.
- `experiments/initialization/paper_initialization_error`: initialization
  scaling studies.
- `experiments/recovery/paper_recovery`: full recovery studies.
- `results/data`: final synthetic experiment data.
- `results/figures`: final portable figures.
- `sanity checks`: small numerical and visualization checks.

The publication experiments construct their sliding-window signal models in
local helper functions, keeping each experiment self-contained. Run
`setup_qtt_paths` from this directory (or after adding this directory to the
MATLAB path) before running an experiment.

Historical scripts, autosaves, raw logs, checkpoints, and intermediate figure
editing utilities are intentionally excluded from the public code tree.
