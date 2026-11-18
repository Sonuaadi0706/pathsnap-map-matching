# Archived: first round with 28 features

This folder holds the first version of PathSnap, kept unchanged for the record. It is **not** used by the
deployed system, the API or the technical report's main results.

| What | Value |
|---|---|
| Candidate table | arcs within 80 m, at most 12 per fix, 28 features (`BASE_FEATURES` in `pathsnap/candidates.py`) |
| Models | six binary classifiers (`models/*.joblib`) and a Bayesian-tuned LightGBM (`lgbm_tuned.joblib`) |
| Final system of that round | tuned LightGBM + HMM/Viterbi + re-alignment: student-test top-1 0.975 |

Two settings of the deployed 46-feature system were first found here, which is why the round is documented
(technical report, Section 4.3):

* the starting hyperparameters of the deployed LightGBM ranker are the best configuration of the Bayesian
  search in `results/optuna_best.json` (copied to `configs/start_hyperparameters.json`);
* the decoder values beta = 20 m and w = 0.25 come from the grid in `results/hmm_grid.csv`.

## Contents

| Folder | Contents |
|---|---|
| `models/` | the seven 28-feature models |
| `results/` | per-model CV / test results, Optuna trial log, HMM grid and ablation, re-alignment and error analysis of that round |
| `figures/` | the figures generated from those results |
| `scripts/` | `bayes_opt.py`, `hmm_eval.py`, `realign_eval.py` as they were |
| `configs/round1_config.json` | model + decoder configuration of that round |

## Reproducing this round

The scripts here expect the repository layout of commit `ba72858` (the last commit before the 46-feature
table became the only one). To re-run them:

```bash
git checkout ba72858
```

The archived models still load from the current code:
`Predictor(model_dir="archive/round1_28_features/models", config="archive/round1_28_features/configs/round1_config.json")`.
`scripts/validate_final.py` uses this to compare the deployed system with this round on the same trips.
