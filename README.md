# AML Mule Account Detection — Phase 2
## Team Tarang | RBIH-NFPC Competition | IIT National Challenge

## Environment
- Python 3.12 (Kaggle CPU-only, 30 GB RAM, No Accelerator)
- Packages: lightgbm, optuna, scikit-learn, pandas, pyarrow, dill, shap, psutil

## Install Dependencies
pip install lightgbm optuna shap dill pyarrow psutil scikit-learn pandas numpy

## Dataset
Place at: /kaggle/input/datasets/abhyudayrbih/rbih-nfpc-phase-2/
Format: Apache Parquet (Snappy). Total: 720 files, 16.2 GB.

## Steps to Reproduce
Run notebook team-tarang-rbih-nfpc-phase2.ipynb cells in order:

Block 1  - Setup, imports, constants (~2 min)
Block 2  - Load all static tables (~1 min)
Block 3A - Stream transactions parts 1-200 (~45 min)
Block 3B - Stream transactions parts 201-396 (~45 min)
Block 3C - Compute transaction features (~5 min)
Cleanup  - del agg + malloc_trim to free RAM
Block 4  - transactions_additional placeholder (RAM constraint)
Block 5  - Static feature engineering (~3 min)
Block 6  - Temporal features + IoU windows (~20 min)
Block 7  - Graph features + propagation (~30 min)
Cleanup  - del agg + malloc_trim to free RAM
Block 8  - Master feature join (~5 min)
Block 9  - Red herring audit + label noise (~15 min)
Fix cell - Remove leaky features (OOF AUC=1.0 detected and resolved)
Block 10 - LightGBM + Optuna 30 trials + 5-fold CV (~35 min)
Block 11 - Isolation Forest + Random Forest (~15 min)
Block 12 - Ensemble + calibration + submission.csv (~5 min)
Block 13 - Visualisations (~3 min)
Block 14B - Performance summary (~1 min)

Total runtime: approximately 4 hours on Kaggle CPU.

## Crash Recovery
All expensive blocks save checkpoints to /kaggle/working/.
On crash: re-run Block 1 + Block 2 only (3 min total).
All other blocks detect checkpoints and skip automatically.

## Approach
- 7-layer feature engineering, 195 features, all 13 mule patterns covered
- Red herring detection: adversarial validation + direction checks + leakage detection
- 11 features removed (3 injected traps + 8 target-encoding leakage features)
- 4-model ensemble: LightGBM + Random Forest + Isolation Forest + Meta-learner
- Isotonic calibration for true probability output
- Weekly z-score temporal IoU windows for 1,052 test accounts

## Model Files
NOTE: Trained model pkl files were generated during training but could not be
included due to Kaggle session constraints. All results are visible in the
notebook outputs and can be fully reproduced by running the notebook end-to-end.

## Results
LightGBM CV AUC      : 0.9553 +/- 0.0076
Random Forest AUC    : 0.9520
Isolation Forest AUC : 0.8603
Final Ensemble AUC   : 0.9541
Best F1              : 0.7150 @ threshold 0.7473
Temporal windows     : 1,052 / 64,062 test accounts
Features used        : 195 from 207 candidates
Red herrings removed : 11 (3 injected + 8 leakage)
