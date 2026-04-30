# Crisis-Period Trading Resilience and Dynamic MSR Reform: A Quantum Kernel Approach to EU ETS Carbon Price Forecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19913059.svg)](https://doi.org/10.5281/zenodo.19913059)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

## Overview

Replication archive for the manuscript:

> **"Crisis-Period Trading Resilience and Dynamic MSR Reform: A Quantum Kernel Approach to EU ETS Carbon Price Forecasting"**
>
> *Submitted to: Energy Policy (Elsevier)*

This repository contains all Python scripts, input data, output results, and auxiliary files required to fully reproduce the empirical results, tables, and figures reported in the manuscript.

---

## Repository Structure

```
carbon_qml_project/
│
├── scripts/                              # Analysis pipeline (run in order)
│   ├── 01_fold_generator.py	
│   ├── 1b_compute_naive_baselines.py  # Random Walk and Historical Mean baselines 
│   ├── 02_qksvr_core.py                 # QK-SVR model (PennyLane fidelity kernel)
│   ├── 03a_optuna_tuning.py             # Hyperparameter tuning — tree/SVM models
│   ├── 03b_tree_svm.py                  # XGBoost, LightGBM, RBF-SVM, Laplacian-SVM
│   ├── 03c_neural_nets.py               # EMD-LSTM, BiLSTM, GRU, Transformer
│   ├── 03d_merge.py                     # Merge all model predictions
│   ├── 04_statistical_validation.py     # DM tests, MCS, Romano-Wolf correction
│   ├── 05_interpretability_            
│   │   expressibility.py               # QKFM + SHAP + kernel expressibility
│   ├── 06_trading_simulation.py         # Trading simulation (Sharpe, Calmar, DA)
│   ├── 07a_ablation_qubits.py          # Ablation AB1: qubit count (N=4 vs N=6)
│   ├── 07b_ablation_depth.py           # Ablation AB2: circuit depth (L=1,2,3)
│   ├── 07c_ablation_entanglement.py    # Ablation AB3: entanglement topology
│   ├── 07d_ablation_reuploading.py     # Ablation AB4: data re-uploading
│   ├── 07e_ablation_merge.py           # Merge ablation results
│   ├── 08_export_manuscript_tables.py  # Export APA-formatted tables to xlsx
│   ├── 09_bootstrap_hedging_ci.py      # Bootstrap 95% CI for hedging savings (Table 10)
│   ├── 10_generate_figures.py          # Generate Figures 1–2 (300 dpi)
│   ├── compute_vif.py                  # Variance Inflation Factor analysis (Appendix C, Table C3)
│   └── descriptive_stats.py            # Descriptive statistics and unit root tests (Table 3)
│
├── data/
│   ├── master_dataset.csv              # Input: 1,302 daily observations (2019–2023)
│   └── README_data.md                  # Data dictionary and variable descriptions
│
├── results/                            # Script outputs (committed for reproducibility)
│   ├── metrics_fold_level.csv          # Fold-level RMSE/MAE/DA/Sharpe (999 rows)
│   ├── metrics_summary.csv             # Full-sample aggregate metrics (Table 5)
│   ├── crisis_subperiod.csv            # Regime-conditional metrics (Table 6)
│   ├── dm_tests.csv                    # Diebold-Mariano test statistics (Appendix C, Table C2)
│   ├── mcs_results.csv                 # Model Confidence Set results
│   ├── trading_simulation.csv          # Trading simulation by regime/horizon (Table 7; Appendix C, Table C1)
│   ├── trading_cumulative_returns.csv  # Cumulative returns time series
│   ├── feature_importance.csv          # QKFM + SHAP importance scores (Table 8)
│   ├── qkfm_raw.csv                    # Raw fold-level QKFM scores
│   ├── expressibility.csv              # Kernel expressibility metrics (Appendix A, Table A1)
│   ├── regime_ari.json                 # Kernel regime ARI
│   ├── ljungbox_diagnostics.csv        # Ljung-Box residual diagnostics
│   ├── vif_results.csv                 # VIF analysis (Appendix C, Table C3)
│   ├── descriptive_stats.csv           # Descriptive statistics (Table 3)
│   ├── policy_implications.csv         # Hedging savings point estimates
│   ├── benchmark_summary.csv           # Benchmark model RMSE summary
│   ├── ablation_ab1_qubits_summary.csv         # Ablation AB1 results (Appendix B, Table B2)
│   ├── ablation_ab2_depth_summary.csv          # Ablation AB2 results (Appendix B, Table B3)
│   ├── ablation_ab3_entanglement_summary.csv   # Ablation AB3 results (Appendix B, Table B4)
│   ├── ablation_ab4_reuploading_summary.csv    # Ablation AB4 results (Appendix B, Table B5)
│   ├── ablation_all_summary.csv                # Merged ablation results (Appendix B, Table B1)
│   └── bootstrap_ci_results.csv               # Bootstrap CI (parametric, n=10,000, seed=42; Table 10)
│
├── figures/                            # Generated figures (300 dpi)
│   ├── Figure1_regime_rmse_h5.png      # Figure 1: Regime-conditional RMSE (H=5)
│   └── Figure2_feature_importance_heatmap.png  # Figure 2: QKFM and TreeSHAP heatmap
│
├── tables/                             # APA-formatted Excel tables
│   ├── Table_A1_Expressibility.xlsx
│   ├── Table_B1_Ablation_Summary.xlsx
│   ├── Table_B2_Ablation_AB1_qubits.xlsx
│   ├── Table_B3_Ablation_AB2_depth.xlsx
│   ├── Table_B4_Ablation_AB3_entanglement.xlsx
│   ├── Table_B5_Ablation_AB4_reuploading.xlsx
│   ├── Table_C1_Trading_Simulation.xlsx
│   ├── Table_C2_DM_Tests.xlsx
│   └── Table_C3_VIF.xlsx
│
├── requirements.txt                    # Python dependencies (pip)
├── environment.yml                     # Conda environment specification
├── .gitignore                          # Git ignore rules
├── LICENSE                             # MIT License
├── CITATION.cff                        # Machine-readable citation metadata
├── CHANGELOG.md                        # Version history
├── CODEBOOK.md                         # Variable definitions and analysis decisions
└── REPRODUCE.md                        # Step-by-step reproduction guide
```

---

## Reproducing the Results

### 1. Environment Setup

**Option A — pip (recommended for most users):**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

**Option B — Conda:**
```bash
conda env create -f environment.yml
conda activate carbon_qml
```

> **Important:** Scripts must be run with **CPython 3.11**, not PyPy.
> On Windows, use the full CPython path if PyPy is the system default:
> ```powershell
> C:\Users\<user>\AppData\Local\Programs\Python\Python311\python.exe scripts\02_qksvr_core.py
> ```

---

### 2. Running the Pipeline

Execute scripts in the following order from the `scripts/` directory:

```bash
# Step 1: Generate walk-forward folds (~1 min)
python 01_fold_generator.py

# Step 2: Train and evaluate QK-SVR (~4–8 hours on CPU)
python 02_qksvr_core.py

# Step 3: Train classical benchmarks (3a–3c can run in parallel)
python 03a_optuna_tuning.py        # ~30–60 min
python 03b_tree_svm.py             # ~30 min
python 03c_neural_nets.py          # ~2–4 hours (GPU recommended)
python 03d_merge.py                # ~2 min

# Step 4: Statistical validation (~10 min)
python 04_statistical_validation.py

# Step 5: Interpretability and expressibility (~20 min)
python 05_interpretability_expressibility.py

# Step 6: Trading simulation (~5 min)
python 06_trading_simulation.py

# Step 7: Ablation studies (7a–7d can run in parallel)
python 07a_ablation_qubits.py
python 07b_ablation_depth.py
python 07c_ablation_entanglement.py
python 07d_ablation_reuploading.py
python 07e_ablation_merge.py       # ~2 min

# Step 8: Export manuscript tables (~2 min)
python 08_export_manuscript_tables.py

# Step 9: Bootstrap CI for Table 10 (~1 min)
python 09_bootstrap_hedging_ci.py --data_dir ../results --n_boot 10000 --seed 42

# Step 10: Generate figures (Figure 1 and Figure 2, 300 dpi)
python 10_generate_figures.py

# Auxiliary: Descriptive statistics and unit root tests (Table 3)
python descriptive_stats.py

# Auxiliary: VIF analysis (Appendix C, Table C3)
python compute_vif.py
```

> **Estimated total runtime:** 6–12 hours on a modern CPU (Intel Core i7/i9 or AMD Ryzen 7/9).
> Steps 02 and 03c are the bottlenecks. Step 02 (QK-SVR) benefits from multi-core
> parallelism (`n_jobs` parameter) but not GPU.

---

### 3. Table-to-Script Mapping

| Manuscript Table/Figure | Generated by | Output file |
|---|---|---|
| Table 2 (Correlations) | `08_export_manuscript_tables.py` | — (from `master_dataset.csv`) |
| Table 3 (Descriptive stats) | `descriptive_stats.py` | `descriptive_stats.csv` |
| Table 4 (Structural breaks) | `04_statistical_validation.py` | `dm_tests.csv` |
| Table 5 (Full-sample RMSE) | `04_statistical_validation.py` | `metrics_summary.csv` |
| Table 6 (Regime-conditional RMSE) | `04_statistical_validation.py` | `crisis_subperiod.csv` |
| Table 7 (Peak Crisis Sharpe) | `06_trading_simulation.py` | `trading_simulation.csv` |
| Table 8 (QKFM attribution) | `05_interpretability_expressibility.py` | `feature_importance.csv` |
| Table 9 (Policy mechanisms) | Manual (manuscript) | — |
| Table 10 (Bootstrap CI) | `09_bootstrap_hedging_ci.py` | `bootstrap_ci_results.csv` |
| Figure 1 (Regime RMSE) | `10_generate_figures.py` | `figures/Figure1_regime_rmse_h5.png` |
| Figure 2 (Feature heatmap) | `10_generate_figures.py` | `figures/Figure2_feature_importance_heatmap.png` |
| Appendix A, Table A1 (Expressibility) | `05_interpretability_expressibility.py` | `expressibility.csv` |
| Appendix B, Table B1 (Ablation summary) | `07e_ablation_merge.py` | `ablation_all_summary.csv` |
| Appendix B, Tables B2–B5 (Ablations AB1–AB4) | `07a–07d` scripts | `ablation_ab*.csv` |
| Appendix C, Table C1 (Trading simulation) | `06_trading_simulation.py` | `trading_simulation.csv` |
| Appendix C, Table C2 (DM test matrix) | `04_statistical_validation.py` | `dm_tests.csv` |
| Appendix C, Table C3 (VIF) | `compute_vif.py` | `vif_results.csv` |
| Appendix D, Table D1 (Roadmap) | Manual (manuscript) | — |

---

## Model Architecture

The QK-SVR circuit uses the following fixed hyperparameters (not tuned via Optuna):

| Parameter | Value | Justification |
|---|---|---|
| Qubits (N) | 4 | Avoids concentration (Thanasilp et al., 2024) |
| Layers (L) | 2 | Expressibility–trainability trade-off |
| Entanglement | Circular CNOT | Hardware-deployable, lower gate count |
| Kernel type | Fidelity Kernel | K(xᵢ,xⱼ) = \|⟨φ(xᵢ)\|φ(xⱼ)⟩\|² |
| Ansatz | Hardware-Efficient (HEA) | NISQ-compatible |
| Data re-uploading | True | Pérez-Salinas et al. (2020) |
| Simulator | PennyLane default.qubit | Noise-free; TRL 3–4 baseline |
| n_seeds | 30 (main); 10 (ablation) | See CODEBOOK.md §10 |

Architectural robustness validated via ablation studies AB1–AB4 (Scripts 07a–07d; Appendix B, Tables B2–B5).

---

## Data Description

See `data/README_data.md` for the full data dictionary.

**`data/master_dataset.csv`** — 1,302 daily observations (2019-01-03 to 2023-12-29):

| Column | Description | Unit |
|---|---|---|
| `date` | Trading date | YYYY-MM-DD |
| `EUA_return` | EU Allowance log-return | Float |
| `GAS_return` | TTF Natural Gas log-return | Float |
| `OIL_return` | Brent Crude Oil log-return | Float |
| `COAL_return` | ARA Coal log-return | Float |
| `ELEC_return` | European Electricity log-return | Float |
| `IP_return` | Industrial Production YoY growth (step-interpolated) | Float |
| `CPI_return` | HICP Inflation YoY growth (step-interpolated) | Float |
| `POLICY_dummy` | ETS regulatory event window (±7 days) | 0/1 Binary |
| `PHASE_dummy` | EU ETS Phase indicator | 0=Phase 3, 1=Phase 4 |

**Sources:** ICE Futures/Sandbag (EUA), Investing.com (TTF), FRED (Brent),
Platts/ICE (Coal ARA), EEX/ENTSO-E (Electricity), Eurostat (IP, CPI),
EUR-Lex/ECB (POLICY), EU Commission (PHASE).

> **Note:** Raw price series are not included due to data provider licensing restrictions.
> The archive distributes only derived log-returns and binary indicators.

---

## Hardware and Runtime Notes

- **QK-SVR (Script 02):** PennyLane noise-free simulator. Runtime scales linearly
  with fold count (37 folds × 3 horizons = 111 model fits). Multi-core parallelism
  supported via `n_jobs` parameter in script header.
- **Neural networks (Script 03c):** GPU-accelerated via PyTorch. Tested on NVIDIA
  RTX 3060+. CPU fallback available but approximately 5× slower.
- **Ablation studies (Scripts 07a–07d):** n_seeds = 10 (vs. n_seeds = 30 for main
  analysis). RMSE values are not directly comparable to Table 5. See CODEBOOK.md §10.

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{TranQuangCanh2026qksvr,
  title   = {Crisis-Period Trading Resilience and Dynamic MSR Reform: 
             A Quantum Kernel Approach to EU ETS Carbon Price Forecasting},
  author  = {Tran Quang Canh},
  journal = {Energy Policy},
  year    = {2026},
  doi     = {10.5281/zenodo.19913059},
  url     = {https://doi.org/10.5281/zenodo.19913059}
}
```

A machine-readable `CITATION.cff` is also provided.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The dataset (`data/master_dataset.csv`) contains processed log-returns derived from
publicly available market data. Original data are subject to the terms of their
respective providers (ICE, Eurostat, FRED, EEX, Platts).

---

## Contact

For questions about replication, please open a GitHub Issue or contact the
corresponding author via the email address listed in the manuscript.
