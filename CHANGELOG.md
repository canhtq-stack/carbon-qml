# Changelog

All notable changes to this replication archive are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-04-30 (Initial Zenodo Release)

### Archive Information
- **Manuscript:** Crisis-Period Trading Resilience and Dynamic MSR Reform: A Quantum Kernel Approach to EU ETS Carbon Price Forecasting
- **Journal:** Energy Policy (Elsevier)
- **Author:** Tran Quang Canh (canhtq@uef.edu.vn)
- **DOI:** **DOI:** https://doi.org/10.5281/zenodo.19913059

---

### Added

#### Pipeline Scripts
- `01_fold_generator.py` — Walk-forward fold construction (37 folds, step=21 days)
- `01b_compute_naive_baselines.py` — Random Walk (RW) and Historical Mean (HM) naïve baselines
- `02_qksvr_core.py` — QK-SVR model (PennyLane fidelity kernel, N=4 qubits, L=2 layers)
- `03a_optuna_tuning.py` — Hyperparameter tuning for tree/SVM models (100 Optuna trials)
- `03b_tree_svm.py` — XGBoost, LightGBM, RBF-SVM, Laplacian-SVM benchmarks
- `03c_neural_nets.py` — EMD-LSTM, BiLSTM, GRU, Transformer benchmarks
- `03d_merge.py` — Merge all model predictions
- `04_statistical_validation.py` — DM tests (HLN-corrected), MCS (α=0.25), Romano-Wolf correction
- `05_interpretability_expressibility.py` — QKFM attribution, TreeSHAP, kernel expressibility
- `06_trading_simulation.py` — Long-short trading simulation (Sharpe, Calmar, DA)
- `07a_ablation_qubits.py` — Ablation AB1: qubit count (N=4 vs N=6)
- `07b_ablation_depth.py` — Ablation AB2: circuit depth (L=1, 2, 3)
- `07c_ablation_entanglement.py` — Ablation AB3: entanglement topology (circular, linear, full)
- `07d_ablation_reuploading.py` — Ablation AB4: data re-uploading (True vs False)
- `07e_ablation_merge.py` — Merge ablation results
- `08_export_manuscript_tables.py` — Export APA-formatted tables to xlsx
- `09_bootstrap_hedging_ci.py` — Bootstrap 95% CI for hedging savings (Table 10)
- `10_generate_figures.py` — Generate Figures 1–2 (300 dpi)
- `compute_vif.py` — Variance Inflation Factor analysis (Appendix C, Table C3)
- `descriptive_stats.py` — Descriptive statistics and unit root tests (Table 3)

#### Data
- `data/master_dataset.csv` — 1,302 daily observations (2019-01-03 to 2023-12-29)
- `data/README_data.md` — Full data dictionary and variable descriptions

#### Results
- All result CSV files matching manuscript Tables 5–10 and Supplementary Tables A1, B1–B5, C1–C3
- `results/bootstrap_ci_results.csv` — Bootstrap CI (parametric, n=10,000, seed=42)
- `results/expressibility.csv` — Kernel expressibility metrics (D_KL, Spectral Gini)
- `results/vif_results.csv` — VIF analysis for all input features
- `results/descriptive_stats.csv` — Descriptive statistics and ADF/KPSS test results

#### Figures
- `figures/Figure1_regime_rmse_h5.png` — Regime-conditional RMSE across four regimes (H=5)
- `figures/Figure2_feature_importance_heatmap.png` — QKFM and TreeSHAP attribution heatmap

#### Documentation
- `CODEBOOK.md` — Complete variable definitions and analysis decisions
- `REPRODUCE.md` — Step-by-step reproduction guide with verification checks
- `CITATION.cff` — Machine-readable citation metadata
- `README.md` — Repository overview and usage instructions
- `README_data.md` — Data dictionary
- `.gitignore` — Git ignore rules
- `LICENSE` — MIT License
- `requirements.txt` — Python dependencies
- `environment.yml` — Conda environment specification

---

### Key Empirical Results (v1.0.0)

Results verified against manuscript Tables 5–10 and Appendix A–D:

#### Full-Sample RMSE (Table 5)
| Model | H=1 | H=5 | H=22 |
|---|---|---|---|
| QK-SVR | 0.0271 | 0.0254 | 0.0288 |
| LightGBM (best) | 0.0202 | 0.0183 | 0.0203 |

#### Peak Crisis Sharpe Ratios (Table 7)
| Model | H=1 | H=5 | H=22 | Positive all 3? |
|---|---|---|---|---|
| QK-SVR | +3.15 | +1.39 | +3.15 | ✓ ONLY |
| All classical benchmarks | — | — | — | ✗ |

#### QKFM Attribution — Peak Crisis (Table 8)
| Feature | QKFM | Rank |
|---|---|---|
| Coal ARA | 39.4% | 1st |
| GAS TTF | 24.0% | 2nd |
| Electricity | 18.8% | 3rd |
| Brent Oil | 17.8% | 4th |

#### Bootstrap CI — Table 10 (n=10,000, seed=42)
| Scenario | Point Est. | Bootstrap Mean | 95% CI | % Positive |
|---|---|---|---|---|
| QK-SVR vs. Laplacian-SVM (H=5) | EUR 0 | EUR 3,662,844 | EUR 0–36,456,364 | 19.5% |
| QK-SVR vs. Transformer (H=5) | EUR 0 | EUR 5,158,527 | EUR 0–42,389,204 | 25.6% |

#### Kernel Expressibility (Appendix A, Table A1)
| Kernel | D_KL from Haar | Spectral Decay Gini |
|---|---|---|
| QK-SVR | 2.817 | −0.945 |
| RBF-SVM | 1.619 | −0.826 |
| Laplacian-SVM | 0.368 | −0.421 |

---

### Configuration
- **Walk-forward:** Expanding window, initial=504 days, step=21 days, 37 folds
- **QK-SVR:** N=4 qubits, L=2 layers, circular CNOT, fidelity kernel, n_seeds=30
- **Ablation:** n_seeds=10 (not directly comparable to main results — see CODEBOOK §10)
- **Bootstrap:** Parametric, n=10,000 replications, seed=42
- **Normalization:** Min-max [0,1] for quantum circuit inputs; RobustScaler for classical models
- **Simulator:** PennyLane default.qubit (noise-free; TRL 3–4)
