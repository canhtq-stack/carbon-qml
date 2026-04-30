# Reproduction Guide

**Manuscript:** Crisis-Period Trading Resilience and Dynamic MSR Reform: A Quantum Kernel Approach to EU ETS Carbon Price Forecasting  
**Journal:** Energy Policy (Elsevier)  
**Archive version:** 2.0.0

This document provides a step-by-step checklist to reproduce every table and figure
in the manuscript from scratch.

---

## Prerequisites

- Python 3.11 (CPython — see `requirements.txt`)
- RAM: ≥ 16 GB (32 GB recommended for neural network training)
- Storage: ≥ 5 GB free
- GPU: Optional but recommended for Script 03c (neural networks)

---

## Step-by-Step Checklist

### ✅ Step 1 — Clone and Install

```bash
git clone https://github.com/[USERNAME]/carbon_qml_project.git
cd carbon_qml_project
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import pennylane; print('PennyLane', pennylane.__version__)"
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "import optuna; print('Optuna', optuna.__version__)"
```

---

### ✅ Step 2 — Verify Input Data

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/master_dataset.csv')
assert df.shape == (1302, 10), f'Expected (1302,10), got {df.shape}'
assert df['date'].min() == '2019-01-03'
assert df['date'].max() == '2023-12-29'
assert df.isnull().sum().sum() == 0, 'Missing values detected'
print('Data verification: PASSED')
"
```

---

### ✅ Step 3 — Run Pipeline

```bash
cd scripts/

# Step 1: Generate walk-forward folds (~1 min)
python 01_fold_generator.py

# Step 1b: Compute naive baselines (Random Walk, Historical Mean)
python 01b_compute_naive_baselines.py

# Step 2: Train and evaluate QK-SVR (~4–8 hours on CPU)
python 02_qksvr_core.py

# Step 3: Train classical benchmarks (3a–3c can run in parallel)
python 03a_optuna_tuning.py         # ~30–60 min
python 03b_tree_svm.py              # ~30 min
python 03c_neural_nets.py           # ~2–4 hours (GPU recommended)
python 03d_merge.py                 # ~2 min

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
python 07e_ablation_merge.py        # ~2 min

# Step 8: Export manuscript tables (~2 min)
python 08_export_manuscript_tables.py

# Step 9: Bootstrap CI for Table 10 (~1 min)
python 09_bootstrap_hedging_ci.py --data_dir ../results --n_boot 10000 --seed 42

# Step 9b: FAR sensitivity analysis for Dual-Trigger MSR Protocol (~1 min)
# Outputs: results/far_sensitivity_K357.csv, results/far_sensitivity_K357.txt
python 09_far_sensitivity.py --input ../results/trading_simulation.csv --outdir ../results

# Step 10: Generate figures (~2 min)
python 10_generate_figures.py

# Auxiliary: Descriptive statistics and unit root tests (Table 3)
python descriptive_stats.py

# Auxiliary: VIF analysis (Appendix C, Table C3)
python compute_vif.py
```

> **Estimated total runtime:** 6–12 hours on a modern CPU (Intel Core i7/i9 or AMD Ryzen 7/9).
> Steps 02 and 03c are the bottlenecks.

> **Windows PowerShell — wrong Python version:**
> ```powershell
> C:\Users\<user>\AppData\Local\Programs\Python\Python311\python.exe scripts\02_qksvr_core.py
> ```

---

### ✅ Step 4 — Verify Key Results

Run this verification script after the full pipeline:

```python
import pandas as pd
import numpy as np

# ── Table 5: QK-SVR full-sample RMSE ─────────────────────────
ms = pd.read_csv('results/metrics_summary.csv')
qk_h1 = ms[(ms.model=='qk_svr')&(ms.horizon==1)]['rmse_mean'].values[0]
qk_h5 = ms[(ms.model=='qk_svr')&(ms.horizon==5)]['rmse_mean'].values[0]
qk_h22 = ms[(ms.model=='qk_svr')&(ms.horizon==22)]['rmse_mean'].values[0]
assert abs(qk_h1  - 0.0271) < 0.001, f"QK-SVR H=1 RMSE: {qk_h1}"
assert abs(qk_h5  - 0.0254) < 0.001, f"QK-SVR H=5 RMSE: {qk_h5}"
assert abs(qk_h22 - 0.0288) < 0.001, f"QK-SVR H=22 RMSE: {qk_h22}"
print("Table 5 verification: PASSED")

# ── Table 6: QK-SVR Peak Crisis RMSE ─────────────────────────
cs = pd.read_csv('results/crisis_subperiod.csv')
qk_pc_h5  = cs[(cs.model=='qk_svr')&(cs.horizon==5) &(cs.regime=='peak_crisis')]['rmse_mean'].values[0]
qk_pc_h22 = cs[(cs.model=='qk_svr')&(cs.horizon==22)&(cs.regime=='peak_crisis')]['rmse_mean'].values[0]
assert abs(qk_pc_h5  - 0.0249) < 0.001, f"QK-SVR Peak Crisis H=5 RMSE: {qk_pc_h5}"
assert abs(qk_pc_h22 - 0.0287) < 0.001, f"QK-SVR Peak Crisis H=22 RMSE: {qk_pc_h22}"
print("Table 6 verification: PASSED")

# ── Table 7: QK-SVR Peak Crisis Sharpe ratios ─────────────────
ts = pd.read_csv('results/trading_simulation.csv')
qk_pc_h1_sharpe  = ts[(ts.model=='qk_svr')&(ts.horizon==1) &(ts.regime=='peak_crisis')]['sharpe_ratio'].values[0]
qk_pc_h5_sharpe  = ts[(ts.model=='qk_svr')&(ts.horizon==5) &(ts.regime=='peak_crisis')]['sharpe_ratio'].values[0]
qk_pc_h22_sharpe = ts[(ts.model=='qk_svr')&(ts.horizon==22)&(ts.regime=='peak_crisis')]['sharpe_ratio'].values[0]
assert abs(qk_pc_h1_sharpe  - 3.15) < 0.05, f"QK-SVR H=1 Peak Crisis Sharpe: {qk_pc_h1_sharpe}"
assert abs(qk_pc_h5_sharpe  - 1.39) < 0.05, f"QK-SVR H=5 Peak Crisis Sharpe: {qk_pc_h5_sharpe}"
assert abs(qk_pc_h22_sharpe - 3.15) < 0.05, f"QK-SVR H=22 Peak Crisis Sharpe: {qk_pc_h22_sharpe}"
print("Table 7 verification: PASSED")

# ── Table 8: QKFM attribution Peak Crisis ────────────────────
fi = pd.read_csv('results/feature_importance.csv')
coal_pc = fi[(fi.feature=='COAL_return')&(fi.regime=='peak_crisis')]['qkfm_importance'].values[0]
gas_pc  = fi[(fi.feature=='GAS_return') &(fi.regime=='peak_crisis')]['qkfm_importance'].values[0]
assert abs(coal_pc - 0.394) < 0.01, f"COAL Peak Crisis QKFM: {coal_pc}"
assert abs(gas_pc  - 0.240) < 0.01, f"GAS Peak Crisis QKFM: {gas_pc}"
print("Table 8 verification: PASSED")

# ── Table 10: Bootstrap CI primary scenario ───────────────────
bci = pd.read_csv('results/bootstrap_ci_results.csv')
lap = bci[bci.scenario.str.contains('Laplacian')].iloc[0]
trf = bci[bci.scenario.str.contains('Transformer')].iloc[0]
assert lap.point_est_eur == 0,                    f"Laplacian point est: {lap.point_est_eur}"
assert abs(lap.bootstrap_mean_eur - 3662844) < 10000, f"Laplacian bootstrap mean: {lap.bootstrap_mean_eur}"
assert abs(trf.bootstrap_mean_eur - 5158527) < 10000, f"Transformer bootstrap mean: {trf.bootstrap_mean_eur}"
assert abs(lap.pct_savings_positive - 19.5) < 0.5, f"Laplacian % positive: {lap.pct_savings_positive}"
assert abs(trf.pct_savings_positive - 25.6) < 0.5, f"Transformer % positive: {trf.pct_savings_positive}"
print("Table 10 verification: PASSED")

# ── Appendix A, Table A1: Expressibility ─────────────────────
ex = pd.read_csv('results/expressibility.csv')
qk_dkl  = ex[ex.kernel.str.contains('Quantum')]['dkl_from_haar'].values[0]
qk_gini = ex[ex.kernel.str.contains('Quantum')]['spectral_decay_gini'].values[0]
assert abs(qk_dkl  - 2.817) < 0.01, f"QK D_KL: {qk_dkl}"
assert abs(qk_gini - (-0.945)) < 0.01, f"QK Gini: {qk_gini}"
print("Table A1 verification: PASSED")

print("\nAll key result checks PASSED ✅")
```

---

## Table-to-Script Mapping

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
| Figure 1 (Regime RMSE, H=5) | `10_generate_figures.py` | `figures/Figure1_regime_rmse_h5.png` |
| Figure 2 (Feature heatmap) | `10_generate_figures.py` | `figures/Figure2_feature_importance_heatmap.png` |
| Appendix A, Table A1 (Expressibility) | `05_interpretability_expressibility.py` | `expressibility.csv` |
| Appendix B, Table B1 (Ablation summary) | `07e_ablation_merge.py` | `ablation_all_summary.csv` |
| Appendix B, Table B2 (Ablation AB1) | `07a_ablation_qubits.py` | `ablation_ab1_qubits_summary.csv` |
| Appendix B, Table B3 (Ablation AB2) | `07b_ablation_depth.py` | `ablation_ab2_depth_summary.csv` |
| Appendix B, Table B4 (Ablation AB3) | `07c_ablation_entanglement.py` | `ablation_ab3_entanglement_summary.csv` |
| Appendix B, Table B5 (Ablation AB4) | `07d_ablation_reuploading.py` | `ablation_ab4_reuploading_summary.csv` |
| Appendix C, Table C1 (Trading simulation) | `06_trading_simulation.py` | `trading_simulation.csv` |
| Appendix C, Table C2 (DM test matrix) | `04_statistical_validation.py` | `dm_tests.csv` |
| Appendix C, Table C3 (VIF) | `compute_vif.py` | `vif_results.csv` |
| Appendix D, Table D1 (Roadmap) | Manual (manuscript) | — |
| Appendix D, Table D1 (Phase 1 KPI — FAR) | `09_far_sensitivity.py` | `far_sensitivity_K357.csv` |

---

## Troubleshooting

**PennyLane version mismatch:**
```bash
pip install --upgrade pennylane pennylane-lightning
```

**PyTorch CUDA out of memory:**
Reduce `batch_size` in `03c_neural_nets.py` (default: 32 → try 16).

**Script 02 very slow:**
Increase `n_jobs` parameter in `02_qksvr_core.py` (default: 1 → set to number of CPU cores - 1).

**Optuna study already exists:**
Delete `*.db` files in `scripts/` or set `--study-name new_study` flag.

**Windows PowerShell — wrong Python version:**
```powershell
C:\Users\<user>\AppData\Local\Programs\Python\Python311\python.exe scripts\02_qksvr_core.py
```

**Verification script assertion fails:**
Check `CODEBOOK.md §10` for known limitations on fold-level vs. aggregate RMSE comparisons.
