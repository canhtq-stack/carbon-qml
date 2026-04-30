# CODEBOOK — Variable Definitions and Analysis Decisions

**Archive:** Quantum Kernel Methods for Carbon Price Forecasting During Energy Crises  
**Manuscript:** Crisis-Period Trading Resilience and Dynamic MSR Reform: A Quantum Kernel Approach to EU ETS Carbon Price Forecasting  
**Journal:** Energy Policy (Elsevier)  
**Author:** Tran Quang Canh (canhtq@uef.edu.vn)

---

## 1. Walk-Forward Validation Design

| Parameter | Value | Justification |
|---|---|---|
| Validation type | Expanding window | Mimics real-time regulatory decision-making |
| Initial training window | 504 trading days (~24 months) | January 2019–November 2020 |
| Step size (S) | 21 trading days (~1 month) | Aligns with MSR monthly auction cycle |
| Forecast horizons | H = {1, 5, 22} days | H=1 (tactical), H=5 (weekly), H=22 (MSR operational) |
| Total folds | 37 per horizon | December 2020–December 2023 |
| Peak-Crisis folds | 13 (H=5, H=22) | test_start in October 2022–March 2023 |
| Normalization (quantum) | Min-max [0, 1], fit on training window only | Required for quantum rotation gates; prevents look-ahead bias |
| Normalization (classical) | RobustScaler, fit on training window only | Prevents look-ahead bias |

---

## 2. Regime Identification

Structural breaks identified using three complementary tests:
- **Bai-Perron (1998)** sequential F-test: three statistically significant breaks
- **Zivot-Andrews (1992)**: confirms February 2022 as primary endogenous break
- **Lee-Strazicich (2003)**: two-break LM test (January 2021, February 2022)

| Regime | Dates | n_folds (H=5) | Characterization |
|---|---|---|---|
| Pre-Crisis | January 2019–January 2022 | 13 | Phase 4 commencement baseline; progressive EUA price appreciation |
| Crisis-Onset | February 2022–September 2022 | 3 | Geopolitical shock; gas supply disruption begins |
| Peak Crisis | October 2022–March 2023 | 13 | Maximum nonlinearity; EUA price peak; panic-driven contagion |
| Post-Crisis | April 2023–December 2023 | 5 | Partial normalization following REPowerEU measures |

> **Note on fold assignment:** Folds are assigned to regimes based on their `test_start` date.
> Regime boundaries are identified via structural break tests validated independently within each fold.

---

## 3. Model Identifiers

| `model` column value | Full name | Script |
|---|---|---|
| `qk_svr` | Quantum Kernel SVR | `02_qksvr_core.py` |
| `rbf_svm` | RBF-SVM | `03b_tree_svm.py` |
| `laplacian_svm` | Laplacian-SVM | `03b_tree_svm.py` |
| `xgboost` | XGBoost | `03b_tree_svm.py` |
| `lightgbm` | LightGBM | `03b_tree_svm.py` |
| `bilstm` | Bidirectional LSTM | `03c_neural_nets.py` |
| `gru` | Gated Recurrent Unit | `03c_neural_nets.py` |
| `transformer` | Transformer (attention) | `03c_neural_nets.py` |
| `emd_lstm` | EMD-decomposed LSTM (Xu et al., 2022) | `03c_neural_nets.py` |
| `random_walk` | Random Walk (RW) naïve baseline | `04_statistical_validation.py` |
| `historical_mean` | Historical Mean (HM) naïve baseline | `04_statistical_validation.py` |

All models optimized via Optuna (100 trials, Tree-structured Parzen Estimator) under identical evaluation conditions.

---

## 4. QK-SVR Architecture (Fixed — not tuned via Optuna)

```
Circuit parameters:
  N         = 4 qubits
  L         = 2 layers (Hardware-Efficient Ansatz)
  Entanglement: Circular CNOT — CNOT(i → (i+1) mod N)
  Kernel:   Fidelity Kernel K(xᵢ,xⱼ) = |⟨φ(xᵢ)|φ(xⱼ)⟩|²
  Re-uploading: True (Pérez-Salinas et al., 2020)
  Simulator: PennyLane default.qubit (noise-free)
  n_seeds:  30 (main analysis); 10 (ablation studies AB1–AB4)
```

Circuit features (entering quantum circuit):
- `GAS_return` (TTF Natural Gas)
- `OIL_return` (Brent Crude)
- `COAL_return` (ARA Coal)
- `ELEC_return` (European Electricity)

Non-circuit features (appended as classical auxiliary inputs to SVR):
- `IP_return`, `CPI_return`, `POLICY_dummy`, `PHASE_dummy`

SVR hyperparameters tuned via Optuna per fold: regularization C, epsilon ε, encoding scale γ.

---

## 5. Statistical Tests

### Diebold-Mariano Test (`dm_tests.csv`)
- **Correction:** Harvey-Leybourne-Newbold (1997) small-sample
- **HAC:** Newey-West, bandwidth lag = 5
- **Multiple testing:** Romano-Wolf stepdown (FWER control, 10 pairwise comparisons per horizon)
- **Null hypothesis:** Equal predictive accuracy between QK-SVR and benchmark
- **Sign convention:** Positive DM = QK-SVR worse; Negative DM = QK-SVR better

### Model Confidence Set (`mcs_results.csv`)
- **Significance level:** α = 0.25 (standard for 9-model MCS)
- **Statistic:** Range statistic T_R
- **Bootstrap:** Block bootstrap with 1,000 replications

### Ljung-Box Test (`ljungbox_diagnostics.csv`)
- **Lags:** 10
- **Null hypothesis:** No residual autocorrelation
- QK-SVR results: H=1 (LB=11.22, p=0.341); H=5 (LB=9.13, p=0.520); H=22 (LB=16.73, p=0.081)

### VIF Analysis (`vif_results.csv`)
- **Script:** `compute_vif.py`
- **Threshold:** VIF < 5.0 = None; 5.0–10.0 = Moderate; > 10.0 = High
- All features confirmed below threshold (maximum: PHASE_dummy VIF = 4.29)

---

## 6. Trading Simulation (`trading_simulation.csv`)

```
Strategy:          Long-short momentum
Position:          sign(forecast) — long if forecast > 0, short otherwise
PnL:               position(t) × realized_return(t+H)
Transaction costs: None (zero-cost benchmark)
Sharpe:            annualised = (mean PnL / std PnL) × √252
Calmar:            annualised_return / |max_drawdown|
Hit rate:          % correctly-signed directional predictions
```

Peak Crisis Sharpe ratios (October 2022–March 2023):

| Model | H=1 | H=5 | H=22 | Positive all 3? |
|---|---|---|---|---|
| QK-SVR | +3.15 | +1.39 | +3.15 | ✓ ONLY |
| All 9 classical benchmarks | varied | varied | varied | ✗ |

---

## 7. QKFM Feature Attribution (`feature_importance.csv`)

**Method:** Quantum Kernel Feature Masking (QKFM)
- For each feature f and fold k: set f to its training-window mean (masked) and recompute QK-SVR prediction
- Importance = increase in fold RMSE when feature is masked
- `qkfm_importance` = normalized importance score (sums to ~1 per regime for circuit features)
- Features with `NaN` in `qkfm_importance` are non-circuit features assessed via TreeSHAP only

Peak Crisis attribution:

| Feature | QKFM | Rank | TreeSHAP | SHAP Rank |
|---|---|---|---|---|
| Coal ARA | 39.4% | 1st | 26.8% | 1st |
| GAS TTF | 24.0% | 2nd | 19.4% | 2nd |
| Electricity | 18.8% | 3rd | 12.3% | 5th |
| Brent Oil | 17.8% | 4th | 16.4% | 3rd |

---

## 8. Bootstrap CI (`bootstrap_ci_results.csv`)

**Method:** Parametric bootstrap (recommended)
- **Distribution:** Normal(μ_regime, σ_regime) per model, drawn from `crisis_subperiod.csv`
- **Resampling unit:** Fold (n=13 Peak-Crisis folds)
- **Replications:** n = 10,000
- **Seed:** 42
- **Savings formula:** max(0, (RMSE_bench − RMSE_QK) / RMSE_bench) × EUR 28B × 0.5%
- **Lower bound:** Capped at EUR 0 (savings cannot be negative by construction)

Results (Table 10):

| Scenario | Point Est. | Bootstrap Mean | 95% CI | % Positive |
|---|---|---|---|---|
| QK-SVR vs. Laplacian-SVM (H=5) | EUR 0 | EUR 3,662,844 | EUR 0–36,456,364 | 19.5% |
| QK-SVR vs. Transformer (H=5) | EUR 0 | EUR 5,158,527 | EUR 0–42,389,204 | 25.6% |
| QK-SVR vs. RBF-SVM (H=22) | EUR 0 | EUR 1,623,620 | EUR 0–23,191,446 | 10.3% |
| QK-SVR vs. EMD-LSTM (Full) | EUR 0 | EUR 111,661 | EUR 0–0 | 1.8% |

---

## 9. Expressibility Metrics (`expressibility.csv`)

| Metric | Definition |
|---|---|
| `dkl_from_haar` | KL divergence between kernel output distribution and Haar-random measure. Higher = more geometrically concentrated (Sim et al., 2019). |
| `spectral_decay_gini` | Gini coefficient of eigenvalue spectrum of the kernel matrix. More negative = greater concentration in dominant eigenmodes. |

Results (Appendix A, Table A1):

| Kernel | D_KL from Haar | Spectral Decay Gini |
|---|---|---|
| QK-SVR | 2.817 | −0.945 |
| RBF-SVM | 1.619 | −0.826 |
| Laplacian-SVM | 0.368 | −0.421 |

---

## 10. Known Limitations

1. **Ablation RMSE not comparable to main results:** Ablation studies use n_seeds=10 vs. n_seeds=30 in the main analysis. RMSE values in Tables B2–B5 should not be directly compared to Table 5. See Supplementary note before Tables B2–B5.

2. **Fold-level vs. aggregate RMSE:** Each H=5 fold covers a single 5-day prediction step. Fold-level RMSE in `metrics_fold_level.csv` reflects log-return scale noise and should not be directly averaged to reproduce `crisis_subperiod.csv` aggregate values (Jensen's inequality gap).

3. **Simulator-only results:** All QK-SVR outputs are from PennyLane noise-free simulators (TRL 3–4). Hardware decoherence is not modeled. See manuscript Section 7 (Limitations).

4. **Raw price data not included:** Original price series (EUA spot, TTF front-month, Brent, ARA Coal, German electricity day-ahead) are not distributed due to data provider licensing restrictions. Only derived log-returns are provided.
