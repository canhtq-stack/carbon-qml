# =============================================================================
# 05_interpretability_expressibility_fixed.py  —  v2
# =============================================================================
# BUG 1 [CRITICAL] — rebuild_qksvr_kernel dùng PennyLane circuit
#   → 02_qksvr_core_fixed dùng numpy FastQuantumKernel
#   → QKFM/expressibility tính trên kernel KHÁC với model thực tế → kết quả không nhất quán
#   Fix: copy FastQuantumKernel từ 02_qksvr_core_fixed để dùng cùng implementation
#
# BUG 2 — No NaN handling khi load dataset
#   Fix: ffill().bfill() sau load
#
# BUG 3 — No parquet fallback cho fold_splits
#   Fix: try parquet → fallback CSV
#
# BUG 4 — No FAST_MODE → QKFM + regime analysis chạy rất lâu
#   Fix: FAST_MODE=True giảm n_folds_sample và n_samples
#
# BUG 5 — within/between ratio O(N^2) Python loops → slow
#   Fix: numpy vectorized với boolean masking
# =============================================================================

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)

# =============================================================================
# CẤU HÌNH — chỉnh tại đây rồi nhấn Run trong VS Code
# =============================================================================
FAST_MODE = True   # True = nhanh ~10 phút | False = full ~60 phút

BASE_DIR    = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project")
# =============================================================================

DATA_PATH   = BASE_DIR / "Data" / "data" / "processed" / "master_dataset.csv"
RESULTS_DIR = BASE_DIR / "Data" / "results"
CONFIG_DIR  = BASE_DIR / "config"
OUTPUT_DIR  = RESULTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "EUA_return"
FEATURES = [
    "GAS_return", "OIL_return", "COAL_return", "ELEC_return",
    "IP_return", "CPI_return", "POLICY_dummy", "PHASE_dummy",
]
N_FEAT   = len(FEATURES)
N_QUBITS = 4    # fixed per paper Section 3.4
N_LAYERS = 2

EU_ETS_MARKET_SIZE_EUR = 28_000_000_000
HEDGING_COST_RATIO     = 0.005

FALLBACK_REGIMES = {
    "pre_crisis":   ("2019-01-01", "2021-12-31"),
    "crisis_onset": ("2022-01-01", "2022-06-30"),
    "peak_crisis":  ("2022-07-01", "2023-06-30"),
    "post_crisis":  ("2023-07-01", "2023-12-31"),
}

if FAST_MODE:
    N_FOLDS_SAMPLE = 5
    N_SAMPLES_EXPR = 50
    N_SAMPLES_REG  = 40
    logging.info("FAST MODE: n_folds=5, n_samples=50 (~10 phut)")
else:
    N_FOLDS_SAMPLE = 20
    N_SAMPLES_EXPR = 150
    N_SAMPLES_REG  = 80
    logging.info("FULL MODE: n_folds=20, n_samples=150 (~60 phut)")


# ── Helpers ───────────────────────────────────────────────────────
def load_folds(results_dir: Path) -> pd.DataFrame:
    """BUG 3 FIX: fallback CSV."""
    pq = results_dir / "fold_splits.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = pq.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No fold_splits in {results_dir}")


def load_break_dates() -> Dict:
    bp = CONFIG_DIR / "break_dates.json"
    if not bp.exists():
        logging.warning("break_dates.json not found → fallback regimes.")
        return FALLBACK_REGIMES
    with open(bp) as f:
        data = json.load(f)
    breaks = sorted(data.get("break_dates", []))
    if len(breaks) < 2:
        return FALLBACK_REGIMES
    return {
        "pre_crisis":   ("2019-01-01", breaks[0]),
        "crisis_onset": (breaks[0], breaks[1]),
        "peak_crisis":  (breaks[1], breaks[2] if len(breaks) > 2 else "2023-06-30"),
        "post_crisis":  (breaks[2] if len(breaks) > 2 else "2023-07-01", "2024-12-31"),
    }


def assign_regime(date: pd.Timestamp, regimes: Dict) -> str:
    for name, (s, e) in regimes.items():
        if pd.Timestamp(s) <= date <= pd.Timestamp(e):
            return name
    return "other"


# ── BUG 1 FIX: numpy FastQuantumKernel (same as 02_qksvr_core_fixed) ─
class FastQuantumKernel:
    """
    Numpy-based quantum kernel — identical to 02_qksvr_core_fixed.
    Ensures QKFM and expressibility use same implementation as actual model.
    """
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim      = 2 ** n_qubits
        rng = np.random.default_rng(seed)
        self.params = rng.uniform(-0.01, 0.01, (n_layers, n_qubits))
        self._cnot_mat = self._build_cnot()
        self._ry_mats  = self._build_ry_all()

    @staticmethod
    def _ry(t):
        c, s = np.cos(t/2), np.sin(t/2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def _kron_all(gs):
        r = gs[0]
        for g in gs[1:]: r = np.kron(r, g)
        return r

    def _build_cnot(self):
        n, dim = self.n_qubits, self.dim
        def cnot(ctrl, tgt):
            U = np.zeros((dim, dim), dtype=complex)
            for col in range(dim):
                cb = (col >> (n-1-ctrl)) & 1
                U[col ^ (1 << (n-1-tgt)) if cb else col, col] = 1
            return U
        U = np.eye(dim, dtype=complex)
        for i in range(n): U = cnot(i, (i+1)%n) @ U
        return U

    def _build_ry_all(self):
        return [self._kron_all([self._ry(self.params[l, q])
                                for q in range(self.n_qubits)])
                for l in range(self.n_layers)]

    def _apply_rx_batch(self, states, angles, qubit):
        N, dim = states.shape
        nq = self.n_qubits
        s  = states.reshape(N, *([2]*nq))
        ax = 1 + qubit
        idx0 = [slice(None)]*(nq+1); idx0[ax] = 0
        idx1 = [slice(None)]*(nq+1); idx1[ax] = 1
        s0, s1 = s[tuple(idx0)], s[tuple(idx1)]
        ca, sa = np.cos(angles/2), np.sin(angles/2)
        for _ in range(nq-1): ca = ca[..., None]; sa = sa[..., None]
        return np.stack([ca*s0 - 1j*sa*s1, -1j*sa*s0 + ca*s1],
                        axis=ax).reshape(N, dim)

    def compute_states(self, X: np.ndarray) -> np.ndarray:
        N = len(X)
        Xc = np.clip(X[:, :self.n_qubits], -np.pi, np.pi)
        s = np.zeros((N, self.dim), dtype=complex); s[:, 0] = 1.0
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                s = self._apply_rx_batch(s, Xc[:, q], q)
            s = s @ self._cnot_mat.T
            s = s @ self._ry_mats[l].T
        return s

    def kernel(self, X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
        sa = self.compute_states(X_a)
        sb = self.compute_states(X_b)
        return np.real(np.abs(sa @ sb.conj().T) ** 2)


# ── Step 1: QKFM ─────────────────────────────────────────────────
def compute_qkfm(df_main, folds, regimes) -> pd.DataFrame:
    """BUG 1 FIX: use FastQuantumKernel instead of PennyLane."""
    fold_h1 = folds[folds["horizon"] == 1].sample(
        min(N_FOLDS_SAMPLE, len(folds[folds["horizon"] == 1])),
        random_state=42,
    )

    fqk     = FastQuantumKernel(N_QUBITS, N_LAYERS, seed=42)
    records = []
    logging.info(f"  QKFM: {len(fold_h1)} folds × {N_QUBITS} features")

    for idx, (_, fold) in enumerate(fold_h1.iterrows()):
        tr_s = pd.to_datetime(fold["train_start"])
        tr_e = pd.to_datetime(fold["train_end"])
        te_s = pd.to_datetime(fold["test_start"])

        X_tr = df_main.loc[tr_s:tr_e, FEATURES[:N_QUBITS]].values
        if len(X_tr) < 10:
            continue

        sc        = RobustScaler(quantile_range=(5, 95)).fit(X_tr)
        X_sc      = sc.transform(X_tr)
        fold_mean = X_sc.mean(axis=0)   # fold-aware masking (FIX 1 original)

        rng    = np.random.default_rng(42 + idx)
        n_samp = min(40, len(X_sc))
        X_samp = X_sc[rng.choice(len(X_sc), n_samp, replace=False)]

        try:
            K_full = fqk.kernel(X_samp, X_samp)
        except Exception as e:
            logging.warning(f"    Fold {idx} kernel failed: {e}")
            continue

        importances = np.zeros(N_QUBITS)
        for i in range(N_QUBITS):
            Xm = X_samp.copy(); Xm[:, i] = fold_mean[i]
            try:
                K_masked = fqk.kernel(Xm, Xm)
                importances[i] = float(np.mean(np.abs(K_full - K_masked)))
            except Exception:
                pass

        s = importances.sum()
        if s > 1e-10:
            importances /= s

        regime = assign_regime(te_s, regimes)
        for i, feat in enumerate(FEATURES[:N_QUBITS]):
            records.append({
                "fold_id":    int(fold["fold_id"]),
                "regime":     regime,
                "feature":    feat,
                "importance": float(importances[i]),
                "method":     "QKFM",
            })
        if (idx+1) % 5 == 0:
            logging.info(f"    QKFM: {idx+1}/{len(fold_h1)}")

    return pd.DataFrame(records)


# ── Step 2: Kernel Regime Analysis ───────────────────────────────
def eigengap_clusters(K: np.ndarray, max_k: int = 6) -> int:
    try:
        from scipy.sparse.linalg import eigsh
        n    = K.shape[0]
        vals = eigsh(K, k=min(max_k+1, n-1), which="LM", return_eigenvectors=False)
        vals = np.sort(vals)[::-1]
        gaps = np.abs(np.diff(vals))
        return int(np.argmax(gaps) + 1)
    except Exception:
        return 4


def compute_regime_analysis(df_main, regimes) -> Dict:
    """BUG 5 FIX: numpy vectorized within/between ratio."""
    fqk    = FastQuantumKernel(N_QUBITS, N_LAYERS, seed=42)
    df_test = df_main.loc["2022-01-01":"2023-12-29", FEATURES[:N_QUBITS]].dropna()
    if len(df_test) < 20:
        return {}

    rng   = np.random.default_rng(42)
    n_s   = min(N_SAMPLES_REG, len(df_test))
    idx_s = np.sort(rng.choice(len(df_test), n_s, replace=False))
    X_raw = df_test.iloc[idx_s].values
    dates = df_test.index[idx_s]

    sc   = RobustScaler(quantile_range=(5, 95)).fit(df_test.values)
    X_sc = sc.transform(X_raw)

    logging.info(f"  Regime kernel on {n_s} samples...")
    try:
        K = fqk.kernel(X_sc, X_sc)
    except Exception as e:
        logging.error(f"Kernel failed: {e}")
        return {}

    regime_keys = list(regimes.keys())
    true_labels = np.array([
        regime_keys.index(assign_regime(d, regimes))
        if assign_regime(d, regimes) in regime_keys else -1
        for d in dates
    ])
    valid = true_labels >= 0
    if valid.sum() < 10:
        return {}

    K_v  = np.clip(K[np.ix_(valid, valid)], 0, None)
    tl   = true_labels[valid]

    # BUG 5 FIX: numpy vectorized within/between
    same    = (tl[:, None] == tl[None, :])          # (N, N) bool
    diff    = ~same
    np.fill_diagonal(same, False)                    # exclude diagonal
    within  = float(K_v[same].mean()) if same.any() else 0.0
    between = float(K_v[diff].mean()) if diff.any() else 0.0

    n_clust = eigengap_clusters(K_v)
    logging.info(f"  Eigengap → n_clusters={n_clust}")

    try:
        pred = SpectralClustering(
            n_clusters=n_clust, affinity="precomputed",
            random_state=42, n_init=10,
        ).fit_predict(K_v)
        ari = float(adjusted_rand_score(tl, pred))
    except Exception as e:
        logging.error(f"Spectral clustering: {e}")
        ari = float("nan")

    result = {
        "ari": ari,
        "n_clusters_inferred": n_clust,
        "within_regime_sim": within,
        "between_regime_sim": between,
        "within_between_ratio": within / (between + 1e-10),
        "n_samples": int(valid.sum()),
    }
    logging.info(f"  ARI={ari:.3f} | W/B={result['within_between_ratio']:.3f}")
    return result


# ── Step 3: Expressibility ────────────────────────────────────────
def kernel_dkl(K: np.ndarray) -> float:
    K_sym = (K + K.T) / 2
    w     = np.clip(np.linalg.eigvalsh(K_sym), 1e-10, None)
    w    /= w.sum()
    n     = len(w)
    unif  = np.ones(n) / n
    return float(np.sum(w * np.log(w / unif + 1e-10)))


def spectral_decay(K: np.ndarray) -> float:
    w   = np.sort(np.clip(np.linalg.eigvalsh(K), 0, None))[::-1]
    w  /= (w.sum() + 1e-10)
    n   = len(w)
    idx = np.arange(1, n+1)
    return float((2 * np.sum(idx * w) - (n+1)) / n)


def compute_expressibility(df_main) -> pd.DataFrame:
    from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel

    rng   = np.random.default_rng(42)
    df_s  = df_main[FEATURES[:N_QUBITS]].dropna()
    n_s   = min(N_SAMPLES_EXPR, len(df_s))
    X_raw = df_s.iloc[rng.choice(len(df_s), n_s, replace=False)].values
    sc    = RobustScaler(quantile_range=(5, 95)).fit(X_raw)
    X_sc  = sc.transform(X_raw)
    X_c   = np.clip(X_sc, -np.pi, np.pi)
    gamma = 1.0 / (N_QUBITS * X_sc.var())

    results = []
    fqk = FastQuantumKernel(N_QUBITS, N_LAYERS, seed=42)

    logging.info("  QK expressibility...")
    try:
        K_qk = fqk.kernel(X_c, X_c)
        results.append({"kernel": "Quantum (QK-SVR)",
                        "dkl_from_haar": kernel_dkl(K_qk),
                        "spectral_decay_gini": spectral_decay(K_qk),
                        "n_samples": n_s})
    except Exception as e:
        logging.warning(f"  QK failed: {e}")
        results.append({"kernel": "Quantum (QK-SVR)", "dkl_from_haar": np.nan,
                        "spectral_decay_gini": np.nan, "n_samples": n_s})

    logging.info("  RBF expressibility...")
    K_rbf = rbf_kernel(X_c, gamma=gamma)
    results.append({"kernel": "RBF (classical)", "dkl_from_haar": kernel_dkl(K_rbf),
                    "spectral_decay_gini": spectral_decay(K_rbf), "n_samples": n_s})

    logging.info("  Laplacian expressibility...")
    K_lap = laplacian_kernel(X_c, gamma=gamma)
    results.append({"kernel": "Laplacian (classical)", "dkl_from_haar": kernel_dkl(K_lap),
                    "spectral_decay_gini": spectral_decay(K_lap), "n_samples": n_s})

    df_e = pd.DataFrame(results)
    logging.info("\n" + df_e[["kernel", "dkl_from_haar", "spectral_decay_gini"]].to_string(index=False))
    return df_e


# ── Step 4: TreeSHAP ─────────────────────────────────────────────
def compute_treeshap(df_main, folds, all_params, regimes) -> pd.DataFrame:
    try:
        import shap
        import xgboost as xgb
    except ImportError:
        logging.warning("shap or xgboost not installed → skip TreeSHAP.")
        return pd.DataFrame()

    params  = all_params.get("xgboost", {})
    fold_h1 = folds[folds["horizon"] == 1].sample(
        min(N_FOLDS_SAMPLE, len(folds[folds["horizon"] == 1])),
        random_state=42,
    )
    records = []
    for _, fold in fold_h1.iterrows():
        tr_s = pd.to_datetime(fold["train_start"])
        tr_e = pd.to_datetime(fold["train_end"])
        te_s = pd.to_datetime(fold["test_start"])

        X_tr = df_main.loc[tr_s:tr_e, FEATURES].values
        y_tr = df_main.loc[tr_s:tr_e, TARGET].values
        X_te = df_main.loc[te_s:te_s, FEATURES].values
        if len(X_tr) < 20 or len(X_te) == 0:
            continue

        sc      = RobustScaler(quantile_range=(5, 95)).fit(X_tr)
        X_tr_sc = sc.transform(X_tr)
        X_te_sc = sc.transform(X_te)

        m = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        m.fit(X_tr_sc, y_tr)

        explainer  = shap.TreeExplainer(m)
        shap_vals  = explainer.shap_values(X_te_sc)
        importance = np.abs(shap_vals).mean(axis=0)
        s = importance.sum()
        if s > 1e-10:
            importance /= s

        regime = assign_regime(te_s, regimes)
        for i, feat in enumerate(FEATURES):
            records.append({
                "fold_id":    int(fold["fold_id"]),
                "regime":     regime,
                "feature":    feat,
                "importance": float(importance[i]),
                "method":     "TreeSHAP (XGBoost)",
            })

    return pd.DataFrame(records)


def merge_importance(df_qkfm, df_shap) -> pd.DataFrame:
    records = []
    for regime in ["pre_crisis", "crisis_onset", "peak_crisis"]:
        for feat in FEATURES:
            q = s = float("nan")
            if len(df_qkfm) > 0:
                v = df_qkfm[(df_qkfm["feature"]==feat) & (df_qkfm["regime"]==regime)]["importance"]
                if len(v): q = float(v.mean())
            if len(df_shap) > 0:
                v = df_shap[(df_shap["feature"]==feat) & (df_shap["regime"]==regime)]["importance"]
                if len(v): s = float(v.mean())
            records.append({"feature": feat, "regime": regime,
                            "qkfm_importance": q, "shap_importance": s})
    df = pd.DataFrame(records)
    for regime in ["pre_crisis", "crisis_onset", "peak_crisis"]:
        mask = df["regime"] == regime
        df.loc[mask, "qkfm_rank"] = df.loc[mask, "qkfm_importance"].rank(ascending=False).astype("Int64")
        df.loc[mask, "shap_rank"] = df.loc[mask, "shap_importance"].rank(ascending=False).astype("Int64")
    return df


# ── Step 5: Policy ────────────────────────────────────────────────
def compute_policy(crisis_path: Path) -> pd.DataFrame:
    if not crisis_path.exists():
        logging.warning(f"crisis_subperiod.csv not found: {crisis_path}")
        return pd.DataFrame()

    df_c    = pd.read_csv(crisis_path)
    records = []
    for h in [1, 5, 22]:
        peak = df_c[(df_c["horizon"]==h) & (df_c["regime"]=="peak_crisis")]
        if len(peak) == 0:
            continue
        qk  = peak[peak["model"]=="qk_svr"]
        bms = peak[peak["model"]!="qk_svr"]
        if len(qk) == 0 or len(bms) == 0:
            continue

        rmse_qk   = float(qk["rmse_mean"].values[0])
        rmse_best = float(bms["rmse_mean"].min())
        da_qk     = float(qk["da_mean"].values[0]) if "da_mean" in qk.columns else np.nan
        da_best   = float(bms["da_mean"].max())    if "da_mean" in bms.columns else np.nan

        uncert_reduc  = (rmse_best - rmse_qk) / (rmse_best + 1e-10)
        hedge_savings = EU_ETS_MARKET_SIZE_EUR * HEDGING_COST_RATIO * max(uncert_reduc, 0)

        records.append({
            "horizon":                    h,
            "regime":                     "peak_crisis",
            "rmse_qk_svr":               round(rmse_qk, 6),
            "rmse_best_classical":        round(rmse_best, 6),
            "uncertainty_reduction_pct":  round(uncert_reduc * 100, 2),
            "da_qk_svr_pct":             round(da_qk, 1),
            "da_best_classical_pct":     round(da_best, 1),
            "da_improvement_pp":         round(da_qk - da_best, 1),
            "hedging_savings_eur":        round(hedge_savings, 0),
            "assumption_market_size":     EU_ETS_MARKET_SIZE_EUR,
            "assumption_cost_ratio_pct":  HEDGING_COST_RATIO * 100,
            "assumption_notes":           "0 transaction cost; savings prop. to RMSE uncertainty reduction",
        })
    return pd.DataFrame(records)


# ── Report ────────────────────────────────────────────────────────
def write_report(df_imp, df_expr, ari, df_pol, out_path):
    lines = ["# Interpretability & Expressibility Report",
             "Generated by: 05_interpretability_expressibility_fixed.py\n"]
    if len(df_expr) > 0:
        lines += ["\n## 1. Expressibility (Table 9)",
                  df_expr[["kernel","dkl_from_haar","spectral_decay_gini"]]
                  .to_markdown(index=False, floatfmt=".4f")]
    if ari:
        lines += [f"\n## 2. Kernel Regime ARI",
                  f"- ARI: {ari.get('ari','N/A'):.3f}",
                  f"- W/B ratio: {ari.get('within_between_ratio','N/A'):.3f}",
                  f"- Inferred clusters: {ari.get('n_clusters_inferred','N/A')}"]
    if len(df_imp) > 0:
        lines += ["\n## 3. Feature Importance (Table 10)",
                  df_imp.to_markdown(index=False, floatfmt=".4f")]
    if len(df_pol) > 0:
        lines += ["\n## 4. Policy Implications (Table 11)",
                  f"EU ETS market size: EUR {EU_ETS_MARKET_SIZE_EUR:,}",
                  df_pol[["horizon","uncertainty_reduction_pct",
                           "hedging_savings_eur","da_improvement_pp"]]
                  .to_markdown(index=False, floatfmt=".2f")]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"Report saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────
def main():
    # BUG 2 FIX: NaN handling
    df_main = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").sort_index()
    n_nan   = df_main[FEATURES + [TARGET]].isna().sum().sum()
    if n_nan > 0:
        df_main[FEATURES + [TARGET]] = df_main[FEATURES + [TARGET]].ffill().bfill()
        logging.info(f"Filled {n_nan} NaN values")

    # BUG 3 FIX: parquet fallback
    folds   = load_folds(RESULTS_DIR)
    regimes = load_break_dates()

    all_params = {}
    p = RESULTS_DIR / "optuna_best_params.json"
    if p.exists():
        with open(p) as f:
            all_params = json.load(f)

    # Step 1: QKFM
    logging.info("\n" + "="*55 + "\nStep 1/5 — QKFM\n" + "="*55)
    df_qkfm = compute_qkfm(df_main, folds, regimes)
    if len(df_qkfm) > 0:
        df_qkfm.to_csv(OUTPUT_DIR / "qkfm_raw.csv", index=False)
        logging.info("  Saved qkfm_raw.csv")

    # Step 2: Regime analysis
    logging.info("\n" + "="*55 + "\nStep 2/5 — Kernel Regime Analysis\n" + "="*55)
    ari = compute_regime_analysis(df_main, regimes)
    if ari:
        with open(OUTPUT_DIR / "regime_ari.json", "w") as f:
            json.dump(ari, f, indent=2)
        logging.info("  Saved regime_ari.json")

    # Step 3: Expressibility
    logging.info("\n" + "="*55 + "\nStep 3/5 — Expressibility\n" + "="*55)
    df_expr = compute_expressibility(df_main)
    df_expr.to_csv(OUTPUT_DIR / "expressibility.csv", index=False)
    logging.info("  Saved expressibility.csv")

    # Step 4: TreeSHAP
    logging.info("\n" + "="*55 + "\nStep 4/5 — TreeSHAP\n" + "="*55)
    df_shap = compute_treeshap(df_main, folds, all_params, regimes)
    df_imp  = merge_importance(df_qkfm, df_shap)
    if len(df_imp) > 0:
        df_imp.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
        logging.info("  Saved feature_importance.csv")

    # Step 5: Policy
    logging.info("\n" + "="*55 + "\nStep 5/5 — Policy\n" + "="*55)
    df_pol = compute_policy(OUTPUT_DIR / "crisis_subperiod.csv")
    if len(df_pol) > 0:
        df_pol.to_csv(OUTPUT_DIR / "policy_implications.csv", index=False)
        logging.info("  Saved policy_implications.csv")

    write_report(df_imp, df_expr, ari, df_pol,
                 OUTPUT_DIR / "interpretability_report.md")

    # Console summary
    print(f"\n{'='*60}")
    print("  FILE 05 — SUMMARY")
    print(f"{'='*60}")
    if len(df_expr) > 0:
        print("\n▶ Expressibility:")
        print(df_expr[["kernel","dkl_from_haar","spectral_decay_gini"]].to_string(index=False))
    if ari:
        print(f"\n▶ ARI={ari.get('ari','N/A'):.3f} | W/B={ari.get('within_between_ratio','N/A'):.3f}")
    if len(df_imp) > 0:
        print("\n▶ Feature importance (peak_crisis, QKFM):")
        peak = df_imp[df_imp["regime"]=="peak_crisis"].sort_values("qkfm_importance", ascending=False)
        print(peak[["feature","qkfm_importance","shap_importance"]].to_string(index=False))
    if len(df_pol) > 0:
        print("\n▶ Policy savings:")
        print(df_pol[["horizon","uncertainty_reduction_pct","hedging_savings_eur"]].to_string(index=False))
    print(f"\n  Outputs: {OUTPUT_DIR}")
    if FAST_MODE:
        print("  FAST MODE done. Set FAST_MODE=False for full run.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
