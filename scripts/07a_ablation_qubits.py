# =============================================================================
# 07a_ablation_qubits_fixed.py  —  v2
# =============================================================================
# BUG 1 [CRITICAL] — Dùng PennyLane serial circuit thay vì numpy FastQuantumKernel
#   → Ablation sẽ mất ~148 giờ với PennyLane; ~3.7 giờ với numpy
#   Fix: copy FastQuantumKernel từ 02_qksvr_core_fixed
#
# BUG 2 — FEATURES thiếu CPI_return và PHASE_dummy (không nhất quán với pipeline)
#   Fix: đồng bộ 8 features với QK-SVR pipeline
#
# BUG 3 — svr_params lấy từ rbf_svm optuna params (không phải ablation intent)
#   Fix: dùng params cố định C=1.0, epsilon=0.1 (paper Section 5.1)
#        Ablation study phải giữ tất cả thứ khác cố định, chỉ thay n_qubits
#
# BUG 4 — N_SEEDS=30 nhưng comment nói 10
#   Fix: N_SEEDS=10 (đủ cho ablation, nhanh hơn 3x)
#
# BUG 5 — y_true/y_pred lưu list thay vì scalar
#   Fix: float(y_te[0]), float(y_pred[0])
#
# BUG 6 — No NaN handling
#   Fix: ffill().bfill() sau load
#
# BUG 7 — No parquet fallback
#   Fix: try parquet → fallback CSV
#
# BUG 8 — No FAST_MODE
#   Fix: FAST_MODE=True → 5 folds, 3 seeds để verify
# =============================================================================

import json
import logging
import time
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)

# =============================================================================
# CẤU HÌNH — chỉnh tại đây rồi nhấn Run trong VS Code
# =============================================================================
FAST_MODE = False   # True = 5 folds, 3 seeds (~5 phút) | False = full (~3.7 giờ)

BASE_DIR = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project")
# =============================================================================

DATA_PATH   = BASE_DIR / "Data" / "data" / "processed" / "master_dataset.csv"
FOLDS_PATH  = BASE_DIR / "Data" / "results" / "fold_splits.parquet"
PARAMS_PATH = BASE_DIR / "Data" / "results" / "optuna_best_params.json"
OUTPUT_DIR  = BASE_DIR / "Data" / "results"
CHECKPOINT  = OUTPUT_DIR / "ablation_ab1_checkpoint.parquet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "EUA_return"
# BUG 2 FIX: đồng bộ 8 features với QK-SVR pipeline
FEATURES = [
    "GAS_return", "OIL_return", "COAL_return", "ELEC_return",
    "IP_return", "CPI_return", "POLICY_dummy", "PHASE_dummy",
]

HORIZONS  = [1, 22]           # chỉ H=1 và H=22 để tiết kiệm thời gian
N_LAYERS  = 2                 # fixed — chỉ thay đổi n_qubits
NYSTROM_M = 100

# BUG 3 FIX: params cố định cho ablation (không dùng optuna rbf_svm)
# Ablation phải cô lập biến: chỉ n_qubits thay đổi, tất cả khác giữ nguyên
SVR_C       = 1.0
SVR_EPSILON = 0.1

# BUG 4 FIX: 10 seeds đủ cho ablation (comment và code nhất quán)
if FAST_MODE:
    N_SEEDS      = 3
    MAX_FOLDS    = 5
    QUBIT_CONFIGS = [4, 6]
    logging.info("FAST MODE: 5 folds, 3 seeds (~5 phút)")
else:
    N_SEEDS      = 10          # BUG 4 FIX: 10 thay vì 30
    MAX_FOLDS    = None
    QUBIT_CONFIGS = [4, 6]
    logging.info(f"FULL MODE: all folds, {N_SEEDS} seeds, qubits={QUBIT_CONFIGS}")


# ── BUG 1 FIX: numpy FastQuantumKernel (same as 02_qksvr_core_fixed) ─
class FastQuantumKernel:
    """
    Numpy vectorized quantum kernel — identical to 02_qksvr_core_fixed.
    Supports arbitrary n_qubits (4 or 6 for ablation).
    ~600x faster than PennyLane default.qubit.
    """
    def __init__(self, n_qubits: int, n_layers: int, seed: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim      = 2 ** n_qubits
        rng = np.random.default_rng(seed)
        self.params = rng.uniform(-0.01, 0.01, (n_layers, n_qubits))
        self._cnot  = self._build_cnot()
        self._ry    = self._build_ry_all()

    @staticmethod
    def _ry_gate(t):
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
        return [self._kron_all([self._ry_gate(self.params[l, q])
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
        s  = np.zeros((N, self.dim), dtype=complex); s[:, 0] = 1.0
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                s = self._apply_rx_batch(s, Xc[:, q], q)
            s = s @ self._cnot.T
            s = s @ self._ry[l].T
        return s

    def kernel(self, X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
        sa = self.compute_states(X_a)
        sb = self.compute_states(X_b)
        return np.real(np.abs(sa @ sb.conj().T) ** 2)


# ── QKSVR runner ──────────────────────────────────────────────────
def run_qksvr(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray,
    n_qubits: int, n_layers: int, seed: int,
    svr_c: float = SVR_C, svr_eps: float = SVR_EPSILON,
) -> np.ndarray:
    """
    Run one QK-SVR fold.
    BUG 1 FIX: uses FastQuantumKernel (numpy) not PennyLane.
    BUG 3 FIX: uses fixed SVR_C, SVR_EPSILON (not optuna rbf_svm params).
    """
    sc      = RobustScaler(quantile_range=(5, 95)).fit(X_tr)
    X_tr_sc = sc.transform(X_tr)
    X_te_sc = sc.transform(X_te)

    # Use first n_qubits features (only these are embedded)
    n_q     = min(n_qubits, X_tr_sc.shape[1])
    X_tr_q  = X_tr_sc[:, :n_q]
    X_te_q  = X_te_sc[:, :n_q]

    fqk = FastQuantumKernel(n_q, n_layers, seed)

    # Nyström landmarks via k-means
    m      = min(NYSTROM_M, len(X_tr_q))
    km     = KMeans(n_clusters=m, random_state=seed, n_init=5).fit(X_tr_q)
    dists  = np.linalg.norm(X_tr_q[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    lm_idx = np.argmin(dists, axis=0)
    X_lm   = X_tr_q[lm_idx]

    # Kernel + Nyström
    K_ll = fqk.kernel(X_lm, X_lm)
    w, v = np.linalg.eigh(K_ll)
    w[w < 1e-8] = 1e-8
    K_ll_inv = np.linalg.pinv(v @ np.diag(w) @ v.T)

    K_tr_lm = fqk.kernel(X_tr_q, X_lm)
    K_nys   = K_tr_lm @ K_ll_inv @ K_tr_lm.T
    w2, v2  = np.linalg.eigh(K_nys)
    w2[w2 < 1e-8] = 1e-8
    K_nys_psd = v2 @ np.diag(w2) @ v2.T

    svr = SVR(kernel="precomputed", C=svr_c, epsilon=svr_eps)
    svr.fit(K_nys_psd, y_tr)

    K_te_lm  = fqk.kernel(X_te_q, X_lm)
    K_te_nys = K_te_lm @ K_ll_inv @ K_tr_lm.T
    return svr.predict(K_te_nys)


# ── Helpers ───────────────────────────────────────────────────────
def load_folds(path: Path) -> pd.DataFrame:
    """BUG 7 FIX: fallback CSV."""
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No fold_splits in {path.parent}")


def save_checkpoint(results: list) -> None:
    try:
        pd.DataFrame(results).to_parquet(CHECKPOINT, index=False, engine="pyarrow")
    except Exception:
        pd.DataFrame(results).to_csv(CHECKPOINT.with_suffix(".csv"), index=False)


# ── Main ──────────────────────────────────────────────────────────
def main():
    # BUG 6 FIX: load + NaN handling
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").sort_index()
    n_nan = df[FEATURES + [TARGET]].isna().sum().sum()
    if n_nan > 0:
        df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].ffill().bfill()
        logging.info(f"Filled {n_nan} NaN values")

    # BUG 7 FIX: parquet fallback
    folds = load_folds(FOLDS_PATH)
    folds = folds[folds["horizon"].isin(HORIZONS)]

    # FAST_MODE: subset folds
    if MAX_FOLDS:
        folds = folds.head(MAX_FOLDS)
        logging.info(f"FAST MODE: {MAX_FOLDS} folds/config")

    # Resume checkpoint
    if CHECKPOINT.exists():
        try:
            df_done = pd.read_parquet(CHECKPOINT)
        except Exception:
            df_done = pd.read_csv(CHECKPOINT.with_suffix(".csv"))
        done_set = set(zip(df_done["n_qubits"], df_done["fold_id"], df_done["horizon"]))
        results  = df_done.to_dict("records")
        logging.info(f"Resume: {len(done_set)} combinations done")
    else:
        done_set, results = set(), []

    total   = len(QUBIT_CONFIGS) * len(folds)
    counter = 0
    t_start = time.time()

    logging.info(f"AB1 Qubit Ablation | Configs: {QUBIT_CONFIGS} | "
                 f"Seeds: {N_SEEDS} | Folds/config: {len(folds)}")

    for n_qubits in QUBIT_CONFIGS:
        logging.info(f"\n{'='*50}\n  n_qubits = {n_qubits}\n{'='*50}")

        for _, fold in folds.iterrows():
            fid  = int(fold["fold_id"])
            h    = int(fold["horizon"])
            counter += 1

            if (n_qubits, fid, h) in done_set:
                continue

            tr_s = pd.to_datetime(fold["train_start"])
            tr_e = pd.to_datetime(fold["train_end"])
            te_s = pd.to_datetime(fold["test_start"])
            te_e = pd.to_datetime(fold["test_end"])

            X_tr = df.loc[tr_s:tr_e, FEATURES].values
            y_tr = df.loc[tr_s:tr_e, TARGET].values
            X_te = df.loc[te_s:te_e, FEATURES].values
            y_te = df.loc[te_s:te_e, TARGET].values

            if len(X_tr) == 0 or len(X_te) == 0:
                logging.warning(f"  Skip fold {fid} H={h}: empty data")
                continue

            t0 = time.time()
            seed_preds: List[np.ndarray] = []
            for seed in range(N_SEEDS):
                try:
                    pred = run_qksvr(X_tr, y_tr, X_te, n_qubits, N_LAYERS, seed)
                    seed_preds.append(pred)
                except Exception as e:
                    logging.warning(f"  seed={seed} failed: {e}")

            if not seed_preds:
                logging.warning(f"  All seeds failed: n_qubits={n_qubits} fold={fid} H={h}")
                continue

            y_pred    = np.stack(seed_preds).mean(axis=0)
            fold_rmse = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
            elapsed   = round(time.time() - t0, 2)

            # ETA
            elapsed_total = time.time() - t_start
            rate    = counter / max(elapsed_total, 1e-6)
            eta_sec = (total - counter) / max(rate, 1e-6)
            eta_str = f"{eta_sec/3600:.1f}h" if eta_sec > 3600 else f"{eta_sec/60:.0f}m"

            results.append({
                "ablation":  "AB1_qubits",
                "config":    f"n_qubits={n_qubits}",
                "n_qubits":  n_qubits,
                "n_layers":  N_LAYERS,
                "fold_id":   fid,
                "horizon":   h,
                "y_true":    float(y_te[0]),    # BUG 5 FIX: scalar
                "y_pred":    float(y_pred[0]),  # BUG 5 FIX: scalar
                "fold_rmse": fold_rmse,
                "n_seeds":   len(seed_preds),
                "time_sec":  elapsed,
            })
            done_set.add((n_qubits, fid, h))

            logging.info(
                f"  [{counter}/{total}] qubits={n_qubits} "
                f"Fold {fid:03d} H={h}d | RMSE={fold_rmse:.5f} | "
                f"{elapsed:.1f}s | ETA {eta_str}"
            )

            if len(results) % 10 == 0:
                save_checkpoint(results)

    # Final save
    df_out   = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "ablation_ab1_qubits.parquet"
    try:
        df_out.to_parquet(out_path, index=False, engine="pyarrow")
    except Exception:
        df_out.to_csv(out_path.with_suffix(".csv"), index=False)
    if CHECKPOINT.exists():
        CHECKPOINT.unlink(missing_ok=True)

    # Summary
    summary = (df_out.groupby(["config", "horizon"])["fold_rmse"]
               .agg(["mean", "std", "count"])
               .reset_index()
               .rename(columns={"mean": "rmse_mean", "std": "rmse_std", "count": "n_folds"}))
    summary.to_csv(OUTPUT_DIR / "ablation_ab1_qubits_summary.csv", index=False)

    total_min = (time.time() - t_start) / 60
    print(f"\n{'='*55}")
    print("  AB1 QUBIT ABLATION SUMMARY")
    print(f"{'='*55}")
    print(summary.to_string(index=False))
    print(f"\n  Time: {total_min:.1f} min")
    print(f"  Saved: {out_path}")
    if FAST_MODE:
        print("\n  FAST MODE done. Set FAST_MODE=False for full run.")
    else:
        print("\n  AB1 xong. Chay tiep 07b, 07c, 07d roi 07e de gop.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
