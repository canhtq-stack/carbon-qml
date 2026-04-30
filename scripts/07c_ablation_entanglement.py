# =============================================================================
# 07c_ablation_entanglement_fixed.py  —  v2
# =============================================================================
# BUG 1 [CRITICAL] — PennyLane serial → numpy FastQuantumKernel
#   Entanglement-specific: numpy precomputes CNOT matrix per pattern 1 lần
#   → full (12 CNOTs) chạy cùng tốc độ circular (4 CNOTs) sau khi precompute
# BUG 3 — svr_params từ rbf_svm → fixed C=1.0, eps=0.1
# BUG 4 — No NaN handling
# BUG 5 — No parquet fallback
# BUG 6 — No FAST_MODE
# FEATURES đã đầy đủ 8 ✅ — không cần sửa
# =============================================================================

import json
import logging
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(asctime)s | %(message)s",
                    datefmt="%H:%M:%S")

# =============================================================================
# CẤU HÌNH
# =============================================================================
FAST_MODE = False   # True = 5 folds, 3 seeds | False = full (~5h)

BASE_DIR = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project")
# =============================================================================

DATA_PATH   = BASE_DIR / "Data" / "data" / "processed" / "master_dataset.csv"
FOLDS_PATH  = BASE_DIR / "Data" / "results" / "fold_splits.parquet"
OUTPUT_DIR  = BASE_DIR / "Data" / "results"
CHECKPOINT  = OUTPUT_DIR / "ablation_ab3_checkpoint.parquet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "EUA_return"
FEATURES = [
    "GAS_return", "OIL_return", "COAL_return", "ELEC_return",
    "IP_return", "CPI_return", "POLICY_dummy", "PHASE_dummy",
]
HORIZONS             = [1, 22]
N_QUBITS             = 4
N_LAYERS             = 2
NYSTROM_M            = 100
SVR_C                = 1.0    # BUG 3 FIX: fixed
SVR_EPSILON          = 0.1
ENTANGLEMENT_CONFIGS = ["linear", "circular", "full"]

if FAST_MODE:
    N_SEEDS = 3;  MAX_FOLDS = 5
    logging.info("FAST MODE: 5 folds, 3 seeds")
else:
    N_SEEDS = 10; MAX_FOLDS = None
    logging.info(f"FULL MODE: all folds, {N_SEEDS} seeds, patterns={ENTANGLEMENT_CONFIGS}")


# ── BUG 1 FIX: FastQuantumKernel hỗ trợ 3 entanglement patterns ──
class FastQuantumKernel:
    """
    Numpy quantum kernel với entanglement pattern configurable.
    Precompute CNOT matrix 1 lần → full/circular/linear đều nhanh như nhau.

    linear  : CNOT(0→1), CNOT(1→2), CNOT(2→3)           — 3 gates
    circular: CNOT(0→1), CNOT(1→2), CNOT(2→3), CNOT(3→0) — 4 gates (baseline)
    full    : CNOT(i→j) for all i≠j                       — 12 gates
    """
    def __init__(self, n_qubits: int, n_layers: int, entanglement: str, seed: int):
        self.n_qubits    = n_qubits
        self.n_layers    = n_layers
        self.entanglement = entanglement
        self.dim         = 2 ** n_qubits
        rng = np.random.default_rng(seed)
        self.params = rng.uniform(-0.01, 0.01, (n_layers, n_qubits))
        # Precompute both CNOT and RY matrices once
        self._cnot = self._build_cnot_layer(entanglement)
        self._ry   = self._build_ry_all()

    @staticmethod
    def _ry_gate(t):
        c, s = np.cos(t/2), np.sin(t/2)
        return np.array([[c,-s],[s,c]], dtype=complex)

    @staticmethod
    def _kron_all(gs):
        r = gs[0]
        for g in gs[1:]: r = np.kron(r, g)
        return r

    def _build_single_cnot(self, ctrl: int, tgt: int) -> np.ndarray:
        n, dim = self.n_qubits, self.dim
        U = np.zeros((dim, dim), dtype=complex)
        for col in range(dim):
            cb = (col >> (n-1-ctrl)) & 1
            U[col ^ (1 << (n-1-tgt)) if cb else col, col] = 1
        return U

    def _build_cnot_layer(self, pattern: str) -> np.ndarray:
        """Build combined CNOT layer matrix for given entanglement pattern."""
        n = self.n_qubits
        U = np.eye(self.dim, dtype=complex)

        if pattern == "linear":
            # CNOT(0→1), (1→2), ..., (n-2→n-1) — no wrap
            pairs = [(i, i+1) for i in range(n-1)]
        elif pattern == "circular":
            # CNOT(0→1), ..., (n-2→n-1), (n-1→0) — wrap
            pairs = [(i, (i+1)%n) for i in range(n)]
        elif pattern == "full":
            # CNOT(i→j) for all i≠j in order
            pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        else:
            raise ValueError(f"Unknown entanglement: {pattern}")

        for ctrl, tgt in pairs:
            U = self._build_single_cnot(ctrl, tgt) @ U

        logging.debug(f"  [{pattern}] CNOT layer: {len(pairs)} gates, "
                      f"matrix shape {U.shape}")
        return U

    def _build_ry_all(self) -> List[np.ndarray]:
        return [self._kron_all([self._ry_gate(self.params[l,q])
                                for q in range(self.n_qubits)])
                for l in range(self.n_layers)]

    def _rx_batch(self, s: np.ndarray, angles: np.ndarray, q: int) -> np.ndarray:
        N, dim = s.shape; nq = self.n_qubits
        s = s.reshape(N, *([2]*nq))
        ax = 1+q
        i0=[slice(None)]*(nq+1); i0[ax]=0
        i1=[slice(None)]*(nq+1); i1[ax]=1
        s0, s1 = s[tuple(i0)], s[tuple(i1)]
        ca, sa = np.cos(angles/2), np.sin(angles/2)
        for _ in range(nq-1): ca=ca[...,None]; sa=sa[...,None]
        return np.stack([ca*s0-1j*sa*s1, -1j*sa*s0+ca*s1], axis=ax).reshape(N,dim)

    def compute_states(self, X: np.ndarray) -> np.ndarray:
        N  = len(X)
        Xc = np.clip(X[:, :self.n_qubits], -np.pi, np.pi)
        s  = np.zeros((N, self.dim), dtype=complex); s[:, 0] = 1.0
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                s = self._rx_batch(s, Xc[:, q], q)
            s = s @ self._cnot.T   # entanglement layer (precomputed)
            s = s @ self._ry[l].T  # trainable rotations
        return s

    def kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sa = self.compute_states(A)
        sb = self.compute_states(B)
        return np.real(np.abs(sa @ sb.conj().T) ** 2)


# ── QKSVR runner ──────────────────────────────────────────────────
def run_qksvr(X_tr, y_tr, X_te, n_qubits, n_layers, entanglement, seed):
    sc  = RobustScaler(quantile_range=(5,95)).fit(X_tr)
    Xtr = sc.transform(X_tr)[:, :n_qubits]
    Xte = sc.transform(X_te)[:, :n_qubits]
    fqk = FastQuantumKernel(n_qubits, n_layers, entanglement, seed)

    m      = min(NYSTROM_M, len(Xtr))
    km     = KMeans(n_clusters=m, random_state=seed, n_init=5).fit(Xtr)
    dists  = np.linalg.norm(Xtr[:,None]-km.cluster_centers_[None], axis=2)
    Xlm    = Xtr[np.argmin(dists, axis=0)]

    Kll    = fqk.kernel(Xlm, Xlm)
    w,v    = np.linalg.eigh(Kll); w[w<1e-8]=1e-8
    Kllinv = np.linalg.pinv(v@np.diag(w)@v.T)

    Ktrlm  = fqk.kernel(Xtr, Xlm)
    Knys   = Ktrlm@Kllinv@Ktrlm.T
    w2,v2  = np.linalg.eigh(Knys); w2[w2<1e-8]=1e-8
    svr    = SVR(kernel="precomputed", C=SVR_C, epsilon=SVR_EPSILON)
    svr.fit(v2@np.diag(w2)@v2.T, y_tr)

    Ktelm  = fqk.kernel(Xte, Xlm)
    return svr.predict(Ktelm@Kllinv@Ktrlm.T)


# ── Helpers ───────────────────────────────────────────────────────
def load_folds(path):
    if path.exists(): return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists(): return pd.read_csv(csv)
    raise FileNotFoundError(f"No fold_splits in {path.parent}")

def save_ckpt(results):
    try: pd.DataFrame(results).to_parquet(CHECKPOINT, index=False, engine="pyarrow")
    except: pd.DataFrame(results).to_csv(CHECKPOINT.with_suffix(".csv"), index=False)


# ── Main ──────────────────────────────────────────────────────────
def main():
    # BUG 4 FIX: NaN handling
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").sort_index()
    n_nan = df[FEATURES+[TARGET]].isna().sum().sum()
    if n_nan > 0:
        df[FEATURES+[TARGET]] = df[FEATURES+[TARGET]].ffill().bfill()
        logging.info(f"Filled {n_nan} NaN")

    # BUG 5 FIX: parquet fallback
    folds = load_folds(FOLDS_PATH)
    folds = folds[folds["horizon"].isin(HORIZONS)]
    if MAX_FOLDS: folds = folds.head(MAX_FOLDS)

    # Resume
    if CHECKPOINT.exists():
        try: df_done = pd.read_parquet(CHECKPOINT)
        except: df_done = pd.read_csv(CHECKPOINT.with_suffix(".csv"))
        done_set = set(zip(df_done["entanglement"],df_done["fold_id"],df_done["horizon"]))
        results  = df_done.to_dict("records")
        logging.info(f"Resume: {len(done_set)} done")
    else:
        done_set, results = set(), []

    total = len(ENTANGLEMENT_CONFIGS)*len(folds); counter = 0
    t0_all = time.time()
    logging.info(f"AB3 Entanglement | patterns={ENTANGLEMENT_CONFIGS} | "
                 f"seeds={N_SEEDS} | folds/pattern={len(folds)}")

    for ent in ENTANGLEMENT_CONFIGS:
        logging.info(f"\n{'='*50}\n  entanglement={ent}\n{'='*50}")
        for _, fold in folds.iterrows():
            fid=int(fold["fold_id"]); h=int(fold["horizon"]); counter+=1
            if (ent,fid,h) in done_set: continue

            tr_s=pd.to_datetime(fold["train_start"]); tr_e=pd.to_datetime(fold["train_end"])
            te_s=pd.to_datetime(fold["test_start"]);  te_e=pd.to_datetime(fold["test_end"])
            X_tr=df.loc[tr_s:tr_e,FEATURES].values; y_tr=df.loc[tr_s:tr_e,TARGET].values
            X_te=df.loc[te_s:te_e,FEATURES].values; y_te=df.loc[te_s:te_e,TARGET].values

            if len(X_tr)==0 or len(X_te)==0:
                logging.warning(f"  Skip fold {fid} H={h}: empty"); continue

            t0 = time.time()
            preds = []
            for seed in range(N_SEEDS):
                try: preds.append(run_qksvr(X_tr,y_tr,X_te,N_QUBITS,N_LAYERS,ent,seed))
                except Exception as e: logging.warning(f"    seed={seed}: {e}")
            if not preds: continue

            y_pred    = np.stack(preds).mean(axis=0)
            fold_rmse = float(np.sqrt(np.mean((y_te-y_pred)**2)))
            elapsed   = round(time.time()-t0, 2)
            rate      = counter/max(time.time()-t0_all,1e-6)
            eta_sec   = (total-counter)/max(rate,1e-6)
            eta_str   = f"{eta_sec/3600:.1f}h" if eta_sec>3600 else f"{eta_sec/60:.0f}m"

            results.append({
                "ablation":     "AB3_entanglement",
                "config":       f"entanglement={ent}",
                "entanglement": ent,
                "n_qubits":     N_QUBITS, "n_layers": N_LAYERS,
                "fold_id":      fid,      "horizon":  h,
                "y_true":       float(y_te[0]),   # scalar
                "y_pred":       float(y_pred[0]), # scalar
                "fold_rmse":    fold_rmse,
                "n_seeds":      len(preds), "time_sec": elapsed,
            })
            done_set.add((ent,fid,h))
            logging.info(f"  [{counter}/{total}] ent={ent:8} Fold {fid:03d} H={h}d | "
                         f"RMSE={fold_rmse:.5f} | {elapsed:.1f}s | ETA {eta_str}")
            if len(results)%10==0: save_ckpt(results)

    df_out = pd.DataFrame(results)
    out    = OUTPUT_DIR / "ablation_ab3_entanglement.parquet"
    try: df_out.to_parquet(out, index=False, engine="pyarrow")
    except: df_out.to_csv(out.with_suffix(".csv"), index=False)
    if CHECKPOINT.exists(): CHECKPOINT.unlink(missing_ok=True)

    summary = (df_out.groupby(["config","horizon"])["fold_rmse"]
               .agg(["mean","std","count"]).reset_index()
               .rename(columns={"mean":"rmse_mean","std":"rmse_std","count":"n_folds"}))
    summary.to_csv(OUTPUT_DIR/"ablation_ab3_entanglement_summary.csv", index=False)

    total_min = (time.time()-t0_all)/60
    print(f"\n{'='*50}\n  AB3 ENTANGLEMENT ABLATION SUMMARY\n{'='*50}")
    print(summary.to_string(index=False))
    print(f"\n  Time: {total_min:.1f} min | Saved: {out}")
    if FAST_MODE: print("\n  FAST MODE done. Set FAST_MODE=False for full run.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
