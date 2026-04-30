# =============================================================================
# 07d_ablation_reuploading_fixed.py  —  v2
# =============================================================================
# BUG 1 [CRITICAL] — PennyLane serial → numpy FastQuantumKernel
#   Re-uploading specific: embedding trong mỗi layer vs chỉ layer đầu
#   Config A (re_upload=True):  RX(x) trong mỗi layer → stronger data encoding
#   Config B (re_upload=False): RX(x) chỉ layer 0 → dữ liệu ít được encode hơn
# BUG 2 — svr_params từ rbf_svm → fixed C=1.0, eps=0.1
# BUG 3 — No NaN handling
# BUG 4 — No parquet fallback
# BUG 5 — No FAST_MODE
# BUG 6 — done_set key dùng str(re_upload) → fragile, dùng bool trực tiếp
# FEATURES đã đầy đủ 8 ✅
# =============================================================================

import logging
import time
import warnings
from pathlib import Path

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
FAST_MODE = False   # True = 5 folds, 3 seeds | False = full (~2h)

BASE_DIR = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project")
# =============================================================================

DATA_PATH   = BASE_DIR / "Data" / "data" / "processed" / "master_dataset.csv"
FOLDS_PATH  = BASE_DIR / "Data" / "results" / "fold_splits.parquet"
OUTPUT_DIR  = BASE_DIR / "Data" / "results"
CHECKPOINT  = OUTPUT_DIR / "ablation_ab4_checkpoint.parquet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "EUA_return"
FEATURES = [
    "GAS_return", "OIL_return", "COAL_return", "ELEC_return",
    "IP_return", "CPI_return", "POLICY_dummy", "PHASE_dummy",
]
HORIZONS          = [1, 22]
N_QUBITS          = 4
N_LAYERS          = 2
NYSTROM_M         = 100
SVR_C             = 1.0    # BUG 2 FIX
SVR_EPSILON       = 0.1
REUPLOAD_CONFIGS  = [True, False]

if FAST_MODE:
    N_SEEDS = 3;  MAX_FOLDS = 5
    logging.info("FAST MODE: 5 folds, 3 seeds")
else:
    N_SEEDS = 10; MAX_FOLDS = None
    logging.info(f"FULL MODE: all folds, {N_SEEDS} seeds")


# ── BUG 1 FIX: FastQuantumKernel với re_upload flag ──────────────
class FastQuantumKernel:
    """
    Numpy quantum kernel với data re-uploading configurable.

    re_upload=True  (baseline): RX(x_q) applied in EVERY layer
      → circuit: [RX(x)|CNOT|RY]^L — data baked deeper into state
      → more expressive encoding (Pérez-Salinas et al. 2020)

    re_upload=False: RX(x_q) applied ONLY at layer 0
      → circuit: RX(x)|CNOT|RY → CNOT|RY → ...
      → pure variational layers after first encoding
    """
    def __init__(self, n_qubits: int, n_layers: int, re_upload: bool, seed: int):
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.re_upload = re_upload
        self.dim       = 2 ** n_qubits
        rng = np.random.default_rng(seed)
        self.params = rng.uniform(-0.01, 0.01, (n_layers, n_qubits))
        self._cnot  = self._build_cnot()
        self._ry    = self._build_ry_all()

    @staticmethod
    def _ry_gate(t):
        c, s = np.cos(t/2), np.sin(t/2)
        return np.array([[c,-s],[s,c]], dtype=complex)

    @staticmethod
    def _kron_all(gs):
        r = gs[0]
        for g in gs[1:]: r = np.kron(r, g)
        return r

    def _build_cnot(self):
        n, dim = self.n_qubits, self.dim
        def cnot(ctrl, tgt):
            U = np.zeros((dim,dim), dtype=complex)
            for col in range(dim):
                cb = (col>>(n-1-ctrl))&1
                U[col^(1<<(n-1-tgt)) if cb else col, col] = 1
            return U
        U = np.eye(dim, dtype=complex)
        for i in range(n): U = cnot(i,(i+1)%n) @ U   # circular
        return U

    def _build_ry_all(self):
        return [self._kron_all([self._ry_gate(self.params[l,q])
                                for q in range(self.n_qubits)])
                for l in range(self.n_layers)]

    def _rx_batch(self, s, angles, q):
        N, dim = s.shape; nq = self.n_qubits
        s = s.reshape(N, *([2]*nq))
        ax = 1+q
        i0=[slice(None)]*(nq+1); i0[ax]=0
        i1=[slice(None)]*(nq+1); i1[ax]=1
        s0,s1 = s[tuple(i0)],s[tuple(i1)]
        ca,sa = np.cos(angles/2),np.sin(angles/2)
        for _ in range(nq-1): ca=ca[...,None]; sa=sa[...,None]
        return np.stack([ca*s0-1j*sa*s1,-1j*sa*s0+ca*s1],axis=ax).reshape(N,dim)

    def compute_states(self, X: np.ndarray) -> np.ndarray:
        N  = len(X)
        Xc = np.clip(X[:, :self.n_qubits], -np.pi, np.pi)
        s  = np.zeros((N, self.dim), dtype=complex); s[:, 0] = 1.0
        for l in range(self.n_layers):
            # BUG 1 FIX: conditionally apply AngleEmbedding per layer
            if self.re_upload or l == 0:
                for q in range(self.n_qubits):
                    s = self._rx_batch(s, Xc[:, q], q)
            s = s @ self._cnot.T
            s = s @ self._ry[l].T
        return s

    def kernel(self, A, B):
        return np.real(np.abs(self.compute_states(A) @ self.compute_states(B).conj().T)**2)


# ── QKSVR runner ──────────────────────────────────────────────────
def run_qksvr(X_tr, y_tr, X_te, n_qubits, n_layers, re_upload, seed):
    sc  = RobustScaler(quantile_range=(5,95)).fit(X_tr)
    Xtr = sc.transform(X_tr)[:, :n_qubits]
    Xte = sc.transform(X_te)[:, :n_qubits]
    fqk = FastQuantumKernel(n_qubits, n_layers, re_upload, seed)

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
    return svr.predict(fqk.kernel(Xte,Xlm)@Kllinv@Ktrlm.T)


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
    # BUG 3 FIX: NaN handling
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").sort_index()
    n_nan = df[FEATURES+[TARGET]].isna().sum().sum()
    if n_nan > 0:
        df[FEATURES+[TARGET]] = df[FEATURES+[TARGET]].ffill().bfill()
        logging.info(f"Filled {n_nan} NaN")

    # BUG 4 FIX: parquet fallback
    folds = load_folds(FOLDS_PATH)
    folds = folds[folds["horizon"].isin(HORIZONS)]
    if MAX_FOLDS: folds = folds.head(MAX_FOLDS)

    # Resume — BUG 6 FIX: dùng bool key thay vì str(bool)
    if CHECKPOINT.exists():
        try: df_done = pd.read_parquet(CHECKPOINT)
        except: df_done = pd.read_csv(CHECKPOINT.with_suffix(".csv"))
        done_set = set(zip(df_done["re_upload"].astype(bool),
                           df_done["fold_id"],
                           df_done["horizon"]))
        results  = df_done.to_dict("records")
        logging.info(f"Resume: {len(done_set)} done")
    else:
        done_set, results = set(), []

    total = len(REUPLOAD_CONFIGS)*len(folds); counter = 0
    t0_all = time.time()
    logging.info(f"AB4 Re-uploading | configs={REUPLOAD_CONFIGS} | "
                 f"seeds={N_SEEDS} | folds/config={len(folds)}")

    for re_upload in REUPLOAD_CONFIGS:
        label = "re_upload=True" if re_upload else "re_upload=False"
        logging.info(f"\n{'='*50}\n  {label}\n{'='*50}")

        for _, fold in folds.iterrows():
            fid=int(fold["fold_id"]); h=int(fold["horizon"]); counter+=1
            if (re_upload, fid, h) in done_set: continue  # BUG 6 FIX: bool key

            tr_s=pd.to_datetime(fold["train_start"]); tr_e=pd.to_datetime(fold["train_end"])
            te_s=pd.to_datetime(fold["test_start"]);  te_e=pd.to_datetime(fold["test_end"])
            X_tr=df.loc[tr_s:tr_e,FEATURES].values; y_tr=df.loc[tr_s:tr_e,TARGET].values
            X_te=df.loc[te_s:te_e,FEATURES].values; y_te=df.loc[te_s:te_e,TARGET].values

            if len(X_tr)==0 or len(X_te)==0:
                logging.warning(f"  Skip fold {fid} H={h}: empty"); continue

            t0 = time.time()
            preds = []
            for seed in range(N_SEEDS):
                try: preds.append(run_qksvr(X_tr,y_tr,X_te,N_QUBITS,N_LAYERS,re_upload,seed))
                except Exception as e: logging.warning(f"    seed={seed}: {e}")
            if not preds: continue

            y_pred    = np.stack(preds).mean(axis=0)
            fold_rmse = float(np.sqrt(np.mean((y_te-y_pred)**2)))
            elapsed   = round(time.time()-t0, 2)
            rate      = counter/max(time.time()-t0_all,1e-6)
            eta_sec   = (total-counter)/max(rate,1e-6)
            eta_str   = f"{eta_sec/3600:.1f}h" if eta_sec>3600 else f"{eta_sec/60:.0f}m"

            results.append({
                "ablation":  "AB4_reuploading",
                "config":    label,
                "re_upload": re_upload,
                "n_qubits":  N_QUBITS, "n_layers": N_LAYERS,
                "fold_id":   fid,      "horizon":  h,
                "y_true":    float(y_te[0]),
                "y_pred":    float(y_pred[0]),
                "fold_rmse": fold_rmse,
                "n_seeds":   len(preds), "time_sec": elapsed,
            })
            done_set.add((re_upload, fid, h))
            logging.info(f"  [{counter}/{total}] {label} Fold {fid:03d} H={h}d | "
                         f"RMSE={fold_rmse:.5f} | {elapsed:.1f}s | ETA {eta_str}")
            if len(results)%10==0: save_ckpt(results)

    df_out = pd.DataFrame(results)
    out    = OUTPUT_DIR / "ablation_ab4_reuploading.parquet"
    try: df_out.to_parquet(out, index=False, engine="pyarrow")
    except: df_out.to_csv(out.with_suffix(".csv"), index=False)
    if CHECKPOINT.exists(): CHECKPOINT.unlink(missing_ok=True)

    summary = (df_out.groupby(["config","horizon"])["fold_rmse"]
               .agg(["mean","std","count"]).reset_index()
               .rename(columns={"mean":"rmse_mean","std":"rmse_std","count":"n_folds"}))
    summary.to_csv(OUTPUT_DIR/"ablation_ab4_reuploading_summary.csv", index=False)

    total_min = (time.time()-t0_all)/60
    print(f"\n{'='*50}\n  AB4 RE-UPLOADING ABLATION SUMMARY\n{'='*50}")
    print(summary.to_string(index=False))
    print(f"\n  Time: {total_min:.1f} min | Saved: {out}")
    if FAST_MODE: print("\n  FAST MODE done. Set FAST_MODE=False for full run.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
