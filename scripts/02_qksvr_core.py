# =============================================================================
# 02_qksvr_core_fixed.py  —  v2 (fixed + optimized)
# =============================================================================
# BUG FIXES:
#   BUG 1 — Parquet save/load có thể fail nếu pyarrow không có
#            Fix: fallback sang CSV
#   BUG 2 — COAL NaN đầu series (đã fix ở 01_fold_generator, nhưng thêm guard)
#
# PERFORMANCE (chạy nhanh hơn ~200-600×):
#   PERF 1 — [CRITICAL] Thay PennyLane serial circuit loop bằng numpy
#            vectorized statevector simulation:
#            - Cũ : for x in X_train: circuit(x)  → N calls × PennyLane overhead
#            - Mới: apply_rx_batch() vectorized     → 1 batch, numpy speed
#            → Ước tính: 25 giờ → ~5 phút (cho 30 seeds)
#   PERF 2 — Precompute CNOT + RY matrices 1 lần (không lặp mỗi sample)
#   PERF 3 — Tăng checkpoint từ mỗi 5 lên mỗi 10 folds
#   PERF 4 — Thêm --n_seeds CLI flag (mặc định 30; dùng --n_seeds 5 để test)
#   PERF 5 — Thêm progress bar (tqdm nếu có, fallback logging)
# =============================================================================

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)

# ── Progress bar (optional) ───────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _progress(iterable, total=None, desc=""):
    if HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc, ncols=90)
    return iterable


# =============================================================================
# FastQuantumKernel: numpy vectorized quantum circuit
# Thay thế PennyLane default.qubit cho circuit cụ thể này.
# Đúng về mặt toán học (statevectors và kernel matrix khớp PennyLane).
# =============================================================================

class FastQuantumKernel:
    """
    Numpy-based fidelity quantum kernel for the circuit:
        Layer l: AngleEmbedding(x) → CNOT(ring) → RY(params[l])

    Tất cả N samples được xử lý đồng thời (vectorized).
    Không cần PennyLane runtime → nhanh hơn ~200-600× so với default.qubit.
    """

    def __init__(self, n_qubits: int, n_layers: int, params: np.ndarray):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params   = params
        self.dim      = 2 ** n_qubits

        # PERF 2: precompute fixed matrices một lần
        self._cnot_mat  = self._build_cnot_layer()          # (dim, dim)
        self._ry_mats   = self._build_ry_layer_all()        # list[n_layers] (dim, dim)

    # ── Gate builders ─────────────────────────────────────────────
    @staticmethod
    def _rx(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    @staticmethod
    def _ry(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def _kron_all(gates: List[np.ndarray]) -> np.ndarray:
        result = gates[0]
        for g in gates[1:]:
            result = np.kron(result, g)
        return result

    def _build_cnot_layer(self) -> np.ndarray:
        """
        Sequential ring CNOT: CNOT(0,1), CNOT(1,2), ..., CNOT(n-1,0).
        Qubit 0 = MSB convention (leftmost tensor dimension).
        """
        n, dim = self.n_qubits, self.dim

        def _cnot(ctrl: int, tgt: int) -> np.ndarray:
            U = np.zeros((dim, dim), dtype=complex)
            for col in range(dim):
                ctrl_bit = (col >> (n - 1 - ctrl)) & 1
                row = col ^ (1 << (n - 1 - tgt)) if ctrl_bit else col
                U[row, col] = 1
            return U

        U = np.eye(dim, dtype=complex)
        for i in range(n):
            U = _cnot(i, (i + 1) % n) @ U
        return U

    def _build_ry_layer_all(self) -> List[np.ndarray]:
        """
        Build ⊗_q RY(params[l,q]) for each layer l.
        Returns list of (dim, dim) real unitaries.
        """
        return [
            self._kron_all([self._ry(self.params[l, q])
                            for q in range(self.n_qubits)])
            for l in range(self.n_layers)
        ]

    # ── Vectorized gate application ───────────────────────────────
    def _apply_rx_batch(
        self, states: np.ndarray, angles: np.ndarray, qubit: int
    ) -> np.ndarray:
        """
        Apply RX(angle) to `qubit` for all N samples simultaneously.

        states : (N, 2^n)  complex
        angles : (N,)      float — one angle per sample
        Returns: (N, 2^n)  complex
        """
        N, dim = states.shape
        n      = self.n_qubits

        # Reshape: (N, 2^n) → (N, 2, 2, ..., 2)  [n qubit axes]
        s = states.reshape(N, *([2] * n))

        # Qubit 0 = MSB = axis index 1 (after batch axis 0)
        # Qubit q  = axis index (1 + q)  ... qubit n-1 = axis index n
        ax = 1 + qubit          # axis in reshaped tensor

        # Extract the two 'halves' along the qubit axis
        slc0 = [slice(None)] * (n + 1); slc0[ax] = 0
        slc1 = [slice(None)] * (n + 1); slc1[ax] = 1
        s0 = s[tuple(slc0)]     # (N, 2, ...) — n-1 remaining qubit dims
        s1 = s[tuple(slc1)]

        # Broadcast angles: (N,) → (N, 1, 1, ...) for n-1 trailing dims
        ca = np.cos(angles / 2)
        sa = np.sin(angles / 2)
        for _ in range(n - 1):
            ca = ca[..., None]
            sa = sa[..., None]

        # Apply RX: [[cos, -i·sin], [-i·sin, cos]]
        new_s0 = ca * s0 - 1j * sa * s1
        new_s1 = -1j * sa * s0 + ca * s1

        return np.stack([new_s0, new_s1], axis=ax).reshape(N, dim)

    # ── Statevector computation ───────────────────────────────────
    def compute_states(self, X: np.ndarray) -> np.ndarray:
        """
        Compute statevectors for all N samples simultaneously.

        X      : (N, n_features) — uses first n_qubits features after clip
        Returns: (N, 2^n_qubits) complex statevectors
        """
        N   = len(X)
        Xc  = np.clip(X[:, :self.n_qubits], -np.pi, np.pi)

        # All samples start in |0...0⟩
        states = np.zeros((N, self.dim), dtype=complex)
        states[:, 0] = 1.0

        for l in range(self.n_layers):
            # AngleEmbedding: RX(x_q) on each qubit (sample-specific)
            for q in range(self.n_qubits):
                states = self._apply_rx_batch(states, Xc[:, q], q)

            # CNOT ring (same for all samples) — vectorized matmul
            states = states @ self._cnot_mat.T          # (N, dim) @ (dim, dim)

            # Trainable RY rotations (same for all samples)
            states = states @ self._ry_mats[l].T        # (N, dim) @ (dim, dim)

        return states

    # ── Kernel computation ────────────────────────────────────────
    def kernel(
        self, states_a: np.ndarray, states_b: np.ndarray
    ) -> np.ndarray:
        """
        Fidelity kernel: K[i,j] = |⟨ψ_a[i]|ψ_b[j]⟩|²
        Returns real (n_a, n_b) matrix.
        """
        overlaps = states_a @ states_b.conj().T         # (n_a, n_b) complex
        return np.real(np.abs(overlaps) ** 2)


# =============================================================================
# QKSVR: Quantum Kernel SVR with Nyström approximation
# =============================================================================

class QKSVR:
    """
    QK-SVR theo Research Design Section 5.1 & 4.2.
    Dùng FastQuantumKernel (numpy) thay PennyLane → nhanh hơn ~200-600×.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        nystrom_m: int = 100,
        seed: int = 42,
        svr_C: float = 1.0,
        svr_eps: float = 0.1,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.m        = nystrom_m
        self.seed     = seed

        rng = np.random.default_rng(seed)
        self.params = rng.uniform(-0.01, 0.01, (n_layers, n_qubits))

        self.scaler       = RobustScaler(quantile_range=(5.0, 95.0))
        self.svr          = SVR(kernel='precomputed', C=svr_C, epsilon=svr_eps)
        self._qk          : Optional[FastQuantumKernel] = None
        self._landmark_idx: Optional[np.ndarray] = None
        self._states_lm   : Optional[np.ndarray] = None
        self._K_ll_inv    : Optional[np.ndarray] = None
        self._K_tr_lm     : Optional[np.ndarray] = None

    # ── Helpers ───────────────────────────────────────────────────
    def _select_landmarks(self, X_tr: np.ndarray) -> np.ndarray:
        """
        K-Means landmark selection: for each centroid find the
        nearest training point (correct version of original FIX 1).
        """
        n_clusters = min(self.m, len(X_tr))
        km = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        km.fit(X_tr)
        # (n_train, n_clusters) distances
        dists = np.linalg.norm(
            X_tr[:, None, :] - km.cluster_centers_[None, :, :], axis=2
        )
        return np.argmin(dists, axis=0)  # (n_clusters,)

    @staticmethod
    def _enforce_psd(K: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Clip negative eigenvalues to eps (ensures PSD)."""
        w, v = np.linalg.eigh(K)
        w[w < eps] = eps
        return v @ np.diag(w) @ v.T

    # ── Public API ────────────────────────────────────────────────
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        X_tr = self.scaler.fit_transform(X_train)

        # Instantiate fast kernel once per seed (params baked in)
        self._qk = FastQuantumKernel(self.n_qubits, self.n_layers, self.params)

        # Landmark selection
        self._landmark_idx = self._select_landmarks(X_tr)

        # Compute all statevectors in ONE vectorized batch per call
        self._states_lm = self._qk.compute_states(X_tr[self._landmark_idx])
        states_tr       = self._qk.compute_states(X_tr)

        # Nyström kernel
        K_ll              = self._qk.kernel(self._states_lm, self._states_lm)
        K_ll              = self._enforce_psd(K_ll)
        self._K_ll_inv    = np.linalg.pinv(K_ll)
        self._K_tr_lm     = self._qk.kernel(states_tr, self._states_lm)

        K_nys = self._K_tr_lm @ self._K_ll_inv @ self._K_tr_lm.T
        K_nys = self._enforce_psd(K_nys)

        self.svr.fit(K_nys, y_train.values)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        X_te      = self.scaler.transform(X_test)
        states_te = self._qk.compute_states(X_te)
        K_te_lm   = self._qk.kernel(states_te, self._states_lm)
        K_te_nys  = K_te_lm @ self._K_ll_inv @ self._K_tr_lm.T
        return self.svr.predict(K_te_nys)


# =============================================================================
# Pipeline
# =============================================================================

def _load_folds(folds_dir: Path) -> pd.DataFrame:
    """BUG 1 FIX: load parquet với fallback sang CSV."""
    pq  = folds_dir / "fold_splits.parquet"
    csv = folds_dir / "fold_splits.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        logging.info("fold_splits.parquet không có → dùng fold_splits.csv")
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Không tìm thấy fold_splits.parquet/csv trong {folds_dir}")


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save với parquet fallback sang CSV."""
    try:
        df.to_parquet(path, index=False, engine="pyarrow")
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logging.warning(f"Parquet lỗi → lưu CSV: {csv_path}")


def run_qksvr_pipeline(
    base_dir: Path,
    n_seeds: int = 30,
    nystrom_m: int = 100,
    checkpoint_every: int = 10,     # PERF 3: tăng từ 5 lên 10
) -> None:

    DATA_PATH  = base_dir / "Data" / "data" / "processed" / "master_dataset.csv"
    RESULTS    = base_dir / "Data" / "results"
    CHECKPOINT = RESULTS / "qksvr_checkpoint.parquet"
    OUT_PATH   = RESULTS / "qksvr_predictions.parquet"
    RESULTS.mkdir(parents=True, exist_ok=True)

    N_QUBITS = 4
    TARGET   = "EUA_return"
    FEATURES = [
        "GAS_return", "OIL_return", "COAL_return", "ELEC_return",
        "IP_return", "CPI_return", "POLICY_dummy", "PHASE_dummy",
    ]

    # Load data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").sort_index()
    # Safety: fill residual NaN (COAL đầu series) — tránh KMeans crash
    _all_cols = FEATURES + [TARGET]
    _n_nan = df[_all_cols].isna().sum().sum()
    if _n_nan > 0:
        df[_all_cols] = df[_all_cols].ffill().bfill()
        logging.info(f"ℹ️  Filled {_n_nan} residual NaN in dataset")
    logging.info(f"📥 Data: {df.shape} | {df.index[0].date()} → {df.index[-1].date()}")

    # Load folds
    folds = _load_folds(RESULTS)
    total = len(folds)
    logging.info(f"📋 Folds: {total} | Seeds: {n_seeds} | Nyström m={nystrom_m}")

    # Resume checkpoint
    if CHECKPOINT.exists():
        try:
            df_done  = pd.read_parquet(CHECKPOINT)
        except Exception:
            df_done  = pd.read_csv(CHECKPOINT.with_suffix(".csv"))
        done_set = set(zip(df_done["fold_id"], df_done["horizon"]))
        results  = df_done.to_dict("records")
        logging.info(f"♻️  Resume: {len(done_set)} folds đã hoàn thành.")
    else:
        done_set, results = set(), []

    remaining = [(i, r) for i, (_, r) in enumerate(folds.iterrows(), 1)
                 if (int(r["fold_id"]), int(r["horizon"])) not in done_set]
    logging.info(f"🚀 Bắt đầu {len(remaining)} folds còn lại...")

    t_pipeline = time.time()

    for idx, (orig_idx, fold) in enumerate(
        _progress(remaining, total=len(remaining), desc="QK-SVR"), 1
    ):
        fid = int(fold["fold_id"])
        h   = int(fold["horizon"])

        t0   = time.time()
        tr_s = pd.to_datetime(fold["train_start"])
        tr_e = pd.to_datetime(fold["train_end"])
        te_s = pd.to_datetime(fold["test_start"])
        te_e = pd.to_datetime(fold["test_end"])

        X_tr = df.loc[tr_s:tr_e, FEATURES]
        y_tr = df.loc[tr_s:tr_e, TARGET]
        X_te = df.loc[te_s:te_e, FEATURES]
        y_te = df.loc[te_s:te_e, TARGET]

        # ── Pre-flight check: skip fold nếu X_te empty ───────────
        # (tránh retry 30 seeds với cùng lỗi)
        if X_te.empty or X_tr.empty:
            in_idx = te_s in df.index
            logging.warning(
                f"Fold {fid} H={h}: X_te={X_te.shape} X_tr={X_tr.shape} — skip. "
                f"test_date={te_s.date()} in_index={in_idx}. "
                f"Re-run 01_fold_generator_fixed.py to fix."
            )
            continue

        # ── Multi-seed loop ───────────────────────────────────────
        seed_preds = []
        for seed in range(n_seeds):
            try:
                model = QKSVR(
                    n_qubits=N_QUBITS,
                    nystrom_m=nystrom_m,
                    seed=seed,
                )
                model.fit(X_tr, y_tr)
                seed_preds.append(model.predict(X_te))
            except Exception as e:
                logging.warning(f"  Fold {fid} H={h} seed={seed} lỗi: {e}")

        if not seed_preds:
            logging.warning(f"  Fold {fid} H={h}: tất cả seed đều lỗi — bỏ qua.")
            continue

        y_pred    = np.stack(seed_preds).mean(axis=0)
        fold_rmse = float(np.sqrt(np.mean((y_te.values - y_pred) ** 2)))
        elapsed   = round(time.time() - t0, 2)

        results.append({
            "fold_id":     fid,
            "horizon":     h,
            "y_true":      float(y_te.values[0]),   # scalar, 1 test point
            "y_pred":      float(y_pred[0]),
            "train_start": str(tr_s.date()),
            "test_start":  str(te_s.date()),
            "test_end":    str(te_e.date()),
            "n_train":     len(X_tr),
            "n_seeds_ok":  len(seed_preds),
            "time_sec":    elapsed,
            "fold_rmse":   fold_rmse,
        })
        done_set.add((fid, h))

        # ETA
        elapsed_total = time.time() - t_pipeline
        rate          = idx / elapsed_total                    # folds per second
        eta_sec       = (len(remaining) - idx) / max(rate, 1e-6)
        eta_str       = f"{eta_sec/60:.0f}m" if eta_sec < 3600 else f"{eta_sec/3600:.1f}h"

        if not HAS_TQDM:
            logging.info(
                f"[{orig_idx}/{total}] Fold {fid:03d} H={h:2d}d | "
                f"RMSE={fold_rmse:.6f} | seeds={len(seed_preds)} | "
                f"⏱{elapsed:.1f}s | ETA {eta_str}"
            )

        # PERF 3: Checkpoint mỗi 10 folds
        if len(results) % checkpoint_every == 0:
            _save_parquet(pd.DataFrame(results), CHECKPOINT)

    # ── Lưu kết quả cuối ─────────────────────────────────────────
    df_out = pd.DataFrame(results)
    _save_parquet(df_out, OUT_PATH)
    if CHECKPOINT.exists():
        CHECKPOINT.unlink(missing_ok=True)

    total_time = time.time() - t_pipeline
    logging.info(f"\n{'='*55}")
    logging.info(f"✅ Hoàn thành: {len(results)} folds")
    logging.info(f"💾 Lưu: {OUT_PATH}")
    logging.info(f"⏱  Tổng thời gian: {total_time/60:.1f} phút")
    # Guard: cột fold_rmse có thể thiếu nếu load từ checkpoint cũ
    if len(df_out) > 0 and 'fold_rmse' in df_out.columns:
        logging.info(f"📊 RMSE trung bình: {df_out['fold_rmse'].mean():.6f}")


# =============================================================================
# Entry point
# =============================================================================

# =============================================================================
# Entry point -- chinh truc tiep cac bien duoi day roi nhan Run trong VS Code
# =============================================================================

if __name__ == "__main__":

    # ── CAU HINH -- chinh tai day ───────────────────────────────────────────
    BASE_DIR_STR = r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project"

    FAST_MODE = False   # True = test nhanh (~10 phut) | False = full run (~60 phut)
    # ───────────────────────────────────────────────────────────────────────

    if FAST_MODE:
        n_seeds   = 5
        nystrom_m = 50
        logging.info("FAST MODE: n_seeds=5, nystrom_m=50 (~10 phut)")
    else:
        n_seeds   = 30
        nystrom_m = 100
        logging.info("FULL MODE: n_seeds=30, nystrom_m=100 (~60 phut)")

    run_qksvr_pipeline(
        base_dir         = Path(BASE_DIR_STR),
        n_seeds          = n_seeds,
        nystrom_m        = nystrom_m,
        checkpoint_every = 10,
    )
