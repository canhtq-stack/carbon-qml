# =============================================================================
# 03c_neural_nets_fixed.py  —  v2
# =============================================================================
# BUG 1 [CRITICAL] — predict_seq() O(N) per-sample loop
#   Fix: batch_predict() — build all sequences at once, 1 forward pass
#
# BUG 2 [CRITICAL] — Transformer dùng params.get("d_model", 64) sai key
#   Optuna tune d_model_mult, code tìm "d_model" → dùng default 64 thay vì 8
#   Fix: d_model = nhead × d_model_mult
#
# BUG 3 — No NaN handling khi load dataset → KMeans/SVR crash
#   Fix: ffill().bfill() sau load
#
# BUG 4 — No parquet fallback cho fold_splits
#   Fix: thử parquet → fallback CSV
#
# BUG 5 — No pre-flight check cho empty X_te/X_tr
#   Fix: skip fold + log rõ lý do
#
# BUG 6 — train_net dùng walrus operator trong lambda sum() → unreadable
#   Fix: rewrite thành vòng lặp rõ ràng
#
# BUG 7 — No FAST_MODE cho VS Code testing
#   Fix: FAST_MODE=True chỉ 5 folds/model, N_SEEDS=3
#
# BUG 8 — y_true/y_pred_mean lưu dạng list (1 test point) → nên scalar
#   Fix: float(y_te[0]), float(y_pred_mean[0])
#
# BUG 9 — EMD fallback dùng gaussian_filter1d thay vì EMD thật
#   Fix: cảnh báo rõ ràng; recommend cài PyEMD; fallback hợp lý hơn
# =============================================================================

import json
import logging
import os
import time
import warnings
from functools import partial
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp
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
FAST_MODE = False   # True = 5 folds/model, 3 seeds (~5 phút) | False = full run
N_JOBS    = 4      # Số seeds chạy song song (parallel) — chỉnh theo số CPU cores máy bạn
               # Xem số cores: os.cpu_count() trong Python
               # Khuyến nghị: N_JOBS = số cores - 1 (để máy không bị treo)

BASE_DIR  = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project")
# =============================================================================

DATA_PATH   = BASE_DIR / "Data" / "data" / "processed" / "master_dataset.csv"
FOLDS_PATH  = BASE_DIR / "Data" / "results" / "fold_splits.parquet"
PARAMS_PATH = BASE_DIR / "Data" / "results" / "optuna_best_params.json"
OUTPUT_DIR  = BASE_DIR / "Data" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT  = OUTPUT_DIR / "predictions_neural_nets_checkpoint.parquet"

TARGET   = "EUA_return"
FEATURES = [
    "GAS_return", "OIL_return", "COAL_return", "ELEC_return",
    "IP_return", "CPI_return", "POLICY_dummy", "PHASE_dummy",
]
N_FEAT = len(FEATURES)
MODELS = ["bilstm", "gru", "transformer", "emd_lstm"]
INIT_SEED = 42

if FAST_MODE:
    N_SEEDS    = 3
    MAX_FOLDS  = 5
    logging.info("FAST MODE: 5 folds/model, 3 seeds (~5 phút)")
else:
    N_SEEDS    = 30
    MAX_FOLDS  = None
    logging.info(f"FULL MODE: {N_SEEDS} seeds × N_JOBS={N_JOBS} parallel (~{9.2/N_JOBS:.1f}h với {N_JOBS} cores)")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {DEVICE}")


# ── Helpers ───────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_folds(path: Path) -> pd.DataFrame:
    """BUG 4 FIX: fallback CSV."""
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        logging.info("fold_splits.parquet not found → using CSV")
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No fold_splits file in {path.parent}")


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False, engine="pyarrow")
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)


def build_train_sequences(X_sc: np.ndarray, seq_len: int):
    """Build (N-sl, sl, F) training sequences using numpy stride tricks (no Python loop)."""
    sl = min(seq_len, len(X_sc) - 1)
    n  = len(X_sc) - sl
    if n <= 0:
        return np.empty((0, sl, X_sc.shape[1]), dtype=np.float32), sl
    shape   = (n, sl, X_sc.shape[1])
    strides = (X_sc.strides[0], X_sc.strides[0], X_sc.strides[1])
    Xs = np.lib.stride_tricks.as_strided(X_sc, shape=shape, strides=strides).copy().astype(np.float32)
    return Xs, sl


def batch_predict(
    net: nn.Module, X_tr_sc: np.ndarray, X_te_sc: np.ndarray, seq_len: int
) -> np.ndarray:
    """
    BUG 1 FIX: batch all test sequences → 1 forward pass instead of N.

    Builds N_te sequences using sliding window over [X_tr tail | X_te].
    """
    N_te     = len(X_te_sc)
    combined = np.vstack([X_tr_sc[-seq_len:], X_te_sc])   # (sl + N_te, F)
    seqs     = np.stack(
        [combined[i: i + seq_len] for i in range(N_te)], axis=0
    ).astype(np.float32)                                    # (N_te, sl, F)

    net.eval()
    with torch.no_grad():
        preds = net(torch.tensor(seqs).to(DEVICE)).cpu().numpy()
    return preds


# ── Network definitions ───────────────────────────────────────────
class BiLSTMNet(nn.Module):
    def __init__(self, n_feat, hidden, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(self.drop(o[:, -1, :])).squeeze(-1)


class GRUNet(nn.Module):
    def __init__(self, n_feat, hidden, n_layers, dropout):
        super().__init__()
        self.gru  = nn.GRU(n_feat, hidden, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.gru(x)
        return self.fc(self.drop(o[:, -1, :])).squeeze(-1)


class TransformerNet(nn.Module):
    def __init__(self, n_feat, d_model, nhead, n_layers, dim_ff, dropout, max_len=2000):
        super().__init__()
        # Ensure d_model divisible by nhead
        d_model = max((d_model // nhead) * nhead, nhead)
        self.proj = nn.Linear(n_feat, d_model)
        # Positional encoding
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x) + self.pe[:, :x.size(1)]
        return self.fc(self.encoder(x)[:, -1, :]).squeeze(-1)


class LSTMNet(nn.Module):
    def __init__(self, n_feat, hidden, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :]).squeeze(-1)


# ── Trainer ───────────────────────────────────────────────────────
def train_net(
    net: nn.Module,
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    params: dict,
) -> nn.Module:
    """
    BUG 6 FIX: rewrite train loop thành vòng lặp rõ ràng thay vì lambda+walrus.
    Thêm gradient clipping và early stopping.
    """
    lr       = params.get("lr", 1e-3)
    bs       = params.get("batch_size", 32)
    epochs   = min(params.get("epochs", 50), 30)   # cap 30 để tiết kiệm thời gian
    patience = min(params.get("patience", 10), 5)  # aggressive early stopping

    opt  = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.MSELoss()
    ds   = torch.utils.data.TensorDataset(
        torch.tensor(X_seq).to(DEVICE),
        torch.tensor(y_seq).to(DEVICE),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

    best_loss  = float("inf")
    wait       = 0
    best_state: Optional[dict] = None

    net.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        n_total    = 0
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * len(xb)
            n_total    += len(xb)

        epoch_loss /= max(n_total, 1)
        if epoch_loss < best_loss - 1e-6:
            best_loss  = epoch_loss
            wait       = 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        net.load_state_dict(best_state)
    return net


# ── Per-model runners ─────────────────────────────────────────────
def run_bilstm(X_tr, y_tr, X_te, params, seed):
    set_seed(seed)
    sc      = RobustScaler(quantile_range=(5, 95)).fit(X_tr)
    Xtr_sc  = sc.transform(X_tr)
    Xte_sc  = sc.transform(X_te)
    sl      = params.get("seq_len", 10)
    X_seq, sl = build_train_sequences(Xtr_sc, sl)
    y_seq   = y_tr[sl:].astype(np.float32)
    net     = BiLSTMNet(N_FEAT,
                        params.get("hidden", 64),
                        params.get("n_layers", 2),
                        params.get("dropout", 0.2)).to(DEVICE)
    net     = train_net(net, X_seq, y_seq, params)
    return batch_predict(net, Xtr_sc, Xte_sc, sl)   # BUG 1 FIX


def run_gru(X_tr, y_tr, X_te, params, seed):
    set_seed(seed)
    sc      = RobustScaler(quantile_range=(5, 95)).fit(X_tr)
    Xtr_sc  = sc.transform(X_tr)
    Xte_sc  = sc.transform(X_te)
    sl      = params.get("seq_len", 10)
    X_seq, sl = build_train_sequences(Xtr_sc, sl)
    y_seq   = y_tr[sl:].astype(np.float32)
    net     = GRUNet(N_FEAT,
                     params.get("hidden", 64),
                     params.get("n_layers", 2),
                     params.get("dropout", 0.2)).to(DEVICE)
    net     = train_net(net, X_seq, y_seq, params)
    return batch_predict(net, Xtr_sc, Xte_sc, sl)   # BUG 1 FIX


def run_transformer(X_tr, y_tr, X_te, params, seed):
    set_seed(seed)
    sc      = RobustScaler(quantile_range=(5, 95)).fit(X_tr)
    Xtr_sc  = sc.transform(X_tr)
    Xte_sc  = sc.transform(X_te)
    sl      = params.get("seq_len", 10)
    X_seq, sl = build_train_sequences(Xtr_sc, sl)
    y_seq   = y_tr[sl:].astype(np.float32)

    # BUG 2 FIX: compute d_model từ d_model_mult × nhead (đúng key)
    nhead   = params.get("nhead", 2)
    d_model = nhead * params.get("d_model_mult", 8)

    net = TransformerNet(
        N_FEAT, d_model, nhead,
        params.get("n_layers", 2),
        params.get("dim_ff", 128),
        params.get("dropout", 0.1),
        max_len=len(X_tr) + sl + 10,
    ).to(DEVICE)
    net = train_net(net, X_seq, y_seq, params)
    return batch_predict(net, Xtr_sc, Xte_sc, sl)   # BUG 1 FIX


def run_emd_lstm(X_tr, y_tr, X_te, params, seed):
    """
    EMD-LSTM: decompose y_train per fold (leakage-free), train LSTM per IMF.
    BUG 9 FIX: cảnh báo rõ ràng nếu PyEMD không có; fallback ≠ gaussian blur.
    """
    try:
        from PyEMD import EMD
        imfs = EMD().emd(y_tr, max_imf=3)   # cap 3: đủ, tránh 6x chậm
    except ImportError:
        logging.warning(
            "PyEMD not installed → EMD-LSTM dùng simple trend decomposition.\n"
            "  Install: pip install EMD-signal\n"
            "  Kết quả sẽ khác với EMD thật. Chỉ dùng cho fast test."
        )
        # Fallback: trend + residual (2 components)
        from scipy.signal import savgol_filter
        try:
            trend = savgol_filter(y_tr, window_length=min(21, len(y_tr)//2*2-1), polyorder=2)
        except Exception:
            trend = np.convolve(y_tr, np.ones(5)/5, mode='same')
        imfs = np.array([y_tr - trend, trend])

    if imfs.ndim == 1:
        imfs = imfs[np.newaxis, :]

    sc         = RobustScaler(quantile_range=(5, 95)).fit(X_tr)
    Xtr_sc     = sc.transform(X_tr)
    Xte_sc     = sc.transform(X_te)
    sl         = params.get("seq_len", 10)
    preds_total = np.zeros(len(X_te))

    for i, imf in enumerate(imfs):
        X_seq, sl_i = build_train_sequences(Xtr_sc, sl)
        y_imf = imf[sl_i:].astype(np.float32)
        if len(X_seq) == 0:
            continue
        set_seed(seed + i)
        net = LSTMNet(N_FEAT,
                      params.get("hidden", 32),
                      params.get("n_layers", 1)).to(DEVICE)
        net = train_net(net, X_seq, y_imf, params)
        preds_total += batch_predict(net, Xtr_sc, Xte_sc, sl_i)  # BUG 1 FIX

    return preds_total


RUNNERS = {
    "bilstm":      run_bilstm,
    "gru":         run_gru,
    "transformer": run_transformer,
    "emd_lstm":    run_emd_lstm,
}


def _run_one_seed(args):
    """
    Worker function cho multiprocessing — chạy 1 seed độc lập.
    Trả về (seed, pred_array) hoặc (seed, None) nếu lỗi.
    """
    model_name, seed, X_tr, y_tr, X_te, params = args
    try:
        runner = RUNNERS[model_name]
        pred   = runner(X_tr, y_tr, X_te, params, seed)
        return seed, pred
    except Exception as e:
        return seed, None


def run_seeds_parallel(
    model_name: str,
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
    params: dict,
    n_seeds: int,
    n_jobs: int,
) -> List[np.ndarray]:
    """
    Chạy n_seeds seeds song song với multiprocessing Pool.
    n_jobs: số processes đồng thời (nên = số CPU cores - 1).
    Tự động fallback về serial nếu multiprocessing gặp vấn đề.
    """
    args_list = [(model_name, seed, X_tr, y_tr, X_te, params)
                 for seed in range(n_seeds)]

    preds = []
    if n_jobs <= 1:
        # Serial fallback
        for args in args_list:
            _, pred = _run_one_seed(args)
            if pred is not None:
                preds.append(pred)
        return preds

    try:
        ctx = mp.get_context("spawn")   # spawn an toàn hơn fork với PyTorch
        with ctx.Pool(processes=n_jobs) as pool:
            results = pool.map(_run_one_seed, args_list)
        for seed, pred in results:
            if pred is not None:
                preds.append(pred)
            else:
                logging.warning(f"    seed={seed} failed in parallel pool")
    except Exception as e:
        logging.warning(f"  Parallel pool failed ({e}) → fallback serial")
        for args in args_list:
            _, pred = _run_one_seed(args)
            if pred is not None:
                preds.append(pred)

    return preds


# ── Main ──────────────────────────────────────────────────────────
def main():
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"{PARAMS_PATH} not found. Run 03a_optuna_tuning_fixed.py first."
        )
    with open(PARAMS_PATH, encoding="utf-8") as f:
        all_params = json.load(f)

    # BUG 3 FIX: NaN handling
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").sort_index()
    n_nan = df[FEATURES + [TARGET]].isna().sum().sum()
    if n_nan > 0:
        df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].ffill().bfill()
        logging.info(f"Filled {n_nan} NaN values")

    # BUG 4 FIX: parquet fallback
    folds = load_folds(FOLDS_PATH)

    # FAST_MODE: subset folds
    if MAX_FOLDS:
        folds = folds.head(MAX_FOLDS)
        logging.info(f"FAST MODE: {MAX_FOLDS} folds/model")

    # Resume checkpoint
    if CHECKPOINT.exists():
        try:
            df_done = pd.read_parquet(CHECKPOINT)
        except Exception:
            df_done = pd.read_csv(CHECKPOINT.with_suffix(".csv"))
        done_set = set(zip(df_done["model"], df_done["fold_id"], df_done["horizon"]))
        results  = df_done.to_dict("records")
        logging.info(f"Resume: {len(done_set)} folds already done")
    else:
        done_set, results = set(), []

    total   = len(MODELS) * len(folds)
    counter = 0
    t_start = time.time()

    logging.info(f"Device: {DEVICE} | Seeds: {N_SEEDS} | Total folds: {total}")

    for model_name in MODELS:
        params = all_params.get(model_name, {})
        runner = RUNNERS[model_name]
        logging.info(f"\n{'='*50}\n  {model_name.upper()}\n{'='*50}")

        for _, fold in folds.iterrows():
            fid = int(fold["fold_id"])
            h   = int(fold["horizon"])
            counter += 1

            if (model_name, fid, h) in done_set:
                continue

            tr_s = pd.to_datetime(fold["train_start"])
            tr_e = pd.to_datetime(fold["train_end"])
            te_s = pd.to_datetime(fold["test_start"])
            te_e = pd.to_datetime(fold["test_end"])

            X_tr = df.loc[tr_s:tr_e, FEATURES].values
            y_tr = df.loc[tr_s:tr_e, TARGET].values
            X_te = df.loc[te_s:te_e, FEATURES].values
            y_te = df.loc[te_s:te_e, TARGET].values

            # BUG 5 FIX: pre-flight check
            if len(X_tr) == 0 or len(X_te) == 0:
                logging.warning(f"  Skip fold {fid} H={h}: empty data")
                continue

            t0 = time.time()
            # PARALLEL SEEDS: chạy N_SEEDS seeds song song với N_JOBS processes
            seed_preds = run_seeds_parallel(
                model_name, X_tr, y_tr, X_te, params,
                n_seeds=N_SEEDS, n_jobs=N_JOBS if not FAST_MODE else 1,
            )

            if not seed_preds:
                logging.error(f"  All seeds failed: {model_name} fold {fid} H={h}")
                continue

            stack        = np.stack(seed_preds, axis=0)
            y_pred_mean  = stack.mean(axis=0)
            y_pred_std   = stack.std(axis=0, ddof=1) if len(seed_preds) > 1 else np.zeros_like(y_pred_mean)
            fold_rmse    = float(np.sqrt(np.mean((y_te - y_pred_mean) ** 2)))
            elapsed      = round(time.time() - t0, 2)

            # ETA
            elapsed_total = time.time() - t_start
            rate    = counter / max(elapsed_total, 1e-6)
            eta_sec = (total - counter) / max(rate, 1e-6)
            eta_str = f"{eta_sec/60:.0f}m"

            results.append({
                "model":       model_name,
                "fold_id":     fid,
                "horizon":     h,
                "y_true":      float(y_te[0]),          # BUG 8 FIX: scalar
                "y_pred_mean": float(y_pred_mean[0]),   # BUG 8 FIX: scalar
                "y_pred_std":  float(y_pred_std[0]),
                "n_seeds":     len(seed_preds),
                "train_start": str(tr_s.date()),
                "test_start":  str(te_s.date()),
                "test_end":    str(te_e.date()),
                "n_train":     len(X_tr),
                "time_sec":    elapsed,
                "fold_rmse":   fold_rmse,
            })
            done_set.add((model_name, fid, h))

            logging.info(
                f"  [{counter}/{total}] {model_name} Fold {fid:03d} H={h:2d}d | "
                f"RMSE={fold_rmse:.5f} | seeds={len(seed_preds)} | "
                f"{elapsed:.1f}s | ETA {eta_str}"
            )

            # Checkpoint mỗi 10 folds
            if len(results) % 10 == 0:
                save_parquet(pd.DataFrame(results), CHECKPOINT)

    # Final save
    df_out   = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "predictions_neural_nets.parquet"
    save_parquet(df_out, out_path)
    if CHECKPOINT.exists():
        CHECKPOINT.unlink(missing_ok=True)

    total_min = (time.time() - t_start) / 60
    print(f"\n{'='*55}")
    print("  NEURAL NETS — RMSE SUMMARY")
    print(f"{'='*55}")
    if len(df_out) > 0:
        summary = (df_out.groupby(["model", "horizon"])["fold_rmse"]
                   .mean().round(5).reset_index())
        print(summary.to_string(index=False))
    print(f"\nTotal time : {total_min:.1f} min")
    print(f"Folds done : {len(df_out)}")
    print(f"Saved      : {out_path}")
    if FAST_MODE:
        print("\nFAST MODE done. Set FAST_MODE=False for full run.")
    else:
        print("\nBuoc 2b xong. Chay 03d_merge.py de gop ket qua.")


if __name__ == "__main__":
    main()
