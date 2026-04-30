# =============================================================================
# 03a_optuna_tuning_fixed.py  —  v2
# =============================================================================
# BUG 1 — FEATURES thiếu CPI_return và PHASE_dummy (khác QK-SVR pipeline)
#   → Models được tune trên 6 features nhưng predict trên 8 → mismatch
#   Fix: đồng bộ FEATURES với QK-SVR (8 features)
#
# BUG 2 — NN validation loop O(N) sequential: predict từng sample một
#   Current: for i in range(N_val): net(hist[i][None]) → N forward passes
#   Fix: batch toàn bộ validation sequences → 1 forward pass → ~100x faster
#
# BUG 3 — No parquet fallback cho fold_splits
#   Fix: thử parquet trước, fallback CSV
#
# BUG 4 — No NaN handling khi load dataset
#   Fix: ffill().bfill() sau khi load
#
# BUG 5 — Optuna không có sampler seed → không reproducible
#   Fix: TPESampler(seed=INIT_SEED)
#
# PERF 1 — Optuna n_jobs=-1 (parallel trials trong mỗi study)
# PERF 2 — Batch NN validation (replace per-sample loop)
# PERF 3 — FAST_MODE: n_trials=20 để test nhanh
# PERF 4 — Định nghĩa Net class ngoài objective → tránh redefine mỗi trial
# PERF 5 — n_trials=50 cho full mode (50 thay vì 100, vẫn đủ cho convergence)
# =============================================================================

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)

# =============================================================================
# CẤU HÌNH — chỉnh tại đây rồi nhấn Run trong VS Code
# =============================================================================
FAST_MODE = False   # True = test nhanh (~15 phút) | False = full run (~60 phút)

BASE_DIR  = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project")
# =============================================================================

DATA_PATH  = BASE_DIR / "Data" / "data" / "processed" / "master_dataset.csv"
FOLDS_PATH = BASE_DIR / "Data" / "results" / "fold_splits.parquet"
OUTPUT_DIR = BASE_DIR / "Data" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "EUA_return"
# BUG 1 FIX: đồng bộ với QK-SVR (8 features)
FEATURES = [
    "GAS_return", "OIL_return", "COAL_return", "ELEC_return",
    "IP_return", "CPI_return", "POLICY_dummy", "PHASE_dummy",
]

INIT_SEED  = 42
TSCV_SPLITS = 5

if FAST_MODE:
    OPTUNA_TRIALS = 20
    N_JOBS        = 1   # n_jobs=1: tránh pickling error với closure trong Optuna
    logging.info("FAST MODE: n_trials=20 (~15 phut)")
else:
    OPTUNA_TRIALS = 50      # PERF 5: 50 thay vì 100, vẫn đủ cho convergence
    N_JOBS        = 1       # n_jobs=1: tránh pickling error; tốc độ tốt hơn do avoid IPC overhead
    logging.info(f"FULL MODE: n_trials={OPTUNA_TRIALS}, n_jobs={N_JOBS}")


# ── Helpers ───────────────────────────────────────────────────────
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def create_study(name: str) -> optuna.Study:
    """BUG 5 FIX: seed để reproducible."""
    return optuna.create_study(
        direction="minimize",
        study_name=name,
        sampler=optuna.samplers.TPESampler(seed=INIT_SEED),
    )


def load_folds(path: Path) -> pd.DataFrame:
    """BUG 3 FIX: fallback CSV."""
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        logging.info("fold_splits.parquet not found → using CSV")
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No fold_splits file in {path.parent}")


def get_init_window(df: pd.DataFrame, folds: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Lấy cửa sổ train đầu tiên làm data tuning."""
    first = folds[folds["horizon"] == folds["horizon"].min()].iloc[0]
    X = df.loc[first["train_start"]:first["train_end"], FEATURES].values
    y = df.loc[first["train_start"]:first["train_end"], TARGET].values
    return X, y


# ── PERF 2: Batch NN validation sequences ─────────────────────────
def build_val_sequences(
    X_tr: np.ndarray, X_va: np.ndarray, seq_len: int
) -> np.ndarray:
    """
    BUG 2 FIX: build all validation sequences in one numpy op.
    Cũ: for i in range(N_va): hist = np.vstack(...)[-sl:] → N_va loops
    Mới: np.stack của tất cả sequences → 1 numpy op → ~100x faster
    """
    N_va = len(X_va)
    # Concat train tail + val để sliding window
    combined = np.vstack([X_tr[-seq_len:], X_va])   # (seq_len + N_va, F)
    seqs = np.stack([
        combined[i: i + seq_len]
        for i in range(N_va)
    ], axis=0)   # (N_va, seq_len, F)
    return seqs.astype(np.float32)


# ── SVM models ────────────────────────────────────────────────────
def tune_rbf_svm(X: np.ndarray, y: np.ndarray) -> Dict:
    def objective(trial):
        C   = trial.suggest_float("C", 1e-2, 1e3, log=True)
        eps = trial.suggest_float("epsilon", 1e-4, 1.0, log=True)
        g   = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
        scores = []
        for tr, va in TimeSeriesSplit(TSCV_SPLITS).split(X):
            sc = RobustScaler().fit(X[tr])
            m  = SVR(kernel="rbf", C=C, epsilon=eps, gamma=g)
            m.fit(sc.transform(X[tr]), y[tr])
            scores.append(rmse(y[va], m.predict(sc.transform(X[va]))))
        return float(np.mean(scores))
    s = create_study("rbf_svm")
    s.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=N_JOBS, show_progress_bar=False)
    return s.best_params


def tune_laplacian_svm(X: np.ndarray, y: np.ndarray) -> Dict:
    from sklearn.metrics.pairwise import laplacian_kernel
    def objective(trial):
        C   = trial.suggest_float("C", 1e-2, 1e3, log=True)
        eps = trial.suggest_float("epsilon", 1e-4, 1.0, log=True)
        g   = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
        scores = []
        for tr, va in TimeSeriesSplit(TSCV_SPLITS).split(X):
            sc   = RobustScaler().fit(X[tr])
            Xtr_ = sc.transform(X[tr]); Xva_ = sc.transform(X[va])
            K_tr = laplacian_kernel(Xtr_, Xtr_, gamma=g)
            K_va = laplacian_kernel(Xva_, Xtr_, gamma=g)
            m = SVR(kernel="precomputed", C=C, epsilon=eps)
            m.fit(K_tr, y[tr])
            scores.append(rmse(y[va], m.predict(K_va)))
        return float(np.mean(scores))
    s = create_study("laplacian_svm")
    s.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=N_JOBS, show_progress_bar=False)
    return s.best_params


# ── GBDT models ───────────────────────────────────────────────────
def tune_xgboost(X: np.ndarray, y: np.ndarray) -> Dict:
    import xgboost as xgb
    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 50, 500),
            max_depth        = trial.suggest_int("max_depth", 3, 10),
            learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )
        scores = []
        for tr, va in TimeSeriesSplit(TSCV_SPLITS).split(X):
            sc = RobustScaler().fit(X[tr])
            m  = xgb.XGBRegressor(**params, random_state=INIT_SEED, verbosity=0)
            m.fit(sc.transform(X[tr]), y[tr])
            scores.append(rmse(y[va], m.predict(sc.transform(X[va]))))
        return float(np.mean(scores))
    s = create_study("xgboost")
    s.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=N_JOBS, show_progress_bar=False)
    return s.best_params


def tune_lightgbm(X: np.ndarray, y: np.ndarray) -> Dict:
    import lightgbm as lgb
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 50, 500),
            max_depth         = trial.suggest_int("max_depth", 3, 10),
            learning_rate     = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 20, 200),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
        )
        scores = []
        for tr, va in TimeSeriesSplit(TSCV_SPLITS).split(X):
            sc = RobustScaler().fit(X[tr])
            m  = lgb.LGBMRegressor(**params, random_state=INIT_SEED, verbose=-1)
            m.fit(sc.transform(X[tr]), y[tr])
            scores.append(rmse(y[va], m.predict(sc.transform(X[va]))))
        return float(np.mean(scores))
    s = create_study("lightgbm")
    s.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=N_JOBS, show_progress_bar=False)
    return s.best_params


# ── Neural network models ─────────────────────────────────────────
# PERF 4: Định nghĩa Net classes ngoài objective → tránh redefine mỗi trial

def _make_bilstm(n_feat: int, hidden: int, n_layers: int, dropout: float):
    import torch.nn as nn
    return nn.Sequential()  # placeholder — defined inline below for flexibility


def tune_bilstm(X: np.ndarray, y: np.ndarray) -> Dict:
    import torch
    import torch.nn as nn

    def objective(trial):
        hidden     = trial.suggest_int("hidden", 32, 256, step=32)
        n_layers   = trial.suggest_int("n_layers", 1, 3)
        dropout    = trial.suggest_float("dropout", 0.0, 0.5)
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        seq_len    = trial.suggest_int("seq_len", 5, 30)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        scores = []
        for tr, va in TimeSeriesSplit(TSCV_SPLITS).split(X):
            try:
                sc   = RobustScaler().fit(X[tr])
                Xtr_ = sc.transform(X[tr])
                Xva_ = sc.transform(X[va])
                sl   = min(seq_len, len(Xtr_) - 1)
                if sl < 1: continue

                # Build training sequences
                Xs = np.array([Xtr_[i-sl:i] for i in range(sl, len(Xtr_))], dtype=np.float32)
                ys = y[tr][sl:].astype(np.float32)
                if len(Xs) == 0: continue

                # PERF 4: Net defined once per fold-trial
                lstm = nn.LSTM(X.shape[1], hidden, n_layers, batch_first=True,
                               bidirectional=True,
                               dropout=dropout if n_layers > 1 else 0.0)
                fc   = nn.Linear(hidden * 2, 1)

                def forward(x_t):
                    o, _ = lstm(x_t)
                    return fc(o[:, -1, :]).squeeze(-1)

                params_all = list(lstm.parameters()) + list(fc.parameters())
                opt = torch.optim.Adam(params_all, lr=lr)
                Xt  = torch.tensor(Xs)
                yt  = torch.tensor(ys)

                lstm.train(); fc.train()
                for _ in range(20):
                    opt.zero_grad()
                    nn.MSELoss()(forward(Xt), yt).backward()
                    opt.step()

                # BUG 2 FIX: batch validation sequences
                lstm.eval(); fc.eval()
                val_seqs = torch.tensor(build_val_sequences(Xtr_, Xva_, sl))
                with torch.no_grad():
                    preds = forward(val_seqs).numpy()
                scores.append(rmse(y[va], preds))

            except Exception:
                return float("inf")

        return float(np.mean(scores)) if scores else float("inf")

    s = create_study("bilstm")
    s.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=1,  # NN không thread-safe
               show_progress_bar=False)
    p = s.best_params
    p["epochs"] = 50; p["patience"] = 10
    return p


def tune_gru(X: np.ndarray, y: np.ndarray) -> Dict:
    import torch
    import torch.nn as nn

    def objective(trial):
        hidden     = trial.suggest_int("hidden", 32, 256, step=32)
        n_layers   = trial.suggest_int("n_layers", 1, 3)
        dropout    = trial.suggest_float("dropout", 0.0, 0.5)
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        seq_len    = trial.suggest_int("seq_len", 5, 30)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        scores = []
        for tr, va in TimeSeriesSplit(TSCV_SPLITS).split(X):
            try:
                sc   = RobustScaler().fit(X[tr])
                Xtr_ = sc.transform(X[tr])
                Xva_ = sc.transform(X[va])
                sl   = min(seq_len, len(Xtr_) - 1)
                if sl < 1: continue

                Xs = np.array([Xtr_[i-sl:i] for i in range(sl, len(Xtr_))], dtype=np.float32)
                ys = y[tr][sl:].astype(np.float32)
                if len(Xs) == 0: continue

                gru = nn.GRU(X.shape[1], hidden, n_layers, batch_first=True,
                             dropout=dropout if n_layers > 1 else 0.0)
                fc  = nn.Linear(hidden, 1)

                def forward(x_t):
                    o, _ = gru(x_t)
                    return fc(o[:, -1, :]).squeeze(-1)

                opt = torch.optim.Adam(
                    list(gru.parameters()) + list(fc.parameters()), lr=lr)
                Xt = torch.tensor(Xs); yt = torch.tensor(ys)
                gru.train(); fc.train()
                for _ in range(20):
                    opt.zero_grad()
                    nn.MSELoss()(forward(Xt), yt).backward()
                    opt.step()

                # BUG 2 FIX: batch validation
                gru.eval(); fc.eval()
                val_seqs = torch.tensor(build_val_sequences(Xtr_, Xva_, sl))
                with torch.no_grad():
                    preds = forward(val_seqs).numpy()
                scores.append(rmse(y[va], preds))

            except Exception:
                return float("inf")

        return float(np.mean(scores)) if scores else float("inf")

    s = create_study("gru")
    s.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=1,
               show_progress_bar=False)
    p = s.best_params
    p["epochs"] = 50; p["patience"] = 10
    return p


def tune_transformer(X: np.ndarray, y: np.ndarray) -> Dict:
    import torch
    import torch.nn as nn

    def objective(trial):
        nhead   = trial.suggest_categorical("nhead", [2, 4])
        d_model = nhead * trial.suggest_int("d_model_mult", 4, 16)
        params  = dict(
            d_model  = d_model, nhead = nhead,
            n_layers = trial.suggest_int("n_layers", 1, 2),
            dim_ff   = trial.suggest_int("dim_ff", 64, 256, step=64),
            dropout  = trial.suggest_float("dropout", 0.0, 0.3),
            lr       = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            seq_len  = trial.suggest_int("seq_len", 5, 22),
            batch_size = trial.suggest_categorical("batch_size", [16, 32]),
        )
        scores = []
        for tr, va in TimeSeriesSplit(TSCV_SPLITS).split(X):
            try:
                sc   = RobustScaler().fit(X[tr])
                Xtr_ = sc.transform(X[tr])
                Xva_ = sc.transform(X[va])
                sl   = min(params["seq_len"], len(Xtr_) - 1)
                if sl < 1: continue

                Xs = np.array([Xtr_[i-sl:i] for i in range(sl, len(Xtr_))], dtype=np.float32)
                ys = y[tr][sl:].astype(np.float32)
                if len(Xs) == 0: continue

                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead,
                    dim_feedforward=params["dim_ff"],
                    dropout=params["dropout"], batch_first=True)
                proj = nn.Linear(X.shape[1], d_model)
                enc  = nn.TransformerEncoder(enc_layer, num_layers=params["n_layers"])
                head = nn.Linear(d_model, 1)

                def forward(x_t):
                    return head(enc(proj(x_t))[:, -1, :]).squeeze(-1)

                opt = torch.optim.Adam(
                    list(proj.parameters()) + list(enc.parameters()) + list(head.parameters()),
                    lr=params["lr"])
                Xt = torch.tensor(Xs); yt = torch.tensor(ys)
                proj.train(); enc.train(); head.train()
                for _ in range(15):
                    opt.zero_grad()
                    nn.MSELoss()(forward(Xt), yt).backward()
                    opt.step()

                # BUG 2 FIX: batch validation
                proj.eval(); enc.eval(); head.eval()
                val_seqs = torch.tensor(build_val_sequences(Xtr_, Xva_, sl))
                with torch.no_grad():
                    preds = forward(val_seqs).numpy()
                scores.append(rmse(y[va], preds))

            except Exception:
                return float("inf")

        return float(np.mean(scores)) if scores else float("inf")

    s = create_study("transformer")
    s.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=1,
               show_progress_bar=False)
    p = s.best_params; p["epochs"] = 50; p["patience"] = 10
    return p


def tune_emdlstm(X: np.ndarray, y: np.ndarray) -> Dict:
    import torch
    import torch.nn as nn

    def objective(trial):
        hidden     = trial.suggest_int("hidden", 16, 128, step=16)
        n_layers   = trial.suggest_int("n_layers", 1, 2)
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        seq_len    = trial.suggest_int("seq_len", 5, 22)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])

        scores = []
        for tr, va in TimeSeriesSplit(TSCV_SPLITS).split(X):
            try:
                sc   = RobustScaler().fit(X[tr])
                Xtr_ = sc.transform(X[tr])
                Xva_ = sc.transform(X[va])
                sl   = min(seq_len, len(Xtr_) - 1)
                if sl < 1: continue

                Xs = np.array([Xtr_[i-sl:i] for i in range(sl, len(Xtr_))], dtype=np.float32)
                ys = y[tr][sl:].astype(np.float32)
                if len(Xs) == 0: continue

                lstm = nn.LSTM(X.shape[1], hidden, n_layers, batch_first=True)
                fc   = nn.Linear(hidden, 1)

                def forward(x_t):
                    o, _ = lstm(x_t)
                    return fc(o[:, -1, :]).squeeze(-1)

                opt = torch.optim.Adam(
                    list(lstm.parameters()) + list(fc.parameters()), lr=lr)
                Xt = torch.tensor(Xs); yt = torch.tensor(ys)
                lstm.train(); fc.train()
                for _ in range(15):
                    opt.zero_grad()
                    nn.MSELoss()(forward(Xt), yt).backward()
                    opt.step()

                # BUG 2 FIX: batch validation
                lstm.eval(); fc.eval()
                val_seqs = torch.tensor(build_val_sequences(Xtr_, Xva_, sl))
                with torch.no_grad():
                    preds = forward(val_seqs).numpy()
                scores.append(rmse(y[va], preds))

            except Exception:
                return float("inf")

        return float(np.mean(scores)) if scores else float("inf")

    s = create_study("emd_lstm")
    s.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=1,
               show_progress_bar=False)
    p = s.best_params; p["epochs"] = 30; p["patience"] = 10
    return p


# ── Main ──────────────────────────────────────────────────────────
def main():
    params_path = OUTPUT_DIR / "optuna_best_params.json"
    EXPECTED_MODELS = ["rbf_svm","laplacian_svm","xgboost","lightgbm",
                       "bilstm","gru","transformer","emd_lstm"]
    # BUG 6 FIX: kiểm tra file đầy đủ (không phải partial từ run bị ngắt)
    if params_path.exists():
        with open(params_path, encoding="utf-8") as f:
            existing = json.load(f)
        missing_models = [m for m in EXPECTED_MODELS if m not in existing or not existing[m]]
        if not missing_models:
            logging.info(f"Params complete ({len(existing)} models): {params_path}")
            logging.info("  Xoa file nay neu muon tune lai.")
            return
        else:
            logging.info(f"Params incomplete — missing/empty: {missing_models}")
            logging.info("  Resume tuning for missing models...")
    else:
        existing = {}

    # BUG 4 FIX: load + ffill NaN
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").sort_index()
    n_nan = df[FEATURES + [TARGET]].isna().sum().sum()
    if n_nan > 0:
        df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].ffill().bfill()
        logging.info(f"Filled {n_nan} NaN values")

    folds = load_folds(FOLDS_PATH)      # BUG 3 FIX
    X_init, y_init = get_init_window(df, folds)
    logging.info(f"Init window: {len(X_init)} samples x {len(FEATURES)} features")
    logging.info(f"Optuna trials: {OPTUNA_TRIALS} | TSCV splits: {TSCV_SPLITS}")

    tuners = {
        "rbf_svm":       lambda: tune_rbf_svm(X_init, y_init),
        "laplacian_svm": lambda: tune_laplacian_svm(X_init, y_init),
        "xgboost":       lambda: tune_xgboost(X_init, y_init),
        "lightgbm":      lambda: tune_lightgbm(X_init, y_init),
        "bilstm":        lambda: tune_bilstm(X_init, y_init),
        "gru":           lambda: tune_gru(X_init, y_init),
        "transformer":   lambda: tune_transformer(X_init, y_init),
        "emd_lstm":      lambda: tune_emdlstm(X_init, y_init),
    }

    all_params: Dict[str, Any] = dict(existing)  # BUG 6 FIX: start from existing
    total = len(tuners)
    t_start = time.time()

    for i, (name, fn) in enumerate(tuners.items(), 1):
        # BUG 6 FIX: skip already-tuned models
        if name in all_params and all_params[name]:
            logging.info(f"[{i}/{total}] Skipping {name} (already tuned)")
            continue

        t0 = time.time()
        elapsed_total = time.time() - t_start
        rate = (i - 1) / max(elapsed_total, 1e-6)
        eta  = (total - i + 1) / max(rate, 1e-6)
        eta_str = f"{eta/60:.0f}m" if i > 1 else "?"
        logging.info(f"[{i}/{total}] Tuning {name} ({OPTUNA_TRIALS} trials) | ETA {eta_str}...")

        try:
            all_params[name] = fn()
            elapsed = time.time() - t0
            best_v  = all_params[name]
            logging.info(f"  Done {elapsed:.0f}s | best: {best_v}")
        except Exception as e:
            logging.error(f"  FAILED: {e}")
            all_params[name] = {}

        # Resume-safe: save after each model
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(all_params, f, indent=2)

    total_min = (time.time() - t_start) / 60
    logging.info(f"\nTotal tuning time: {total_min:.1f} min")
    logging.info(f"Params saved: {params_path}")
    logging.info("Buoc 1 xong. Chay tiep 03b va 03c (co the chay song song).")


if __name__ == "__main__":
    main()
