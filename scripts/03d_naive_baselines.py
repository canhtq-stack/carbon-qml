# =============================================================================
# 03e_naive_baselines_fixed.py
# =============================================================================
# Tính Random Walk (RW) và Historical Mean (HM) — naïve baselines
# để so sánh với QK-SVR và các classical models (Section 4.4).
#
# Random Walk (RW):
#   y_hat[t+h] = y[train_end]  ← last observed log-return
#   Rationale: nếu log-return là martingale difference (IID), thì
#   forecast tốt nhất là giá trị cuối cùng quan sát được.
#
# Historical Mean (HM):
#   y_hat[t+h] = mean(y_train)  ← expanding window mean (no leakage)
#   Rationale: unconditional mean predictor (benchmark cho Sharpe ratio).
#
# Protocol:
#   - Cùng walk-forward folds với tất cả models khác (fold_splits.parquet)
#   - Không cần training (< 1 phút)
#   - Output format nhất quán với 03b (predictions_tree_svm.parquet)
#   - 03d_merge_fixed.py sẽ tự động merge file này vào benchmark_predictions.parquet
#
# Input : results/fold_splits.parquet
#         data/processed/master_dataset.csv
# Output: results/predictions_naive_baselines.parquet
#         results/predictions_naive_baselines_summary.csv
#
# Chạy : Nhấn ▶️ Run trong VS Code (< 1 phút)
# =============================================================================

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)

# =============================================================================
# CẤU HÌNH — chỉnh BASE_DIR rồi nhấn Run
# =============================================================================
BASE_DIR = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project")
# =============================================================================

DATA_PATH  = BASE_DIR / "Data" / "data" / "processed" / "master_dataset.csv"
FOLDS_PATH = BASE_DIR / "Data" / "results" / "fold_splits.parquet"
OUTPUT_DIR = BASE_DIR / "Data" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "EUA_return"
FEATURES = [                           # cần để load dataset đầy đủ
    "GAS_return", "OIL_return", "COAL_return", "ELEC_return",
    "IP_return", "CPI_return", "POLICY_dummy", "PHASE_dummy",
]
MODELS   = ["rw", "hm"]               # Random Walk, Historical Mean
HORIZONS = [1, 5, 22]


# ── Helpers ───────────────────────────────────────────────────────
def load_folds(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        logging.info("fold_splits.parquet not found → using CSV")
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No fold_splits in {path.parent}")


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False, engine="pyarrow")
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)
        logging.warning(f"Parquet failed → saved CSV: {path.with_suffix('.csv').name}")


# ── Naive predictors ──────────────────────────────────────────────
def predict_rw(y_train: np.ndarray) -> float:
    """
    Random Walk: forecast = last observed value.
    For log-returns (EUA): this is y[train_end].
    Economic rationale: log-returns follow martingale hypothesis.
    """
    return float(y_train[-1])


def predict_hm(y_train: np.ndarray) -> float:
    """
    Historical Mean: forecast = mean of entire training window.
    Expanding window — no data leakage.
    Economic rationale: unconditional mean is optimal under constant
    drift hypothesis; commonly used in asset return forecasting.
    """
    return float(np.mean(y_train))


PREDICTORS = {
    "rw": predict_rw,
    "hm": predict_hm,
}

MODEL_NAMES = {
    "rw": "Random Walk",
    "hm": "Historical Mean",
}


# ── Main ──────────────────────────────────────────────────────────
def main():
    # Load dataset with NaN handling
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").sort_index()
    n_nan = df[FEATURES + [TARGET]].isna().sum().sum()
    if n_nan > 0:
        df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].ffill().bfill()
        logging.info(f"Filled {n_nan} NaN values")
    logging.info(f"Dataset: {df.shape} | {df.index[0].date()} → {df.index[-1].date()}")

    # Load folds
    folds = load_folds(FOLDS_PATH)
    logging.info(f"Folds: {len(folds)} | Horizons: {sorted(folds['horizon'].unique())}")

    total   = len(MODELS) * len(folds)
    counter = 0
    results = []
    t_start = time.time()

    for model_name in MODELS:
        predict_fn  = PREDICTORS[model_name]
        display     = MODEL_NAMES[model_name]
        logging.info(f"\n{'='*50}\n  {display} ({model_name})\n{'='*50}")

        for _, fold in folds.iterrows():
            fid = int(fold["fold_id"])
            h   = int(fold["horizon"])
            counter += 1

            tr_s = pd.to_datetime(fold["train_start"])
            tr_e = pd.to_datetime(fold["train_end"])
            te_s = pd.to_datetime(fold["test_start"])
            te_e = pd.to_datetime(fold["test_end"])

            y_tr = df.loc[tr_s:tr_e, TARGET].values
            y_te = df.loc[te_s:te_e, TARGET].values

            if len(y_tr) == 0 or len(y_te) == 0:
                logging.warning(f"  Skip fold {fid} H={h}: empty data")
                continue

            t0 = time.time()

            # No leakage: predict from train data only
            y_pred_val = predict_fn(y_tr)
            y_pred     = np.array([y_pred_val])

            fold_rmse = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
            elapsed   = round(time.time() - t0, 4)

            results.append({
                "model":       model_name,
                "fold_id":     fid,
                "horizon":     h,
                "y_true":      float(y_te[0]),    # scalar — consistent with 03b
                "y_pred":      float(y_pred[0]),  # scalar
                "y_pred_std":  0.0,               # deterministic — no variance
                "n_seeds":     1,
                "train_start": str(tr_s.date()),
                "test_start":  str(te_s.date()),
                "test_end":    str(te_e.date()),
                "n_train":     len(y_tr),
                "time_sec":    elapsed,
                "fold_rmse":   fold_rmse,
            })

            # Log every 20 folds hoặc fold đầu tiên
            if counter <= 2 or counter % 60 == 0:
                rate    = counter / max(time.time() - t_start, 1e-6)
                eta     = (total - counter) / max(rate, 1e-6)
                eta_str = f"{eta:.0f}s"
                logging.info(
                    f"  [{counter}/{total}] {model_name} Fold {fid:03d} H={h:2d}d | "
                    f"RMSE={fold_rmse:.5f} | pred={y_pred_val:.5f} | ETA {eta_str}"
                )

    # Save
    df_out   = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "predictions_naive_baselines.parquet"
    save_parquet(df_out, out_path)

    # Summary
    summary = (
        df_out.groupby(["model", "horizon"])["fold_rmse"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "rmse_mean", "std": "rmse_std", "count": "n_folds"})
        .round({"rmse_mean": 5, "rmse_std": 5})
    )
    summary_path = OUTPUT_DIR / "predictions_naive_baselines_summary.csv"
    summary.to_csv(summary_path, index=False)

    total_time = time.time() - t_start
    print(f"\n{'='*55}")
    print("  NAIVE BASELINES SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Model':<20} {'H':>4} {'RMSE_mean':>10} {'RMSE_std':>10} {'N':>5}")
    print("  " + "-" * 50)
    for _, r in summary.iterrows():
        print(f"  {MODEL_NAMES.get(r['model'],r['model']):<20} {int(r['horizon']):>4} "
              f"{r['rmse_mean']:>10.5f} {r['rmse_std']:>10.5f} {int(r['n_folds']):>5}")

    print(f"\n  Folds done : {len(df_out)}")
    print(f"  Time       : {total_time:.1f}s")
    print(f"  Saved      : {out_path}")
    print(f"  Saved      : {summary_path}")
    print(f"\n{'='*55}")
    print("  ✅ Bước tiếp theo: chạy lại 03d_merge_fixed.py")
    print("     để merge RW + HM vào benchmark_predictions.parquet")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
