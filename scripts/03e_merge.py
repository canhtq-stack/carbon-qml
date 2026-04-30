# =============================================================================
# 03d_merge_fixed.py  —  v2
# =============================================================================
# BUG 1 — No parquet fallback khi đọc input files
#   Fix: try parquet → fallback CSV cho mỗi input
#
# BUG 2 — No parquet fallback khi save output
#   Fix: save_parquet() helper với fallback CSV
#
# BUG 3 [CRITICAL] — Column mismatch sau concat:
#   03b: y_pred (scalar)
#   03c: y_pred_mean, y_pred_std (scalar)
#   → After concat: y_pred=NaN cho neural nets, y_pred_mean=NaN cho tree/svm
#   Fix: standardize tất cả thành y_pred + y_pred_std trước khi concat
#
# BUG 4 [CRITICAL] — Không merge qksvr_predictions.parquet
#   → benchmark_predictions.parquet thiếu QK-SVR → file 04 compare sai
#   Fix: thêm QK-SVR vào FILES dict; chuẩn hóa format QK-SVR
#
# BUG 5 — pivot_table fail nếu thiếu model (fast mode data)
#   Fix: thêm error handling; in bảng dạng groupby thay vì pivot
# =============================================================================

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DIR   = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project")
OUTPUT_DIR = BASE_DIR / "Data" / "results"

# BUG 4 FIX: thêm QK-SVR vào danh sách files
FILES = {
    "QK-SVR":      OUTPUT_DIR / "qksvr_predictions.parquet",
    "Tree & SVM":  OUTPUT_DIR / "predictions_tree_svm.parquet",
    "Neural Nets": OUTPUT_DIR / "predictions_neural_nets.parquet",
    "Naive":       OUTPUT_DIR / "predictions_naive_baselines.parquet",
}

# Models expected in final benchmark
EXPECTED_MODELS = [
    "qk_svr",
    "rbf_svm", "laplacian_svm", "xgboost", "lightgbm",
    "bilstm", "gru", "transformer", "emd_lstm",
]
HORIZONS = [1, 5, 22]


# ── Helpers ───────────────────────────────────────────────────────
def load_parquet(path: Path) -> pd.DataFrame:
    """BUG 1 FIX: fallback CSV."""
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        logging.info(f"  Parquet not found → using CSV: {csv.name}")
        return pd.read_csv(csv)
    return None


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """BUG 2 FIX: fallback CSV."""
    try:
        df.to_parquet(path, index=False, engine="pyarrow")
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logging.warning(f"Parquet failed → saved: {csv_path}")


def standardize_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    BUG 3 + BUG 4 FIX: chuẩn hóa tất cả dataframes về cùng schema:
    model, fold_id, horizon, y_true, y_pred, y_pred_std, n_seeds,
    train_start, test_start, test_end, n_train, time_sec, fold_rmse

    - 03b (tree/svm): có y_pred, không có y_pred_std → thêm y_pred_std=0
    - 03c (neural nets): có y_pred_mean, y_pred_std → rename y_pred_mean→y_pred
    - 02 (qk_svr): có y_pred, không có y_pred_std → thêm y_pred_std=0
                   model column cần thêm nếu không có
    """
    df = df.copy()

    # QK-SVR: model column có thể không có, thêm vào
    if "model" not in df.columns:
        df["model"] = "qk_svr"

    # Standardize y_pred column
    if "y_pred_mean" in df.columns and "y_pred" not in df.columns:
        # 03c format → rename
        df = df.rename(columns={"y_pred_mean": "y_pred"})
    elif "y_pred" not in df.columns:
        logging.warning(f"  [{source}] No y_pred or y_pred_mean column found!")
        df["y_pred"] = np.nan

    # Standardize y_pred_std
    if "y_pred_std" not in df.columns:
        df["y_pred_std"] = 0.0

    # Ensure scalar types (not lists)
    for col in ["y_true", "y_pred", "y_pred_std"]:
        if col in df.columns:
            sample = df[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                df[col] = df[col].apply(
                    lambda x: float(x[0]) if hasattr(x, '__len__') else float(x)
                )

    # Ensure n_seeds exists
    if "n_seeds" not in df.columns:
        df["n_seeds"] = 1

    # Recompute fold_rmse nếu cần (từ y_true và y_pred)
    if "fold_rmse" not in df.columns:
        df["fold_rmse"] = np.sqrt((df["y_true"] - df["y_pred"]) ** 2)

    return df


def main():
    dfs = []
    found_models = []

    for label, path in FILES.items():
        df = load_parquet(path)
        if df is None:
            logging.warning(f"⚠️  {label}: {path.name} không tìm thấy — bỏ qua.")
            continue

        # Standardize schema (BUG 3 + 4 FIX)
        df = standardize_df(df, label)
        models_in = df["model"].unique().tolist()
        found_models.extend(models_in)

        logging.info(
            f"✅ {label}: {len(df)} rows | "
            f"models: {models_in} | "
            f"folds: {df.groupby('horizon').size().to_dict()}"
        )
        dfs.append(df)

    if not dfs:
        raise RuntimeError("Không có file nào để gộp. Chạy 03b, 03c, 02 trước.")

    # Concat với schema thống nhất
    COLS = [
        "model", "fold_id", "horizon",
        "y_true", "y_pred", "y_pred_std",
        "n_seeds", "train_start", "test_start", "test_end",
        "n_train", "time_sec", "fold_rmse",
    ]
    df_all = pd.concat(dfs, ignore_index=True)

    # Chỉ giữ cột cần thiết (bỏ các cột thừa từ các sources khác nhau)
    cols_available = [c for c in COLS if c in df_all.columns]
    df_all = df_all[cols_available].copy()

    # Kiểm tra và xử lý trùng lặp
    dupes = df_all.duplicated(subset=["model", "fold_id", "horizon"]).sum()
    if dupes > 0:
        logging.warning(f"⚠️  {dupes} dòng trùng lặp → giữ lần đầu.")
        df_all = df_all.drop_duplicates(
            subset=["model", "fold_id", "horizon"], keep="first"
        )

    # Save benchmark_predictions
    out_path = OUTPUT_DIR / "benchmark_predictions.parquet"
    save_parquet(df_all, out_path)

    # Summary stats
    summary = (
        df_all.groupby(["model", "horizon"])["fold_rmse"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "rmse_mean", "std": "rmse_std", "count": "n_folds"})
        .round({"rmse_mean": 5, "rmse_std": 5})
    )
    summary_path = OUTPUT_DIR / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    # BUG 5 FIX: print groupby format thay vì pivot (tránh fail nếu thiếu model)
    print(f"\n{'='*65}")
    print("  BENCHMARK RMSE SUMMARY (mean ± std per model per horizon)")
    print(f"{'='*65}")
    print(f"{'Model':<18} {'H':>4} {'RMSE_mean':>10} {'RMSE_std':>10} {'N_folds':>8}")
    print("-" * 55)
    for _, row in summary.sort_values(["horizon", "rmse_mean"]).iterrows():
        print(f"  {row['model']:<16} {int(row['horizon']):>4} "
              f"{row['rmse_mean']:>10.5f} {row['rmse_std']:>10.5f} "
              f"{int(row['n_folds']):>8}")

    # Completeness check
    print(f"\n{'='*65}")
    print("  COMPLETENESS CHECK")
    print(f"{'='*65}")
    missing = [m for m in EXPECTED_MODELS if m not in found_models]
    if missing:
        print(f"  ⚠️  Models chưa có trong benchmark: {missing}")
        print("      → Chạy lại file tương ứng trước khi chạy file 04.")
    else:
        print(f"  ✅ Tất cả {len(EXPECTED_MODELS)} models có mặt.")

    folds_check = df_all.groupby(["model", "horizon"]).size()
    incomplete = folds_check[folds_check < 37]
    if len(incomplete) > 0:
        print(f"\n  ⚠️  Một số model/horizon chưa đủ 37 folds:")
        print(incomplete.to_string())
        print("      → Có thể đang chạy fast mode. OK nếu dùng full mode sau.")
    else:
        print(f"  ✅ Tất cả model/horizon đủ 37 folds.")

    print(f"\n  Saved: {out_path}")
    print(f"  Saved: {summary_path}")
    print(f"\n  Tổng rows: {len(df_all)}")
    print(f"  Models: {sorted(df_all['model'].unique())}")
    print("\n  Bước tiếp theo: chạy 04_statistical_validation.py")


if __name__ == "__main__":
    main()
