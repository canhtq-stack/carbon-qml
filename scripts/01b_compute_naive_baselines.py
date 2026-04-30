# ─────────────────────────────────────────────────────────────────────────────
# Script: compute_naive_baselines_v2.py
# Purpose: Tính RMSE/MAE của RW và HM với ĐÚNG 37 folds, khớp pipeline chính
#          Fix: giới hạn MAX_FOLDS=37 và dùng N=1285
#
# Run: C:\Users\tranq\AppData\Local\Programs\Python\Python311\python.exe compute_naive_baselines_v2.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH  = r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project\Data\data\processed\eua_daily_processed.csv"
TARGET_COL = "EUA_return"

INITIAL_TRAIN = 504
STEP          = 21
MAX_FOLDS     = 37    # <── khớp chính xác với pipeline chính
HORIZONS      = [1, 5, 22]

OUTPUT_PATH = r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project\naive_baselines_metrics_v2.csv"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print(f"Loading: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File không tìm thấy.")
        return

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    if TARGET_COL not in df.columns:
        print(f"ERROR: Không tìm thấy cột '{TARGET_COL}'.")
        print(f"Các cột: {list(df.columns)}")
        return

    # Lấy đúng 1285 obs đầu tiên (khớp với pipeline)
    returns = df[TARGET_COL].dropna().iloc[:1285]
    N = len(returns)
    print(f"N = {N} (sau khi giới hạn 1285 obs)")
    print(f"Date range: {returns.index[0]} → {returns.index[-1]}")
    print(f"Walk-forward: initial={INITIAL_TRAIN}, step={STEP}, max_folds={MAX_FOLDS}")

    rows = []

    for h in HORIZONS:
        rw_rmse_list, rw_mae_list = [], []
        hm_rmse_list, hm_mae_list = [], []

        for fold in range(MAX_FOLDS):
            train_end  = INITIAL_TRAIN + fold * STEP
            test_start = train_end
            test_end   = train_end + h

            if test_end > N:
                print(f"  H={h}, fold {fold+1}: test_end={test_end} > N={N}, dừng. Folds hoàn chỉnh: {fold}")
                break

            y_train = returns.iloc[:train_end].values
            y_test  = returns.iloc[test_start:test_end].values

            # Random Walk: forecast = 0
            rw_err  = y_test - 0.0
            rw_rmse_list.append(np.sqrt(np.mean(rw_err**2)))
            rw_mae_list.append(np.mean(np.abs(rw_err)))

            # Historical Mean: forecast = expanding mean
            hm_val  = np.mean(y_train)
            hm_err  = y_test - hm_val
            hm_rmse_list.append(np.sqrt(np.mean(hm_err**2)))
            hm_mae_list.append(np.mean(np.abs(hm_err)))

        for model, rmse_l, mae_l in [
            ('rw', rw_rmse_list, rw_mae_list),
            ('hm', hm_rmse_list, hm_mae_list),
        ]:
            rows.append({
                'model'    : model,
                'horizon'  : h,
                'rmse_mean': np.mean(rmse_l),
                'rmse_std' : np.std(rmse_l),
                'mae_mean' : np.mean(mae_l),
                'mae_std'  : np.std(mae_l),
                'n_folds'  : len(rmse_l),
            })

    out = pd.DataFrame(rows).sort_values(['model', 'horizon'])
    print("\n=== Kết quả ===")
    print(out.to_string(index=False))
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nĐã lưu: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
