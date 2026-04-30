# =============================================================================
# 03b_tree_svm_fixed.py  —  v2
# =============================================================================
# BUG 1 — done += 1 xuất hiện 2 lần trong loop → counter tăng gấp đôi
#   Fix: xóa dòng thừa
#
# BUG 2 — y_true/y_pred lưu dạng list → nên lưu scalar (1 test point/fold)
#   Fix: float(y_te[0]) và float(y_pred[0])
#
# BUG 3 — y_pred_std = [0.0] hardcoded → xóa cột misleading này
#   Fix: bỏ y_pred_std khỏi output
#
# BUG 4 — No parquet fallback cho fold_splits
#   Fix: thử parquet → fallback CSV
#
# BUG 5 — No NaN handling khi load dataset
#   Fix: ffill().bfill() sau load
#
# BUG 6 — No pre-flight check cho empty X_te / X_tr
#   Fix: skip fold nếu empty, log rõ lý do
#
# PERF — No FAST_MODE cho VS Code
#   Fix: FAST_MODE=True chỉ chạy 10 folds/model để verify
#
# PERF — parquet save fallback CSV nếu pyarrow không có
# =============================================================================

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import laplacian_kernel
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
FAST_MODE = False   # True = chỉ 10 folds/model để verify | False = full 111 folds

BASE_DIR  = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project")
# =============================================================================

DATA_PATH   = BASE_DIR / "Data" / "data" / "processed" / "master_dataset.csv"
FOLDS_PATH  = BASE_DIR / "Data" / "results" / "fold_splits.parquet"
PARAMS_PATH = BASE_DIR / "Data" / "results" / "optuna_best_params.json"
OUTPUT_DIR  = BASE_DIR / "Data" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "EUA_return"
FEATURES = [
    "GAS_return", "OIL_return", "COAL_return", "ELEC_return",
    "IP_return", "CPI_return", "POLICY_dummy", "PHASE_dummy",
]
MODELS = ["rbf_svm", "laplacian_svm", "xgboost", "lightgbm"]
INIT_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────
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
    """Save với fallback CSV nếu pyarrow không có."""
    try:
        df.to_parquet(path, index=False, engine="pyarrow")
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logging.warning(f"Parquet failed → saved CSV: {csv_path}")


# ── Model predictors ──────────────────────────────────────────────
def predict_rbf_svm(X_tr, y_tr, X_te, params):
    sc  = RobustScaler(quantile_range=(5, 95)).fit(X_tr)
    svr = SVR(
        kernel="rbf",
        C=params.get("C", 1.0),
        epsilon=params.get("epsilon", 0.1),
        gamma=params.get("gamma", "scale"),
    )
    svr.fit(sc.transform(X_tr), y_tr)
    return svr.predict(sc.transform(X_te))


def predict_laplacian_svm(X_tr, y_tr, X_te, params):
    sc   = RobustScaler(quantile_range=(5, 95)).fit(X_tr)
    Xtr_ = sc.transform(X_tr)
    Xte_ = sc.transform(X_te)
    g    = params.get("gamma", 1.0)
    K_tr = laplacian_kernel(Xtr_, Xtr_, gamma=g)
    K_te = laplacian_kernel(Xte_, Xtr_, gamma=g)
    svr  = SVR(
        kernel="precomputed",
        C=params.get("C", 1.0),
        epsilon=params.get("epsilon", 0.1),
    )
    svr.fit(K_tr, y_tr)
    return svr.predict(K_te)


def predict_xgboost(X_tr, y_tr, X_te, params):
    import xgboost as xgb
    sc = RobustScaler(quantile_range=(5, 95)).fit(X_tr)
    m  = xgb.XGBRegressor(
        **params, random_state=INIT_SEED, verbosity=0, n_jobs=-1
    )
    m.fit(sc.transform(X_tr), y_tr)
    return m.predict(sc.transform(X_te))


def predict_lightgbm(X_tr, y_tr, X_te, params):
    import lightgbm as lgb
    sc = RobustScaler(quantile_range=(5, 95)).fit(X_tr)
    m  = lgb.LGBMRegressor(
        **params, random_state=INIT_SEED, verbose=-1, n_jobs=-1
    )
    m.fit(sc.transform(X_tr), y_tr)
    return m.predict(sc.transform(X_te))


PREDICTORS = {
    "rbf_svm":       predict_rbf_svm,
    "laplacian_svm": predict_laplacian_svm,
    "xgboost":       predict_xgboost,
    "lightgbm":      predict_lightgbm,
}


# ── Main ──────────────────────────────────────────────────────────
def main():
    out_path   = OUTPUT_DIR / "predictions_tree_svm.parquet"
    checkpoint = OUTPUT_DIR / "predictions_tree_svm_checkpoint.parquet"

    # Load params
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"{PARAMS_PATH} not found. Run 03a_optuna_tuning_fixed.py first."
        )
    with open(PARAMS_PATH, encoding="utf-8") as f:
        all_params = json.load(f)

    # BUG 5 FIX: load + ffill NaN
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").sort_index()
    n_nan = df[FEATURES + [TARGET]].isna().sum().sum()
    if n_nan > 0:
        df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].ffill().bfill()
        logging.info(f"Filled {n_nan} NaN values in dataset")

    # BUG 4 FIX: load folds with fallback
    folds = load_folds(FOLDS_PATH)

    # FAST_MODE: chỉ lấy 10 folds đầu tiên để verify
    if FAST_MODE:
        folds = folds.head(10)
        logging.info("FAST MODE: running 10 folds only for verification")

    # Resume checkpoint
    if checkpoint.exists():
        try:
            df_done  = pd.read_parquet(checkpoint)
        except Exception:
            df_done  = pd.read_csv(checkpoint.with_suffix(".csv"))
        done_set = set(zip(df_done["model"], df_done["fold_id"], df_done["horizon"]))
        results  = df_done.to_dict("records")
        logging.info(f"Resume: {len(done_set)} folds already done")
    else:
        done_set, results = set(), []

    total = len(MODELS) * len(folds)
    counter = 0   # BUG 1 FIX: single counter, incremented once per fold
    t_start = time.time()

    for model_name in MODELS:
        params     = all_params.get(model_name, {})
        predict_fn = PREDICTORS[model_name]
        logging.info(f"\n{'='*50}\n  {model_name.upper()}\n{'='*50}")

        for _, fold in folds.iterrows():
            fid  = int(fold["fold_id"])
            h    = int(fold["horizon"])

            if (model_name, fid, h) in done_set:
                counter += 1
                continue

            tr_s = pd.to_datetime(fold["train_start"])
            tr_e = pd.to_datetime(fold["train_end"])
            te_s = pd.to_datetime(fold["test_start"])
            te_e = pd.to_datetime(fold["test_end"])

            X_tr = df.loc[tr_s:tr_e, FEATURES].values
            y_tr = df.loc[tr_s:tr_e, TARGET].values
            X_te = df.loc[te_s:te_e, FEATURES].values
            y_te = df.loc[te_s:te_e, TARGET].values

            # BUG 6 FIX: pre-flight check
            if len(X_tr) == 0 or len(X_te) == 0:
                logging.warning(
                    f"  Skip fold {fid} H={h}: "
                    f"X_tr={X_tr.shape} X_te={X_te.shape}"
                )
                counter += 1
                continue

            t0 = time.time()
            try:
                y_pred = predict_fn(X_tr, y_tr, X_te, params)
            except Exception as e:
                logging.warning(f"  {model_name} fold {fid} H={h} error: {e}")
                y_pred = np.array([np.nan])

            elapsed   = round(time.time() - t0, 3)
            fold_rmse = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
            counter  += 1   # BUG 1 FIX: only increment once

            results.append({
                "model":       model_name,
                "fold_id":     fid,
                "horizon":     h,
                "y_true":      float(y_te[0]),    # BUG 2 FIX: scalar
                "y_pred":      float(y_pred[0]),  # BUG 2 FIX: scalar
                # BUG 3 FIX: removed y_pred_std (was hardcoded 0.0)
                "n_seeds":     1,
                "train_start": str(tr_s.date()),
                "test_start":  str(te_s.date()),
                "test_end":    str(te_e.date()),
                "n_train":     len(X_tr),
                "time_sec":    elapsed,
                "fold_rmse":   fold_rmse,
            })
            done_set.add((model_name, fid, h))

            # ETA
            elapsed_total = time.time() - t_start
            rate    = counter / max(elapsed_total, 1e-6)
            eta_sec = (total - counter) / max(rate, 1e-6)
            eta_str = f"{eta_sec/60:.0f}m"

            logging.info(
                f"  [{counter}/{total}] Fold {fid:03d} H={h:2d}d | "
                f"RMSE={fold_rmse:.5f} | {elapsed:.2f}s | ETA {eta_str}"
            )

            # Checkpoint mỗi 40 folds
            if counter % 40 == 0:
                save_parquet(pd.DataFrame(results), checkpoint)

    # Save final
    df_out = pd.DataFrame(results)
    save_parquet(df_out, out_path)
    if checkpoint.exists():
        checkpoint.unlink(missing_ok=True)

    # Summary
    total_min = (time.time() - t_start) / 60
    print(f"\n{'='*55}")
    print("  TREE & SVM — RMSE SUMMARY")
    print(f"{'='*55}")
    if len(df_out) > 0:
        summary = (df_out.groupby(["model", "horizon"])["fold_rmse"]
                   .mean().round(5).reset_index())
        print(summary.to_string(index=False))
    print(f"\nTotal time  : {total_min:.1f} min")
    print(f"Folds done  : {len(df_out)}")
    print(f"Saved       : {out_path}")
    if FAST_MODE:
        print("\nFAST MODE complete. Set FAST_MODE=False for full run.")
    else:
        print("\nBuoc 2a xong. Chay tiep 03d_merge.py sau khi 03c xong.")


if __name__ == "__main__":
    main()
