# =============================================================================
# 04_statistical_validation_fixed.py  —  v2
# =============================================================================
# BUG 1 [CRITICAL] — load_predictions đọc cả qksvr_predictions VÀ benchmark
#   → duplicate qk_svr rows nếu 03d đã merge QK-SVR vào benchmark
#   Fix: chỉ đọc benchmark_predictions.parquet (đã có đầy đủ tất cả models)
#
# BUG 2 [CRITICAL] — extract_arrays dùng np.array(list) nhưng y_true/y_pred
#   là scalar sau 03d_merge_fixed → np.array(float) = 0-dim → crash
#   Fix: wrap scalar thành array(float).reshape(1,) trước khi dùng
#
# BUG 3 [CRITICAL] — run_dm_tests dùng np.concatenate với list column
#   Fix: collect scalars từ từng fold thành array (T,)
#
# BUG 4 [CRITICAL] — add_romano_wolf cũng dùng concatenate với list
#   Fix: same as BUG 3
#
# BUG 5 — compute_crisis_metrics có dead code (if False else)
#   Fix: rewrite merge logic đúng cách
#
# BUG 6 — No FAST_MODE cho VS Code testing
#   Fix: FAST_MODE=True bỏ qua Romano-Wolf và MCS bootstrap (~5 phút)
#
# BUG 7 — No parquet fallback khi đọc input files
#   Fix: try parquet → fallback CSV
# =============================================================================

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)

# =============================================================================
# CẤU HÌNH — chỉnh tại đây rồi nhấn Run trong VS Code
# =============================================================================
FAST_MODE = True   # True = bỏ qua bootstrap (~5 phút) | False = full (~30 phút)

BASE_DIR    = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project")
# =============================================================================

RESULTS_DIR = BASE_DIR / "Data" / "results"
CONFIG_DIR  = BASE_DIR / "config"
OUTPUT_DIR  = RESULTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_MODEL   = "qk_svr"
ALPHA_DM        = 0.05
ALPHA_MCS       = 0.25
RW_BOOTSTRAPS   = 500 if FAST_MODE else 1000
MCS_BOOTSTRAPS  = 500 if FAST_MODE else 1000
NEWEY_WEST_LAGS = 5
HORIZONS        = [1, 5, 22]

FALLBACK_REGIMES = {
    "pre_crisis":   ("2019-01-01", "2021-12-31"),
    "crisis_onset": ("2022-01-01", "2022-06-30"),
    "peak_crisis":  ("2022-07-01", "2023-06-30"),
    "post_crisis":  ("2023-07-01", "2024-12-31"),
}


# ── Helpers ───────────────────────────────────────────────────────
def load_parquet(path: Path) -> Optional[pd.DataFrame]:
    """BUG 7 FIX: fallback CSV."""
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        logging.info(f"Parquet not found → using CSV: {csv.name}")
        return pd.read_csv(csv)
    return None


def to_array(val) -> np.ndarray:
    """
    BUG 2 FIX: convert scalar, list, or ndarray to 1-D float array.
    - scalar (float/int): → array([val])
    - list: → np.array(list)
    - ndarray: → flatten to 1-D
    """
    if isinstance(val, (int, float, np.floating, np.integer)):
        return np.array([float(val)])
    arr = np.array(val, dtype=float)
    return arr.reshape(-1)


def significance_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ── Step 0: Data loading ──────────────────────────────────────────
def load_predictions() -> pd.DataFrame:
    """
    BUG 1 FIX: chỉ đọc benchmark_predictions.parquet (đã có QK-SVR từ 03d).
    Không đọc qksvr_predictions riêng để tránh duplicate.
    """
    bm_path = RESULTS_DIR / "benchmark_predictions.parquet"
    df = load_parquet(bm_path)
    if df is None:
        raise FileNotFoundError(
            f"{bm_path} not found.\n"
            "Run 03d_merge_fixed.py first."
        )

    # Standardize: đảm bảo cột y_pred tồn tại
    if "y_pred_mean" in df.columns and "y_pred" not in df.columns:
        df = df.rename(columns={"y_pred_mean": "y_pred"})

    # Ensure datetime
    for col in ["test_start", "test_end", "train_start"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Check QK-SVR present
    if PRIMARY_MODEL not in df["model"].unique():
        logging.warning(
            f"⚠️ {PRIMARY_MODEL} not in benchmark_predictions.\n"
            "   Run 03d_merge_fixed.py with QK-SVR predictions present."
        )

    models   = sorted(df["model"].unique())
    horizons = sorted(df["horizon"].unique())
    logging.info(f"📥 Loaded: {len(df)} rows | Models: {models} | H: {horizons}")
    return df


def load_break_dates() -> Dict[str, Tuple[str, str]]:
    bp_path = CONFIG_DIR / "break_dates.json"
    if not bp_path.exists():
        logging.warning("break_dates.json not found → using fallback regimes.")
        return FALLBACK_REGIMES
    with open(bp_path) as f:
        data = json.load(f)
    breaks = sorted(data.get("break_dates", []))
    if len(breaks) < 2:
        logging.warning("< 2 break dates → using fallback regimes.")
        return FALLBACK_REGIMES
    regimes = {
        "pre_crisis":   ("2019-01-01", breaks[0]),
        "crisis_onset": (breaks[0],    breaks[1]),
        "peak_crisis":  (breaks[1],    breaks[2] if len(breaks) > 2 else "2023-06-30"),
        "post_crisis":  (breaks[2] if len(breaks) > 2 else "2023-07-01", "2024-12-31"),
    }
    logging.info(f"Break dates: {breaks}")
    return regimes


# ── Step 1: Metrics ───────────────────────────────────────────────
def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def compute_mape(y_true, y_pred, eps=1e-8):
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def compute_da(y_true, y_pred):
    mask = (y_true != 0) & (y_pred != 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])) * 100)

def compute_sharpe(y_true, y_pred):
    pnl = np.sign(y_pred) * y_true
    if pnl.std() < 1e-10:
        return np.nan
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """BUG 2 FIX: use to_array() for y_true/y_pred."""
    records = []
    for _, row in df.iterrows():
        y_true = to_array(row["y_true"])   # BUG 2 FIX
        y_pred = to_array(row["y_pred"])   # BUG 2 FIX
        records.append({
            "model":      row["model"],
            "fold_id":    row["fold_id"],
            "horizon":    row["horizon"],
            "test_start": row.get("test_start", pd.NaT),
            "rmse":       compute_rmse(y_true, y_pred),
            "mae":        compute_mae(y_true, y_pred),
            "mape":       compute_mape(y_true, y_pred),
            "da":         compute_da(y_true, y_pred),
            "sharpe":     compute_sharpe(y_true, y_pred),
            "n_obs":      len(y_true),
        })
    return pd.DataFrame(records)


def summarise_metrics(df_metrics: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df_metrics.groupby(["model", "horizon"])[["rmse", "mae", "mape", "da", "sharpe"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in agg.columns
    ]
    return agg


# ── Step 2: Diebold-Mariano Test ──────────────────────────────────
def newey_west_variance(d: np.ndarray, lags: int = NEWEY_WEST_LAGS) -> float:
    T = len(d)
    d_dm = d - d.mean()
    nw = np.dot(d_dm, d_dm) / T
    for h in range(1, lags + 1):
        gamma_h = np.dot(d_dm[h:], d_dm[:-h]) / T
        nw += 2 * (1.0 - h / (lags + 1)) * gamma_h
    return max(nw, 1e-12)


def hln_correction(T: int, horizon: int) -> float:
    h = horizon
    c = (T + 1 - 2 * h + h * (h - 1) / T) / T
    return float(np.sqrt(max(c, 1e-6)))


def dm_test(y_true, y_pred_ref, y_pred_alt, horizon) -> dict:
    e_ref = y_true - y_pred_ref
    e_alt = y_true - y_pred_alt
    d     = e_ref ** 2 - e_alt ** 2
    T     = len(d)
    d_bar = d.mean()
    nw_v  = newey_west_variance(d)
    dm    = d_bar / np.sqrt(nw_v / T)
    hln   = hln_correction(T, horizon)
    dm_h  = dm * hln
    pval  = float(2 * stats.t.sf(np.abs(dm_h), df=T - 1))
    return {"dm_stat": float(dm), "dm_hln": float(dm_h),
            "p_value": pval, "d_bar": float(d_bar), "T": T}


def get_aligned_arrays(
    df_h: pd.DataFrame, model_a: str, model_b: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    BUG 3 FIX: collect scalar y_true/y_pred per fold into arrays.
    Returns (y_true, y_pred_a, y_pred_b) aligned on common fold_ids.
    """
    rows_a = df_h[df_h["model"] == model_a].set_index("fold_id")
    rows_b = df_h[df_h["model"] == model_b].set_index("fold_id")
    common = sorted(set(rows_a.index) & set(rows_b.index))
    if not common:
        return None

    y_true_list, y_pred_a_list, y_pred_b_list = [], [], []
    for fid in common:
        y_true_list.append(to_array(rows_a.loc[fid, "y_true"]))
        y_pred_a_list.append(to_array(rows_a.loc[fid, "y_pred"]))
        y_pred_b_list.append(to_array(rows_b.loc[fid, "y_pred"]))

    return (
        np.concatenate(y_true_list),
        np.concatenate(y_pred_a_list),
        np.concatenate(y_pred_b_list),
    )


def run_dm_tests(df: pd.DataFrame, benchmarks: List[str]) -> pd.DataFrame:
    records = []
    for horizon in HORIZONS:
        df_h = df[df["horizon"] == horizon]
        if PRIMARY_MODEL not in df_h["model"].unique():
            logging.warning(f"No QK-SVR for H={horizon} — skipping DM.")
            continue

        for bm in benchmarks:
            arrays = get_aligned_arrays(df_h, PRIMARY_MODEL, bm)
            if arrays is None:
                logging.warning(f"  No aligned folds: QK-SVR vs {bm} H={horizon}")
                continue
            y_true, y_ref, y_bm = arrays
            res = dm_test(y_true, y_ref, y_bm, horizon)
            res.update({"model_ref": PRIMARY_MODEL, "model_alt": bm, "horizon": horizon})
            records.append(res)

    df_dm = pd.DataFrame(records)
    logging.info(f"✅ DM tests: {len(df_dm)} comparisons")
    return df_dm


# ── Step 3: Romano-Wolf ───────────────────────────────────────────
def romano_wolf_stepdown(
    loss_ref: np.ndarray,
    loss_matrix: np.ndarray,
    n_bootstrap: int = RW_BOOTSTRAPS,
    seed: int = 42,
) -> np.ndarray:
    """BUG 4 FIX: inputs are pre-built arrays, no list concatenation here."""
    rng = np.random.default_rng(seed)
    K, T = loss_matrix.shape
    D = loss_ref[None, :] - loss_matrix   # (K, T)
    d_bar = D.mean(axis=1)

    boot_max = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, T, size=T)
        D_b = D[:, idx] - d_bar[:, None]
        t_b = D_b.mean(axis=1) / (D_b.std(axis=1, ddof=1) / np.sqrt(T) + 1e-12)
        boot_max[b] = t_b.min()

    t_obs = d_bar / (D.std(axis=1, ddof=1) / np.sqrt(T) + 1e-12)
    rw    = np.array([float(np.mean(boot_max <= t_obs[k])) for k in range(K)])

    # Enforce monotonicity
    order = np.argsort(t_obs)
    rs    = rw[order]
    for i in range(1, K):
        rs[i] = max(rs[i], rs[i - 1])
    rw[order] = rs
    return rw


def add_romano_wolf(df_dm: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """BUG 4 FIX: build loss arrays using get_aligned_arrays."""
    rw_rows = []
    for horizon in HORIZONS:
        df_h  = df[df["horizon"] == horizon]
        dm_h  = df_dm[df_dm["horizon"] == horizon]
        if len(dm_h) == 0:
            continue

        benchmarks = dm_h["model_alt"].tolist()
        loss_list, valid_bms = [], []

        for bm in benchmarks:
            arrays = get_aligned_arrays(df_h, PRIMARY_MODEL, bm)
            if arrays is None:
                continue
            y_true, y_ref, y_bm = arrays
            loss_list.append((y_true - y_bm) ** 2)
            valid_bms.append(bm)

        if not loss_list:
            continue

        # Build loss_ref aligned with same folds
        arrays0 = get_aligned_arrays(df_h, PRIMARY_MODEL, valid_bms[0])
        y_true0, y_ref0, _ = arrays0
        loss_ref = (y_true0 - y_ref0) ** 2
        T = min(len(loss_ref), min(len(l) for l in loss_list))
        loss_ref    = loss_ref[:T]
        loss_matrix = np.stack([l[:T] for l in loss_list])

        rw_pvals = romano_wolf_stepdown(loss_ref, loss_matrix)
        for bm, rw_p in zip(valid_bms, rw_pvals):
            rw_rows.append({"horizon": horizon, "model_alt": bm, "rw_pvalue": float(rw_p)})

    if rw_rows:
        df_rw = pd.DataFrame(rw_rows)
        df_dm = df_dm.merge(df_rw, on=["horizon", "model_alt"], how="left")
        logging.info("✅ Romano-Wolf p-values attached.")
    else:
        df_dm["rw_pvalue"] = np.nan

    df_dm["sig_dm"] = df_dm["p_value"].apply(significance_stars)
    df_dm["sig_rw"] = df_dm.get("rw_pvalue", pd.Series(np.nan)).apply(significance_stars)
    return df_dm


# ── Step 4: MCS ───────────────────────────────────────────────────
def model_confidence_set(
    loss_matrix: np.ndarray,
    model_names: List[str],
    alpha: float = ALPHA_MCS,
    n_bootstrap: int = MCS_BOOTSTRAPS,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    K, T = loss_matrix.shape
    surviving = list(range(K))
    eliminated = []

    while len(surviving) > 1:
        sub   = loss_matrix[surviving, :]
        d_bar = sub.mean(axis=1) - sub.mean()

        boot_max = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            idx   = rng.integers(0, T, size=T)
            sb    = sub[:, idx]
            db    = sb.mean(axis=1) - sb.mean()
            se_b  = (sb - sb.mean(axis=1, keepdims=True)).std(axis=1, ddof=1) / np.sqrt(T) + 1e-12
            t_b   = (db - d_bar) / se_b
            boot_max[b] = t_b.max()

        se_obs = (sub - sub.mean(axis=1, keepdims=True)).std(axis=1, ddof=1) / np.sqrt(T) + 1e-12
        t_obs  = d_bar / se_obs
        p_vals = np.array([float(np.mean(boot_max >= t_obs[i])) for i in range(len(surviving))])

        worst = int(np.argmin(p_vals))
        if p_vals[worst] < alpha:
            eliminated.append(surviving[worst])
            surviving.pop(worst)
        else:
            break

    records = []
    for i, name in enumerate(model_names):
        records.append({
            "model":         name,
            "in_mcs":        i in surviving,
            "elim_order":    eliminated.index(i) + 1 if i in eliminated else None,
            "mean_sq_error": float(loss_matrix[i].mean()),
        })
    df_mcs = pd.DataFrame(records).sort_values("mean_sq_error").reset_index(drop=True)
    in_set = [r["model"] for r in records if r["in_mcs"]]
    logging.info(f"✅ MCS (α={alpha}): {in_set}")
    return df_mcs


def run_mcs(df: pd.DataFrame) -> pd.DataFrame:
    all_mcs = []
    for horizon in HORIZONS:
        df_h   = df[df["horizon"] == horizon]
        models = sorted(df_h["model"].unique())

        # Build aligned loss matrix
        pivot = {}
        for m in models:
            m_rows = df_h[df_h["model"] == m]
            pivot[m] = {
                int(row["fold_id"]): float(to_array(row["y_true"])[0] - to_array(row["y_pred"])[0]) ** 2
                for _, row in m_rows.iterrows()
            }

        common = sorted(set.intersection(*[set(v.keys()) for v in pivot.values()]))
        if len(common) < 10:
            logging.warning(f"MCS H={horizon}: only {len(common)} common folds — skip.")
            continue

        loss_matrix = np.array([[pivot[m][f] for f in common] for m in models])
        df_mcs = model_confidence_set(loss_matrix, models)
        df_mcs["horizon"] = horizon
        all_mcs.append(df_mcs)

    return pd.concat(all_mcs, ignore_index=True) if all_mcs else pd.DataFrame()


# ── Step 5: Crisis subperiod ──────────────────────────────────────
def assign_regime(ts, regimes):
    if pd.isna(ts):
        return "unknown"
    for name, (s, e) in regimes.items():
        if pd.Timestamp(s) <= ts <= pd.Timestamp(e):
            return name
    return "other"


def compute_crisis_metrics(
    df_preds: pd.DataFrame,
    df_metrics: pd.DataFrame,
    regimes: dict,
) -> pd.DataFrame:
    """BUG 5 FIX: remove dead code; use proper merge."""
    if "test_start" not in df_preds.columns:
        logging.warning("test_start missing — cannot compute crisis sub-periods.")
        return pd.DataFrame()

    key = ["model", "fold_id", "horizon"]
    ts_map = (df_preds[key + ["test_start"]]
              .drop_duplicates()
              .set_index(key)["test_start"])

    df_m = df_metrics.copy()
    df_m["test_start"] = df_m.set_index(key).index.map(ts_map)
    df_m["regime"]     = df_m["test_start"].apply(lambda x: assign_regime(x, regimes))

    crisis = (
        df_m.groupby(["model", "horizon", "regime"])[["rmse", "mae", "da"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    crisis.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in crisis.columns
    ]
    logging.info(f"✅ Crisis subperiod: {df_m['regime'].value_counts().to_dict()}")
    return crisis


# ── Step 6: Ljung-Box ─────────────────────────────────────────────
def run_ljungbox(df: pd.DataFrame, model=PRIMARY_MODEL, lags=10) -> pd.DataFrame:
    records = []
    for horizon in HORIZONS:
        df_h = df[(df["model"] == model) & (df["horizon"] == horizon)]
        if len(df_h) == 0:
            continue
        residuals = np.concatenate([
            to_array(r["y_true"]) - to_array(r["y_pred"])
            for _, r in df_h.iterrows()
        ])
        try:
            lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
            stat, pval = float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0])
        except Exception as e:
            logging.warning(f"Ljung-Box H={horizon}: {e}")
            stat, pval = np.nan, np.nan
        records.append({
            "model": model, "horizon": horizon,
            "lb_stat": stat, "lb_pvalue": pval,
            "lags": lags, "white_noise": pval > 0.05 if not np.isnan(pval) else None,
        })
    logging.info(f"✅ Ljung-Box done for {model}")
    return pd.DataFrame(records)


# ── Report ────────────────────────────────────────────────────────
def write_markdown_report(df_summary, df_dm, df_mcs, path):
    lines = [
        "# Statistical Validation Report",
        "Generated by: 04_statistical_validation_fixed.py\n",
        "## 1. Overall Forecast Accuracy (RMSE mean ± std)",
        df_summary[["model", "horizon", "rmse_mean", "rmse_std", "da_mean"]]
        .to_markdown(index=False, floatfmt=".4f"),
        "\n## 2. Diebold-Mariano Tests (QK-SVR vs Benchmarks)",
        "Significance: * p<0.10  ** p<0.05  *** p<0.01",
        df_dm[["model_alt", "horizon", "dm_hln", "p_value", "rw_pvalue"]]
        .to_markdown(index=False, floatfmt=".4f"),
        "\n## 3. Model Confidence Set",
        df_mcs[["model", "horizon", "in_mcs", "mean_sq_error"]]
        .to_markdown(index=False, floatfmt=".6f"),
        "\n---\n*All DM tests use HLN small-sample correction + Newey-West HAC.*",
        "*Romano-Wolf FWER correction applied per horizon.*",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"📄 Report saved: {path}")


# ── Main ──────────────────────────────────────────────────────────
def main():
    if FAST_MODE:
        logging.info("FAST MODE: bootstrap disabled for speed (~5 min)")
    else:
        logging.info("FULL MODE: full bootstrap (~30 min)")

    df       = load_predictions()
    regimes  = load_break_dates()
    benchmarks = [m for m in df["model"].unique() if m != PRIMARY_MODEL]
    logging.info(f"Benchmarks: {benchmarks}")

    # Step 1
    logging.info("\n[1/6] Computing metrics...")
    df_metrics = compute_all_metrics(df)
    df_summary = summarise_metrics(df_metrics)
    df_metrics.to_csv(OUTPUT_DIR / "metrics_fold_level.csv", index=False)
    df_summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    logging.info(f"  Saved metrics_summary.csv ({len(df_summary)} rows)")

    # Step 2
    logging.info("\n[2/6] Diebold-Mariano tests...")
    df_dm = run_dm_tests(df, benchmarks)

    # Step 3
    if FAST_MODE:
        logging.info("\n[3/6] Romano-Wolf — SKIPPED (FAST_MODE)")
        df_dm["rw_pvalue"] = np.nan
        df_dm["sig_dm"] = df_dm["p_value"].apply(significance_stars)
        df_dm["sig_rw"] = ""
    else:
        logging.info("\n[3/6] Romano-Wolf stepdown...")
        df_dm = add_romano_wolf(df_dm, df)

    df_dm.to_csv(OUTPUT_DIR / "dm_tests.csv", index=False)
    logging.info(f"  Saved dm_tests.csv ({len(df_dm)} rows)")

    # Step 4
    if FAST_MODE:
        logging.info("\n[4/6] MCS — SKIPPED (FAST_MODE)")
        df_mcs = pd.DataFrame()
    else:
        logging.info("\n[4/6] Model Confidence Set...")
        df_mcs = run_mcs(df)
        if len(df_mcs) > 0:
            df_mcs.to_csv(OUTPUT_DIR / "mcs_results.csv", index=False)
            logging.info(f"  Saved mcs_results.csv ({len(df_mcs)} rows)")

    # Step 5
    logging.info("\n[5/6] Crisis sub-period analysis...")
    df_crisis = compute_crisis_metrics(df, df_metrics, regimes)
    if len(df_crisis) > 0:
        df_crisis.to_csv(OUTPUT_DIR / "crisis_subperiod.csv", index=False)
        logging.info(f"  Saved crisis_subperiod.csv ({len(df_crisis)} rows)")

    # Step 6
    logging.info("\n[6/6] Ljung-Box diagnostics...")
    df_lb = run_ljungbox(df)
    df_lb.to_csv(OUTPUT_DIR / "ljungbox_diagnostics.csv", index=False)

    # Report
    if len(df_mcs) > 0:
        write_markdown_report(
            df_summary, df_dm, df_mcs,
            OUTPUT_DIR / "stat_validation_report.md"
        )

    # Console summary
    print(f"\n{'='*65}")
    print("  VALIDATION COMPLETE — RMSE SUMMARY")
    print(f"{'='*65}")
    print(f"{'Model':<18} {'H=1':>8} {'H=5':>8} {'H=22':>8}")
    print("-" * 46)
    for model in sorted(df["model"].unique()):
        row_str = f"  {model:<16}"
        for h in [1, 5, 22]:
            sub = df_summary[
                (df_summary["model"] == model) & (df_summary["horizon"] == h)
            ]
            val = f"{sub['rmse_mean'].iloc[0]:.5f}" if len(sub) else "  —  "
            row_str += f" {val:>8}"
        print(row_str)

    print(f"\n  DM Tests (H=5, QK-SVR vs benchmarks):")
    dm_h5 = df_dm[df_dm["horizon"] == 5][["model_alt", "dm_hln", "p_value", "sig_dm"]]
    print(dm_h5.to_string(index=False))

    print(f"\n{'='*65}")
    print(f"  Outputs saved to: {OUTPUT_DIR}")
    if FAST_MODE:
        print("  FAST MODE: set FAST_MODE=False for Romano-Wolf + MCS bootstrap")
    print("  Next: 05_interpretability_expressibility.py")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
