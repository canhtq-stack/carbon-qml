# =============================================================================
# bootstrap_hedging_ci_fixed.py  —  v2
# =============================================================================
# BUG 1 [PERF] — nonparametric_bootstrap O(n_boot) Python loop → vectorized
#   Fix: numpy (n_boot, n) index matrix → ~100x faster
#
# BUG 2 — import pandas as _pd bên trong hàm → tái import mỗi lần gọi
#   Fix: import ở đầu file
#
# BUG 3 — nonparametric dùng "rmse" column nhưng metrics_fold_level có "fold_rmse"
#   Fix: thử cả hai column names
#
# BUG 4 — N_PEAK_FOLDS_H5=12 hardcoded → không match với data thực
#   Fix: đọc n_folds từ crisis_subperiod.csv
#
# BUG 5 — PEAK_CRISIS_YEAR=2022 hardcoded cho nonparametric filter
#   Fix: đọc từ break_dates.json nếu có, fallback về 2022
#
# BUG 6 — argparse → không chạy được bằng VS Code Run button
#   Fix: thêm CONFIG section ở đầu, argparse chỉ override nếu có CLI args
# =============================================================================

import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# CẤU HÌNH — chỉnh tại đây rồi nhấn Run trong VS Code
# (Hoặc chạy CLI: python bootstrap_hedging_ci_fixed.py --data_dir /path/to/results)
# =============================================================================
DATA_DIR  = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project\Data\results")
METHOD    = "parametric"     # "parametric" hoặc "nonparametric"
N_BOOT    = 10_000           # số bootstrap replications
SEED      = 42
OUTPUT    = "bootstrap_ci_results.csv"
# =============================================================================

# Hằng số kinh tế
EUR_MARKET_SIZE  = 28_000_000_000
HEDGE_COST_RATIO = 0.005


# ── CLI override (optional) ───────────────────────────────────────
def _apply_cli():
    """Override CONFIG bằng CLI args nếu chạy từ terminal."""
    global DATA_DIR, METHOD, N_BOOT, SEED, OUTPUT
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--method",   default=None, choices=["parametric","nonparametric"])
    p.add_argument("--n_boot",   default=None, type=int)
    p.add_argument("--seed",     default=None, type=int)
    p.add_argument("--output",   default=None)
    args, _ = p.parse_known_args()
    if args.data_dir: DATA_DIR = Path(args.data_dir)
    if args.method:   METHOD   = args.method
    if args.n_boot:   N_BOOT   = args.n_boot
    if args.seed:     SEED     = args.seed
    if args.output:   OUTPUT   = args.output


# ── Load helpers ──────────────────────────────────────────────────
def _load(filename, parse_dates=None, required=True):
    path = DATA_DIR / filename
    if not path.exists():
        if required:
            sys.exit(f"ERROR: File không tìm thấy: {path}")
        return None
    kw = {"parse_dates": parse_dates} if parse_dates else {}
    return pd.read_csv(path, **kw)


def _load_break_dates():
    """BUG 5 FIX: đọc peak crisis start year từ break_dates.json."""
    bp = DATA_DIR.parent / "config" / "break_dates.json"
    if bp.exists():
        with open(bp) as f:
            data = json.load(f)
        breaks = sorted(data.get("break_dates", []))
        if len(breaks) >= 2:
            # peak_crisis starts at breaks[1]
            peak_year = int(str(breaks[1])[:4])
            return peak_year, breaks
    return 2022, []   # fallback


def _get_n_folds(cs_df, model, horizon, regime):
    """BUG 4 FIX: lấy n_folds thực từ crisis_subperiod.csv."""
    sub = cs_df[(cs_df["model"]==model) &
                (cs_df["horizon"]==horizon) &
                (cs_df["regime"]==regime)]
    if "n_folds" in sub.columns and not sub.empty:
        return int(sub["n_folds"].iloc[0])
    # fallback: đọc từ rmse_count nếu có
    if "rmse_count" in sub.columns and not sub.empty:
        return int(sub["rmse_count"].iloc[0])
    return 12   # hardcode fallback


def _get_rmse_col(df):
    """BUG 3 FIX: tìm đúng column name cho RMSE."""
    for col in ["rmse", "fold_rmse", "rmse_mean"]:
        if col in df.columns:
            return col
    raise KeyError(f"Không tìm thấy cột RMSE. Columns: {list(df.columns)}")


# ── Savings formula ───────────────────────────────────────────────
def savings_formula(mean_qk, mean_bench):
    if mean_bench <= 0: return 0.0
    return max(0.0, (mean_bench - mean_qk) / mean_bench) * EUR_MARKET_SIZE * HEDGE_COST_RATIO


# ── Bootstrap functions ───────────────────────────────────────────
def parametric_bootstrap(mu_qk, sd_qk, mu_bench, sd_bench, n_folds, n_boot, rng):
    """Vectorized parametric bootstrap (Normal draws)."""
    qk    = np.maximum(rng.normal(mu_qk,   sd_qk,   (n_boot, n_folds)), 1e-8)
    bench = np.maximum(rng.normal(mu_bench, sd_bench, (n_boot, n_folds)), 1e-8)
    improvement = (bench.mean(1) - qk.mean(1)) / bench.mean(1)
    return np.maximum(0.0, improvement) * EUR_MARKET_SIZE * HEDGE_COST_RATIO


def nonparametric_bootstrap(rmse_qk, rmse_bench, n_boot, rng):
    """
    BUG 1 FIX: vectorized nonparametric bootstrap.
    Cũ: for b in range(n_boot) → O(n_boot) Python loop → ~100ms
    Mới: numpy (n_boot, n) index matrix → ~1ms
    """
    n   = len(rmse_qk)
    idx = rng.integers(0, n, size=(n_boot, n))         # (n_boot, n)
    qk_means    = rmse_qk[idx].mean(axis=1)             # (n_boot,)
    bench_means = rmse_bench[idx].mean(axis=1)
    improvement = (bench_means - qk_means) / np.maximum(bench_means, 1e-10)
    return np.maximum(0.0, improvement) * EUR_MARKET_SIZE * HEDGE_COST_RATIO


# ── Summary / print ───────────────────────────────────────────────
def summarise(dist, pt_savings, label):
    return {
        "scenario":             label,
        "point_est_eur":        round(pt_savings),
        "bootstrap_mean_eur":   round(dist.mean()),
        "ci_lower_95":          round(np.percentile(dist,  2.5)),
        "ci_upper_95":          round(np.percentile(dist, 97.5)),
        "ci_lower_90":          round(np.percentile(dist,  5.0)),
        "ci_upper_90":          round(np.percentile(dist, 95.0)),
        "pct_savings_positive": round(100.0 * np.mean(dist > 0), 1),
        "std_eur":              round(dist.std()),
        "_dist":                dist,
    }


def print_result(r):
    print(f"    Point estimate   : EUR {r['point_est_eur']:>12,.0f}")
    print(f"    Bootstrap mean   : EUR {r['bootstrap_mean_eur']:>12,.0f}")
    print(f"    95% CI           : [EUR {r['ci_lower_95']:>10,.0f} , EUR {r['ci_upper_95']:>10,.0f}]")
    print(f"    90% CI           : [EUR {r['ci_lower_90']:>10,.0f} , EUR {r['ci_upper_90']:>10,.0f}]")
    print(f"    P(savings > 0)   : {r['pct_savings_positive']:.1f}%")


# ── Main ──────────────────────────────────────────────────────────
def main():
    _apply_cli()   # BUG 6 FIX: CLI override optional
    rng = np.random.default_rng(SEED)

    peak_year, breaks = _load_break_dates()  # BUG 5 FIX

    print("=" * 65)
    print("QK-SVR Hedging Savings — Bootstrap CI")
    print(f"  Phương pháp  : {METHOD}")
    print(f"  n_boot       : {N_BOOT:,}   |   seed : {SEED}")
    print(f"  EU ETS volume: EUR {EUR_MARKET_SIZE/1e9:.0f}B")
    print(f"  Hedge ratio  : {HEDGE_COST_RATIO*100:.1f}%")
    print(f"  Peak Crisis  : {peak_year} (from {'break_dates.json' if breaks else 'default'})")
    print("=" * 65)

    # Xác minh point estimate
    pi = _load("policy_implications.csv", required=False)
    if pi is not None:
        h5 = pi[(pi["horizon"]==5) & (pi["regime"]=="peak_crisis")]
        if not h5.empty:
            pt = float(h5["hedging_savings_eur"].values[0])
            print(f"\n  Point estimate từ policy_implications.csv:")
            print(f"    H=5 Peak Crisis: EUR {pt:,.0f}  (manuscript: EUR 821,907)")

    results = []

    # ── Parametric ────────────────────────────────────────────────
    if METHOD == "parametric":
        cs = _load("crisis_subperiod.csv")
        ms = _load("metrics_summary.csv")   # BUG 2 FIX: import pd at top

        def get_mu_sd(model, h, regime="peak_crisis"):
            if regime == "full_sample":
                r = ms[(ms["model"]==model) & (ms["horizon"]==h)]
                if r.empty:
                    sys.exit(f"ERROR: {model} H={h} không có trong metrics_summary.csv")
                return float(r["rmse_mean"].iloc[0]), float(r["rmse_std"].iloc[0])
            r = cs[(cs["model"]==model) & (cs["horizon"]==h) & (cs["regime"]==regime)]
            if r.empty:
                sys.exit(f"ERROR: {model} H={h} {regime} không có trong crisis_subperiod.csv")
            return float(r["rmse_mean"].values[0]), float(r["rmse_std"].values[0])

        # BUG 4 FIX: lấy n_folds thực từ data
        n_peak_folds_h5 = _get_n_folds(cs, "qk_svr", 5, "peak_crisis")
        print(f"\n  n_folds (peak_crisis, H=5): {n_peak_folds_h5} (từ crisis_subperiod.csv)")

        scenarios = [
            {
                "label":  "QK-SVR vs. Laplacian-SVM | H=5 | Peak Crisis",
                "desc":   "PRIMARY — best classical (point est: EUR 0.82M)",
                "bench":  "laplacian_svm", "h": 5, "regime": "peak_crisis",
                "n_folds": n_peak_folds_h5,
            },
            {
                "label":  "QK-SVR vs. Transformer | H=5 | Peak Crisis",
                "desc":   "DL benchmark (21.0% RMSE improvement)",
                "bench":  "transformer", "h": 5, "regime": "peak_crisis",
                "n_folds": n_peak_folds_h5,
            },
            {
                "label":  "QK-SVR vs. RBF-SVM | H=22 | Peak Crisis",
                "desc":   "H=22 (QK-SVR underperforms → savings = 0)",
                "bench":  "rbf_svm", "h": 22, "regime": "peak_crisis",
                "n_folds": _get_n_folds(cs, "qk_svr", 22, "peak_crisis"),
            },
            {
                "label":  "QK-SVR vs. EMD-LSTM | H=5 | Full Sample",
                "desc":   "Full-sample (CI from bootstrap)",
                "bench":  "emd_lstm", "h": 5, "regime": "full_sample",
                "n_folds": 37,
            },
        ]

        for sc in scenarios:
            print(f"\n  [{sc['label']}]")
            print(f"  {sc['desc']}")
            mu_qk,    sd_qk    = get_mu_sd("qk_svr",   sc["h"], sc["regime"])
            mu_bench, sd_bench = get_mu_sd(sc["bench"], sc["h"], sc["regime"])
            pt = savings_formula(mu_qk, mu_bench)

            print(f"    μ_QK-SVR    = {mu_qk:.5f}  (σ = {sd_qk:.5f})")
            print(f"    μ_{sc['bench']:18s} = {mu_bench:.5f}  (σ = {sd_bench:.5f})")
            print(f"    RMSE impr.  : {(mu_bench-mu_qk)/mu_bench*100:+.2f}%")
            print(f"    n_folds     : {sc['n_folds']}")

            dist = parametric_bootstrap(mu_qk, sd_qk, mu_bench, sd_bench,
                                        sc["n_folds"], N_BOOT, rng)
            r = summarise(dist, pt, sc["label"])
            print_result(r)
            results.append(r)

    # ── Nonparametric ─────────────────────────────────────────────
    else:
        mfl = _load("metrics_fold_level.csv", parse_dates=["test_start"])
        rmse_col = _get_rmse_col(mfl)   # BUG 3 FIX
        mfl["year"] = mfl["test_start"].dt.year

        # BUG 5 FIX: dùng peak_year từ break_dates.json
        pc_ids = sorted(
            mfl[(mfl["year"]==peak_year) &
                (mfl["model"]=="qk_svr") &
                (mfl["horizon"]==5)]["fold_id"].unique()
        )
        if len(pc_ids) == 0:
            print(f"\n  ⚠️ Không có peak-crisis folds trong năm {peak_year}")
            print("     Thử các năm lân cận...")
            for yr in [peak_year-1, peak_year+1]:
                pc_ids = sorted(mfl[(mfl["year"]==yr) &
                                    (mfl["model"]=="qk_svr") &
                                    (mfl["horizon"]==5)]["fold_id"].unique())
                if len(pc_ids) > 0:
                    print(f"     Tìm thấy {len(pc_ids)} folds trong năm {yr}")
                    break

        print(f"\n  Peak-crisis fold IDs ({len(pc_ids)}): {pc_ids}")
        print("  CẢNH BÁO: CI rộng vì n_obs nhỏ.\n")

        def get_fold_arr(model, h):
            sub = mfl[(mfl["model"]==model) &
                      (mfl["horizon"]==h) &
                      (mfl["fold_id"].isin(pc_ids))]
            return sub.set_index("fold_id")[rmse_col]   # BUG 3 FIX

        np_scenarios = [
            ("QK-SVR vs. Laplacian-SVM | H=5 | Peak Crisis (NP)", "laplacian_svm", 5),
            ("QK-SVR vs. Transformer   | H=5 | Peak Crisis (NP)", "transformer",   5),
        ]

        for label, bench, h in np_scenarios:
            print(f"\n  [{label}]")
            qk_s = get_fold_arr("qk_svr", h)
            bm_s = get_fold_arr(bench, h)
            shared = sorted(set(qk_s.index) & set(bm_s.index))
            if not shared:
                print("    SKIP: không có shared folds"); continue

            qk_arr  = qk_s.loc[shared].values.astype(float)
            bm_arr  = bm_s.loc[shared].values.astype(float)
            pt      = savings_formula(qk_arr.mean(), bm_arr.mean())

            print(f"    n_folds          = {len(shared)}")
            print(f"    Mean RMSE QK-SVR = {qk_arr.mean():.5f}")
            print(f"    Mean RMSE bench  = {bm_arr.mean():.5f}  [{bench}]")

            dist = nonparametric_bootstrap(qk_arr, bm_arr, N_BOOT, rng)  # BUG 1 FIX
            r = summarise(dist, pt, label)
            print_result(r)
            results.append(r)

    # ── Summary table ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TABLE 10 — Hedging Savings Estimates")
    print("=" * 65)
    print(f"\n{'Scenario':<48} {'Point':>8}  {'95% CI':>25}")
    print("─" * 88)
    for r in results:
        lbl = r["scenario"].split("|")[0].strip()
        print(f"{lbl:<48} EUR {r['point_est_eur']/1e6:>4.2f}M  "
              f"EUR {r['ci_lower_95']/1e6:.2f}M – EUR {r['ci_upper_95']/1e6:.2f}M")

    # ── Manuscript guidance ───────────────────────────────────────
    print("\n" + "─" * 65)
    print("HƯỚNG DẪN CẬP NHẬT MANUSCRIPT:")
    print("─" * 65)

    primary = next((r for r in results if "Laplacian" in r["scenario"]
                    and "(NP)" not in r["scenario"]), None)
    if primary:
        lo, hi, pt = primary["ci_lower_95"], primary["ci_upper_95"], primary["point_est_eur"]
        print(f"\n  Abstract/Highlights (vs. Laplacian-SVM, H=5, Peak Crisis):")
        print(f"    Point: EUR {pt/1e6:.2f}M")
        print(f"    95% CI: EUR {lo/1e6:.2f}M – EUR {hi/1e6:.2f}M")

    tf = next((r for r in results if "Transformer" in r["scenario"]
               and "(NP)" not in r["scenario"]), None)
    if tf:
        print(f"\n  Table 10 (vs. Transformer, H=5, Peak Crisis):")
        print(f"    Point: EUR {tf['point_est_eur']/1e6:.2f}M | "
              f"95% CI: EUR {tf['ci_lower_95']/1e6:.2f}M–EUR {tf['ci_upper_95']/1e6:.2f}M")

    h22 = next((r for r in results if "H=22" in r["scenario"]), None)
    if h22:
        print(f"\n  Table 10 (H=22, underperforms): EUR 0 (95% CI: 0 – EUR {h22['ci_upper_95']/1e6:.2f}M)")

    # ── Save ──────────────────────────────────────────────────────
    if results:
        clean = [{k: v for k, v in r.items() if k != "_dist"} for r in results]
        out_path = DATA_DIR / OUTPUT
        pd.DataFrame(clean).to_csv(out_path, index=False, float_format="%.2f")
        print(f"\n  Saved: {out_path}")

        dist_path = out_path.with_stem(out_path.stem + "_distributions")
        pd.DataFrame({r["scenario"]: r["_dist"] for r in results}).to_csv(
            dist_path, index=False, float_format="%.2f")
        print(f"  Saved: {dist_path}")

    print("\nHoàn thành.")


if __name__ == "__main__":
    main()
