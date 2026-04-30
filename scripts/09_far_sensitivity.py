"""
09_far_sensitivity.py
=====================
False Alarm Rate (FAR) Sensitivity Analysis for the Dual-Trigger Dynamic MSR Protocol.

Estimates the annualised false alarm rate of the QK-SVR directional trigger
(Trigger 2) across K ∈ {3, 5, 7} consecutive-day thresholds, using fold-level
hit rates from walk-forward backtesting in non-crisis regimes.

Methodology
-----------
A "false alarm" is defined as K consecutive trading days on which QK-SVR
signals downward EUA price pressure during a non-crisis regime (Pre-Crisis or
Post-Crisis). The Bernoulli sliding-window model is applied:

    P(trigger in fold) = (D - K + 1) × p_down^K

where:
    D         = trading days per fold (21)
    p_down    = 1 - hit_rate  (fraction of fold-level observations with
                               incorrect direction in non-crisis regimes)
    K         = consecutive-day threshold

FAR/year = (expected triggered folds) / (total non-crisis observation years)

Limitation: Bernoulli model assumes independence across consecutive days.
Actual autocorrelation in QK-SVR signals may yield different empirical FAR.
Trigger 1 (coal-gas volatility gate) provides additional filtering not
reflected here; actual deployment FAR is expected to be substantially lower.

Reference: Manuscript Section 6.2; Appendix D, Table D1 (Phase 1 KPI).

Usage
-----
    python 09_far_sensitivity.py

    # Custom input file:
    python 09_far_sensitivity.py --input path/to/trading_simulation.csv

    # Custom output directory:
    python 09_far_sensitivity.py --outdir results/

Output
------
    results/far_sensitivity_K357.csv   — FAR table for K ∈ {3,5,7}, all horizons
    results/far_sensitivity_K357.txt   — Human-readable summary (plain text)

Requirements
------------
    pandas >= 1.3
    numpy  >= 1.21

Author : Tran Quang Canh (canhtq@uef.edu.vn)
Date   : 2026-04-30
License: MIT
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_INPUT  = "results/trading_simulation.csv"
DEFAULT_OUTDIR = "results"

DAYS_PER_FOLD   = 21          # trading days per walk-forward fold (step = 1 month)
TRADING_DAYS    = 252         # trading days per year
K_VALUES        = [3, 5, 7]  # consecutive-day thresholds to evaluate
HORIZONS        = [1, 5, 22] # forecasting horizons (days)

NON_CRISIS_REGIMES = ["pre_crisis", "post_crisis"]
MODEL_NAME         = "qk_svr"

# ──────────────────────────────────────────────────────────────────────────────
# Core Functions
# ──────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load and validate trading simulation CSV."""
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] Input file not found: {path}")

    df = pd.read_csv(path)
    required_cols = {"model", "horizon", "regime", "n_obs", "hit_rate_pct"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing required columns: {missing}")

    return df


def compute_far(p_down: float, K: int, n_folds: int,
                days_per_fold: int = DAYS_PER_FOLD,
                trading_days: int = TRADING_DAYS) -> dict:
    """
    Compute FAR for a given p_down and K using Bernoulli sliding-window model.

    Parameters
    ----------
    p_down       : P(downward signal in one fold-level observation)
    K            : consecutive-day threshold
    n_folds      : number of non-crisis folds
    days_per_fold: trading days per fold
    trading_days : trading days per year

    Returns
    -------
    dict with keys: p_trigger_in_fold, expected_triggered_folds,
                    observation_years, far_per_year
    """
    windows              = days_per_fold - K + 1          # sliding windows per fold
    p_trigger_in_fold    = min(windows * (p_down ** K), 1.0)
    expected_triggered   = n_folds * p_trigger_in_fold
    observation_years    = (n_folds * days_per_fold) / trading_days
    far_per_year         = expected_triggered / observation_years if observation_years > 0 else np.nan

    return {
        "p_trigger_in_fold"    : p_trigger_in_fold,
        "expected_triggered_folds": expected_triggered,
        "observation_years"    : observation_years,
        "far_per_year"         : far_per_year,
    }


def run_far_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run FAR sensitivity analysis for all horizons and K values.

    Parameters
    ----------
    df : trading simulation DataFrame (all models, all regimes)

    Returns
    -------
    pd.DataFrame with FAR estimates
    """
    qk = df[(df["model"] == MODEL_NAME) &
            (df["regime"].isin(NON_CRISIS_REGIMES))].copy()

    records = []

    for horizon in HORIZONS:
        subset = qk[qk["horizon"] == horizon]

        if subset.empty:
            print(f"  [WARN] No non-crisis data for H={horizon} — skipping")
            continue

        # Weighted average p_down across non-crisis regimes
        total_folds = subset["n_obs"].sum()
        if total_folds == 0:
            continue

        # Weighted p_down
        p_down_weighted = (
            ((1 - subset["hit_rate_pct"] / 100) * subset["n_obs"]).sum()
            / total_folds
        )

        # Per-regime breakdown for transparency
        regime_breakdown = {}
        for _, row in subset.iterrows():
            regime_breakdown[row["regime"]] = {
                "n_folds"   : int(row["n_obs"]),
                "hit_rate"  : row["hit_rate_pct"] / 100,
                "p_down"    : 1 - row["hit_rate_pct"] / 100,
            }

        observation_years = (total_folds * DAYS_PER_FOLD) / TRADING_DAYS

        for K in K_VALUES:
            result = compute_far(p_down_weighted, K, total_folds)

            note = ""
            if K == 3:
                note = "Faster detection; higher intervention fatigue risk"
            elif K == 5:
                note = "Baseline — optimal balance (manuscript recommendation)"
            elif K == 7:
                note = "Lower FAR; may exceed 20-day statutory MSR window"

            records.append({
                "horizon"                  : horizon,
                "K"                        : K,
                "n_non_crisis_folds"       : total_folds,
                "observation_years"        : round(observation_years, 2),
                "p_down_weighted"          : round(p_down_weighted, 4),
                "p_trigger_in_fold"        : round(result["p_trigger_in_fold"], 4),
                "expected_triggered_folds" : round(result["expected_triggered_folds"], 2),
                "far_per_year_prefilter"   : round(result["far_per_year"], 2),
                "note"                     : note,
            })

    return pd.DataFrame(records)


def print_summary(results: pd.DataFrame) -> str:
    """Generate human-readable summary text."""
    lines = []
    lines.append("=" * 70)
    lines.append("FAR SENSITIVITY ANALYSIS — Dual-Trigger Dynamic MSR Protocol")
    lines.append("QK-SVR Directional Trigger (Trigger 2)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("METHOD : Bernoulli sliding-window model on walk-forward backtesting")
    lines.append(f"MODEL  : {MODEL_NAME.upper()}")
    lines.append(f"REGIMES: {', '.join(NON_CRISIS_REGIMES)} (non-crisis folds only)")
    lines.append(f"K      : {K_VALUES} consecutive trading days")
    lines.append(f"H      : {HORIZONS} forecasting horizons (days)")
    lines.append("")
    lines.append("NOTE: FAR reported is PRE-FILTER (Trigger 1 volatility gate not applied).")
    lines.append("Actual deployment FAR expected substantially lower.")
    lines.append("")

    for horizon in HORIZONS:
        subset = results[results["horizon"] == horizon]
        if subset.empty:
            continue

        obs_years = subset.iloc[0]["observation_years"]
        n_folds   = subset.iloc[0]["n_non_crisis_folds"]
        p_down    = subset.iloc[0]["p_down_weighted"]

        lines.append(f"─── H = {horizon} day(s) ───────────────────────────────────────────")
        lines.append(f"  Non-crisis folds : {n_folds} ({obs_years:.1f} years)")
        lines.append(f"  p_down (weighted): {p_down:.4f}")
        lines.append("")
        lines.append(f"  {'K':>4}  {'FAR/year (pre-filter)':>22}  Note")
        lines.append(f"  {'─'*4}  {'─'*22}  {'─'*45}")

        for _, row in subset.iterrows():
            lines.append(
                f"  {int(row['K']):>4}  {row['far_per_year_prefilter']:>22.2f}  {row['note']}"
            )
        lines.append("")

    lines.append("─" * 70)
    lines.append("MANUSCRIPT REFERENCE")
    lines.append("  Section 6.2 (K calibration paragraph)")
    lines.append("  Appendix D, Table D1, Phase 1 Success KPI: FAR < 1.0/year")
    lines.append("")
    lines.append("CITATION")
    lines.append("  Tran Quang Canh (2026). Crisis-Period Trading Resilience and")
    lines.append("  Dynamic MSR Reform: A Quantum Kernel Approach to EU ETS Carbon")
    lines.append("  Price Forecasting. Energy Policy.")
    lines.append("=" * 70)

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FAR sensitivity analysis for the Dual-Trigger Dynamic MSR Protocol."
    )
    parser.add_argument(
        "--input",  default=DEFAULT_INPUT,
        help=f"Path to trading_simulation.csv (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--outdir", default=DEFAULT_OUTDIR,
        help=f"Output directory (default: {DEFAULT_OUTDIR})"
    )
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"[1/4] Loading data from: {args.input}")
    df = load_data(args.input)
    print(f"      Rows: {len(df)} | Models: {df['model'].nunique()} | "
          f"Horizons: {sorted(df['horizon'].unique())}")

    # ── Analyse ───────────────────────────────────────────────────────────────
    print("[2/4] Computing FAR estimates ...")
    results = run_far_analysis(df)

    if results.empty:
        sys.exit("[ERROR] No results produced — check input data.")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "far_sensitivity_K357.csv")
    results.to_csv(csv_path, index=False)
    print(f"[3/4] CSV saved  → {csv_path}")

    # ── Save TXT ──────────────────────────────────────────────────────────────
    summary = print_summary(results)
    txt_path = os.path.join(args.outdir, "far_sensitivity_K357.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[4/4] Report saved → {txt_path}")

    # ── Print to console ──────────────────────────────────────────────────────
    print()
    print(summary)


if __name__ == "__main__":
    main()
