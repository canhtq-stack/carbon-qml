# =============================================================================
# 01_fold_generator_fixed.py  —  v2
# =============================================================================
# BUG 1 — COAL NaN drop làm mất 1 fold (36 thay vì 37/horizon)
#   Nguyên nhân: load_and_validate_data() drop ANY row có NaN ở feature
#   COAL có 9 NaN đầu series → N: 1285→1276 → folds/horizon: 37→36
#   Fix: ffill NaN ở feature columns TRƯỚC khi drop (giữ N=1285, 37 folds)
#
# BUG 2 — pyarrow không được cài sẵn → to_parquet() crash
#   Fix: try parquet trước, fallback sang CSV nếu pyarrow không có
#
# BUG 3 — fold_id interleaves horizons (1=h1, 2=h5, 3=h22, 4=h1...)
#   → downstream scripts dễ nhầm khi filter by fold_id
#   Fix: thêm cột window_id (unique per window position) vào output
#
# PERF 1 — np.arange(0, start_idx) tạo array mới 111 lần
#   Fix: yield slice indices thay vì full arrays; materialization ở module 02/03
#
# PERF 2 — Parquet engine tường minh → tránh auto-detect overhead
# =============================================================================

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterator

import numpy as np
import pandas as pd
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ── Hằng số ──────────────────────────────────────────────────────
MANUSCRIPT_FOLDS_PER_HORIZON = 37   # target khớp manuscript


# ── ExpandingWindowCV ─────────────────────────────────────────────
class ExpandingWindowCV:
    """
    Expanding Window Cross-Validator — Research Design Section 5.3.

    Parameters
    ----------
    n_initial : int  Initial training window (504 business days ≈ 2 years)
    step      : int  Walk-forward step (21 business days ≈ 1 month)
    horizons  : list Forecast horizons {1, 5, 22} days (direct multi-step)

    Non-overlapping guarantee
    -------------------------
    For each horizon h, the test set = exactly ONE observation at position
    (train_end + h). Avoids overlap when h > step (h=22 > step=21).
    """

    def __init__(
        self,
        n_initial: int = 504,
        step: int = 21,
        horizons: Optional[List[int]] = None,
    ):
        self.n_initial = n_initial
        self.step      = step
        self.horizons  = sorted(horizons) if horizons else [1, 5, 22]

    # PERF 1: yield (window_id, fold_id, h, train_end, test_idx) as lightweight tuples
    # Callers materialize np.arange(0, train_end) only when needed.
    def split(self, n_samples: int) -> Iterator[Tuple[int, int, int, int, int]]:
        """
        Yields
        ------
        (window_id, fold_id, horizon, train_end_idx, test_idx)

        train indices  = np.arange(0, train_end_idx + 1)  — materialize in caller
        test  index    = test_idx  (scalar, single point)
        """
        max_h = self.horizons[-1]
        if n_samples < self.n_initial + max_h:
            raise ValueError(
                f"Dataset too short. Need >= {self.n_initial + max_h} "
                f"samples, got {n_samples}."
            )

        fold_id   = 0
        window_id = 0

        for start_idx in range(self.n_initial, n_samples - max_h, self.step):
            window_id += 1
            train_end = start_idx - 1   # last training index (inclusive)

            for h in self.horizons:
                # Direct multi-step: predict exactly h steps ahead
                test_idx = min(start_idx + h - 1, n_samples - 1)
                fold_id += 1
                yield window_id, fold_id, h, train_end, test_idx

    def n_windows(self, n_samples: int) -> int:
        """Number of walk-forward windows for given dataset size."""
        max_h = self.horizons[-1]
        return len(range(self.n_initial, n_samples - max_h, self.step))


# ── Data loading ──────────────────────────────────────────────────
def load_and_validate_data(
    data_path: Path,
    target_col: str,
    feature_cols: List[str],
    ffill_limit: int = 5,       # BUG 1 FIX: ffill NaN trước khi drop
) -> pd.DataFrame:
    """
    Load master dataset, validate và enforce leakage-free indexing.

    BUG 1 FIX: forward-fill NaN ở feature columns (COAL đầu series)
    trước khi drop. Giữ N=1285 → 37 folds/horizon (khớp manuscript).
    Chỉ drop row nếu vẫn còn NaN sau ffill (gap thực sự, không phải
    missing đầu series).
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    logging.info(f"📥 Loading: {data_path.name}")
    df = pd.read_csv(data_path, parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)

    required = [target_col] + feature_cols
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # BUG 1 FIX: ffill + bfill để xử lý NaN ở cả đầu lẫn giữa series
    # ffill: lấp NaN giữa series (ffill từ giá trị trước)
    # bfill: lấp NaN đầu series (COAL Jan-2019 chưa có data tháng trước)
    n_nan_before = df[required].isna().sum().sum()
    if n_nan_before > 0:
        df[required] = df[required].ffill(limit=ffill_limit).bfill()
        n_nan_after = df[required].isna().sum().sum()
        logging.info(
            f"ℹ️  ffill+bfill NaN: {n_nan_before} → {n_nan_after} "
            f"(limit={ffill_limit}). Per column:"
        )
        for col in required:
            n = df[col].isna().sum()
            if n > 0:
                logging.info(f"   {col}: {n} NaN còn lại → sẽ drop")

    # Drop rows vẫn còn NaN sau ffill
    valid_mask = df[required].notna().all(axis=1)
    n_dropped  = (~valid_mask).sum()
    df = df[valid_mask].copy()
    if n_dropped > 0:
        logging.warning(f"⚠️  Dropped {n_dropped} rows với NaN không thể ffill.")

    # Business day gap check (BUG FIX from original: báo cả start lẫn end gap)
    date_diffs = df.index.to_series().diff().dropna()
    max_gap    = date_diffs.max()
    if max_gap > timedelta(days=21):
        gap_end   = date_diffs.idxmax()
        gap_start = df.index[df.index.get_loc(gap_end) - 1]
        logging.warning(
            f"⚠️  Date gap > 21 days: {max_gap.days} days "
            f"({gap_start.date()} → {gap_end.date()}). "
            "Verify calendar alignment."
        )
    else:
        logging.info(f"✅ Date frequency OK. Max gap: {max_gap.days} days.")

    logging.info(
        f"📊 Shape: {df.shape} | "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )
    return df


# ── Fold generation ───────────────────────────────────────────────
def generate_fold_splits(
    df: pd.DataFrame,
    output_dir: Path,
    cv: Optional[ExpandingWindowCV] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate expanding window splits và save structured metadata.

    BUG 3 FIX: thêm cột window_id để downstream scripts dễ group folds
    theo window position (không bị nhầm với fold_id interleaved).
    """
    if cv is None:
        cv = ExpandingWindowCV()

    n_windows_expected = cv.n_windows(len(df))
    logging.info(
        f"🔄 Generating splits: init={cv.n_initial}, step={cv.step}, "
        f"horizons={cv.horizons}, N={len(df)}, "
        f"expected windows={n_windows_expected}"
    )

    # Warn nếu số folds khác manuscript
    if n_windows_expected != MANUSCRIPT_FOLDS_PER_HORIZON:
        logging.warning(
            f"⚠️  Folds/horizon={n_windows_expected} ≠ manuscript target "
            f"{MANUSCRIPT_FOLDS_PER_HORIZON}. Kiểm tra N dataset!"
        )
    else:
        logging.info(f"✅ Folds/horizon={n_windows_expected} khớp manuscript.")

    fold_records: List[Dict[str, Any]] = []

    for window_id, fold_id, horizon, train_end, test_idx in cv.split(len(df)):
        fold_records.append({
            "window_id":   window_id,      # BUG 3 FIX: unique per window
            "fold_id":     fold_id,
            "horizon":     horizon,
            "train_start": df.index[0].strftime("%Y-%m-%d"),
            "train_end":   df.index[train_end].strftime("%Y-%m-%d"),
            "test_start":  df.index[test_idx].strftime("%Y-%m-%d"),
            "test_end":    df.index[test_idx].strftime("%Y-%m-%d"),
            "n_train":     train_end + 1,
            "n_test":      1,
        })

    df_folds = pd.DataFrame(fold_records)

    # Metadata — BUG FIX original: dùng cv attributes thay vì hardcode
    metadata = {
        "generator": "01_fold_generator_fixed.py",
        "research_design_version": "5.1",
        "config": {
            "n_initial":     cv.n_initial,
            "step":          cv.step,
            "horizons":      cv.horizons,
            "forecast_mode": "direct_multi_step",
            "ffill_limit":   5,
        },
        "data_summary": {
            "total_rows": len(df),
            "date_range": [
                df.index[0].strftime("%Y-%m-%d"),
                df.index[-1].strftime("%Y-%m-%d"),
            ],
        },
        "fold_stats": {
            "total_folds":       len(df_folds),
            "windows":           n_windows_expected,
            "folds_per_horizon": {
                str(h): int((df_folds["horizon"] == h).sum())
                for h in cv.horizons
            },
            "manuscript_target": MANUSCRIPT_FOLDS_PER_HORIZON,
            "matches_manuscript": n_windows_expected == MANUSCRIPT_FOLDS_PER_HORIZON,
        },
        "usage_note": (
            "Each fold has exactly ONE test observation (direct multi-step). "
            "Modules 02/03: slice X/y using train_end (inclusive) for train, "
            "test_start == test_end for prediction. "
            "Fit ALL scalers ONLY on train rows. "
            "window_id groups all horizons of the same walk-forward step."
        ),
    }

    # Export — BUG 2 FIX: fallback CSV nếu pyarrow không có
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_parquet = False
    fold_path_pq  = output_dir / "fold_splits.parquet"
    fold_path_csv = output_dir / "fold_splits.csv"
    meta_path     = output_dir / "fold_metadata.json"

    try:
        df_folds.to_parquet(fold_path_pq, index=False, engine="pyarrow")
        logging.info(f"✅ Fold splits (parquet): {fold_path_pq}")
        saved_parquet = True
    except Exception as e:
        logging.warning(f"⚠️  Parquet save failed ({e}). Saving CSV fallback...")
        df_folds.to_csv(fold_path_csv, index=False)
        logging.info(f"✅ Fold splits (CSV):     {fold_path_csv}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logging.info(f"✅ Metadata:              {meta_path}")

    # Summary per horizon
    logging.info("📊 Fold summary:")
    for h in cv.horizons:
        sub = df_folds[df_folds["horizon"] == h]
        logging.info(
            f"   H={h:>2}: {len(sub):>3} folds | "
            f"train_n [{sub['n_train'].min()}–{sub['n_train'].max()}] | "
            f"test [{sub['test_start'].iloc[0]} … {sub['test_start'].iloc[-1]}]"
        )

    return df_folds, metadata


# ── Validation helper ─────────────────────────────────────────────
def validate_no_leakage(df_folds: pd.DataFrame) -> bool:
    """
    Kiểm tra test observations không bị leak vào train của cùng fold.
    Mỗi fold: test_start phải > train_end.
    """
    ok = True
    for _, row in df_folds.iterrows():
        if row["test_start"] <= row["train_end"]:
            logging.error(
                f"❌ LEAKAGE! fold_id={row['fold_id']} h={row['horizon']}: "
                f"test_start={row['test_start']} <= train_end={row['train_end']}"
            )
            ok = False
    if ok:
        logging.info("✅ No data leakage detected across all folds.")
    return ok


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_PATH = Path(
        r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU"
        r"\carbon_qml_project\Data\data\processed\master_dataset.csv"
    )
    OUTPUT_DIR = Path(
        r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU"
        r"\carbon_qml_project\Data\results"
    )
    TARGET_COL   = "EUA_return"
    FEATURE_COLS = [
        "GAS_return", "OIL_return", "COAL_return",
        "ELEC_return", "IP_return", "CPI_return",
        "POLICY_dummy", "PHASE_dummy",
    ]

    # Load & validate
    df = load_and_validate_data(DATA_PATH, TARGET_COL, FEATURE_COLS)

    # Generate folds
    df_folds, meta = generate_fold_splits(df, OUTPUT_DIR)

    # Leakage check
    validate_no_leakage(df_folds)

    # Final report
    print("\n" + "=" * 60)
    print("🔍 FOLD GENERATION COMPLETE")
    print(f"📁 Output dir : {OUTPUT_DIR}")
    print(f"📊 Total folds: {len(df_folds)} "
          f"({meta['fold_stats']['folds_per_horizon']} per horizon)")
    print(f"✅ Matches manuscript: {meta['fold_stats']['matches_manuscript']}")
    if not meta["fold_stats"]["matches_manuscript"]:
        print(f"   ⚠️  Got {meta['fold_stats']['windows']} windows/horizon, "
              f"expected {MANUSCRIPT_FOLDS_PER_HORIZON}")
        print("   → Kiểm tra lại N trong master_dataset.csv")
    print("=" * 60)
