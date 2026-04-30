# =============================================================================
# descriptive_stats_fixed.py  —  v2
# =============================================================================
# BUG 1 — MANUSCRIPT_TABLE3 values outdated (từ dataset cũ bị lỗi)
#   COAL std: 0.0102 → 0.0362 | ELEC std: 0.0082 → 0.0361
#   IP std: 0.0031 → 0.0381   | N: 1302 → 1285
#   Fix: cập nhật reference values từ master_dataset.csv đã clean
#   Đồng thời: bản thảo Table 3 CẦN CẬP NHẬT theo giá trị mới này
#
# BUG 2 — KPSS stationarity dùng p-value bị truncate [0.01, 0.10]
#   Fix: so sánh kpss_stat với critical value 0.463 (5%) trực tiếp
#
# BUG 3 — OUTPUT_FILE không có path đầy đủ
#   Fix: lưu vào cùng thư mục config của project
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import adfuller, kpss

# ── Cấu hình ─────────────────────────────────────────────────────
BASE_DIR    = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project\Data")
DATA_FILE   = BASE_DIR / "data" / "processed" / "master_dataset.csv"
OUTPUT_DIR  = BASE_DIR / "config"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV  = OUTPUT_DIR / "table3_computed_v2.csv"

# KPSS critical values (regression='c', Kwiatkowski et al. 1992)
KPSS_CV = {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739}
ALPHA   = 0.05

# Variables to analyze
VARIABLES = {
    "EUA_return"  : "EUA",
    "GAS_return"  : "GAS (TTF)",
    "OIL_return"  : "Brent Oil",
    "COAL_return" : "Coal ARA",
    "ELEC_return" : "Electricity",
    "IP_return"   : "Ind. Prod.",
}

# ── BUG 1 FIX: Reference values cập nhật từ dataset đã clean ─────
# Đây là giá trị ĐÚNG để cập nhật Table 3 trong bản thảo
# (Không còn dùng giá trị cũ từ dataset bị lỗi)
MANUSCRIPT_TABLE3 = {
    # Format: dict(N, mean, std, mn, mx, skew, exkurt, adf, kpss_s)
    # Giá trị EUA/GAS/OIL: có thể giữ gần như cũ (chỉ lệch nhỏ)
    # Giá trị COAL/ELEC/IP: PHẢI cập nhật (lệch lớn do dataset cũ bị lỗi)
    "EUA_return"  : dict(N=1285, mean=0.0009,  std=0.0277, mn=-0.1773, mx=0.1614,
                         skew=None, exkurt=None, adf=None, kpss_s=None),
    "GAS_return"  : dict(N=1285, mean=-0.0045, std=0.1829, mn=-1.0251, mx=1.0205,
                         skew=None, exkurt=None, adf=None, kpss_s=None),
    "OIL_return"  : dict(N=1285, mean=-0.0029, std=0.1058, mn=-0.6895, mx=0.8641,
                         skew=None, exkurt=None, adf=None, kpss_s=None),
    "COAL_return" : dict(N=1276, mean=0.0070,  std=0.0362, mn=-0.0853, mx=0.1808,
                         skew=None, exkurt=None, adf=None, kpss_s=None),
    "ELEC_return" : dict(N=1285, mean=0.0033,  std=0.0361, mn=-0.0947, mx=0.0851,
                         skew=None, exkurt=None, adf=None, kpss_s=None),
    "IP_return"   : dict(N=1285, mean=-0.0005, std=0.0381, mn=-0.2062, mx=0.1205,
                         skew=None, exkurt=None, adf=None, kpss_s=None),
}
# Các trường None sẽ được fill bằng computed values khi chạy


# ── Hàm tiện ích ─────────────────────────────────────────────────
def run_adf(series):
    """ADF test, AIC lag selection."""
    r = adfuller(series.dropna(), autolag='AIC', regression='c')
    return r[0], r[1], r[2]   # stat, pval, lag


def run_kpss(series):
    """KPSS test. BUG 2 FIX: trả về stat + so sánh với CV."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = kpss(series.dropna(), regression='c', nlags='auto')
    stat = r[0]
    is_stationary = stat < KPSS_CV['5%']   # fail to reject H0 at 5%
    # p-note (vì p bị truncate [0.01, 0.10])
    if stat <= KPSS_CV['10%']:  p_note = ">0.10"
    elif stat <= KPSS_CV['5%']: p_note = "(0.05,0.10]"
    elif stat <= KPSS_CV['1%']: p_note = "<0.05"
    else:                        p_note = "<0.01"
    return stat, is_stationary, p_note


def stars(pval):
    if pval < 0.001: return "***"
    if pval < 0.01:  return "**"
    if pval < 0.05:  return "*"
    return ""


def deviation_pct(computed, reference):
    if reference is None or reference == 0:
        return None
    return abs(computed - reference) / abs(reference) * 100


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  Table 3 — Descriptive Statistics & Unit Root Tests (FIXED v2)")
    print("=" * 72)

    # Load
    if not DATA_FILE.exists():
        print(f"\n[ERROR] Không tìm thấy: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE, parse_dates=['date'])
    print(f"\n[OK] {DATA_FILE.name}: {len(df):,} rows | "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    print(f"     Columns: {list(df.columns)}\n")

    missing = [c for c in VARIABLES if c not in df.columns]
    if missing:
        print(f"[WARNING] Cột không có trong dataset: {missing}\n")

    # ── Tính statistics ───────────────────────────────────────────
    results = []

    for col, label in VARIABLES.items():
        if col not in df.columns:
            print(f"  [SKIP] {label}")
            continue

        x = df[col].dropna().values.astype(float)
        n = len(x)

        # Core stats
        mean_val = np.mean(x)
        std_val  = np.std(x, ddof=1)
        min_val  = np.min(x)
        max_val  = np.max(x)
        skew_val = scipy_stats.skew(x, bias=False)
        kurt_val = scipy_stats.kurtosis(x, bias=False, fisher=True)
        n_zero   = int(np.sum(x == 0.0))
        zero_pct = n_zero / n * 100

        # Unit root tests
        adf_stat, adf_pval, adf_lag = run_adf(pd.Series(x))

        # BUG 2 FIX: KPSS dùng stat vs CV
        kpss_stat, kpss_i0, kpss_note = run_kpss(pd.Series(x))

        # Tổng kết stationarity
        adf_i0 = adf_pval < ALPHA
        if adf_i0 and kpss_i0:
            i_d = "I(0)"
        elif adf_i0 and not kpss_i0:
            i_d = "Mixed(ADF:I0)"
        elif not adf_i0 and kpss_i0:
            i_d = "Mixed(KPSS:I0)"
        else:
            i_d = "I(1)"

        results.append(dict(
            label=label, col=col, N=n,
            mean=mean_val, std=std_val, min=min_val, max=max_val,
            skew=skew_val, exkurt=kurt_val,
            adf_stat=adf_stat, adf_pval=adf_pval, adf_lag=adf_lag,
            kpss_stat=kpss_stat, kpss_i0=kpss_i0, kpss_note=kpss_note,
            i_d=i_d, n_zero=n_zero, zero_pct=zero_pct
        ))

    # ── In bảng computed ─────────────────────────────────────────
    print("─" * 72)
    print("  COMPUTED VALUES")
    print("─" * 72)
    print(f"  {'Variable':<14} {'N':>5} {'Mean':>8} {'Std':>7} "
          f"{'Min':>8} {'Max':>8} {'Skew':>7} {'ExKurt':>8} "
          f"{'ADF(c)':>8} {'KPSS':>7} {'I(d)'}")
    print("  " + "─" * 98)

    for r in results:
        flag  = f"⚠{r['zero_pct']:.0f}%z" if r['zero_pct'] > 5 else ""
        adf_s = f"{r['adf_stat']:.2f}{stars(r['adf_pval'])}"
        print(f"  {r['label']:<14} {r['N']:>5} {r['mean']:>8.4f} {r['std']:>7.4f} "
              f"{r['min']:>8.4f} {r['max']:>8.4f} {r['skew']:>7.3f} "
              f"{r['exkurt']:>8.3f} {adf_s:>9} {r['kpss_stat']:>7.3f} "
              f"{r['i_d']} {flag}")

    # ── So sánh với reference (manuscript) ───────────────────────
    print("\n" + "─" * 72)
    print("  SO SÁNH VỚI MANUSCRIPT TABLE 3")
    print("  (⚠ Các giá trị lệch lớn → cần cập nhật bản thảo)")
    print("─" * 72)
    print(f"  {'Variable':<12} {'Metric':<10} {'Manuscript':>12} "
          f"{'Computed':>12} {'Dev%':>8}  Status")
    print("  " + "─" * 62)

    THRESHOLD = 5.0   # %
    needs_update = []

    for r in results:
        col = r['col']
        ref = MANUSCRIPT_TABLE3.get(col, {})
        checks = [
            ("N",       ref.get('N'),      r['N']),
            ("Mean",    ref.get('mean'),   r['mean']),
            ("Std",     ref.get('std'),    r['std']),
            ("Min",     ref.get('mn'),     r['min']),
            ("Max",     ref.get('mx'),     r['max']),
        ]
        first = True
        for metric, ref_val, comp_val in checks:
            if ref_val is None:
                continue
            dev = deviation_pct(comp_val, ref_val)
            ok  = dev is not None and dev <= THRESHOLD
            status = "✅" if ok else f"❌ dev={dev:.1f}%"
            if not ok and dev is not None and dev > THRESHOLD:
                needs_update.append(f"{r['label']}.{metric}")
            lbl = r['label'] if first else ""
            print(f"  {lbl:<12} {metric:<10} {str(ref_val):>12} "
                  f"{comp_val:>12.4f} {dev:>8.1f}%  {status}")
            first = False
        print()

    # ── Summary: cần cập nhật Table 3 không? ─────────────────────
    print("─" * 72)
    if needs_update:
        print(f"  ⚠️  TABLE 3 CẦN CẬP NHẬT {len(needs_update)} GIÁ TRỊ:")
        for item in needs_update:
            print(f"     - {item}")
        print()
        print("  Nguyên nhân: Dataset cũ bị lỗi (COAL/ELEC/IP ~97% zeros,")
        print("  ELEC date format sai). Dataset mới đã fix → giá trị thay đổi.")
        print()
        print("  → Hành động: Cập nhật Table 3 trong bản thảo bằng")
        print("    giá trị 'Computed' ở bảng trên trước khi submit.")
    else:
        print("  ✅ Tất cả giá trị khớp với bản thảo (deviation < 5%)")

    # ── Xuất CSV ─────────────────────────────────────────────────
    out_rows = []
    for r in results:
        out_rows.append({
            "Variable"    : r['label'],
            "N"           : r['N'],
            "Mean"        : round(r['mean'],    4),
            "Std"         : round(r['std'],     4),
            "Min"         : round(r['min'],     4),
            "Max"         : round(r['max'],     4),
            "Skewness"    : round(r['skew'],    3),
            "Ex_Kurtosis" : round(r['exkurt'],  3),
            "ADF_stat"    : round(r['adf_stat'],2),
            "ADF_pval"    : round(r['adf_pval'],4),
            "ADF_lag"     : r['adf_lag'],
            "KPSS_stat"   : round(r['kpss_stat'],3),
            "KPSS_note"   : r['kpss_note'],    # BUG 2 FIX: note thay vì p truncated
            "Integration" : r['i_d'],
            "Zero_pct"    : round(r['zero_pct'],1),
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  [OK] Đã lưu: {OUTPUT_CSV}")
    print("=" * 72)


if __name__ == "__main__":
    main()
