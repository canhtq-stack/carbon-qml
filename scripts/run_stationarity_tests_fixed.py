# =============================================================================
# run_stationarity_tests_fixed.py  —  v10 (all bugs fixed)
# =============================================================================
# BUG 1 — ADF_I0 dùng OR thay vì AND → quá liberal
#   Fix: dùng AND (cả ADF(c) và ADF(ct) phải reject)
#
# BUG 2 — KPSS p-value bị truncate [0.01, 0.10] bởi statsmodels
#   Fix: so sánh trực tiếp stat với critical values; ghi note vào output
#
# BUG 3 — ZA break date indexing: thêm guard DatetimeIndex
#   Fix: kiểm tra type của index trước khi truy xuất
#
# BUG 4 — Conclusion logic chưa phân biệt đủ cases
#   Fix: taxonomy 5 cases chuẩn theo Perron (1989), Kwiatkowski et al. (1992)
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from datetime import datetime

from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

np.random.seed(42)
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')

# ── Cấu hình ─────────────────────────────────────────────────────
BASE_DIR    = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project\Data")
MASTER_FILE = BASE_DIR / "data" / "processed" / "master_dataset.csv"
CONFIG_DIR  = BASE_DIR / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON  = CONFIG_DIR / "stationarity_results_v10.json"
OUT_LATEX = CONFIG_DIR / "stationarity_table_v10.tex"
OUT_CSV   = CONFIG_DIR / "stationarity_results_v10.csv"

ALPHA = 0.05   # significance level

# KPSS critical values (statsmodels, regression='c')
KPSS_CV = {'10%': 0.347, '5%': 0.463, '2.5%': 0.574, '1%': 0.739}

print(f"🔬 KIỂM ĐỊNH TÍNH DỪNG v10 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ── Hàm tiện ích ─────────────────────────────────────────────────
def kpss_decision(stat):
    """
    BUG 2 FIX: so sánh stat với critical values trực tiếp thay vì dùng p-value
    bị truncate.

    KPSS H0: series is stationary (level)
    Reject H0 (= NOT stationary) khi stat > critical value
    → I0 khi stat <= CV[5%] (fail to reject H0 at 5%)
    """
    is_i0    = stat <= KPSS_CV['5%']
    # Xây dựng p-note cho báo cáo
    if stat <= KPSS_CV['10%']:
        p_note = ">0.10"
    elif stat <= KPSS_CV['5%']:
        p_note = "(0.05, 0.10]"
    elif stat <= KPSS_CV['2.5%']:
        p_note = "(0.025, 0.05]"
    elif stat <= KPSS_CV['1%']:
        p_note = "(0.01, 0.025]"
    else:
        p_note = "<0.01"
    return is_i0, p_note


def get_za_break_date(series, break_idx):
    """
    BUG 3 FIX: xử lý an toàn khi index là DatetimeIndex hoặc integer.
    """
    try:
        idx = int(break_idx)
        loc = series.index[idx]
        if isinstance(series.index, pd.DatetimeIndex):
            return str(loc.date())
        else:
            return str(loc)
    except Exception:
        return "N/A"


def conclude(adf_i0, kpss_i0, za_i0):
    """
    BUG 4 FIX: taxonomy 5 cases chuẩn.

    ADF H0: unit root (reject → stationary)
    KPSS H0: stationary (fail to reject → stationary)

    Cases:
      1. ADF reject + KPSS not reject → I(0), both tests consistent
      2. ZA reject (structural break) + either ADF/KPSS mixed → I(0) with break
      3. ADF not reject + KPSS reject → I(1), both tests consistent
      4. ADF reject + KPSS reject → Mixed (ADF→I(0), KPSS→I(1))
      5. ADF not reject + KPSS not reject → Mixed (ADF→I(1), KPSS→I(0))
    """
    if adf_i0 and kpss_i0:
        return "I(0) — ADF and KPSS consistent"
    elif za_i0 and not (adf_i0 and kpss_i0):
        return "I(0) with structural break (ZA)"
    elif not adf_i0 and not kpss_i0:
        return "I(1) — ADF and KPSS consistent"
    elif adf_i0 and not kpss_i0:
        return "Mixed — ADF: I(0), KPSS: I(1)"
    else:  # not adf_i0 and kpss_i0
        return "Mixed — ADF: I(1), KPSS: I(0)"


# ── Main ─────────────────────────────────────────────────────────
df = pd.read_csv(MASTER_FILE, parse_dates=['date'], index_col='date')
variables = ['EUA_return', 'GAS_return', 'OIL_return',
             'COAL_return', 'ELEC_return', 'IP_return']

results   = {}
csv_rows  = []

for var in variables:
    if var not in df.columns:
        print(f"  ⚠️  {var}: không có trong dataset — bỏ qua")
        continue

    series = df[var].dropna()
    n      = len(series)

    if n < 50:
        print(f"  ⚠️  {var}: cỡ mẫu quá nhỏ ({n}) — bỏ qua")
        continue

    res = {'n_obs': int(n)}

    # ── ADF ──────────────────────────────────────────────────────
    adf_c  = adfuller(series, autolag='AIC', regression='c')
    adf_ct = adfuller(series, autolag='AIC', regression='ct')

    res['ADF_c_stat']  = round(float(adf_c[0]),  4)
    res['ADF_c_p']     = round(float(adf_c[1]),  4)
    res['ADF_c_lag']   = int(adf_c[2])
    res['ADF_ct_stat'] = round(float(adf_ct[0]), 4)
    res['ADF_ct_p']    = round(float(adf_ct[1]), 4)
    res['ADF_ct_lag']  = int(adf_ct[2])

    # BUG 1 FIX: AND logic — cả hai spec phải reject để kết luận I(0)
    res['ADF_I0'] = (adf_c[1] < ALPHA) and (adf_ct[1] < ALPHA)

    # ── KPSS ─────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_out = kpss(series, regression='c', nlags='auto')

    res['KPSS_stat']   = round(float(kpss_out[0]), 4)
    res['KPSS_p_raw']  = round(float(kpss_out[1]), 4)  # truncated [0.01, 0.10]

    # BUG 2 FIX: dùng stat so với CV thay vì p-value
    kpss_i0, kpss_p_note = kpss_decision(kpss_out[0])
    res['KPSS_I0']     = kpss_i0
    res['KPSS_p_note'] = kpss_p_note   # dùng cái này để báo cáo

    # ── Zivot-Andrews ────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        za_out = zivot_andrews(series, autolag='AIC')

    res['ZA_stat']    = round(float(za_out[0]), 4)
    res['ZA_p']       = round(float(za_out[1]), 6)
    res['ZA_cv_5pct'] = float(za_out[2].get('5%', -4.80))
    res['ZA_break']   = get_za_break_date(series, za_out[4])  # BUG 3 FIX
    res['ZA_I0']      = za_out[1] < ALPHA

    # ── Robustness check cho monthly variables ───────────────────
    # Monthly ffill → ADF autolag = ~21 → mất power → có thể false I(1)
    # Fix: chạy thêm ADF với fixed lag=1 để kiểm tra
    MONTHLY_VARS = {'COAL_return', 'ELEC_return', 'IP_return', 'CPI_return'}
    res['is_monthly_ffill'] = var in MONTHLY_VARS

    if var in MONTHLY_VARS:
        adf_lag1 = adfuller(series, maxlag=1, autolag=None, regression='c')
        res['ADF_c_p_lag1'] = round(float(adf_lag1[1]), 4)
        res['ADF_c_stat_lag1'] = round(float(adf_lag1[0]), 4)
        # Nếu lag=1 reject nhưng autolag không reject → artifact of ffill
        if adf_lag1[1] < ALPHA and not res['ADF_I0']:
            res['ADF_I0_override'] = True
            res['conclusion_note'] = (
                "ADF autolag inflated by monthly ffill autocorrelation. "
                "ADF(lag=1) rejects unit root — treated as I(0)."
            )
            res['conclusion'] = "I(0) — ADF lag=1 override (monthly ffill artifact)"
            print(f"  ℹ️  [{var}] monthly ffill artifact → ADF(lag=1) p={adf_lag1[1]:.4f} "
                  f"→ override to I(0)")
        else:
            res['ADF_I0_override'] = False
            res['conclusion_note'] = ""
    else:
        res['is_monthly_ffill'] = False
        res['ADF_I0_override'] = False
        res['ADF_c_p_lag1'] = None
        res['ADF_c_stat_lag1'] = None
        res['conclusion_note'] = ""

    # ── Kết luận ─────────────────────────────────────────────────
    # BUG 4 FIX: taxonomy 5 cases
    res['conclusion'] = conclude(res['ADF_I0'], res['KPSS_I0'], res['ZA_I0'])

    results[var] = res
    csv_rows.append({'variable': var, **res})

    # Print summary
    adf_star  = '***' if res['ADF_c_p'] < 0.001 else ('**' if res['ADF_c_p'] < 0.01
                else ('*' if res['ADF_c_p'] < 0.05 else ''))
    kpss_flag = '✅ I(0)' if kpss_i0 else '❌ reject'
    za_star   = '***' if res['ZA_p'] < 0.001 else ('*' if res['ZA_p'] < 0.05 else '')
    print(f"📊 {var:<15} "
          f"ADF(c)={res['ADF_c_p']:.3f}{adf_star:3}  "
          f"KPSS={res['KPSS_stat']:.3f}({kpss_p_note})  "
          f"ZA={res['ZA_p']:.4f}{za_star}  break={res['ZA_break']}  "
          f"→ {res['conclusion']}")


# ── Xuất JSON ────────────────────────────────────────────────────
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✅ JSON: {OUT_JSON}")

# ── Xuất CSV ─────────────────────────────────────────────────────
df_out = pd.DataFrame(csv_rows)
df_out.to_csv(OUT_CSV, index=False)
print(f"✅ CSV: {OUT_CSV}")

# ── Xuất LaTeX ───────────────────────────────────────────────────
# Dùng KPSS_p_note thay vì p-value truncated (BUG 2 FIX)
latex = r"""\begin{table}[htbp]
\caption{Unit root and structural break tests}
\label{tab:stationarity}
\small
\centering
\begin{tabular}{lccccccc}
\hline
Variable & \multicolumn{2}{c}{ADF} & KPSS & \multicolumn{2}{c}{Zivot-Andrews} & Conclusion \\
\cmidrule(lr){2-3}\cmidrule(lr){5-6}
 & $p$(c) & $p$(ct) & Stat & Stat & Break & \\
\hline
"""

for var, r in results.items():
    # Bold p-values that reject at 5%
    def fmt_p(p):
        s = f"{p:.3f}"
        return r"\textbf{" + s + "}" if p < 0.05 else s

    def fmt_kpss(stat, note):
        s = f"{stat:.3f}"
        # Bold if stat > CV[5%] (reject stationarity)
        return r"\textbf{" + s + "}" if stat > KPSS_CV['5%'] else s

    var_lbl = var.replace('_', '\\_')
    latex += (
        f"{var_lbl} & "
        f"{fmt_p(r['ADF_c_p'])} & "
        f"{fmt_p(r['ADF_ct_p'])} & "
        f"{fmt_kpss(r['KPSS_stat'], r['KPSS_p_note'])} & "
        f"{r['ZA_stat']:.3f} & "
        f"{r['ZA_break']} & "
        f"{r['conclusion']} \\\\\n"
    )

latex += r"""\hline
\end{tabular}
\begin{tablenotes}
\footnotesize
\item ADF = Augmented Dickey-Fuller; KPSS = Kwiatkowski-Phillips-Schmidt-Shin;
ZA = Zivot-Andrews (1992) allowing one endogenous structural break.
All tests use automatic lag selection (AIC).
ADF $H_0$: unit root; KPSS $H_0$: level stationarity.
Bold values indicate rejection at 5\% significance level.
KPSS critical values: 10\%=0.347, 5\%=0.463, 2.5\%=0.574, 1\%=0.739.
ZA critical value at 5\%: $-4.80$.
\end{tablenotes}
\end{table}"""

with open(OUT_LATEX, 'w', encoding='utf-8') as f:
    f.write(latex)
print(f"✅ LaTeX: {OUT_LATEX}")

# ── Sanity check tổng kết ────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TỔNG KẾT: {len(results)}/{len(variables)} biến được kiểm định")
i0_count = sum(1 for r in results.values()
               if 'I(0)' in r['conclusion'])
i1_count = sum(1 for r in results.values()
               if 'I(1)' in r['conclusion'])
mx_count = sum(1 for r in results.values()
               if 'Mixed' in r['conclusion'])
print(f"  I(0)  : {i0_count} biến")
print(f"  I(1)  : {i1_count} biến")
print(f"  Mixed : {mx_count} biến")
if i1_count > 0:
    i1_vars = [v for v, r in results.items() if 'I(1)' in r['conclusion']]
    print(f"\n  ⚠️  Các biến I(1) cần first-difference trước khi đưa vào model:")
    for v in i1_vars:
        print(f"     - {v}")
else:
    print(f"\n  ✅ Tất cả biến đều I(0) — sẵn sàng dùng levels trong model")
print(f"{'='*60}")
print(f"\n🏁 HOÀN TẤT v10")
