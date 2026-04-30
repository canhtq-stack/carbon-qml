# =============================================================================
# merge_and_synchronize_FIXED.py  —  v3 (final)
# =============================================================================
# BUG 1  — Date parsing DD-MM-YY sai tháng         → dayfirst=True
# BUG 2  — Monthly vars 97% zeros                  → log-return trước, ffill sau
# BUG 3  — Thiếu CPI_return, PHASE_dummy           → bổ sung
# BUG 4  — Month-end rơi weekend → reindex miss    → shift_to_bday + dedup
# BUG 5  — NaN đầu series monthly                  → bfill() no limit
# BUG 6  — np.log với giá âm/zero                  → safe_log()
# BUG 7  — Duplicate dates sau shift               → drop_duplicates
# BUG 8  — N=1101: calendar freq='B' ≠ EUA trading → dùng EUA dates làm master
# BUG 9  — CPI là tỷ lệ YoY (%), không phải price → load_rate_variable()
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

print("🔧 MERGE & SYNCHRONIZE — v3 (final)")
print("=" * 60)

# ── Cấu hình ─────────────────────────────────────────────────────
BASE_DIR      = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project\Data")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR       = BASE_DIR / "data" / "raw"
MASTER_FILE   = PROCESSED_DIR / "master_dataset.csv"

DATE_START = "2019-01-01"
DATE_END   = "2023-12-31"

# DAILY_IDX sẽ được xác định từ EUA trading days thực tế (BUG 8 FIX)
# Khởi tạo tạm để các hàm dùng được; sẽ gán lại trong main()
DAILY_IDX = None

SKIP_KW = {'return', 'log', 'dummy', 'policy', 'phase', 'date'}


# ── Hàm tiện ích ─────────────────────────────────────────────────
def find_price_col(df, filepath):
    col = next(
        (c for c in df.columns
         if not any(k in c.lower() for k in SKIP_KW)
         and pd.api.types.is_numeric_dtype(df[c])),
        None
    )
    if col is None:
        raise ValueError(
            f"Không tìm thấy cột giá trị trong {filepath.name}.\n"
            f"Columns: {list(df.columns)}"
        )
    return col


def detect_date_format(series):
    """
    Phát hiện format ngày tự động từ sample value.
    Xử lý cả DD-MM-YY (EUA, GAS...) và MM-DD-YY (ELEC...).

    Logic:
    - Nếu phần đầu (trước dấu -) > 12: chắc chắn là DAY → DD-MM-YY
    - Nếu phần giữa (sau dấu - đầu tiên) = "01": khả năng cao là 1st of month → MM-DD-YY
    - Thử cả hai format, chọn cái cho kết quả hợp lý hơn (unique months > unique days)
    """
    sample = str(series.dropna().iloc[0]).strip()

    # Thử tách để đọc parts
    parts = sample.replace('/', '-').split('-')
    if len(parts) >= 2:
        try:
            p1 = int(parts[0])
            p2 = int(parts[1])
            # Nếu part1 > 12: chắc chắn là ngày (DD-MM-YY)
            if p1 > 12:
                return '%d-%m-%y' if len(parts[2]) == 2 else '%d-%m-%Y'
            # Nếu part2 == 1 (ngày 1 của tháng): có thể là MM-01-YY
            # Thử cả hai và chọn cái cho nhiều unique months hơn
            if p2 == 1 and p1 <= 12:
                try:
                    s_mm = pd.to_datetime(series, format='%m-%d-%y')
                    s_dd = pd.to_datetime(series, format='%d-%m-%y')
                    # MM-DD-YY: sẽ cho unique months = số unique của part1
                    # DD-MM-YY: sẽ cho part1 là ngày (1-12 hoặc 1-31)
                    n_months_mm = s_mm.dt.month.nunique()
                    n_months_dd = s_dd.dt.month.nunique()
                    if n_months_mm > n_months_dd:
                        return '%m-%d-%y'
                    else:
                        return '%d-%m-%y'
                except Exception:
                    pass
        except (ValueError, IndexError):
            pass

    # Fallback: thử từng format phổ biến
    for fmt in ('%d-%m-%y', '%m-%d-%y', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%Y'):
        try:
            pd.to_datetime(sample, format=fmt)
            return fmt
        except Exception:
            continue

    return None  # unknown


def parse_dates(series):
    """Parse dates với format detection tự động."""
    fmt = detect_date_format(series)
    if fmt:
        try:
            result = pd.to_datetime(series, format=fmt)
            # Sanity check: năm phải trong khoảng 2015-2030
            years = result.dt.year
            if years.between(2015, 2030).mean() > 0.95:
                return result
        except Exception:
            pass
    # Fallback
    return pd.to_datetime(series, dayfirst=True)


def safe_log(series, name=""):
    """BUG 6 FIX: kiểm tra giá âm/zero trước khi log."""
    n_bad = (series <= 0).sum()
    if n_bad > 0:
        print(f"  ⚠️  [{name}] {n_bad} giá trị <= 0 → dùng |price|")
        series = series.abs().replace(0, np.nan)
    return np.log(series)


def shift_to_bday(date):
    """BUG 4 FIX: nếu là weekend, lùi về business day trước."""
    ts = pd.Timestamp(date)
    return ts if ts.dayofweek < 5 else ts - pd.tseries.offsets.BDay(1)


# ── Hàm load daily price variable ────────────────────────────────
def load_daily_return(filepath, col_name):
    """Tính log-return từ daily price. Áp dụng BUG 1, 6."""
    df = pd.read_csv(filepath)
    df['date']    = parse_dates(df['date'])
    price_col     = find_price_col(df, filepath)
    print(f"     → cột giá: '{price_col}'")

    df = df.sort_values('date').reset_index(drop=True)
    df['log_price'] = safe_log(df[price_col], col_name)
    df['return']    = df['log_price'].diff()
    df = df.dropna(subset=['return'])

    n_inf = np.isinf(df['return']).sum()
    if n_inf > 0:
        print(f"  ⚠️  [{col_name}] {n_inf} inf → thay NaN")
        df['return'] = df['return'].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['return'])

    # Lọc trong phạm vi nghiên cứu
    df = df[(df['date'] >= DATE_START) & (df['date'] <= DATE_END)]
    df = df.reset_index(drop=True)

    print(f"  ✅ {filepath.name}: {len(df)} rows | "
          f"std={df['return'].std():.4f} | "
          f"[{df['return'].min():.4f}, {df['return'].max():.4f}]")

    return df[['date', 'return']].rename(columns={'return': col_name})


# ── Hàm load monthly price variable (COAL, ELEC, OIL nếu monthly) ─
def load_monthly_return(filepath, col_name):
    """
    Tính log-return từ monthly price GỐC rồi ffill sang daily.
    Áp dụng BUG 1, 2, 4, 5, 6, 7.
    """
    global DAILY_IDX
    df = pd.read_csv(filepath)
    df['date']    = parse_dates(df['date'])
    price_col     = find_price_col(df, filepath)
    print(f"     → cột giá: '{price_col}'")

    df = df.sort_values('date').reset_index(drop=True)

    # BUG 4 FIX: shift weekend → business day
    df['date'] = df['date'].apply(shift_to_bday)

    # BUG 7 FIX: dedup sau shift
    n_dup = df.duplicated(subset='date').sum()
    if n_dup > 0:
        print(f"  ⚠️  [{col_name}] {n_dup} duplicate sau shift → giữ cuối")
        df = df.drop_duplicates(subset='date', keep='last').reset_index(drop=True)

    # BUG 2 + BUG 6 FIX: log-return từ monthly price gốc
    df['log_price']      = safe_log(df[price_col], col_name)
    df['monthly_return'] = df['log_price'].diff()
    df = df.dropna(subset=['monthly_return']).reset_index(drop=True)

    # Expand → BUG 5 FIX: ffill + bfill đầu series
    df_indexed = df.set_index('date')[['monthly_return']]
    df_daily   = df_indexed.reindex(DAILY_IDX)
    df_daily['monthly_return'] = (df_daily['monthly_return']
                                  .ffill().bfill())
    df_daily = df_daily.reset_index()
    df_daily.columns = ['date', col_name]

    zero_pct = (df_daily[col_name] == 0).mean() * 100
    nan_pct  = df_daily[col_name].isna().mean() * 100
    print(f"  ✅ {filepath.name}: {len(df_daily)} rows | "
          f"std={df_daily[col_name].std():.4f} | "
          f"zeros={zero_pct:.1f}% | NaN={nan_pct:.1f}%")

    return df_daily


# ── Hàm load monthly rate variable (CPI, IP nếu là YoY %) ────────
def load_rate_variable(filepath, col_name, scale=1.0):
    """
    BUG 9 FIX: dùng cho CPI, IP khi giá trị là tỷ lệ YoY (%)
    KHÔNG tính log-return. Dùng giá trị trực tiếp (chia scale để ra decimal).

    Ví dụ CPI: 1.4 (%) → chia 100 → 0.014
    Ví dụ IP:  nếu đã là decimal thì scale=1.0
    """
    global DAILY_IDX
    df = pd.read_csv(filepath)
    df['date']  = parse_dates(df['date'])
    val_col     = find_price_col(df, filepath)
    print(f"     → cột giá trị: '{val_col}' (scale=1/{scale})")

    df = df.sort_values('date').reset_index(drop=True)

    # BUG 4 + 7 FIX
    df['date'] = df['date'].apply(shift_to_bday)
    n_dup = df.duplicated(subset='date').sum()
    if n_dup > 0:
        df = df.drop_duplicates(subset='date', keep='last').reset_index(drop=True)

    # Dùng giá trị trực tiếp, chia scale
    df['rate'] = df[val_col] / scale

    # Expand → ffill + bfill
    df_indexed = df.set_index('date')[['rate']]
    df_daily   = df_indexed.reindex(DAILY_IDX)
    df_daily['rate'] = df_daily['rate'].ffill().bfill()
    df_daily = df_daily.reset_index()
    df_daily.columns = ['date', col_name]

    nan_pct = df_daily[col_name].isna().mean() * 100
    print(f"  ✅ {filepath.name}: {len(df_daily)} rows | "
          f"mean={df_daily[col_name].mean():.4f} | "
          f"std={df_daily[col_name].std():.4f} | "
          f"[{df_daily[col_name].min():.4f}, {df_daily[col_name].max():.4f}] | "
          f"NaN={nan_pct:.1f}%")

    return df_daily


# ── Hàm load monthly variable có sẵn cột return ──────────────────
def load_precomputed_return(filepath, col_name):
    """
    Dùng cho IP_processed.csv: file đã có sẵn cột IP_return (MoM log-return).
    Đọc cột return trực tiếp, forward-fill sang daily trading days.
    KHÔNG tính lại từ price để tránh sai.
    """
    global DAILY_IDX
    df = pd.read_csv(filepath)
    df['date'] = parse_dates(df['date'])

    # Tìm cột return đã có sẵn
    return_col = next(
        (c for c in df.columns if 'return' in c.lower()),
        None
    )
    if return_col is None:
        print(f"  ⚠️  [{col_name}] Không tìm thấy cột return → fallback sang load_monthly_return")
        return load_monthly_return(filepath, col_name)

    print(f"     → dùng cột có sẵn: '{return_col}'")
    df = df.sort_values('date').dropna(subset=[return_col]).reset_index(drop=True)

    # BUG 4 + 7 FIX
    df['date'] = df['date'].apply(shift_to_bday)
    n_dup = df.duplicated(subset='date').sum()
    if n_dup > 0:
        df = df.drop_duplicates(subset='date', keep='last').reset_index(drop=True)

    # Expand → ffill + bfill
    df_indexed = df.set_index('date')[[return_col]]
    df_daily   = df_indexed.reindex(DAILY_IDX)
    df_daily[return_col] = df_daily[return_col].ffill().bfill()
    df_daily = df_daily.reset_index()
    df_daily.columns = ['date', col_name]

    zero_pct = (df_daily[col_name] == 0).mean() * 100
    nan_pct  = df_daily[col_name].isna().mean() * 100
    print(f"  ✅ {filepath.name}: {len(df_daily)} rows | "
          f"mean={df_daily[col_name].mean():.4f} | "
          f"std={df_daily[col_name].std():.4f} | "
          f"[{df_daily[col_name].min():.4f}, {df_daily[col_name].max():.4f}] | "
          f"zeros={zero_pct:.1f}% | NaN={nan_pct:.1f}%")

    return df_daily


# ── Main ──────────────────────────────────────────────────────────
def main():
    global DAILY_IDX

    # ── BUG 8 FIX: Dùng EUA trading days làm master calendar ─────
    print("\n📅 BƯỚC 0: Xây dựng master calendar từ EUA trading days...")
    # Thử file trong processed/ trước, nếu không đủ rows thì báo lỗi rõ ràng
    eua_file = PROCESSED_DIR / 'eua_daily_processed.csv'
    if not eua_file.exists():
        raise FileNotFoundError(f"Không tìm thấy: {eua_file}")

    df_eua_cal = pd.read_csv(eua_file)
    df_eua_cal['date'] = parse_dates(df_eua_cal['date'])
    df_eua_cal = df_eua_cal[
        (df_eua_cal['date'] >= DATE_START) &
        (df_eua_cal['date'] <= DATE_END)
    ].sort_values('date').drop_duplicates('date').reset_index(drop=True)

    n_eua = len(df_eua_cal)
    if n_eua < 1200:
        raise ValueError(
            f"eua_daily_processed.csv chỉ có {n_eua} rows sau filter 2019-2023.\n"
            f"File này thiếu dữ liệu — hãy copy file gốc đúng (1543 rows) vào:\n"
            f"{eua_file}\n"
            f"Lệnh copy:\n"
            f"copy \"<đường dẫn file gốc>\\eua_daily_processed.csv\" \"{eua_file}\""
        )

    DAILY_IDX = df_eua_cal['date']
    print(f"  ✅ EUA calendar: {len(DAILY_IDX)} trading days | "
          f"{DAILY_IDX.iloc[0].date()} → {DAILY_IDX.iloc[-1].date()}")

    df_master = pd.DataFrame({'date': DAILY_IDX}).reset_index(drop=True)

    # ── Bước 1: Daily variables ───────────────────────────────────
    print("\n📥 BƯỚC 1: Daily variables (EUA, GAS, OIL)...")
    for col_name, fname in {
        'EUA_return': 'eua_daily_processed.csv',
        'GAS_return': 'GAS_processed.csv',
        'OIL_return': 'OIL_processed.csv',
    }.items():
        fpath = PROCESSED_DIR / fname
        if not fpath.exists():
            print(f"  ⚠️  Không tìm thấy: {fname}")
            continue
        df_master = df_master.merge(
            load_daily_return(fpath, col_name), on='date', how='left')

        # Forward-fill: GAS/OIL có thể không giao dịch một số ngày EUA có
        # (khác lịch nghỉ giữa ICE TTF/Brent và EU ETS) → dùng giá ngày trước
        n_nan = df_master[col_name].isna().sum()
        if n_nan > 0:
            df_master[col_name] = df_master[col_name].ffill()
            remaining = df_master[col_name].isna().sum()
            print(f"  ℹ️  [{col_name}] ffill {n_nan} ngày thiếu"
                  f"{' | vẫn còn NaN: '+str(remaining) if remaining else ' ✅'}")

    # ── Bước 2: Monthly price variables (COAL, ELEC) ─────────────
    print("\n📥 BƯỚC 2: Monthly price variables (COAL, ELEC)...")
    for col_name, fname in {
        'COAL_return': 'COAL_processed.csv',
        'ELEC_return': 'ELEC_processed.csv',
    }.items():
        fpath = PROCESSED_DIR / fname
        if not fpath.exists():
            print(f"  ⚠️  Không tìm thấy: {fname}")
            continue
        df_master = df_master.merge(
            load_monthly_return(fpath, col_name), on='date', how='left')

    # ── Bước 3: Monthly rate variables (IP, CPI) — BUG 9 FIX ─────
    print("\n📥 BƯỚC 3: Monthly rate variables (IP, CPI)...")

    # IP: kiểm tra format — nếu đã là decimal (0.02) thì scale=1
    #                      nếu là % (2.0) thì scale=100
    ip_file = PROCESSED_DIR / 'IP_processed.csv'
    if ip_file.exists():
        # IP file đã có sẵn cột IP_return (MoM log-return) → dùng trực tiếp
        df_master = df_master.merge(
            load_precomputed_return(ip_file, 'IP_return'),
            on='date', how='left')
    else:
        print("  ⚠️  Không tìm thấy: IP_processed.csv")

    # CPI: giá trị là YoY % (1.4, 1.5...) → chia 100 → decimal
    cpi_file = PROCESSED_DIR / 'CPI_processed.csv'
    if cpi_file.exists():
        df_master = df_master.merge(
            load_rate_variable(cpi_file, 'CPI_return', scale=100.0),
            on='date', how='left')
    else:
        print("  ⚠️  Không tìm thấy: CPI_processed.csv")

    # ── Bước 4: POLICY_dummy ─────────────────────────────────────
    print("\n📅 BƯỚC 4: POLICY_dummy...")
    df_master['POLICY_dummy'] = 0
    policy_file = RAW_DIR / 'policy_events.csv'
    if policy_file.exists():
        df_policy = pd.read_csv(policy_file)
        date_col  = ('event_date' if 'event_date' in df_policy.columns
                     else df_policy.columns[0])
        df_policy['target_date'] = pd.to_datetime(
            df_policy[date_col], errors='coerce')
        for _, row in df_policy.dropna(subset=['target_date']).iterrows():
            mask = ((df_master['date'] >= row['target_date'] - timedelta(days=3)) &
                    (df_master['date'] <= row['target_date'] + timedelta(days=3)))
            df_master.loc[mask, 'POLICY_dummy'] = 1
        print(f"  ✅ {df_master['POLICY_dummy'].sum()} ngày = 1")
    else:
        print("  ⚠️  policy_events.csv không có → POLICY_dummy = 0")

    # ── Bước 5: PHASE_dummy ──────────────────────────────────────
    print("\n📅 BƯỚC 5: PHASE_dummy (Phase 3=0 / Phase 4=1)...")
    df_master['PHASE_dummy'] = (df_master['date'].dt.year >= 2021).astype(int)
    print(f"  ✅ Phase 4: {df_master['PHASE_dummy'].sum()} | "
          f"Phase 3: {(df_master['PHASE_dummy']==0).sum()}")

    # ── Bước 6: Dropna core + lưu ────────────────────────────────
    print("\n🔍 BƯỚC 6: Kiểm tra & lưu...")
    core   = [c for c in ['EUA_return','GAS_return','OIL_return']
              if c in df_master.columns]
    before = len(df_master)
    df_master = df_master.dropna(subset=core).reset_index(drop=True)
    dropped = before - len(df_master)
    if dropped > 0:
        print(f"  ⚠️  Dropped {dropped} rows thiếu core variables")
    print(f"  Rows cuối: {len(df_master)}")

    df_master['date'] = df_master['date'].dt.strftime('%Y-%m-%d')
    df_master.to_csv(MASTER_FILE, index=False)

    # ── Báo cáo ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ MASTER DATASET ĐÃ LƯU")
    print(f"   File : {MASTER_FILE}")
    print(f"   Shape: {len(df_master)} rows × {len(df_master.columns)} cols")
    print(f"   Cols : {list(df_master.columns)}")
    print(f"   Range: {df_master['date'].iloc[0]} → {df_master['date'].iloc[-1]}")

    print("\n📊 THỐNG KÊ CHẤT LƯỢNG DỮ LIỆU:")
    print(f"  {'Column':<16} {'N':>5} {'Mean':>8} {'Std':>8} "
          f"{'Min':>8} {'Max':>8} {'Zeros%':>7} {'NaN%':>6}")
    print("  " + "-" * 68)
    for col in df_master.columns:
        if col == 'date':
            continue
        v   = pd.to_numeric(df_master[col], errors='coerce')
        nn  = v.notna().sum()
        print(f"  {col:<16} {nn:>5} {v.mean():>8.4f} {v.std():>8.4f} "
              f"{v.min():>8.4f} {v.max():>8.4f} "
              f"{(v==0).mean()*100:>6.1f}% {v.isna().mean()*100:>5.1f}%")

    # ── Sanity checks ─────────────────────────────────────────────
    print("\n🔍 SANITY CHECKS:")
    df_num = pd.read_csv(MASTER_FILE)
    checks = [
        ("N rows",       len(df_num),                              1200, 1400, "rows"),
        ("EUA std",      df_num['EUA_return'].std(),               0.020, 0.040, ""),
        ("EUA min",      df_num['EUA_return'].min(),              -0.20, -0.10, ""),
        ("EUA max",      df_num['EUA_return'].max(),               0.10,  0.20, ""),
        ("GAS std",      df_num['GAS_return'].std(),               0.10,  0.25, ""),
        ("CPI mean",     df_num['CPI_return'].mean(),              0.000, 0.050,
         "→ ≈0.02 nếu avg CPI ~2%"),
        ("CPI std",      df_num['CPI_return'].std(),               0.000, 0.040,
         "→ 0.033 ok: CPI 2022 đạt 10.6%"),
    ]
    all_ok = True
    for name, val, lo, hi, note in checks:
        ok = lo <= val <= hi
        status = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        print(f"  {status} {name:<16} = {val:>8.4f}  expected [{lo:.3f}, {hi:.3f}] {note}")

    if all_ok:
        print("\n  🎉 TẤT CẢ SANITY CHECKS ĐÃ PASS — dataset sẵn sàng cho model")
    else:
        print("\n  ⚠️  Một số checks FAIL — xem lại trước khi chạy model")


if __name__ == '__main__':
    main()
