# ─────────────────────────────────────────────────────────────────────────────
# Script: compute_vif.py
# Purpose: Tính Variance Inflation Factor (VIF) cho các features đầu vào
#          để kiểm tra multicollinearity — báo cáo trong Section 5.5
#
# Input:  File features đã processed (cùng file dùng trong pipeline)
# Output: vif_results.csv
#
# Run: C:\Users\tranq\AppData\Local\Programs\Python\Python311\python.exe compute_vif.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project\Data\data\processed\master_dataset.csv"

# Các features đầu vào của mô hình (không bao gồm target EUA_return)
FEATURE_COLS = [
    "GAS_return",
    "OIL_return",
    "COAL_return",
    "ELEC_return",
    "IP_return",
    "CPI_return",
    "POLICY_dummy",
    "PHASE_dummy",
]

OUTPUT_PATH = r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project\vif_results.csv"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print(f"Loading: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File không tìm thấy.")
        return

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).iloc[:1285]

    # Kiểm tra cột
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: Các cột sau không tìm thấy: {missing}")
        print(f"Các cột hiện có: {list(df.columns)}")
        return

    X = df[FEATURE_COLS].dropna()
    print(f"N = {len(X)} observations, {len(FEATURE_COLS)} features")

    # Tính VIF cho từng feature
    vif_data = []
    for i, col in enumerate(FEATURE_COLS):
        vif_val = variance_inflation_factor(X.values, i)
        vif_data.append({
            "feature": col,
            "VIF": round(vif_val, 4),
            "multicollinearity": (
                "None" if vif_val < 5 else
                "Moderate" if vif_val < 10 else
                "High"
            )
        })

    out = pd.DataFrame(vif_data)
    print("\n=== VIF Results ===")
    print(out.to_string(index=False))

    # Kiểm tra max VIF
    max_vif = out["VIF"].max()
    print(f"\nMax VIF = {max_vif:.4f}")
    if max_vif < 5:
        print("✅ Tất cả VIF < 5: không có multicollinearity đáng lo ngại")
    elif max_vif < 10:
        print("⚠️  Có VIF 5–10: multicollinearity trung bình")
    else:
        print("❌ Có VIF > 10: multicollinearity cao, cần xem xét")

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nĐã lưu: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
