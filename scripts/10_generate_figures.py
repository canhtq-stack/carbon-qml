# =============================================================================
# 09_generate_figures_fixed.py
# =============================================================================
# Generate tất cả 5 figures cho manuscript theo chuẩn TFSC / APA 7th Edition
#
# Figure 1: EUA Carbon Price & Return Series (time series + regime shading)
# Figure 2: Walk-Forward Cross-Validation Scheme (schematic diagram)
# Figure 3: Quantum Kernel Circuit Architecture (circuit diagram)
# Figure 4: Crisis Sub-Period RMSE Comparison (grouped bar chart)
# Figure 5: Feature Importance — QKFM vs TreeSHAP (heatmap)
#
# TFSC Requirements:
#   - 300 DPI, PNG format (+ EPS cho submission)
#   - Font: Times New Roman / serif
#   - APA style: no gridlines, horizontal lines only in tables
#   - Color: muted palette (accessible)
#   - Width: double-column = 17.4cm = 6.85 inches
#
# Chạy: Nhấn ▶️ Run trong VS Code (~2 phút)
# =============================================================================

import json
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — không cần display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrow, FancyBboxPatch
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(asctime)s | %(message)s",
                    datefmt="%H:%M:%S")

# =============================================================================
# CẤU HÌNH — chỉnh BASE_DIR rồi nhấn Run
# =============================================================================
BASE_DIR = Path(r"D:\CANH\Nghiên cứu khoa học\VIẾT BÁO - CHẠY SỐ LIỆU\carbon_qml_project")
# =============================================================================

RESULTS_DIR = BASE_DIR / "Data" / "results"
DATA_DIR    = BASE_DIR / "Data" / "data" / "processed"
FIGURES_DIR = BASE_DIR / "Data" / "manuscript_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI        = 300
FIG_W      = 6.85   # double-column width (inches)
FIG_W_HALF = 3.35   # single-column width

# ── APA / TFSC color palette ──────────────────────────────────────
C_PRIMARY  = "#1F4E79"    # dark blue — QK-SVR, primary
C_ACCENT   = "#E74C3C"    # red — highlight / crisis
C_NEUTRAL  = "#7F8C8D"    # grey — baselines
C_GREEN    = "#27AE60"    # green — best model
C_ORANGE   = "#F39C12"    # orange — QK-SVR bars
C_LIGHT    = "#D6E4F0"    # light blue — background shading

REGIME_COLORS = {
    "pre_crisis":   "#EBF5FB",   # light blue
    "crisis_onset": "#FDEBD0",   # light orange
    "peak_crisis":  "#FADBD8",   # light red
    "post_crisis":  "#EAFAF1",   # light green
}

REGIME_DATES = {
    "pre_crisis":   ("2019-01-01", "2021-12-31"),
    "crisis_onset": ("2022-01-01", "2022-06-30"),
    "peak_crisis":  ("2022-07-01", "2023-06-30"),
    "post_crisis":  ("2023-07-01", "2023-12-31"),
}

REGIME_LABELS = {
    "pre_crisis":   "Pre-Crisis\n(2019–2021)",
    "crisis_onset": "Crisis Onset\n(2022 H1)",
    "peak_crisis":  "Peak Crisis\n(2022 H2–2023 H1)",
    "post_crisis":  "Post-Crisis\n(2023 H2)",
}

MODEL_DISPLAY = {
    "qk_svr":        "QK-SVR",
    "rbf_svm":       "RBF-SVM",
    "laplacian_svm": "Laplacian-SVM",
    "xgboost":       "XGBoost",
    "lightgbm":      "LightGBM",
    "bilstm":        "BiLSTM",
    "gru":           "GRU",
    "transformer":   "Transformer",
    "emd_lstm":      "EMD-LSTM",
    "rw":            "RW",
    "hm":            "HM",
}

MODEL_ORDER = ["qk_svr","rbf_svm","laplacian_svm","xgboost","lightgbm",
               "bilstm","gru","transformer","emd_lstm","rw","hm"]


# ── Matplotlib style setup ────────────────────────────────────────
def setup_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":          9,
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "figure.dpi":         DPI,
        "savefig.dpi":        DPI,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.facecolor":  "white",
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })


def add_regime_shading(ax, df_dates=None):
    """Add regime background shading to a time-series axis."""
    for regime, (s, e) in REGIME_DATES.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=REGIME_COLORS[regime], alpha=0.5, zorder=0)


def save_fig(fig, name, tight=True):
    path_png = FIGURES_DIR / f"{name}.png"
    path_eps = FIGURES_DIR / f"{name}.eps"
    if tight:
        fig.tight_layout()
    fig.savefig(path_png, dpi=DPI, format="png")
    try:
        fig.savefig(path_eps, format="eps")
    except Exception:
        pass  # EPS may fail with some elements
    plt.close(fig)
    logging.info(f"  Saved: {path_png.name}")


# =============================================================================
# FIGURE 1: EUA Carbon Price & Return Series
# =============================================================================
def figure1_eua_series():
    logging.info("[Figure 1] EUA Price & Return Series...")

    # Load EUA data
    eua_path = DATA_DIR / "eua_daily_processed.csv"
    if not eua_path.exists():
        logging.warning(f"  {eua_path.name} not found — skipping Figure 1")
        return

    df = pd.read_csv(eua_path)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date").reset_index(drop=True)

    # Filter to sample period
    mask = (df["date"] >= "2019-01-01") & (df["date"] <= "2023-12-31")
    df   = df[mask].copy()

    # Find price and return columns
    price_col  = next((c for c in df.columns if "price" in c.lower()
                       and "return" not in c.lower()), None)
    return_col = next((c for c in df.columns if "return" in c.lower()), None)

    if price_col is None or return_col is None:
        logging.warning(f"  Cannot find price/return columns. Cols: {list(df.columns)}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 4.5),
                                    gridspec_kw={"height_ratios": [1.6, 1]},
                                    sharex=True)

    # ── Panel A: Price level ──
    add_regime_shading(ax1)
    ax1.plot(df["date"], df[price_col], color=C_PRIMARY, linewidth=0.8, zorder=2)
    ax1.set_ylabel("EUA Price (EUR/tCO₂)", fontsize=9)
    ax1.set_title("A", loc="left", fontweight="bold", fontsize=9)

    # Price peak annotation
    peak_idx = df[price_col].idxmax()
    ax1.annotate(f"Peak: EUR {df[price_col][peak_idx]:.0f}",
                 xy=(df["date"][peak_idx], df[price_col][peak_idx]),
                 xytext=(20, -30), textcoords="offset points",
                 fontsize=7, color=C_ACCENT,
                 arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=0.8))

    # ── Panel B: Log-return ──
    add_regime_shading(ax2)
    ax2.plot(df["date"], df[return_col], color=C_NEUTRAL, linewidth=0.5,
             alpha=0.8, zorder=2)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.set_ylabel("Log-Return", fontsize=9)
    ax2.set_xlabel("Date", fontsize=9)
    ax2.set_title("B", loc="left", fontweight="bold", fontsize=9)

    # ── Regime legend ──
    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.7, label=REGIME_LABELS[r].replace("\n"," "))
               for r in ["pre_crisis","crisis_onset","peak_crisis","post_crisis"]]
    ax1.legend(handles=patches, loc="upper left", fontsize=7,
               frameon=False, ncol=2, handlelength=1.5)

    # X-axis formatting
    ax2.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    fig.suptitle("Figure 1. EU ETS Carbon Price and Log-Return (2019–2023)",
                 fontsize=10, fontweight="bold", y=1.01)
    save_fig(fig, "Figure_01_EUA_Series")


# =============================================================================
# FIGURE 2: Walk-Forward Cross-Validation Scheme
# =============================================================================
def figure2_wfcv_scheme():
    logging.info("[Figure 2] Walk-Forward CV Scheme...")

    fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(-0.5, 5.5)
    ax.axis("off")

    n_shown = 5   # show 5 example folds

    for i in range(n_shown):
        y       = n_shown - 1 - i
        train_w = 3.0 + i * 0.6   # growing train window
        test_x  = train_w
        test_w  = 0.4

        # Train bar
        train_box = FancyBboxPatch((0.2, y + 0.15), train_w, 0.6,
                                   boxstyle="round,pad=0.02",
                                   facecolor=C_LIGHT, edgecolor=C_PRIMARY,
                                   linewidth=0.8)
        ax.add_patch(train_box)
        ax.text(0.2 + train_w/2, y + 0.45, "Train", ha="center", va="center",
                fontsize=7.5, color=C_PRIMARY, fontweight="bold")

        # Test bar
        test_box = FancyBboxPatch((test_x + 0.25, y + 0.15), test_w, 0.6,
                                  boxstyle="round,pad=0.02",
                                  facecolor=C_ACCENT, edgecolor=C_ACCENT,
                                  linewidth=0.8, alpha=0.85)
        ax.add_patch(test_box)
        ax.text(test_x + 0.25 + test_w/2, y + 0.45,
                "T", ha="center", va="center", fontsize=7, color="white", fontweight="bold")

        # Fold label
        ax.text(0.0, y + 0.45, f"Fold {i+1}", ha="right", va="center",
                fontsize=7.5, color="black")

    # Step arrow
    ax.annotate("", xy=(0.2 + 3.6, -0.2), xytext=(0.2 + 3.0, -0.2),
                arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=1.2))
    ax.text(0.2 + 3.3, -0.38, "Step = 21 days", ha="center", fontsize=7.5,
            color=C_PRIMARY)

    # Timeline arrow
    ax.annotate("", xy=(9.5, -0.05), xytext=(0.0, -0.05),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax.text(9.7, -0.05, "t", ha="left", va="center", fontsize=9, style="italic")

    # Labels
    ax.text(2.5, 5.2, "Expanding Training Window\n(initial = 504 days ≈ 2 years)",
            ha="center", fontsize=7.5, color=C_PRIMARY,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C_LIGHT, edgecolor=C_PRIMARY, lw=0.5))
    ax.text(7.0, 5.2, "Test (H ∈ {1,5,22})",
            ha="center", fontsize=7.5, color=C_ACCENT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FADBD8", edgecolor=C_ACCENT, lw=0.5))

    ax.set_title("Figure 2. Walk-Forward Cross-Validation Design\n"
                 "(37 windows × 3 horizons = 111 folds)",
                 fontsize=9, fontweight="bold")
    save_fig(fig, "Figure_02_WFCV_Scheme", tight=False)


# =============================================================================
# FIGURE 3: Quantum Kernel Circuit
# =============================================================================
def figure3_circuit():
    logging.info("[Figure 3] Quantum Kernel Circuit...")

    fig, ax = plt.subplots(figsize=(FIG_W, 3.2))
    ax.set_xlim(-0.5, 12); ax.set_ylim(-0.5, 4.5)
    ax.axis("off")

    n_qubits = 4
    n_layers = 2

    # Wire positions (top to bottom = qubit 0 to 3)
    y_q = [3.5, 2.5, 1.5, 0.5]
    x_end = 11.5

    # Draw wires
    for q, y in enumerate(y_q):
        ax.annotate("", xy=(x_end, y), xytext=(-0.3, y),
                    arrowprops=dict(arrowstyle="-", color="black", lw=1.0))
        ax.text(-0.4, y, f"|0⟩", ha="right", va="center", fontsize=9)

    # Qubit labels
    for q, y in enumerate(y_q):
        ax.text(-0.4, y - 0.25, f"q{q}", ha="right", va="center",
                fontsize=7, color=C_NEUTRAL, style="italic")

    def gate_box(ax, x, y, label, color=C_PRIMARY, textcolor="white", w=0.55, h=0.45):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor="black", linewidth=0.7, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=7, color=textcolor, fontweight="bold", zorder=4)

    def cnot(ax, x, ctrl, tgt):
        yc, yt = y_q[ctrl], y_q[tgt]
        # vertical line
        ax.plot([x, x], [min(yc,yt), max(yc,yt)], "k-", lw=1.0, zorder=2)
        # control dot
        ax.plot(x, yc, "ko", markersize=5, zorder=3)
        # target circle with ⊕
        circle = plt.Circle((x, yt), 0.18, color="white", ec="black", lw=1.0, zorder=3)
        ax.add_patch(circle)
        ax.text(x, yt, "⊕", ha="center", va="center", fontsize=9, zorder=4)

    x = 0.5
    for l in range(n_layers):
        # Layer bracket label
        x_bracket_start = x - 0.2

        # AngleEmbedding: RX(x_i)
        for q in range(n_qubits):
            gate_box(ax, x, y_q[q], f"RX(x{q})", color="#2E86AB")
        ax.text(x, -0.3, "AngleEmb.", ha="center", fontsize=7, color="#2E86AB")
        x += 1.1

        # CNOT ring
        for q in range(n_qubits):
            cnot(ax, x, q, (q+1) % n_qubits)
        ax.text(x, -0.3, "CNOT(ring)", ha="center", fontsize=7, color="black")
        x += 1.1

        # RY trainable
        for q in range(n_qubits):
            gate_box(ax, x, y_q[q], f"RY(θ{l}{q})", color=C_PRIMARY)
        ax.text(x, -0.3, "Trainable RY", ha="center", fontsize=7, color=C_PRIMARY)
        x += 1.1

        # Layer bracket
        x_bracket_end = x - 0.2
        if l == 0:
            ax.annotate("", xy=(x_bracket_end, 4.1),
                        xytext=(x_bracket_start, 4.1),
                        arrowprops=dict(arrowstyle="<->", color="gray", lw=0.8))
            ax.text((x_bracket_start + x_bracket_end)/2, 4.25,
                    f"Layer {l+1}", ha="center", fontsize=7.5, color="gray")
        else:
            ax.annotate("", xy=(x_bracket_end, 4.1),
                        xytext=(x_bracket_start, 4.1),
                        arrowprops=dict(arrowstyle="<->", color="gray", lw=0.8))
            ax.text((x_bracket_start + x_bracket_end)/2, 4.25,
                    f"Layer {l+1}", ha="center", fontsize=7.5, color="gray")

        # Dashed separator between layers
        if l < n_layers - 1:
            ax.axvline(x - 0.05, ymin=0.08, ymax=0.92,
                       color="gray", linestyle="--", lw=0.6, alpha=0.5)

    # Measurement
    for q in range(n_qubits):
        gate_box(ax, x, y_q[q], "M", color="#8E44AD", w=0.45)
    ax.text(x, -0.3, "⟨ψ|ψ'⟩", ha="center", fontsize=7.5, color="#8E44AD",
            fontweight="bold")

    # Legend
    legend_items = [
        mpatches.Patch(color="#2E86AB", label="Data encoding (AngleEmbedding)"),
        mpatches.Patch(color="white", ec="black", label="Entanglement (CNOT ring)"),
        mpatches.Patch(color=C_PRIMARY, label="Trainable rotations (RY)"),
        mpatches.Patch(color="#8E44AD", label="Fidelity measurement |⟨ψ|ψ'⟩|²"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=6.5,
              frameon=True, framealpha=0.9, edgecolor="gray",
              bbox_to_anchor=(1.0, 0.0))

    ax.set_title("Figure 3. Quantum Kernel Circuit Architecture\n"
                 "(n_qubits=4, n_layers=2, circular entanglement)",
                 fontsize=9, fontweight="bold")
    save_fig(fig, "Figure_03_Circuit", tight=False)


# =============================================================================
# FIGURE 4: Crisis Sub-Period RMSE Comparison
# =============================================================================
def figure4_crisis_rmse():
    logging.info("[Figure 4] Crisis RMSE Comparison...")

    path = RESULTS_DIR / "crisis_subperiod.csv"
    if not path.exists():
        logging.warning(f"  {path.name} not found — skipping Figure 4")
        return

    df = pd.read_csv(path)
    df["model_display"] = df["model"].map(lambda x: MODEL_DISPLAY.get(x, x))

    # Focus: H=5 (most interesting horizon for QK-SVR)
    df_h5 = df[df["horizon"] == 5].copy()

    regimes = ["pre_crisis","crisis_onset","peak_crisis","post_crisis"]
    models_plot = [m for m in MODEL_ORDER if m in df["model"].unique()]
    n_models = len(models_plot)
    n_regimes = len(regimes)

    fig, ax = plt.subplots(figsize=(FIG_W, 3.8))

    x     = np.arange(n_models)
    width = 0.18
    colors = [REGIME_COLORS["pre_crisis"], REGIME_COLORS["crisis_onset"],
              REGIME_COLORS["peak_crisis"], REGIME_COLORS["post_crisis"]]
    edge_colors = [C_NEUTRAL, C_ORANGE, C_ACCENT, C_GREEN]

    for ri, (regime, ec) in enumerate(zip(regimes, edge_colors)):
        df_r = df_h5[df_h5["regime"] == regime]
        vals = []
        for m in models_plot:
            row = df_r[df_r["model"] == m]
            vals.append(float(row["rmse_mean"].values[0]) if len(row) > 0 else np.nan)

        offset = (ri - n_regimes/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=REGIME_LABELS[regime].replace("\n"," "),
                      color=colors[ri], edgecolor=ec, linewidth=0.8, alpha=0.9)

        # Highlight QK-SVR bar in peak crisis
        if regime == "peak_crisis":
            qk_idx = models_plot.index("qk_svr") if "qk_svr" in models_plot else None
            if qk_idx is not None:
                bars[qk_idx].set_edgecolor(C_ACCENT)
                bars[qk_idx].set_linewidth(2.0)
                ax.text(x[qk_idx] + offset,
                        vals[qk_idx] + 0.0005,
                        "★", ha="center", va="bottom",
                        fontsize=8, color=C_ACCENT)

    ax.set_xlabel("Model", fontsize=9)
    ax.set_ylabel("RMSE", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models_plot],
                       rotation=35, ha="right", fontsize=7.5)
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)

    # QK-SVR annotation
    ax.text(0.02, 0.97,
            "★ QK-SVR lowest RMSE\nat H=5 Peak Crisis",
            transform=ax.transAxes, fontsize=7, va="top",
            color=C_ACCENT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_ACCENT, lw=0.8, alpha=0.9))

    ax.set_title("Figure 4. Crisis Sub-Period Forecast Accuracy (H=5)\n"
                 "RMSE across four structural regimes, all models",
                 fontsize=9, fontweight="bold")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    save_fig(fig, "Figure_04_Crisis_RMSE")


# =============================================================================
# FIGURE 5: Feature Importance Heatmap
# =============================================================================
def figure5_feature_importance():
    logging.info("[Figure 5] Feature Importance Heatmap...")

    path = RESULTS_DIR / "feature_importance.csv"
    if not path.exists():
        logging.warning(f"  {path.name} not found — skipping Figure 5")
        return

    df = pd.read_csv(path)

    # Pivot QKFM importance
    feat_order  = ["GAS_return","OIL_return","COAL_return","ELEC_return"]
    feat_labels = ["Natural Gas (TTF)","Brent Oil","Coal ARA","Electricity"]
    regime_order  = ["pre_crisis","crisis_onset","peak_crisis"]
    regime_labels = ["Pre-Crisis","Crisis Onset","Peak Crisis"]

    # Build matrix
    def build_matrix(df, method_col):
        mat = np.full((len(regime_order), len(feat_order)), np.nan)
        for ri, reg in enumerate(regime_order):
            df_r = df[df["regime"]==reg]
            for fi, feat in enumerate(feat_order):
                row = df_r[df_r["feature"]==feat]
                if len(row) > 0 and method_col in row.columns:
                    mat[ri, fi] = float(row[method_col].values[0])
        return mat

    mat_qkfm = build_matrix(df, "qkfm_importance")
    mat_shap = build_matrix(df, "shap_importance")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, 2.8))

    def plot_heatmap(ax, mat, title, cmap="Blues"):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0,
                       vmax=np.nanmax(mat) if not np.all(np.isnan(mat)) else 1)
        ax.set_xticks(range(len(feat_order)))
        ax.set_xticklabels(feat_labels, rotation=30, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(regime_order)))
        ax.set_yticklabels(regime_labels, fontsize=8)
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)

        # Value annotations
        for ri in range(len(regime_order)):
            for fi in range(len(feat_order)):
                val = mat[ri, fi]
                if not np.isnan(val):
                    ax.text(fi, ri, f"{val:.3f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if val > np.nanmax(mat)*0.6 else "black",
                            fontweight="bold" if val == np.nanmax(mat) else "normal")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Importance")
        return im

    plot_heatmap(ax1, mat_qkfm, "A. QKFM (Quantum Kernel\nFeature Masking)", cmap="Blues")
    plot_heatmap(ax2, mat_shap, "B. TreeSHAP (XGBoost)", cmap="Oranges")

    fig.suptitle("Figure 5. Feature Importance: QKFM vs. TreeSHAP Across Regimes",
                 fontsize=9, fontweight="bold", y=1.02)
    save_fig(fig, "Figure_05_Feature_Importance")


# =============================================================================
# MAIN
# =============================================================================
def main():
    setup_style()

    print("=" * 60)
    print("  Generating Manuscript Figures (300 DPI)")
    print(f"  Output: {FIGURES_DIR}")
    print("=" * 60)

    figure1_eua_series()
    figure2_wfcv_scheme()
    figure3_circuit()
    figure4_crisis_rmse()
    figure5_feature_importance()

    # Summary
    pngs = list(FIGURES_DIR.glob("Figure_*.png"))
    print(f"\n{'='*60}")
    print(f"  ✅ {len(pngs)}/5 figures generated:")
    for p in sorted(pngs):
        size_kb = p.stat().st_size / 1024
        print(f"     {p.name}  ({size_kb:.0f} KB)")
    print(f"\n  Figures sẵn sàng cho manuscript submission.")
    print(f"  Format: PNG (300 DPI) + EPS (nếu có thể)")
    print(f"{'='*60}")


if __name__ == "__main__":
    import matplotlib.dates
    main()
