import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path

# =========================
# 0. Global style config
# =========================
FIG_WIDTH = 7.2
FIG_HEIGHT = 5.0

FONT_SIZE = 18
LABEL_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 12

LINE_WIDTH = 2.8
GRID_ALPHA = 0.22

# layout margins
LEFT_MARGIN = 0.12
RIGHT_MARGIN = 0.88
BOTTOM_MARGIN = 0.16
TOP_MARGIN = 0.96

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# 1. Data
#    - Top3 removed
#    - Source: outputs/analysis/20260526_074635/full_structured_budget_scaling_summary.json
# =========================
rows = [
    # HotpotQA
    {"dataset": "HotpotQA", "budget": "500",  "input_tok": 593.5,  "f1": 0.3642, "f1_per_1k_input": 0.6135},
    {"dataset": "HotpotQA", "budget": "1000", "input_tok": 1063.9, "f1": 0.3898, "f1_per_1k_input": 0.3664},
    {"dataset": "HotpotQA", "budget": "1500", "input_tok": 1477.2, "f1": 0.4214, "f1_per_1k_input": 0.2853},
    {"dataset": "HotpotQA", "budget": "2000", "input_tok": 1708.7, "f1": 0.4333, "f1_per_1k_input": 0.2536},
    {"dataset": "HotpotQA", "budget": "Full", "input_tok": 1834.7, "f1": 0.4495, "f1_per_1k_input": 0.2450},

    # 2Wiki
    {"dataset": "2Wiki", "budget": "500",  "input_tok": 591.7,  "f1": 0.1517, "f1_per_1k_input": 0.2564},
    {"dataset": "2Wiki", "budget": "1000", "input_tok": 1057.4, "f1": 0.1443, "f1_per_1k_input": 0.1365},
    {"dataset": "2Wiki", "budget": "1500", "input_tok": 1453.9, "f1": 0.1708, "f1_per_1k_input": 0.1174},
    {"dataset": "2Wiki", "budget": "2000", "input_tok": 1699.5, "f1": 0.1674, "f1_per_1k_input": 0.0985},
    {"dataset": "2Wiki", "budget": "Full", "input_tok": 1857.3, "f1": 0.1771, "f1_per_1k_input": 0.0954},
]

df = pd.DataFrame(rows)

budget_order = ["500", "1000", "1500", "2000", "Full"]
df["budget"] = pd.Categorical(df["budget"], categories=budget_order, ordered=True)

color_map = {
    "HotpotQA": "#1f77b4",
    "2Wiki": "#d62728",
}

# =========================
# 2. Plot
# =========================
fig, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
ax2 = ax1.twinx()

for dataset, g in df.groupby("dataset"):
    g = g.sort_values("input_tok")
    color = color_map[dataset]

    # F1: solid line
    ax1.plot(
        g["input_tok"],
        g["f1"],
        color=color,
        linewidth=LINE_WIDTH,
        linestyle="-",
    )

    # F1/1K Input: dashed line
    ax2.plot(
        g["input_tok"],
        g["f1_per_1k_input"],
        color=color,
        linewidth=LINE_WIDTH,
        linestyle="--",
    )

# Axis labels
ax1.set_xlabel("Input tokens")
ax1.set_ylabel("F1")
ax2.set_ylabel("F1 / 1K input tokens")

# Grid
ax1.grid(True, alpha=GRID_ALPHA)

# Optional: make axis spines slightly thicker
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# =========================
# 3. Unified top legend: 1 x 4
# =========================
legend_handles = [
    # Dataset markers
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markersize=7.5,
        markerfacecolor=color_map["HotpotQA"],
        markeredgecolor=color_map["HotpotQA"],
        label="HotpotQA",
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markersize=7.5,
        markerfacecolor=color_map["2Wiki"],
        markeredgecolor=color_map["2Wiki"],
        label="2Wiki",
    ),

    # Metric line styles
    Line2D(
        [0], [0],
        color="black",
        linewidth=LINE_WIDTH,
        linestyle="-",
        label="F1",
    ),
    Line2D(
        [0], [0],
        color="black",
        linewidth=LINE_WIDTH,
        linestyle="--",
        label="F1 / 1K Input",
    ),
]

ax1.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    ncol=4,
    frameon=True,
    framealpha=0.86,
    facecolor="white",
    edgecolor="none",
    borderpad=0.25,
    borderaxespad=0.15,
    columnspacing=0.35,
    handlelength=1.05,
    handletextpad=0.25,
    labelspacing=0.25,
)

# No subplot subtitle / no panel label
fig.subplots_adjust(
    left=LEFT_MARGIN,
    right=RIGHT_MARGIN,
    bottom=BOTTOM_MARGIN,
    top=TOP_MARGIN,
)

# =========================
# 4. Save
# =========================
out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)

png_path = out_dir / "ace_rag_budget_tradeoff.png"
pdf_path = out_dir / "ace_rag_budget_tradeoff.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")

plt.show()
