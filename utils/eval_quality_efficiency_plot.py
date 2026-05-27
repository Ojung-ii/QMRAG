import os
from pathlib import Path

mpl_config_dir = Path("/tmp/matplotlib-ace_rag")
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================
# 0. Global style config
#    Matched to utils/eval_plot.py
# =========================
FIG_WIDTH = 8.4
FIG_HEIGHT = 6

FONT_SIZE = 18
LABEL_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 15

LINE_WIDTH = 2.8
GRID_ALPHA = 0.22

LEFT_MARGIN = 0.12
RIGHT_MARGIN = 0.95
BOTTOM_MARGIN = 0.17
TOP_MARGIN = 0.82

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# =========================
# 1. Data
#    Best common-prompt baseline -> ACE-RAG-Compact
# =========================
rows = [
    {
        "dataset": "HotpotQA",
        "baseline": "LightRAG",
        "baseline_f1": 0.3229,
        "baseline_ctx": 641,
        "ace_rag_f1": 0.379,
        "ace_rag_ctx": 458,
    },
    {
        "dataset": "2Wiki",
        "baseline": "LightRAG",
        "baseline_f1": 0.095,
        "baseline_ctx": 784,
        "ace_rag_f1": 0.143,
        "ace_rag_ctx": 462,
    },
    {
        "dataset": "MuSiQue",
        "baseline": "HippoRAG2",
        "baseline_f1": 0.055,
        "baseline_ctx": 749,
        "ace_rag_f1": 0.070,
        "ace_rag_ctx": 592,
    },
    {
        "dataset": "PopQA",
        "baseline": "Dense RAG",
        "baseline_f1": 0.417,
        "baseline_ctx": 739,
        "ace_rag_f1": 0.476,
        "ace_rag_ctx": 436,
    },
]

color_map = {
    "HotpotQA": "#1f77b4",
    "2Wiki": "#d62728",
    "MuSiQue": "#2ca02c",
    "PopQA": "#9467bd",
}

label_offsets = {
    "HotpotQA": (18, 8),
    "2Wiki": (18, 0),
    "MuSiQue": (18, 6),
    "PopQA": (18, 6),
}


# =========================
# 2. Plot
# =========================
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

for row in rows:
    dataset = row["dataset"]
    color = color_map[dataset]
    x0, y0 = row["baseline_ctx"], row["baseline_f1"]
    x1, y1 = row["ace_rag_ctx"], row["ace_rag_f1"]

    ax.scatter(
        [x0],
        [y0],
        marker="o",
        s=95,
        color=color,
        facecolors="none",
        linewidths=2.2,
        zorder=3,
    )
    ax.scatter(
        [x1],
        [y1],
        marker="o",
        s=110,
        color=color,
        zorder=4,
    )
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linewidth": LINE_WIDTH,
            "shrinkA": 6,
            "shrinkB": 8,
        },
        zorder=2,
    )

    ox, oy = label_offsets[dataset]
    ax.annotate(
        dataset,
        xy=(x1, y1),
        xytext=(ox, oy),
        textcoords="offset points",
        color=color,
        fontsize=15,
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6},
    )

# Axis labels
ax.set_xlabel("Context tokens")
ax.set_ylabel("F1")
ax.set_xlim(380, 830)
ax.set_ylim(0.035, 0.50)

# Grid and spines
ax.grid(True, alpha=GRID_ALPHA)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)


# =========================
# 3. Unified top legend: 1 x 4
# =========================
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=10,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=2.0,
        label="Best baseline",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=10,
        markerfacecolor="black",
        markeredgecolor="black",
        label="ACE-RAG-Compact",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        linewidth=LINE_WIDTH,
        linestyle="-",
        label="Improvement",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        linewidth=0,
        label="Common prompt",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.9),
    ncol=4,
    frameon=False,
    columnspacing=1.2,
    handlelength=1.8,
    handletextpad=0.5,
)

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

png_path = out_dir / "ace_rag_quality_efficiency.png"
pdf_path = out_dir / "ace_rag_quality_efficiency.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")
