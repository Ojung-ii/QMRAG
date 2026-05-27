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
FIG_HEIGHT = 8.4
TWO_PANEL_WIDTH = 13.8
TWO_PANEL_HEIGHT = 8.0

FONT_SIZE = 26
LABEL_SIZE = 31
TICK_SIZE = 27
LEGEND_SIZE = 30
PANEL_LABEL_SIZE = 34
LEGEND_MARKER_SIZE = 14
X_LABEL_PAD = 10
PANEL_LABEL_Y = -0.28

GRID_ALPHA = 0.22
MARKER_SIZE = 235
EDGE_WIDTH = 2.2
ACE_EDGE_WIDTH = 3.2
DATASET_LEGEND_EDGE_WIDTH = 2.6

LEFT_MARGIN = 0.12
RIGHT_MARGIN = 0.82
BOTTOM_MARGIN = 0.17
TOP_MARGIN = 0.76

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
# 1. Common-prompt paper-table data
#    Source: outputs/final_baseline_aware/20260526_053914/final_common_prompt_table.md
#    Values are fractions, not percentages.
#    ACE-RAG uses the paper main row: ACE-RAG-Compact = common_qa/top3.
#    Native/method-prompt and ACE-RAG-Scaled top8 rows are intentionally excluded.
# =========================
rows = [
    # HotpotQA
    {"dataset": "HotpotQA", "method": "BM25", "recall": 0.676, "f1": 0.217, "ctx": 749.5},
    {"dataset": "HotpotQA", "method": "Dense RAG", "recall": 0.944, "f1": 0.314, "ctx": 657.6},
    {"dataset": "HotpotQA", "method": "RAPTOR", "recall": 0.410, "f1": 0.150, "ctx": 3268.4},
    {"dataset": "HotpotQA", "method": "HippoRAG2", "recall": 0.951, "f1": 0.313, "ctx": 668.1},
    {"dataset": "HotpotQA", "method": "LightRAG", "recall": 0.922, "f1": 0.323, "ctx": 641.0},
    {"dataset": "HotpotQA", "method": "ACE-RAG", "recall": 0.970, "f1": 0.379, "ctx": 458.2},
    # 2Wiki
    {"dataset": "2Wiki", "method": "BM25", "recall": 0.531, "f1": 0.042, "ctx": 1111.2},
    {"dataset": "2Wiki", "method": "Dense RAG", "recall": 0.765, "f1": 0.081, "ctx": 793.9},
    {"dataset": "2Wiki", "method": "RAPTOR", "recall": 0.561, "f1": 0.075, "ctx": 3097.6},
    {"dataset": "2Wiki", "method": "HippoRAG2", "recall": 0.897, "f1": 0.089, "ctx": 845.3},
    {"dataset": "2Wiki", "method": "LightRAG", "recall": 0.802, "f1": 0.095, "ctx": 784.0},
    {"dataset": "2Wiki", "method": "ACE-RAG", "recall": 0.881, "f1": 0.143, "ctx": 462.1},
    # MuSiQue
    {"dataset": "MuSiQue", "method": "BM25", "recall": 0.284, "f1": 0.022, "ctx": 835.4},
    {"dataset": "MuSiQue", "method": "Dense RAG", "recall": 0.693, "f1": 0.043, "ctx": 741.6},
    {"dataset": "MuSiQue", "method": "RAPTOR", "recall": 0.423, "f1": 0.049, "ctx": 3284.1},
    {"dataset": "MuSiQue", "method": "HippoRAG2", "recall": 0.722, "f1": 0.055, "ctx": 749.4},
    {"dataset": "MuSiQue", "method": "LightRAG", "recall": 0.540, "f1": 0.045, "ctx": 705.3},
    {"dataset": "MuSiQue", "method": "ACE-RAG", "recall": 0.757, "f1": 0.070, "ctx": 592.2},
    # PopQA
    {"dataset": "PopQA", "method": "BM25", "recall": 0.381, "f1": 0.350, "ctx": 773.2},
    {"dataset": "PopQA", "method": "Dense RAG", "recall": 0.511, "f1": 0.417, "ctx": 738.9},
    {"dataset": "PopQA", "method": "RAPTOR", "recall": 0.386, "f1": 0.272, "ctx": 3263.5},
    {"dataset": "PopQA", "method": "HippoRAG2", "recall": 0.516, "f1": 0.416, "ctx": 743.0},
    {"dataset": "PopQA", "method": "LightRAG", "recall": 0.355, "f1": 0.391, "ctx": 741.1},
    {"dataset": "PopQA", "method": "ACE-RAG", "recall": 0.561, "f1": 0.476, "ctx": 436.3},
]

method_colors = {
    "BM25": "#7f7f7f",
    "Dense RAG": "#8c564b",
    "RAPTOR": "#9467bd",
    "HippoRAG2": "#1f77b4",
    "LightRAG": "#ff7f0e",
    "ACE-RAG": "#2ca02c",
}

dataset_markers = {
    "HotpotQA": "o",
    "2Wiki": "s",
    "MuSiQue": "^",
    "PopQA": "D",
}


def style_axis(ax):
    ax.grid(True, alpha=GRID_ALPHA)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def draw_scatter(ax, x_key, xlabel, xlim, panel_label=None):
    for row in rows:
        method = row["method"]
        dataset = row["dataset"]
        is_ace = method == "ACE-RAG"
        ax.scatter(
            row[x_key],
            row["f1"],
            marker=dataset_markers[dataset],
            s=MARKER_SIZE if not is_ace else MARKER_SIZE + 95,
            facecolor=method_colors[method],
            edgecolor="black" if is_ace else "white",
            linewidth=ACE_EDGE_WIDTH if is_ace else EDGE_WIDTH,
            alpha=0.92,
            zorder=4 if is_ace else 3,
        )

    ax.set_xlabel(xlabel, labelpad=X_LABEL_PAD)
    ax.set_ylabel("F1")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 0.52)
    style_axis(ax)

    if panel_label:
        ax.text(
            0.0,
            1.035,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
        )


def method_legend_handles():
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=LEGEND_MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=EDGE_WIDTH,
            label=method,
        )
        for method, color in method_colors.items()
    ]


def dataset_legend_handles():
    return [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="None",
            markersize=LEGEND_MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=DATASET_LEGEND_EDGE_WIDTH,
            label=dataset,
        )
        for dataset, marker in dataset_markers.items()
    ]


def blank_legend_handle():
    return Line2D(
        [0],
        [0],
        marker="",
        linestyle="None",
        color="none",
        label=" ",
    )


def two_panel_legend_handles():
    methods = method_legend_handles()
    datasets = dataset_legend_handles()

    # Matplotlib fills multi-row legends by columns. This interleaving yields:
    # row 1: six methods
    # row 2: blank, four datasets, blank
    return [
        methods[0],
        blank_legend_handle(),
        methods[1],
        datasets[0],
        methods[2],
        datasets[1],
        methods[3],
        datasets[2],
        methods[4],
        datasets[3],
        methods[5],
        blank_legend_handle(),
    ]


def add_two_panel_top_legend(fig):
    fig.legend(
        handles=two_panel_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        frameon=False,
        columnspacing=0.45,
        handletextpad=0.22,
        handlelength=0.8,
        borderaxespad=0.0,
    )


def add_legends(fig, ax=None, two_panel=False, dataset_loc="lower right", dataset_bbox=None):
    method_bbox = (0.5, 0.97)
    fig.legend(
        handles=method_legend_handles(),
        loc="upper center",
        bbox_to_anchor=method_bbox,
        ncol=3,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    if ax is not None:
        legend_kwargs = {
            "handles": dataset_legend_handles(),
            "loc": dataset_loc,
            "frameon": False,
            "borderpad": 0.2,
            "labelspacing": 0.35,
            "handletextpad": 0.35,
        }
        if dataset_bbox is not None:
            legend_kwargs["bbox_to_anchor"] = dataset_bbox
        ax.legend(
            **legend_kwargs,
        )


def save(fig, stem):
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


def plot_ctx_f1():
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    draw_scatter(ax, "ctx", "Context tokens", (300, 3450))
    add_legends(fig, ax=ax, dataset_loc="center left", dataset_bbox=(1.02, 0.50))
    fig.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        bottom=BOTTOM_MARGIN,
        top=TOP_MARGIN,
    )
    save(fig, "common_prompt_ctx_f1_scatter")


def plot_recall_f1():
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    draw_scatter(ax, "recall", "Recall@5", (0.25, 1.0))
    add_legends(fig, ax=ax, dataset_loc="center left", dataset_bbox=(1.01, 0.52))
    fig.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        bottom=BOTTOM_MARGIN,
        top=TOP_MARGIN,
    )
    save(fig, "common_prompt_recall_f1_scatter")


def add_bottom_panel_label(ax, label):
    ax.text(
        0.5,
        PANEL_LABEL_Y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
    )


def plot_two_panel():
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(TWO_PANEL_WIDTH, TWO_PANEL_HEIGHT),
        sharey=True,
    )
    draw_scatter(axes[0], "recall", "Recall@5", (0.25, 1.0))
    draw_scatter(axes[1], "ctx", "Context tokens", (300, 3450))
    axes[1].set_ylabel("")
    add_bottom_panel_label(axes[0], "(a)")
    add_bottom_panel_label(axes[1], "(b)")
    add_two_panel_top_legend(fig)
    fig.subplots_adjust(
        left=0.045,
        right=0.998,
        bottom=0.20,
        top=0.80,
        wspace=0.09,
    )
    save(fig, "common_prompt_answerability_2panel")


if __name__ == "__main__":
    plot_ctx_f1()
    plot_recall_f1()
    plot_two_panel()
