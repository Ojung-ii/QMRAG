import os
from pathlib import Path

mpl_config_dir = Path("/tmp/matplotlib-qmrag")
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =========================
# Data: HotpotQA F1
# =========================
data = {
    "RAPTOR": {
        "Common Prompt": 0.1502,
        "Method Prompt": 0.2142,
    },
    "HippoRAG2": {
        "Common Prompt": 0.3126,
        "Method Prompt": 0.6118,
    },
    "LightRAG": {
        "Common Prompt": 0.3229,
        "Method Prompt": 0.1300,
    },
    "BRACE-RAG": {
        "Common Prompt": 0.4495,
        "Method Prompt": 0.5246,
    },
}

# =========================
# Output paths
# =========================
out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)

png_path = out_dir / "prompt_sensitivity_hotpotqa.png"
pdf_path = out_dir / "prompt_sensitivity_hotpotqa.pdf"

# =========================
# Plot settings
# =========================
x_labels = ["Common Prompt", "Method Prompt"]
x = [0, 1]

colors = {
    "RAPTOR": "#7f7f7f",
    "HippoRAG2": "#1f77b4",
    "LightRAG": "#ff7f0e",
    "BRACE-RAG": "#2ca02c",
}

markers = {
    "RAPTOR": "o",
    "HippoRAG2": "s",
    "LightRAG": "^",
    "BRACE-RAG": "D",
}

plt.figure(figsize=(5.2, 3.2))

for method, scores in data.items():
    y = [scores["Common Prompt"], scores["Method Prompt"]]

    plt.plot(
        x,
        y,
        marker=markers[method],
        linewidth=2.2,
        markersize=6,
        color=colors[method],
        label=method,
    )

    # Label values near points
    for xi, yi in zip(x, y):
        plt.text(
            xi,
            yi + 0.015,
            f"{yi:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors[method],
        )

# =========================
# Axis / labels
# =========================
plt.xticks(x, x_labels, fontsize=10)
plt.ylabel("F1", fontsize=11)
plt.ylim(0.05, 0.68)

plt.grid(axis="y", linestyle="--", alpha=0.35)
plt.legend(
    frameon=False,
    fontsize=8.5,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12),
    ncol=4,
    columnspacing=1.0,
    handlelength=1.8,
)


plt.tight_layout()

# =========================
# Save
# =========================
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")
