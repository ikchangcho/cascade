import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

root = os.path.dirname(os.path.abspath(__file__))
conc_folder = os.path.join(root, "concentrations")
plot_folder = os.path.join(root, "plots")

batches = ["batch1", "batch4"]
data = {b: pd.read_csv(os.path.join(conc_folder, f"total_DNA_{b}_20261021.csv")) for b in batches}

# single-hue sequential ramp (light -> dark blue), shared by every panel
blues = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
blues.set_bad("#f0efec")

rows = list("ABCDEFGH")
cols = [f"{c:02d}" for c in range(1, 13)]


def annotate(ax, M, fmt="{:.0f}"):
    """Value in each cell, ink color flipped on the dark end of the ramp."""
    vmin, vmax = ax.images[0].get_clim()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            dark = (v - vmin) / (vmax - vmin) > 0.6
            label = fmt.format(v)
            if label.lstrip("-").strip("0.") == "":  # no "-0" for blanks
                label = label.lstrip("-")
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="#ffffff" if dark else "#2c2c29")


# ---------------------------------------------------------------- plate layout
# shared scale across both batches so the two plates are directly comparable
plate_max = max(d["Concentration"].max() for d in data.values())

for batch, d in data.items():
    M = np.full((8, 12), np.nan)
    for _, r in d.iterrows():
        M[rows.index(r["Well"][0]), int(r["Well"][1:]) - 1] = r["Concentration"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(M, cmap=blues, vmin=0, vmax=plate_max)
    ax.set_xticks(range(12), cols)
    ax.set_yticks(range(8), rows)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, 12), minor=True)
    ax.set_yticks(np.arange(-0.5, 8), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=2)
    ax.tick_params(which="minor", length=0)

    annotate(ax, M)
    ax.set_title(f"Total DNA, {batch} plate layout (20261021)", pad=28, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Concentration (ng/mL)")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_folder, f"20261021_total_DNA_{batch}_plate_heatmap.png"), dpi=300)
    plt.close(fig)


# --------------------------------------------------- condition grid (T11 only)
no2_levels = sorted(data["batch1"].query("Sample_type == 'T11'")["Nitrite_input"].unique())
no3_levels = sorted(data["batch1"].query("Sample_type == 'T11'")["Nitrate_input"].unique())

grids = {}
for batch, d in data.items():
    for chl in [0, 1]:
        sub = d[(d["Sample_type"] == "T11") & (d["Chloramphenicol"] == chl)]
        mean = sub.groupby(["Nitrite_input", "Nitrate_input"])["Concentration"].mean()
        M = np.full((len(no2_levels), len(no3_levels)), np.nan)
        for (no2, no3), v in mean.items():
            M[no2_levels.index(no2), no3_levels.index(no3)] = v
        grids[(batch, chl)] = M

cond_max = np.nanmax([M for M in grids.values()])

for batch in batches:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, chl in zip(axes, [0, 1]):
        M = grids[(batch, chl)]
        im = ax.imshow(M, cmap=blues, vmin=0, vmax=cond_max, origin="lower")
        ax.set_xticks(range(len(no3_levels)), [f"{v:g}" for v in no3_levels])
        ax.set_yticks(range(len(no2_levels)), [f"{v:g}" for v in no2_levels])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(no3_levels)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(no2_levels)), minor=True)
        ax.grid(which="minor", color="#ffffff", linewidth=2)
        ax.tick_params(which="minor", length=0)
        annotate(ax, M, fmt="{:.1f}")
        ax.set_xlabel("Nitrate input (mM)")
        ax.set_title(f"Chloramphenicol {chl}", fontsize=10)
    axes[0].set_ylabel("Nitrite input (mM)")
    fig.suptitle(f"Total DNA at T11, mean of 3 replicates, {batch} (20261021)", x=0.02, ha="left")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Concentration (ng/mL)")
    fig.savefig(os.path.join(plot_folder, f"20261021_total_DNA_{batch}_condition_heatmap.png"), dpi=300)
    plt.close(fig)

print("wrote 4 heatmaps to", plot_folder)
