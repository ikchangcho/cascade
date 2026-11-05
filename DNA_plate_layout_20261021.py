import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle

root = os.path.dirname(os.path.abspath(__file__))
meta = pd.read_csv(os.path.join(root, "absorbances", "20261021_DNA_samples_metadata.csv"))

rows = list("ABCDEFGH")
cols = range(1, 13)

# categorical slots 1-3, each as a light fill + full-strength edge
style = {
    "T11": ("#cde2fb", "#2a78d6"),
    "T0": ("#fbdfd2", "#eb6834"),
    "Blank": ("#cceee0", "#1baf7a"),
}

fig, ax = plt.subplots(figsize=(11, 6.2))

for _, r in meta.iterrows():
    i, j = rows.index(r["Well"][0]), int(r["Well"][1:]) - 1
    fill, edge = style[r["Sample_type"]]
    # 2px surface gap between wells
    ax.add_patch(Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, facecolor=fill,
                           edgecolor=edge, linewidth=1.2))
    if r["Sample_type"] == "T11":
        label = f"({r['Nitrate_input']:g}, {r['Nitrite_input']:g})"
    else:
        label = r["Sample_type"]
    ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=8, color="#0b0b0b")

# row letters, tight against the plate
for i, row in enumerate(rows):
    ax.text(-0.3, i + 0.5, row, ha="center", va="center", fontsize=10, color="#0b0b0b")

# chloramphenicol split: rows A-D vs E-H
ax.plot([0, 12], [4, 4], color="#0b0b0b", linewidth=2.5)
for y0, y1, text in [(0, 4, "CHL +"), (4, 8, "CHL -")]:
    ax.annotate("", xy=(-0.8, y0 + 0.1), xytext=(-0.8, y1 - 0.1),
                arrowprops=dict(arrowstyle="-", color="#52514e", linewidth=1.5))
    ax.text(-1.1, (y0 + y1) / 2, text, rotation=90, ha="center", va="center",
            fontsize=10, color="#52514e")

ax.set_xlim(-1.5, 12)
ax.set_ylim(8, 0)
ax.set_aspect("equal")
ax.set_xticks([c - 0.5 for c in cols], [f"{c:02d}" for c in cols])
ax.set_yticks([])
ax.xaxis.set_ticks_position("top")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

ax.legend(handles=[Patch(facecolor=f, edgecolor=e, label=k) for k, (f, e) in style.items()],
          loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)

fig.tight_layout()
out = os.path.join(root, "plots", "20261021_DNA_samples_plate_layout.png")
fig.savefig(out, dpi=300)
plt.close(fig)
print("wrote", out)
