import csv
import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "absorbances")

# 1. load all standard files
files = sorted(glob.glob(os.path.join(folder, "20261021_*_Ik_DNA_BR_standard.CSV")))
N = len(files)
print(f"N = {N} files")

# plate row (line 10 = row A) -> concentration in ng/mL, per 20261021_DNA_standards_metadata.csv
concentrations = {"A": 100, "B": 50, "C": 25, "D": 12.5}

times = []
readings = {c: [] for c in concentrations.values()}
for f in files:
    name = os.path.basename(f)
    # 3. timestamp from filename: 20261021_16-21-18_...
    date_str, time_str = name.split("_")[:2]
    times.append(datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H-%M-%S"))

    # 2. plate rows A-D start at line 10; columns B, C, D hold the three replicates
    with open(f, newline="") as fh:
        rows = list(csv.reader(fh))
    for k, conc in enumerate(concentrations.values()):
        readings[conc].append([float(x) for x in rows[9 + k][1:4]])

t0 = min(times)
elapsed_min = [(t - t0).total_seconds() / 60 for t in times]

for name, e, r in zip(map(os.path.basename, files), elapsed_min, readings[100]):
    print(f"{name}  {e:6.2f} min  {r}")

# 4. one line per replicate, one figure per concentration
for conc, conc_readings in readings.items():
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(3):
        ax.plot(elapsed_min, [r[i] for r in conc_readings], marker="o", label=f"Replicate {i + 1}")
    ax.set_xlabel("Time since first measurement (min)")
    ax.set_title(f"Fluorescence of {conc} ng/mL dsDNA | color = replicate")
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(folder), "plots", f"fluorescence_vs_time_{conc}.png"), dpi=300)
    plt.close(fig)
