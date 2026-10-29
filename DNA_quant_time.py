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

# 5. batch samples, six wells per plate row (columns 1-6), one line per well
for batch in ["batch1", "batch4"]:
    batch_files = sorted(glob.glob(os.path.join(folder, f"20261021_*_Ik_DNA_BR_{batch}.CSV")))
    batch_times, batch_rows = [], []
    for f in batch_files:
        date_str, time_str = os.path.basename(f).split("_")[:2]
        batch_times.append(datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H-%M-%S"))
        with open(f, newline="") as fh:
            batch_rows.append(list(csv.reader(fh)))

    batch_elapsed_min = [(t - min(batch_times)).total_seconds() / 60 for t in batch_times]

    for row_letter in "AE":
        line = 9 + "ABCDEFGH".index(row_letter)
        fig, ax = plt.subplots(figsize=(6, 4))
        for w in range(6):
            ax.plot(batch_elapsed_min, [float(rows[line][1 + w]) for rows in batch_rows],
                    marker="o", label=f"{row_letter}{w + 1:02d}")
        ax.set_xlabel("Time since first measurement (min)")
        ax.set_title(f"Fluorescence of {batch} {row_letter}01-{row_letter}06 | color = well")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(folder), "plots", f"fluorescence_vs_time_{batch}_{row_letter}01-{row_letter}06.png"), dpi=300)
        plt.close(fig)
