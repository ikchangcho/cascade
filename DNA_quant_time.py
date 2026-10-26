import csv
import glob
import os
import statistics
from datetime import datetime

import matplotlib.pyplot as plt

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "absorbances")

# 1. load all standard files
files = sorted(glob.glob(os.path.join(folder, "20261021_*_Ik_DNA_BR_standard.CSV")))
N = len(files)
print(f"N = {N} files")

times, readings = [], []
for f in files:
    name = os.path.basename(f)
    # 3. timestamp from filename: 20261021_16-21-18_...
    date_str, time_str = name.split("_")[:2]
    times.append(datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H-%M-%S"))

    # 2. row 10, columns B, C, D (spreadsheet coordinates -> 0-indexed row 9, cols 1-3)
    with open(f, newline="") as fh:
        rows = list(csv.reader(fh))
    readings.append([float(x) for x in rows[9][1:4]])

t0 = min(times)
elapsed_min = [(t - t0).total_seconds() / 60 for t in times]

for name, e, r in zip(map(os.path.basename, files), elapsed_min, readings):
    print(f"{name}  {e:6.2f} min  {r}")

# 4. scatter plot, three replicates
color = "tab:blue"
means = [statistics.mean(r) for r in readings]
sds = [statistics.stdev(r) for r in readings]

fig, ax = plt.subplots(figsize=(6, 4))
# for i in range(3):
#     ax.scatter(elapsed_min, [r[i] for r in readings], color=color, alpha=0.25, s=15)
ax.errorbar(elapsed_min, means, yerr=sds, color=color, marker="o", capsize=3)
ax.set_xlabel("Time since first measurement (min)")
ax.set_title(f"Flueoresence of 100 ng/mL standard | mean ± SD | faint points = replicates")
fig.tight_layout()
plt.savefig('plots/fluorescence_vs_time.png', dpi=300)
