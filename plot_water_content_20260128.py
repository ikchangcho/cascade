import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Create an array of the intervals between Nov 24 12:56, Dec 1 13:00, Dec 8 9:35, Dec 15 12:58, Dec 19 11:00, Jan 14 11:15, in days
datetime_array = [
    datetime(2024, 11, 24, 12, 56),
    datetime(2024, 12, 1, 13, 0),
    datetime(2024, 12, 8, 9, 35),
    datetime(2024, 12, 15, 12, 58),
    datetime(2024, 12, 19, 11, 0),
    datetime(2025, 1, 14, 11, 15),
    datetime(2025, 2, 23, 11, 37)]
times = [0]
for i in range(1, len(datetime_array)):
    time_diff = datetime_array[i] - datetime_array[0]
    times.append(time_diff.total_seconds() / 3600 / 24)

water_contents =  [98.9, 62.5, 34.4, 7.44, 7.10, 5.16, 4.74]

plt.figure()
sizes = [50, 50, 50, 10, 50, 50, 50]
plt.scatter(times, water_contents, s=sizes)
plt.plot(times, water_contents, linestyle='--', alpha=0.5)
plt.xlabel("Time (days)", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylabel("Water content (%whc)", fontsize=14)
plt.tight_layout()
plt.show()
