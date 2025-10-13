import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

time_days2 = [0, 1.970138889, 4.020138889, 6.916666667, 10.19305556, 13.14236111, 16.07361111, 19.04236111]

dish_data = {
    1: [99.61481481, 89.00740741, 78.25185185, 64.2962963, 48.5037037, 34.93333333, 22.51851852, 10.51851852], 
    2: [99.7037037, 89.54074074, 79.11111111, 64.68148148, 49.12592593, 35.46666667, 23.05185185, 11.55555556],
    3: [99.73333333, 89.33333333, 78.54814815, 63.55555556, 47.37777778, 33.92592593, 21.27407407, 9.511111111],
    4: [99.82222222, 89.54074074, 78.37037037, 63.55555556,	48.35555556, 35.43703704, 23.55555556, 12.38518519],
    5: [99.91111111, 88.91851852, 77.0962963, 60.88888889, 43.4962963, 28.47407407, 13.57037037, 2.311111111]
}

pair_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

plt.figure(figsize=(8,5))
for dish, values in dish_data.items():
    color = pair_colors[(dish - 1) // 1]
    marker = 'o' #if dish % 2 == 1 else 'D'
    plt.plot(time_days2, values, marker=marker, label=f'Dish #{dish}', color=color)

plt.title('Drying Curve of La Bagh Wood Soil 50 g', fontsize=15)
plt.xlabel('Time (day)', fontsize=15)
plt.ylabel('Water Content (%WHC)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(alpha=0.3)
plt.legend()
#legend = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=15)
plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt

# def load_sensor_data(path):
#     # Data start at row 3; temperature = 3rd col, RH = 4th col
#     df = pd.read_csv(path, header=None, skiprows=2)
#     temperature = pd.to_numeric(df.iloc[:, 2], errors='coerce')
#     humidity = pd.to_numeric(df.iloc[:, 3], errors='coerce')
#     time = pd.to_timedelta(range(len(df)), unit='min')
#     return time, temperature, humidity

# sensor_files = [
#     ("ThermoProSensor_export_TP351S-1_09032025.csv", "Sensor 1"),
#     ("ThermoProSensor_export_TP351S-2_09032025.csv", "Sensor 2"),
# ]

# series = []
# for fname, label in sensor_files:
#     fpath = Path(fname)
#     if not fpath.exists():
#         print(f"Warning: {fname} not found, skipping.")
#         continue
#     t, temp, rh = load_sensor_data(fpath)
#     series.append((label, t, temp, rh))

# if not series:
#     raise SystemExit("No sensor data loaded.")

# # Convert time to hours for nicer axis scaling
# def to_days(td_index):
#     return [td.total_seconds() / 3600 / 24 for td in td_index]

# plt.figure(figsize=(9, 4.5))
# for label, t, temp, _ in series:
#     plt.plot(to_days(t), temp, label=label)
# plt.xlabel("Time (days)", fontsize=15)
# plt.ylabel("Temperature (°C)", fontsize=15)
# plt.ylim(25.7, 26.7)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(fontsize=15)
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(9, 4.5))
# for label, t, _, rh in series:
#     plt.plot(to_days(t), rh, label=label)
# plt.xlabel("Time (days)", fontsize=15)
# plt.ylabel("Relative Humidity (%)", fontsize=15)
# plt.ylim(75, 100)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(fontsize=15)
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()
