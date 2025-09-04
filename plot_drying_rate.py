#!/usr/bin/env python3
import matplotlib.pyplot as plt

# # Data
# time_days = [
#     0,
#     0.966666667,
#     1.965277778,
#     3.882638889,
#     5.840972222,
#     6.8125,
#     8.814583333,
#     9.804166667,
#     12.06111111,
#     13.9375,
#     14.84722222,
#     16.97291667,
#     18.75763889,
#     21.16736111,
#     22.96041667,
# ]
# water_pink = [
#     88.88888889,
#     84.77037037,
#     80.68148148,
#     71.91111111,
#     65.36296296,
#     61.00740741,
#     53.33333333,
#     48.82962963,
#     40.38518519,
#     31.97037037,
#     27.79259259,
#     25.6,
#     24.5037037,
#     23.08148148,
#     21.98518519,
# ]
# water_white = [
#     89.15555556,
#     85.27407407,
#     81.6,
#     73.98518519,
#     68.32592593,
#     64.71111111,
#     58.1037037,
#     54.07407407,
#     46.84444444,
#     38.99259259,
#     35.2,
#     32.88888889,
#     31.58518519,
#     30.07407407,
#     28.74074074,
# ]

# def main():
#     plt.figure(figsize=(8, 5))
#     plt.plot(time_days, water_pink, marker='o', label='Water in Pink (%WHC)', color='red')
#     plt.plot(time_days, water_white, marker='o', label='Water in White (%WHC)', color='black')

#     plt.title('Drying Curve of La Bagh Wood Soil 45.45 g', fontsize=15)
#     plt.xlabel('Time (day)', fontsize=15)
#     plt.ylabel('Water Content (%WHC)', fontsize=15)

#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.legend(fontsize=15)
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()

# # Additional plot for dishes 1-10
# time_days2 = [0, 1.976388889, 3.761111111, 6.170833333, 7.963888889]

# dish_data = {
#     1: [120, 116.6518519, 115.0518519, 113.037037, 111.3185185],
#     2: [120, 116.8, 115.2888889, 112.3851852, 110.9925926],
#     3: [120, 116.4444444, 114.6074074, 112.4444444, 110.6074074],
#     4: [120, 116.8592593, 115.2592593, 113.362963, 111.8814815],
#     5: [120, 115.762963, 113.4518519, 110.5777778, 108.0592593],
#     6: [120, 115.8814815, 114.0148148, 111.6740741, 109.8074074],
#     7: [120, 114.5481481, 111.0814815, 106.9037037, 102.9037037],
#     8: [120, 115.0814815, 112.4444444, 109.2740741, 105.1259259],
#     9: [120, 109.5111111, 101.0962963, 91.37777778, 81.68888889],
#     10:[120, 112.2962963, 106.3111111, 99.2, 91.76296296],
# }

# pair_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# plt.figure(figsize=(8,5))
# for dish, values in dish_data.items():
#     color = pair_colors[(dish - 1) // 2]
#     marker = 'o' if dish % 2 == 1 else 'D'
#     plt.plot(time_days2, values, marker=marker, label=f'Dish #{dish}', color=color)

# plt.title('Drying Curve of La Bagh Wood Soil 50 g', fontsize=15)
# plt.xlabel('Time (day)', fontsize=15)
# plt.ylabel('Water Content (%WHC)', fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.grid(alpha=0.3)

# legend = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=15)
# plt.tight_layout()
# plt.show()

