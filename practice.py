import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot 1 / (1 - np.exp(-x)) - (1 / x)
x = np.linspace(-10, 10, 1000)  # Avoid x=0
y = 1 / (1 - np.exp(-x)) - (1 / x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('1 / (1 - exp(-x)) - 1 / x')
ax.grid(True)
plt.show()