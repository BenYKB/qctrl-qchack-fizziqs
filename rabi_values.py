import matplotlib.pyplot as plt
import numpy as np

rabi_values = [2.0 * np.pi / 45, 2.0 * np.pi / 27, 2.0 * np.pi / 20]  # rad / ns
amplitudes = [0.4, 0.7, 1]

plt.plot(amplitudes, rabi_values, 'o')

m, b = np.polyfit(amplitudes, rabi_values, 1)

print(m, b)

x = np.linspace(0, 1, 10)
y = m*x+b

plt.plot(x, y)

plt.show()
