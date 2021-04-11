import matplotlib.pyplot as plt
import numpy as np
import scipy_filter as sf


def planck_taper_window(input_array):
    return_array = np.copy(input_array)
    return_array[0] = 0
    return_array[-1] = 0

    eps = 0.05
    N = np.size(input_array)

    for n in range(1, int(np.floor(eps * N))):
        coefficient = (1 + np.exp(eps * N / n - eps * N / (eps * N - n))) ** -1
        return_array[n] = coefficient * input_array[n]
        return_array[-1 * n - 1] = coefficient * input_array[-1 * n - 1]

    return return_array


if __name__ == "__main__":
    values = np.array([3.0, 9.0, 2.0, 4.0, 1.0, 8.0])
    plt.figure()
    plt.plot(values)
    filtered = sf.filtering(values=values)
    plt.figure()
    plt.plot(filtered, '-')
    tapered = planck_taper_window(filtered)
    plt.figure()
    plt.plot(tapered, '-')
    plt.show()
