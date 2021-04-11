import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def filtering(values=np.repeat([0., 1., 0.], 100)):
    real_values = np.real(values)
    imag_values = np.imag(values)
    #sig = np.repeat([0., 1., 0.], 100)
    print(real_values)
    print(imag_values)
    sig_temp_real = []
    sig_temp_imag = []
    dt = 10
    for x in values:
        for _ in range(dt):
            sig_temp_real.append(np.real(x))
            sig_temp_imag.append(np.imag(x))

    sig_real = np.array(sig_temp_real)
    sig_imag = np.array(sig_temp_imag)

    win = signal.windows.hann(dt)
    filtered_real = signal.convolve(sig_real, win, mode='same') / sum(win)
    filtered_imag = signal.convolve(sig_imag, win, mode='same') / sum(win)
    print(filtered_real)

    plt.figure()
    plt.plot(sig_real, 'o')
    plt.figure()
    plt.plot(sig_imag, 'o')
    plt.figure()
    plt.plot(win, 'o')
    plt.figure()
    plt.plot(filtered_imag, 'o')
    plt.figure()
    plt.plot(filtered_real, 'o')
    plt.show()
    return filtered_real, filtered_imag

if __name__ == "__main__":
    values = np.array([1.0+0j, 0.5+0.2j, 0.7-0.1j, 0.9+0j, 0.8, 0.3-0.3j, 0.1+0.5j])
    filtering(values=values)