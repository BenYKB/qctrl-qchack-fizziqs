import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def filtering(values=np.repeat([0., 1., 0.], 100)):
    #sig = np.repeat([0., 1., 0.], 100)
    print(values)
    sig_temp = []
    dt = 10
    for x in values:
        for _ in range(dt):
            sig_temp.append(x)
    sig = np.array(sig_temp)

    win = signal.windows.hann(dt)
    filtered = signal.convolve(sig, win, mode='same') / sum(win)
    print(filtered)

    plt.figure()
    plt.plot(sig, 'o')
    plt.figure()
    plt.plot(win, 'o')
    plt.figure()
    plt.plot(filtered, 'o')
    plt.show()

if __name__ == "__main__":
    values = np.array([1.0, 0.5, 0.7, 0.9, 0.8, 0.3, 0.1])
    filtering(values=values)