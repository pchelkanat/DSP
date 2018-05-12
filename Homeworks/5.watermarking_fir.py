import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sgn


# PART 1
def createSignal(Fs, myfreqs):
    t = np.linspace(0, 1, Fs)
    waves = np.zeros((len(myfreqs), Fs))
    for i in range(len(myfreqs)):
        waves[i] = np.sin(2 * np.pi * myfreqs[i] * t)
    totalwave = np.sum(waves, axis=0)
    # print(totalwave)
    return totalwave

# PART 2


def __init__():
#PART 1
    Fs = 200
    myfreqs = [15, 40, 60, 90]
    order = 50
    #f=myf/Fs*2
    freq = [0, 0.15, 0.4, 0.6, 0.9, 1]
    gain = [0, 0, 1, 1, 0, 0]

    totalwave = createSignal(Fs, myfreqs)
    b = sgn.firwin2(order, freq, gain)

    w, h = sgn.freqz(b)
    sig_ff = sgn.filtfilt(b, 1, totalwave)
    # sig_lf = sgn.lfilter(b, 1, totalwave)

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(w / np.pi, abs(h), label="Передаточная функция фильтра")
    plt.legend()
    plt.title("PART #1")

    plt.subplot(2, 1, 2)
    plt.plot(totalwave, label="Оригинал")
    plt.plot(sig_ff, label="Фильтрованный")
    # plt.plot(sig_lf)
    plt.legend()

#PART 2


    plt.show()


__init__()
