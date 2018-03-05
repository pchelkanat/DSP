import matplotlib.pyplot as plt
import numpy as np

N = 100  # 500 #1000
quantBits = 8  # 16 #32
quantLevels = 2 ** quantBits / 2
quantStep = 1. / quantLevels


def analog(N):
    # генерация, построение, нормализация [-1; 1]
    x = np.random.randn(N)
    x = np.cumsum(x)
    x = x / (np.max(np.abs(x)))
    # print(np.size(x))
    return x


def digital(signal):
    # округление в большую сторону?? или обычное
    y = np.ceil(signal / quantStep) * quantStep;
    # y   = np.round (x / quantStep) * quantStep;
    # print(np.size(y))
    return y


def snrPr(x, y):
    e, snr = [], []
    for i in range(N):
        e.append(x[i] - y[i])

    dx = np.var(x)
    de = np.var(e)
    snr = 10 * np.log10(np.abs(dx / de))
    return snr


def snrTh(quantBits):
    # теоритическое значение
    # 16 bit ~ 90 dB
    return (6 * quantBits - 7.2)


##   MAIN
x = analog(N)
y = digital(x)

print("Variance of analog: ", np.var(x))
print("Variance of digital: ", np.var(y))

print("Theorecical snr: ", snrTh(quantBits))
print("Practical snr: ", snrPr(x, y))

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(x, label="Analog")
plt.plot(y, label="Digital")
plt.legend()
plt.title("Analog to digital signal conversion")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(y, label=str(quantBits) + " bits")
plt.stem(y)
plt.legend()

plt.show()
