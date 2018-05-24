import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as sw
import scipy.signal as sgn


# Нахождение матрицы A по векторам альфа
def MatrA(origin, shape, N):
    Matr = np.zeros((N, N))
    print("Find Matr ...")
    for i in range(shape - N + 1):
        alphaM = np.array(origin[i:i + N])
        # print(i)
        Matr += np.dot(alphaM.T, alphaM)
    # print("Matr", Matr)
    return Matr


def betha(Matr):
    print("Find betha ...")
    vals, vecs = np.linalg.eig(Matr)
    # print("vals", vals)
    # print("vecs", vecs)
    betha = vecs[:, -1]
    # print("betha", betha)
    return betha


def Watermarking(origin, Pos, wtrmark, coef):
    norigin = origin
    for i in range(len(wtrmark)):
        norigin[i + Pos] += wtrmark[i] * coef

    return norigin


def __init__():
    fs, origin = sw.read("voice.wav")
    coef = 5  # np.max(origin)
    origin = np.float64(origin)
    origin = origin / (2 ** 15)  # нормировка необходима во избежание переполнения памяти

    Pos = 123456
    N = 127  # 255 за 3 минуты, 127 за 2 минуты

    shape = np.shape(origin)[0]
    # print(shape)
    Matr = MatrA(origin, shape, N)
    bth = betha(Matr)

    w, h = sgn.freqz(bth)
    w1, h1 = sgn.freqz(bth[::-1])

    norigin = Watermarking(origin, Pos, bth, coef)
    corr = np.correlate(norigin, bth, "valid")
    MyPos = np.argmax(corr)
    print("My Position", MyPos)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(w / np.pi, abs(h), label="Передат. функция betha")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(w1 / np.pi, abs(h1), label="Передат. функция reverse betha")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(corr, label="Корреляция\nПозиция: " + str(MyPos))
    plt.legend()
    plt.show()


__init__()
