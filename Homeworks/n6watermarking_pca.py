import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as sw
import scipy.signal as sgn


def alphaMatr(origin, shape, N):
    alphaM = np.zeros((1, N), dtype=np.float64)
    print("Find alphaM ...")
    for i in range(shape - N + 1):
        alphaM = np.vstack((alphaM, origin[i:i + N]))
        # print(i)
    alphaM = alphaM[1:]
    # print("alphaM",alphaM)
    return alphaM


def MatrA(alphaM, shape, N):
    Matr = np.zeros((N, N))
    print("Find Matr ...")
    for i in range(shape - N + 1):
        temp = np.array([alphaM[i]])  # чтобы можно было умножать .T
        # print(i)
        Matr += np.dot(temp.T, temp)
    # print("Matr", Matr)
    return Matr


def betha(Matr):
    print("Find betha ...")
    vals, vecs = np.linalg.eig(Matr)
    # print("vals", vals)
    # print("vecs", vecs)
    betha = vecs[:, -1]
    print("betha", betha)
    return betha


def Watermarking(origin, Pos, N, wtrmark, coef):
    norigin = origin
    for i in range(N):
        norigin[i + Pos] += wtrmark[i] * coef

    return norigin


def __init__():
    fs, origin = sw.read("voice.wav")
    origin = np.int64(origin)
    coef = np.max(origin)
    Pos = 123456
    N = 127
    n = 100

    origin2 = origin[Pos:Pos + 2 * N]
    shape = np.shape(origin2)[0]
    # print(shape)

    alphaM = alphaMatr(origin2, shape, N)
    Matr = MatrA(alphaM, shape, N)
    bth = betha(Matr)

    w, h = sgn.freqz(bth)
    w1, h1 = sgn.freqz(bth[::-1])
    norigin = Watermarking(origin, Pos, N, bth, coef)
    corr = np.correlate(norigin, bth, "valid")
    MyPos = np.argmax(corr)
    print("My Position", MyPos)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(w / np.pi, abs(h), label="Передат. функция")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(w1 / np.pi, abs(h1), label="Передат. функция reverse")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(corr[Pos - n:Pos + N + n], label="Корреляция\nПозиция: " + str(MyPos))
    plt.legend()

    plt.figure()
    plt.plot(norigin[Pos - n:Pos + N + n], label="Внедренный wtrmark")
    plt.plot(origin[Pos - n:Pos + N + n], label="Оригинал")
    plt.legend()
    plt.show()


__init__()
