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
    # print("betha", betha)
    return betha


def Watermarking(filter, Pos, wtrmark, coef):
    print("Watermarking ...")
    norigin = np.float64(filter)
    for i in range(len(wtrmark)):
        norigin[i + Pos] += wtrmark[i]

    return norigin


def __init__():
    fs, origin = sw.read("voice.wav")
    # origin = np.int32(origin)
    coef = np.max(origin)
    Pos = 12345
    N = 55
    n = 100

    origin2 = origin[Pos:Pos + 2 * N]
    shape = np.shape(origin2)[0]
    # print(shape)
    alphaM = alphaMatr(origin2, shape, N)
    Matr = MatrA(alphaM, shape, N)
    bth = betha(Matr)

    b, a = sgn.iirfilter(N,0.7,btype='highpass')
    #b, a = sgn.butter(N, 0.7, "highpass")
    # print("b", b)
    # print("a", a)
    c = np.random.random(N)

    pc = np.poly1d(c, True)
    b_new = pc.coeffs.real

    y = sgn.filtfilt(b_new, a, origin)
    # print(np.shape(y), y)
    norigin = Watermarking(y, Pos, bth, coef)
    corr1 = np.correlate(norigin, bth, "valid")

    v = sgn.filtfilt(a, b_new, norigin)
    corr2 = np.correlate(v, bth, "valid")

    u = sgn.filtfilt(b_new, a, v)
    corr3 = np.correlate(u, bth, "valid")
    MyPos = np.argmax(corr3)

    w, h = sgn.freqz(b, a)
    w1, h1 = sgn.freqz(a, b_new)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(w / np.pi, abs(h), label="Передат. функция")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(w1 / np.pi-1, abs(h1), label="Передат. функция new")
    plt.legend()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(origin[Pos - n:Pos + N + n], label="Оригинал")
    plt.plot(norigin[Pos - n:Pos + N + n], label="Фильрованный c wtrmark")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(v[Pos - n:Pos + N + n], label="Фильтрованный скрытый wtrmark")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(corr1, label="Коррелляция")
    plt.plot(corr2, label="Коррелляция фильтрованного")
    plt.plot(corr3, label="Коррелляция\nПозиция" + str(MyPos))
    plt.legend()
    plt.show()


__init__()
