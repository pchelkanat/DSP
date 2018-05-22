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
    norigin = filter
    for i in range(len(wtrmark)):
        norigin[i + Pos] += wtrmark[i] * coef

    return norigin


def FreqResponse(a, b, b_new):
    w, h = sgn.freqz(b, a)  # учтойчив
    w1, h1 = sgn.freqz(a, b_new)  # устойчивость пропала

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(w / np.pi, abs(h), label="Передат. функция")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(w1 / np.pi, abs(h1), label="Передат. функция new")
    plt.legend()


def __init__():
    fs, origin = sw.read("voice.wav")
    # origin = np.float64(origin)
    coef = np.max(origin)
    Pos = 123456
    order = 15  # для устойчивости значения поменьше, но >=11
    N = 255
    n = 100

    origin2 = origin[Pos:Pos + 2 * N]
    shape = np.shape(origin2)[0]
    # print(shape)
    alphaM = alphaMatr(origin2, shape, N)
    Matr = MatrA(alphaM, shape, N)
    bth = betha(Matr)

    b, a = sgn.iirfilter(order, 0.3, btype="low")
    # b, a = sgn.butter(order, 0.3, btype="low")
    # print("b", b)
    # print("a", a)
    c = np.random.random(order)

    pc = np.poly1d(c, True)
    b_new = pc.coeffs.real

    FreqResponse(a, b, b_new)

    y = sgn.lfilter(b_new, a, origin)  # фильтрация
    # print(np.shape(y), type(y[1]))
    norigin = Watermarking(y, Pos, bth, coef)
    # print(np.shape(norigin),type(norigin[1]))
    corr1 = np.correlate(norigin, bth, "valid")
    MyPos1 = np.argmax(corr1)  # обнаружил
    print(MyPos1)

    v = sgn.lfilter(a, b_new, norigin)  # сокрытие обратным
    # print(np.shape(v), type(v[1]))
    corr2 = np.correlate(v, bth, "valid")
    MyPos2 = np.argmax(corr2)  # обнаружил
    print(MyPos2)

    u = sgn.lfilter(b_new, a, v)  # для обнаружения
    # print(np.shape(u), type(u[1]))
    corr3 = np.correlate(u, bth, "valid")
    MyPos3 = np.argmax(corr3)  # обнаружил
    print(MyPos3)

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(origin[Pos - n:Pos + N + n], label="Оригинал")
    plt.legend()
    plt.subplot(4, 1, 2)
    # plt.plot(y[Pos-n:Pos + N + n], label="Фильтр y")
    # plt.legend()
    # plt.subplot(5, 1, 3)
    plt.plot(norigin[Pos - n:Pos + N + n], label="Фильтр + wtrmark")
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(v[Pos - n:Pos + N + n], label="Обратный фильтр v")
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(u[Pos - n:Pos + N + n], label="Для определения wtrmark u")
    plt.legend()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(corr1[Pos - n:Pos + N + n], label="Коррелляция norigin\n" + str(MyPos1))
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(corr2[Pos - n:Pos + N + n], label="Корр. обр. фильтра v\n" + str(MyPos2))
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(corr3[Pos - n:Pos + N + n], label="Коррелляция u\n" + str(MyPos3))
    plt.legend()
    plt.show()


__init__()
