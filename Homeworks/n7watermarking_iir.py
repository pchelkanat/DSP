import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as sw
import scipy.signal as sgn


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
    return betha


def Watermarking(filter, Pos, wtrmark, coef):
    print("Watermarking ...")
    norigin = filter
    for i in range(len(wtrmark)):
        norigin[i + Pos] += wtrmark[i] * coef

    return norigin


def FreqResponse(a, b, b_new):
    w, h = sgn.freqz(b, a)
    w1, h1 = sgn.freqz(a, b_new)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(w / np.pi, abs(h), label="Передат. функция")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(w1 / np.pi, abs(h1), label="Передат. функция new")
    plt.legend()


def __init__():
    fs, origin = sw.read("voice.wav")
    coef = 5  # np.max(origin)
    origin = np.float64(origin)
    origin = origin / (2 ** 15)

    Pos = 123456
    order = 15  # значения >=11 по условию
    N = 127

    shape = np.shape(origin)[0]
    Matr = MatrA(origin, shape, N)
    wtrmark = betha(Matr).real  # чтобы не были комплексными

    b, a = sgn.iirfilter(order, 0.3, btype="lowpass")
    # b, a = sgn.butter(order, 0.3, btype="low")
    c = np.random.random(order)
    # print("b", b)
    # print("a", a)
    # print("c", c)

    pc = np.poly1d(c, True)  # корни <1
    b_new = pc.coeffs.real
    # print("b_new", b_new)

    FreqResponse(a, b, b_new)

    y = sgn.lfilter(b_new, a, origin)  # фильтрация
    norigin = Watermarking(y, Pos, wtrmark, coef)
    corr1 = np.correlate(norigin, wtrmark, "valid")
    MyPos1 = np.argmax(corr1)  # обнаружил
    print(MyPos1)

    v = sgn.lfilter(a, b_new, norigin)  # сокрытие обратным
    corr2 = np.correlate(v, wtrmark, "valid")
    MyPos2 = np.argmax(corr2)  # не обнаружил
    print(MyPos2)

    u = sgn.lfilter(b_new, a, v)  # для обнаружения
    corr3 = np.correlate(u, wtrmark, "valid")
    MyPos3 = np.argmax(corr3)  # обнаружил
    print(MyPos3)

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(origin[Pos - 100:Pos + 100], label="Оригинал")
    plt.legend()
    plt.subplot(4, 1, 2)
    # plt.plot(y[Pos-n:Pos + N + n], label="Фильтр y")
    # plt.legend()
    # plt.subplot(5, 1, 3)
    plt.plot(norigin[Pos - 100:Pos + 100], label="Фильтр + wtrmark")
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(v[Pos - 100:Pos + 100], label="Обратный фильтр v")
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(u[Pos - 100:Pos + 100], label="Для определения wtrmark u")
    plt.legend()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(corr1[Pos - 100:Pos + 100], label="Коррелляция norigin\n" + str(MyPos1))
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(corr2[Pos - 100:Pos + 100], label="Корр. обр. фильтра v\n" + str(MyPos2))
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(corr3[Pos - 100:Pos + 100], label="Коррелляция u\n" + str(MyPos3))
    plt.legend()
    plt.show()


__init__()
