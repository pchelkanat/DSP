import bitstring as bs
import numpy as np
import scipy.io.wavfile as sw


# Преобразование wtr в бинарный вид
def s2bit(s):
    # print(len(s))
    sbytes = s.encode("utf-8")
    sbits = bs.BitArray(sbytes).bin
    N = len(sbits)
    return sbits, N


def findA(q):
    A = np.eye(11 - 1, 11, k=1, dtype=np.int32)  # опр. матрица
    A = np.vstack((A, q))
    return A


def restoreWtr(A, D, S, y):
    N = len(y)
    x = np.zeros(N, dtype=np.int32)
    for k in range(N):
        x[k] = (y[k] + np.linalg.multi_dot([D, A, S[k]])) % 2
    xS = np.array2string(x).replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")
    return xS  # x


# Преобразование в массив по 160
def originM(origin, Size, win):
    or_mass = np.zeros((Size, win), dtype=np.int32)
    for k in range(Size):  # from 0 to  by 160, the last is 176*160
        or_mass[k] = origin[k * 160:(k + 1) * 160]
    # print(or_mass)
    # print(np.shape(or_mass), type(or_mass[0, 0]))
    return or_mass


# Нахождение изменения в последовательности
def findY(sqror1, sqror2, Len, win):
    orM1 = originM(sqror1, Len, win)
    orM2 = originM(sqror2, Len, win)
    sum1 = np.sum(orM1, axis=1)
    sum2 = np.sum(orM2, axis=1)
    #print(sum1, len(sum1), np.size(sum1), type(sum1))
    #print(sum2, len(sum2), type(sum2[0]))

    y=[]
    for i in range(len(sum1)):
        if sum2[i]>sum1[i]:
            y.append(1)
        elif sum2[i]<sum1[i]:
            y.append(0)
        else:
            continue
    print(y)
    return y


# B-Spline
def B_spline(win):
    t = np.linspace(0, 1, win)
    # (t, np.size(t))
    spline = []
    for i in range(160):
        if 0 <= t[i] <= 1 / 3:
            spline.append(9 / 2 * t[i] ** 2)
        elif 1 / 3 < t[i] <= 2 / 3:
            spline.append(-9 * t[i] ** 2 + 9 * t[i] - 3 / 2)
        elif 2 / 3 < t[i] <= 1:
            spline.append(9 / 2 * (1 - t[i]) ** 2)
    #print(spline)
    return spline


def u_plus(spline, coefA):
    u_plus = spline
    for t in range(spline):
        u_plus[t] = 1 + coefA * spline[t]
    return u_plus


def u_minus(spline, coefA):
    u_minus = spline
    for t in range(spline):
        u_minus[t] = 1 - coefA * spline[t]
    return u_minus


def __init__():
    # 1+x^9+x^11
    q1 = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    A = findA(q1)
    D = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    S = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    fs1, origin1 = sw.read("watermark.wav")
    fs2, origin2 = sw.read("voice.wav")
    win = int(fs1 / 100)
    Size = int(np.shape(origin1)[0] / win)
    print(Size)

    sqr_origin1 = np.int32(origin1 ** 2)  # sqr for the power
    sqr_origin2 = np.int32(origin2 ** 2)

    y = findY(sqr_origin1, sqr_origin2, Size, win)

    #xS = restoreWtr(A, D, S, y)

    # print(xS, len(xS))

    # x = B_spline(win)

    # plt.figure()
    # plt.plot(x, 'k')
    # plt.show()


__init__()
