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


# Опр. матрица А с полиномом q
def findA(q):
    A = np.eye(11 - 1, 11, k=1, dtype=np.int32)
    A = np.vstack((A, q))
    return A


# Возврат в первоначальное представление битпоследовательности wtr
def restoreWtr(A, D, S, y):
    N = len(y)
    x = np.zeros(N, dtype=np.int32)
    for k in range(N):
        x[k] = (y[k] + np.linalg.multi_dot([D, A, S[k]])) % 2
    xS = np.array2string(x).replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")
    return xS  # x


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
    # print(spline)
    return spline


# ??коэфицент А из max значения амплитуды или приемлемый набор параметров сводится к выбору A [0.1,0.5]
def cA(origin):
    coef = np.max(origin)
    return coef


# для увеличения
def u_plus(spline, coefA):
    u_plus = spline
    for t in range(spline):
        u_plus[t] = 1 + coefA * spline[t]
    return u_plus


# для уменьшения
def u_minus(spline, coefA):
    u_minus = spline
    for t in range(spline):
        u_minus[t] = 1 - coefA * spline[t]
    return u_minus


# Преобразование в массив по 160 - размер окна
def originM(origin, win_num, win):
    or_mass = np.zeros((win_num, win), dtype=np.int32)
    for k in range(win_num):  # from 0 to  by 160, the last is 176*160
        or_mass[k] = origin[k * 160:(k + 1) * 160]
    # print(or_mass)
    # print(np.shape(or_mass), type(or_mass[0, 0]))
    return or_mass


# Нахождение изменения в последовательности по мощности
def findY(norigin, origin, win_num, win):
    coefA = cA(origin)
    spline = B_spline(win)
    sqr_nor=np.int32(norigin**2)
    sqr_or= np.int32(origin**2)
    norM = originM(sqr_nor, win_num, win)
    norM2 = np.ones_like(norM)
    orM = originM(sqr_or, win_num, win)

    # находим сумму квадратов отсчетов в одном окне
    sum_nor = np.sum(norM, axis=1)
    sum_or = np.sum(orM, axis=1)
    #print(sum_nor, type(sum_nor[0]), np.size(sum_nor), type(sum_nor))
    #print(sum_or, len(sum_or), type(sum_or[0]))

    y = []
    yi = []
    for i in range(len(sum_nor)):
        if sum_or[i] > sum_nor[i]:
            y.append(1)
            yi.append(i*160)
            #norM2 = norM * u_plus(spline, coefA)
        elif sum_or[i] < sum_nor[i]:
            y.append(0)
            yi.append(i*160)
            #norM2 = norM * u_minus(spline, coefA)
        else:
            continue
    # print(y)
    return y, yi


def __init__():
    # 1+x^9+x^11
    q1 = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    A = findA(q1)
    D = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    S = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    fs1, norigin = sw.read("watermark.wav")
    fs2, origin = sw.read("voice.wav")

    win = int(fs1 / 100)
    win_num = int(np.shape(norigin)[0] / win)
    print(win_num)

    y, yi = findY(norigin, origin, win_num, win)
    print("y", y)
    print("yi",yi)
    # xS = restoreWtr(A, D, S, y)

    # print(xS, len(xS))

    # x = B_spline(win)

    # plt.figure()
    # plt.plot(x, 'k')
    # plt.show()


__init__()
