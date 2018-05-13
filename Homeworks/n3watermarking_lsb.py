import bitstring as bs
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as sw


# Преобразование wtr в бинарный вид
def s2bit(s):
    # print(len(s))
    sbytes = s.encode("utf-8")
    sbits = bs.BitArray(sbytes).bin
    N = len(sbits)
    return sbits, N


# Преобразование амплитуд в бинарный вид
def origin2orbit(origin, Len):
    originbytes = np.zeros_like(origin, dtype=list)

    for i in range(Len[0]):
        originbytes[i] = (format(origin[i], 'b').replace("-", "").zfill(8))
    # print(originbytes[1],originbytes[1][len(originbytes[0])-1])
    # print("orbyte",originbytes, type(originbytes), type(originbytes[0]))
    return originbytes


# Компоненты
def ADBS(q):
    A = np.eye(11 - 1, 11, k=1, dtype=np.int32)  # опр. матрица
    A = np.vstack((A, q))
    # print(A)

    # D = np.zeros(N, dtype=np.int64)  # вектор-строка 1хN
    D = np.random.randint(0, 2, 11, dtype=np.int32)
    D[0] = 1

    B = D.T  # вектор-столбец Nх1
    # print(D, np.shape(D))
    # print(B, np.shape(B))

    S = np.random.randint(0, 2, (1, 11),
                          dtype=np.int32)  # ненулевой вектор-столбец, Nх1. Для представления можно использовать строки.
    if (S[0].sum() == 0):
        S[0, 0] = 1
    # print("S[0]",S[0], np.shape(S))
    return A, D, B, S


# Преобразование wtr в Y
def convertWtr(A, D, S, sbits):
    B=D.T
    N = len(sbits)
    # print(N)
    y = np.zeros(N, dtype=np.int32)
    for k in range(N):
        # print(np.shape(A), np.shape(S[k]), np.shape(B))
        # print("ss", S[k],np.shape(S[k]))
        # print(k, sbits[k])
        temp1 = (np.dot(A, S[k]) + (B * int(sbits[k]))) % 2  # NxN * Nx1 = Nx1, Nx1 * число = Nx1
        """
        t1 = np.dot(A, S[k].T)
        t2 = B * int(sbits[k])
        print('t1', t1, np.shape(t1))
        print('t2', t2, np.shape(t2))
        """
        # print("temp", temp1, np.shape(temp1))
        S = np.vstack((S, temp1))  # сохраняем в строку, но для y[k] он считается столбцом
        y[k] = np.dot(D, temp1) % 2  # 1xN * Nx1 = число
    # print('y',y,S, np.shape(S))
    return y, S


# Находим позицию, младший бит которой нужно заменить
def findPosition(A, S, sbits):
    N = len(sbits)
    pos = list()
    for k in range(N):
        temp2 = (np.dot(A, S[k])) % 2
        S = np.vstack(((S, temp2)))

    for k in range(1, N + 1, 1):
        st = ""
        # print(S[k])
        for i in range(len(S[k])):
            st = st + str(S[k, i])
            # print("st",st,type(st))
        st = int(st, 2)
        # print(st, type(st))
        pos.append(st)
    # print(pos)
    # print(len(pos))
    return pos


# Восстановление бинарной последовательности Х wtr
def restoreWtr(A, D, S, y):
    N = len(y)
    x = np.zeros(N, dtype=np.int32)
    for k in range(N):
        x[k] = (y[k] + np.linalg.multi_dot([D, A, S[k]])) % 2
    xS = np.array2string(x).replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")
    return xS


def LPM1(s, q):
    sbits, N = s2bit(s)
    # print(sbits, len(sbits))
    A, D, B, S = ADBS(q)
    y, S = convertWtr(A, D, S, sbits)
    # x = restoreWtr(A, D, S, y)

    # преобразование массива в последовательность
    yS = np.array2string(y).replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")
    # print("A", A)
    # print("D", D)
    # print("B", B)
    # print("S", S)

    # print("y", yS)
    # print("x", x)
    # print(sbits)
    return yS, sbits


def LPM2(origin, Len, y, x, q):
    A1, D1, B1, S1 = ADBS(q)
    pos = findPosition(A1, S1, x)

    newbytes = origin2orbit(origin, Len)
    for k in range(len(pos)):
        temp = newbytes[pos[k]]
        temp = temp[- 1] + y[k]
        newbytes[pos[k]] = temp

    new = np.zeros_like(newbytes, dtype=np.int32)
    for i in range(Len[0]):
        if origin[i] < 0:
            new[i] = -1 * int(newbytes[i], 2)
        else:
            new[i] = int(newbytes[i], 2)
    # print(new, type(new), type(new[0]))

    for i in range(Len[0]):
        newbytes[i] = int(newbytes[i])

    return new, newbytes


def __init__():
    fs, origin = sw.read("voice.wav")
    origin = np.int32(origin)

    s = "pchelkanat's watermark"
    # print(len(list(s)), len(s))

    # q1,q2 - последние 2 на стр 261; степень M=11
    q1 = np.array([[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]])
    q2 = np.array([[1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1]])

    ym, sbits = LPM1(s, q1)
    # print(ym)
    #print(sbits)
    # print()

    newwave, newwavebyte = LPM2(origin, np.shape(origin), ym, sbits, q2)
    print("newwavebyte", newwavebyte)

    sbits=list(sbits)
    for i in range(len(sbits)):
        sbits[i]=int(sbits[i])
    print ("wtrmark in bits",sbits)

    corr = np.correlate(newwavebyte, sbits, "valid")
    # print(len(corr), corr)

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(origin, label="Origin")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(newwave, label="With wtrmark")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(corr, label="Correlation")
    plt.legend()
    plt.show()


__init__()
