import bitstring as bs
import numpy as np
import scipy.io.wavfile as sw


def s2bit(s):
    # print(len(s))
    sbytes = s.encode("utf-8")
    sbits = bs.BitArray(sbytes).bin
    N = len(sbits)
    return sbits, N


def origin2orbit(origin, Len):
    originbytes = np.zeros_like(origin, dtype=list)

    for i in range(Len[0]):
        originbytes[i] = (format(origin[i], 'b').replace("-", "").zfill(8))
    # print(originbytes[1],originbytes[1][len(originbytes[0])-1])
    # print(originbytes)
    return originbytes


def ADBS(N):
    A = np.eye(N - 1, N, k=1, dtype=np.int32)  # опр. матрица
    q = (-1 * np.random.randint(0, 2, (1, N), dtype=np.int32)) % 2  # многочлен
    q[0, 0] = 1
    # print("q", q)
    A = np.vstack((A, q))
    # print(A)

    # D = np.zeros(N, dtype=np.int64)  # вектор-столбец Nx1
    D = np.random.randint(0, 2, N, dtype=np.int32)
    D[0] = 1

    B = D.T  # вектор-строка 1xN
    # print(D, np.shape(D))
    # print(B, np.shape(B))

    S = np.ones((N + 1, N), dtype=np.int32)
    S[0] = np.random.randint(0, 2, (1, N), dtype=np.int32)  # ненулевой вектор-строка, 1xN
    # в дальнейшем чтобы получить столбец будем транспонировать
    if (S[0].sum() == 0):
        S[0, 0] = 1
    # print("S[0]", S[0], np.shape(S))
    return A, D, B, S


def convertWtr(A, D, B, S, sbits):
    N = len(sbits)
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
        S[k + 1] = temp1  # сохраняем опять в строку, но для y[k] он столбец
        y[k] = np.dot(D, temp1) % 2  # 1xN * Nx1 = число
        # print('y',y[k])
    return y, S


def restoreWtr(A, D, S, y):
    N = len(y)
    x = np.zeros(N, dtype=np.int32)
    for k in range(N):
        x[k] = (y[k] + np.linalg.multi_dot([D, A, S[k]])) % 2
    return x


def LPM1(s):
    sbits, N = s2bit(s)
    A, D, B, S = ADBS(N)
    y, S = convertWtr(A, D, B, S, sbits)
    x = restoreWtr(A, D, S, y)

    # print("A", A)
    # print("D", D)
    # print("B", B)
    # print("S", S)

    # print("y", y)
    # print("x", x)
    # print(sbits)
    return y, x


def LPM2(origin, Len, y, x):
    A1, D1, B1, S1 = ADBS(len(y))
    yy, S1 = convertWtr(A1, D1, B1, S1, x)
    # print("A", A1)
    # print("D", D1)
    # print("B", B1)
    # print("S", S1)

    originbytes = origin2orbit(origin, Len)
    print(np.size(origin))
    for k in range(np.shape(S1)[0]):
        yS = np.array2string(S1[k]).replace(" ", "").replace("[", "").replace("]", "").replace("\n", "")
        yS=int(yS,2)
        print(yS)
        #ТАКОГО ПРОСТО НЕ СУЩЕСТВУЕТ
        originbytes[yS] = int(originbytes[yS])
        orBitLen = len(originbytes[yS])
        print(originbytes[yS][orBitLen - 1],type(originbytes[yS][orBitLen - 1]))
        originbytes[yS][orBitLen - 1] = str(y[k])
    return originbytes


def __init__():
    fs, origin = sw.read("voice.wav")
    origin = np.int32(origin)
    shapes = np.shape(origin)

    s = "Natalya's watermark"
    ym, xm = LPM1(s)

    LPM2(origin, shapes, ym, xm)


#    orBytes=getAddress(origin, y1)
#    return orBytes

__init__()

"""
sbytes_d = int(sbits, 2).to_bytes(len(sbytes), 'big')  # начало с MSB
msg = sbytes_d.decode('utf-8')
"""
