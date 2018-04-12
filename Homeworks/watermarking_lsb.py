import bitstring as bs
import numpy as np
import scipy.io.wavfile as sw

fs, origin = sw.read("voice.wav")
origin = np.int64(origin)
Len = np.shape(origin)
print(origin, Len)

s = "Natalya's watermark"
# print(len(s))
sbytes = s.encode("utf-8")
sbits = bs.BitArray(sbytes).bin
N = len(sbits)

# def Origin2Orbit(origin):
originbytes = np.zeros_like(origin, dtype=list)
for i in range(Len[0]):
    originbytes[i] = list(format(origin[i], 'b').replace("-", "").zfill(16))
# print(originbytes, type(originbytes))
# return originbytes



A = np.eye(10 - 1, 10, k=1, dtype=np.int64)  # опр. матрица
q = (-1 * np.random.randint(0, 2, (1, 10), dtype=np.int64)) % 2  # многочлен
q[0, 0] = 1
A = np.vstack((A, q))
# print(A)

D = np.zeros((10, 1), dtype=np.int64)  # вектор-столбец Nx1
D[0][0] = 1
B = D.T  # вектор-строка 1xN
print(D, np.shape(D))
print(B, np.shape(B))

S = np.ones((11, 10), dtype=np.int64)
S[0] = np.random.randint(0, 2, (1, 10), dtype=np.int64)  # ненулевой вектор-строка, 1xN
# в дальнейшем чтобы получить столбец будем транспонировать
if (S[0].sum() == 0):
    S[0, 0] = 1
print()

y = np.zeros((1, 10), dtype=np.int64)
x = np.zeros((1, 10), dtype=np.int64)
for k in range(10):
    # print(np.shape(A), np.shape(S[k]), np.shape(B))
    # print("ss", S[k],np.shape(S[k]))
    print(k, sbits[k])
    temp1 = (np.dot(A, S[k].T) + (B * int(sbits[k]))) % 2  # NxN * Nx1 = Nx1, Nx1 * число = Nx1
    """
    t1 = np.dot(A, S[k].T)
    t2 = B * int(sbits[k])
    print('t1', t1, np.shape(t1))
    print('t2', t2, np.shape(t2))
    """

    print(temp1, np.shape(temp1))
    S[k + 1] = temp1  # сохраняем опять в строку, но для y[k] он столбец
    y[k] = np.dot(D.T, temp1.T)  # 1xN * Nx1 = число
    print('y', y[k])
    #x[k] = (y[k] + np.linalg.multi_dot([D.T, A, S])) % 2

print()
print("s", S)
print("YY", y)

sbytes_d = int(sbits, 2).to_bytes(len(sbytes), 'big')  # начало с MSB
msg = sbytes_d.decode('utf-8')
"""
"""
# print(list(sbits), N, type(sbits), N)
# print(152 / 19)



"""
v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(v)
m = np.copy(v)
c = np.array([[2, 3, 4]])
d = c.T
m[0] = [9, 8, 7]
print(m - v)

print(v, np.shape(v))
print(c, np.shape(c))
print(d, np.shape(d))
print(c[0])
print(d[0])
print(v[0,1])
print(np.dot(v,d))
print(np.dot(v,d)+d*2)
"""
