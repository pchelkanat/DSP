import random

import bitstring as bs
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as sw
import scipy.signal as sgn


# PART 1


def createSignal(Fs, myfreqs):
    t = np.linspace(0, 1, Fs)
    waves = np.zeros((len(myfreqs), Fs))
    for i in range(len(myfreqs)):
        waves[i] = np.sin(2 * np.pi * myfreqs[i] * t)
    totalwave = np.sum(waves, axis=0)
    # print(totalwave)
    return totalwave


# PART 2
def s2bit(s):
    # print(len(s))
    sbytes = s.encode("utf-8")
    sbits = bs.BitArray(sbytes).bin
    N = len(sbits)
    return sbits, N


# Амплитуды -> бинарный вид
def origin2orbit(origin, Len):
    originbytes = np.zeros_like(origin, dtype=list)

    for i in range(Len[0]):
        originbytes[i] = (format(origin[i], 'b').replace("-", "").zfill(8))
    # print(originbytes[1],originbytes[1][len(originbytes[0])-1])
    # print("orbyte",originbytes, type(originbytes), type(originbytes[0]))
    return originbytes  # type = list of strings


# Спектр фрагмента, куда будет внедряться wtrmark
def Spectrum(origin, N, Pos):
    frag = origin[Pos:Pos + 3 * N]  # длина фрагмента превышает длину фильтра в 2-3 раза
    F_frag = np.abs(np.fft.fft(frag))
    F_frag = F_frag[:int(3 * N / 2)]
    # print(F_frag, np.shape(F_frag))
    return F_frag


def freq_cutoff(fs, N, pos):
    # 50 позиция определена исходя из спектра
    freq = fs * pos / N
    c = 2 * freq / fs

    # частота отсечения
    freq_fir = 0
    while freq_fir < c:
        freq_fir = random.random()
    # print("c", c)
    print("cutoff freq", freq_fir)
    return freq_fir


# Внедрение wtrmark
def Watermarking(filter, Pos, norigin, coef):
    norigin = np.float64(norigin)
    wtrmark = filter[::-1]  # коэф. фильтра в обратном порядке
    # print("wtrmark", wtrmark)#float64, len=N
    for i in range(len(wtrmark)):
        norigin[Pos + i] += coef * wtrmark[i]
    return wtrmark, norigin


def Part1():
    # PART 1
    Fs = 200
    myfreqs = [15, 40, 60, 90]
    order = 50
    # f=myf/Fs*2
    freqs = [0, 0.15, 0.4, 0.6, 0.9, 1]
    gain = [0, 0, 1, 1, 0, 0]

    totalwave = createSignal(Fs, myfreqs)
    b = sgn.firwin2(order, freqs, gain)

    w, h = sgn.freqz(b)
    sig_ff = sgn.filtfilt(b, 1, totalwave)
    # sig_lf = sgn.lfilter(b, 1, totalwave)


    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(w / np.pi, abs(h), label="Передаточная функция фильтра")  # w/np.pi?
    plt.legend()
    plt.title("PART #1")

    plt.subplot(2, 1, 2)
    plt.plot(totalwave, label="Оригинал")
    plt.plot(sig_ff, label="Фильтрованный")
    # plt.plot(sig_lf)
    plt.legend()
    plt.show()


def Part2():
    fs, origin = sw.read("voice.wav")
    N = 2047  # длина фильтра, должна быть нечетной, иначе не работает
    Pos = 10000  # позиция внеднения
    n = 100  # вспомогательная величина

    F_frag = Spectrum(origin, N, Pos)
    freq_fir = freq_cutoff(fs, N, 500)

    # Создание фильтра
    myfilter = sgn.firwin(N, freq_fir, pass_zero=False)
    w, h = sgn.freqz(myfilter)

    # Внедрение
    # print((np.linalg.norm(myfilter))**2) #если малое, то coef должен быть большим
    wtrmark, norigin = Watermarking(myfilter, Pos, origin, 5000)
    corr = np.correlate(norigin, wtrmark, "valid")
    sw.write("fir_filter.wav", fs, np.int16(norigin))

    # Фильтрация для обнаружения wtrmark
    sig_ff = sgn.filtfilt(myfilter, 1, norigin)
    corr2 = np.correlate(sig_ff, wtrmark, "valid")
    MyPos = np.argmax(corr2)
    print("My Position", MyPos)

    # print(sig_ff)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(F_frag, label="Спектр внедрения")
    plt.legend()
    plt.title("PART #2.1")
    plt.subplot(3, 1, 2)
    plt.plot(w / np.pi, abs(h), label="Передат. функция")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(corr[Pos - n:Pos + N + n], label="Корреляция")
    plt.legend()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(origin[Pos - n:Pos + N + n], label="Оригинал")
    plt.plot(norigin[Pos - n:Pos + N + n], label="Внедренный wtrmark")
    plt.legend()
    plt.title("PART #2.2")
    plt.subplot(2, 1, 2)
    plt.plot(sig_ff[Pos - n:Pos + N + n], label="Фильтрованный\nПозиция: " + str(MyPos))
    plt.legend()

    plt.show()


def __init__():
    Part1()
    Part2()


__init__()
