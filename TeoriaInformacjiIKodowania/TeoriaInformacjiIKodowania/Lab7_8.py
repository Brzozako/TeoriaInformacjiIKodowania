import numpy as np
from scipy.io import wavfile

# Wczytanie pliku audio
fs, e = wavfile.read("ATrain.wav")
e = e.astype(float)

e = np.floor(e * (2**16) + 0.5)

# Algorytm nr 1
ep = np.zeros_like(e)
for n in range(len(e)):
    ep = np.where(e >= 0, 2 * e, -2 * e - 1)


# Obliczanie s
s = np.mean(ep)

# Obliczanie p
if s >= 2:
    p = (s - 1) / s
else:
    p = 0.5

# Obliczanie k
k = np.log((np.sqrt(5) - 1) / 2) / np.log(p)
k = np.ceil(k)

# Obliczanie entropii dla każdego kanału
ni = np.zeros((2**17, 2), dtype=int)
for i in range(len(ep)):
    ni[int(ep[i, 0] + 2**16), 0] += 1
    ni[int(ep[i, 1] + 2**16), 1] += 1

HS = []
for channel in range(2):
    sum_entropy = 0
    for i in range(2**17):
        p = ni[i, channel] / len(ep)
        if p != 0:
            sum_entropy += p * np.log2(p)
    sum_entropy = -sum_entropy
    HS.append(sum_entropy)

# Konwersja e(n) na u i v
coded_e = []
u = np.zeros(ep.shape, dtype=int)
v = np.zeros(ep.shape, dtype=int)
for i in range(ep.shape[1]):
    u[:, i] = np.floor(ep[:, i] / 2**k)
    v[:, i] = ep[:, i] - u[:, i] * 2**k

# Zakodowany strumień bitów
encoded_bits = ''
for i in range(len(u)):
    for j in range(ep.shape[1]):
        unar = '0' * u[i, j] + '1'
        bin_str = np.binary_repr(v[i, j], width=int(k))
        encoded_bits += unar + bin_str

Lw = len(coded_e) / (2 * len(ep))
