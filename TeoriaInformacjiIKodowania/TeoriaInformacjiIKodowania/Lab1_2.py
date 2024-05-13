import numpy as np
import cv2
import matplotlib.pyplot as plt

# Załadowanie obrazu (należy zmienić ścieżkę do pliku)
M = cv2.imread("lennagrey.bmp", cv2.IMREAD_GRAYSCALE)
M = M.astype(float)

# Wstępne przygotowanie
numRows, numCols = M.shape
D = np.zeros_like(M) # Obraz różnicowy

# Kodowanie różnicowe
D[0,:] = np.diff(np.concatenate(([0], M[0,:])))
D[:,0] = np.diff(np.concatenate(([0], M[:,0])))
for i in range(1, numRows):
    for j in range(1, numCols):
        D[i,j] = M[i,j] - M[i,j-1]

# Dekodowanie różnicowe
R = np.zeros_like(M)
R[0,:] = np.cumsum(D[0,:])
R[:,0] = np.cumsum(D[:,0])
for i in range(1, numRows):
    for j in range(1, numCols):
        R[i,j] = D[i,j] + R[i,j-1]

# Obliczenie histogramów
histOriginal = np.histogram(M.flatten(), bins=np.arange(257), density=True)[0]
histDiff = np.histogram(D.flatten(), bins=np.arange(-255.5, 256.6), density=True)[0]

# Obliczenie entropii obrazu różnicowego
histDiff_nonzero = histDiff[np.where(histDiff > 0)]
entropyDiff = -np.sum(histDiff_nonzero * np.log2(histDiff_nonzero))

# Wykres porównawczy
plt.plot(np.arange(256), histOriginal, 'r', linewidth=1)
plt.plot(np.arange(-255.5, 256.5), histDiff, 'k', linewidth=1)

plt.xlim([-255, 255])
plt.show()

# Wyniki
print("Entropia obrazu różnicowego:", entropyDiff, "bits/pixel")
