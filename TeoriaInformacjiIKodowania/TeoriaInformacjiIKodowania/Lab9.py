import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import librosa

def analyze_entropy(audio_file):
    # Wczytanie próbki audio
    y, sr = librosa.load(audio_file, sr=None, mono=False)

    # Podział danych na kanały lewy i prawy
    left_channel = y[0]
    right_channel = y[1]

    # Przeskalowanie próbek do zakresu od -215 do 215-1
    left_channel_scaled = np.floor(left_channel * 2**15).astype(np.int16)
    right_channel_scaled = np.floor(right_channel * 2**15).astype(np.int16)

    # Obliczenie entropii dla każdego kanału osobno
    epsilon = 1e-10
    entropy_left = entropy(np.histogram(left_channel_scaled, bins=np.arange(-2**15, 2**15, dtype=np.int16))[0] + epsilon, base=2)
    entropy_right = entropy(np.histogram(right_channel_scaled, bins=np.arange(-2**15, 2**15, dtype=np.int16))[0] + epsilon, base=2)

    # Obliczenie średniej arytmetycznej entropii z obu kanałów
    average_entropy = (entropy_left + entropy_right) / 2

    # Tworzenie tabeli z wynikami
    df = pd.DataFrame({
        'Entropy_Left': [entropy_left],
        'Entropy_Right': [entropy_right],
        'Average_Entropy': [average_entropy]
    })

    return df


def encode_diff(audio_file):
    # Wczytanie próbki audio
    y, sr = librosa.load(audio_file, sr=None, mono=False)

    # Konwersja próbek do typu int16 i przeskalowanie do zakresu -32768 do 32767
    y_scaled = np.floor(y * 2**15).astype(np.int16)

    # Inicjalizacja tablicy różnic
    diff = np.zeros_like(y_scaled)

    # Obliczanie różnic między kolejnymi próbkami
    diff[:, 1:] = y_scaled[:, 1:] - y_scaled[:, :-1]

    return diff

def calculate_entropy(diff):
    # Obliczenie entropii dla każdego kanału osobno
    epsilon = 1e-10
    entropy_channel = []
    for channel_diff in diff:
        hist, _ = np.histogram(channel_diff, bins=np.arange(-2**15, 2**15, dtype=np.int16))
        entropy_channel.append(entropy(hist + epsilon, base=2))

    # Obliczenie średniej arytmetycznej entropii z obu kanałów
    average_entropy = np.mean(entropy_channel)

    # Tworzenie tabeli z wynikami
    df = pd.DataFrame({
        'Entropy_Left': [entropy_channel[0]],
        'Entropy_Right': [entropy_channel[1]],
        'Average_Entropy': [average_entropy]
    })

    return df

def algorytm1(e,n):
    e_prim = []
    if e(n)>=0:
        e_prim(n)=2*e(n)
    else:
        e_prim(n)=-2*e(n)-1

    return e_prim

N = 2**17 - 1

for i in range(N):
    S=(1/N)*algorytm1(i)

if(S>=2):
    p = (S-1)/S
elif(S<2):
    p = 0.5

m = np.ceil(-(np.log10(1+p))/(np.log10(p)))

Ug = np.floor(algorytm1(e,n)/m)

Vg = algorytm1(e,n) - Ug * m






















