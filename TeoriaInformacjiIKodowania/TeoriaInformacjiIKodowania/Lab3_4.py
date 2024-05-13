import librosa
import numpy as np
import pandas as pd
from scipy.stats import entropy
import scipy.io.wavfile as wav

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

# Przykład użycia funkcji analyze_entropy dla pojedynczego pliku audio
audio_file = "ATrain.wav"
result_df = analyze_entropy(audio_file)
print("ATrain:", result_df)

audio_file = "BeautySlept.wav"
result_df = analyze_entropy(audio_file)
print("BeautySlept:",result_df)

audio_file = "chanchan.wav"
result_df = analyze_entropy(audio_file)
print("chanchan:",result_df)

audio_file = "death2.wav"
result_df = analyze_entropy(audio_file)
print("death2:",result_df)

audio_file = "experiencia.wav"
result_df = analyze_entropy(audio_file)
print("experiencia:",result_df)

audio_file = "female_speech.wav"
result_df = analyze_entropy(audio_file)
print("female_speech:",result_df)

audio_file = "FloorEssence.wav"
result_df = analyze_entropy(audio_file)
print("FloorEssence:",result_df)

audio_file = "ItCouldBeSweet.wav"
result_df = analyze_entropy(audio_file)
print(f"{audio_file}",result_df)

audio_file = "Layla.wav"
result_df = analyze_entropy(audio_file)
print(f"{audio_file}",result_df)

audio_file = "LifeShatters.wav"
result_df = analyze_entropy(audio_file)
print(f"{audio_file}",result_df)

audio_file = "macabre.wav"
result_df = analyze_entropy(audio_file)
print(f"{audio_file}",result_df)

audio_file = "male_speech.wav"
result_df = analyze_entropy(audio_file)
print(f"{audio_file}",result_df)

audio_file = "SinceAlways.wav"
result_df = analyze_entropy(audio_file)
print(f"{audio_file}",result_df)

audio_file = "thear1.wav"
result_df = analyze_entropy(audio_file)
print(f"{audio_file}",result_df)

audio_file = "TomsDiner.wav"
result_df = analyze_entropy(audio_file)
print(f"{audio_file}",result_df)

audio_file = "velvet.wav"
result_df = analyze_entropy(audio_file)
print(f"{audio_file}",result_df)


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


audio_file = "ATrain.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "BeautySlept.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "chanchan.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "death2.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "experiencia.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "female_speech.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "FloorEssence.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "ItCouldBeSweet.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "Layla.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "LifeShatters.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "macabre.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "male_speech.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "SinceAlways.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "thear1.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "TomsDiner.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)
audio_file = "velvet.wav"
diff = encode_diff(audio_file)
result_df = calculate_entropy(diff)
print(f"{audio_file}", result_df)



