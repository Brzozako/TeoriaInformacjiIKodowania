import math

file_path = '61'

def rice_encoder(file_path):
    with open(file_path, 'r') as file:
        A = [int(c) for c in file.read() if c.isdigit()]

    # Obliczamy sumę wystąpień 1
    ones_count = sum(1 for bit in A if bit == 1)

    # Obliczamy prawdopodobieństwo dla wystąpienia 1 lub 0
    p = ones_count / len(A)

    # Obliczamy stałą k
    k = math.ceil(math.log2((math.sqrt(5) - 1) / 2) / math.log2(1 - p))

    # Konwersja na liczby całkowite n
    n = []
    number = 0
    for bit in A:
        if bit == 0:
            number += 1
        if bit == 1:
            n.append(number)
            number = 0

    # Wyznaczanie u i v
    u = [n_i // 2**k for n_i in n]
    v = [n_i % 2**k for n_i in n]

    # Zakodowany strumień bitów
    encoded_bits = ''
    for i in range(len(u)):
        # Dodajemy kod unarny dla 'u' plus jedynka na końcu
        unar = '0' * u[i] + '1'
        # Dodajemy kod binarny dla 'v'
        bin_str = format(v[i], '0' + str(int(k)) + 'b')
        # Sklejamy i dodajemy do zakodowanego ciągu
        encoded_bits += unar + bin_str

    return encoded_bits, p, k

def rice_decoder(u, v, k):
    decoded_bits = []
    for idx in range(len(u)):
        num_zeros = u[idx] * 2**k + v[idx]  # Obliczanie liczby zer przed jedynką
        decoded_bits.extend([0] * num_zeros + [1])  # Dodawanie zer i jedynki do listy zdekodowanych bitów
    # Usunięcie nadmiarowej jedynki, jeśli taka istnieje
    if len(decoded_bits) > len(u) + sum(u) * 2**k + sum(v):
        decoded_bits.pop()
    return decoded_bits

def main():
    encoded_bits, p, k = rice_encoder(file_path)
    print("Encoded bits:", len(encoded_bits))
    print("p:", p)
    print("k:", k)
    L_R = (1 - p) * (k + 1 / (1 - p**2))
    E_R = (-(p * math.log2(p)) - ((1 - p) * math.log2(1 - p))) / L_R * 100
    print("Efficiency:", E_R)

if __name__ == "__main__":
    main()

def main():
    with open(file_path, 'r') as file:
        A = [int(c) for c in file.read() if c.isdigit()]

    ones_count = sum(1 for bit in A if bit == 1)

    p = ones_count / len(A)

    k = math.ceil(math.log2((math.sqrt(5) - 1) / 2) / math.log2(1 - p))
    n = []
    number = 0
    for bit in A:
        if bit == 0:
            number += 1
        if bit == 1:
            n.append(number)
            number = 0
    u = [n_i // 2 ** k for n_i in n]
    v = [n_i % 2 ** k for n_i in n]
    decoded_bits = rice_decoder(u, v, k)
    print("Decoded bits:", decoded_bits)

if __name__ == "__main__":
    main()
