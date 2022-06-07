import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
sns.set()

rv = scipy.stats.uniform()


### Импорт данных из файла
data_imported = pd.read_csv('A3.txt')

data_imported.columns = [f'A{i + 1}' for i in range(len(data_imported.columns))]
###

### Характеристики сигнала
NG = 92
n = 8
N = 2 ** n
M = 3
i = np.arange(N)
signal_2 = 2 * rv.rvs(N) + NG * np.cos(2 * M * np.pi * i / N) * (1 + 0.1 * rv.rvs(N)) + 17 * np.cos(4 * M * np.pi * i / N + rv.rvs(N)) + 3 * np.cos(5 * M * np.pi * i / N) * rv.rvs(N) * (rv.rvs(N) + NG)
###

### Преобразование Фурье
def fourier_transform(signal):
    temp_a = np.array([(2 / N) * np.sum(signal * np.cos(2 * np.pi * np.arange(N) * l / N))for l in range(1, int(N / 2 - 1))])
    a0 = np.array([(1 / N) * np.sum(signal * np.cos(0))])
    an = np.array([(1 / N) * np.sum(signal * np.cos(np.pi * np.arange(N)))])

    a = np.append(np.insert(temp_a, 0, a0), an)
    b = np.array([(2 / N) * np.sum(signal * np.sin(2 * np.pi * np.arange(N) * j / N)) for j in range(int(N / 2))])
    c = np.sqrt(a ** 2 + a ** 2)

    return a, b, c
###

def reverse_fourier(a, b):
    j = np.arange(int(N / 2))

    return np.array([np.sum(a * np.cos(2 * np.pi * j * i / N))
                     + np.sum(b * np.sin(2 * np.pi * j * i / N))
                     for i in range(N)])


def plot(s1, a, b, r1, color='red'):
    fig, ax = plt.subplots(3, 1, figsize=(20, 20))
    fig2, ax2 = plt.subplots(2, 1, figsize=(20, 20))
    ax[0].set_title("Изначальный сигнал")
    ax[0].plot(s1, color=color)

    ax[1].set_title("Вариант - А")
    ax[1].plot(a, color=color)

    ax[2].set_title("Вариант - Б")
    ax[2].plot(b, color=color)

    ax2[0].set_title("Обратное преобразование")
    ax2[0].plot(r1, color=color)

    ax2[1].set_title("Разница сигналов")
    ax2[1].plot(s1 - r1, color=color)



    plt.show()

signal_1 = signal_2

a, b, c = fourier_transform(signal_1)
r1 = reverse_fourier(a, b)
plot(s1=signal_1, a=a, b=b, r1=r1)

va = np.array([0.42 - 0.5 * np.cos(2 * np.pi * i / N) + 0.08 * np.cos(4 * np.pi * i / N) for i in range(N)])
vb = np.array([0.54 - 0.46 * np.cos(2 * np.pi * i / N) for i in range(N)])

signal_1 = signal_2 * va
a, b, c = fourier_transform(signal_1)
r1 = reverse_fourier(a, b)
plot(s1=signal_1, a=a, b=b, r1=r1)

signal_1 = signal_2 * vb
a, b, c = fourier_transform(signal_1)
r1 = reverse_fourier(a, b)
plot(s1=signal_1, a=a, b=b, r1=r1)