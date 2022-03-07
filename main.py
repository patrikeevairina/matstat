from math import factorial
from math import exp
from math import sqrt
from scipy.stats import laplace
import math
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.stats import norm
import scipy

v = 86  # номер варианта
par_a = pow(-1, v)*0.1*v
sko = round(sqrt(0.005*v+1), 5)
print("номер варианта, значения параметров распределения: ", v, par_a, sko)
random.seed(5)


def empirical_distr(x, selection):
    #считаем сначала w
    local_n = np.unique(selection, return_counts=True)
    arr_size = local_n[1].size
    local_w = [i for i in range(arr_size)]

    for j in range(arr_size):
        local_w[j] = round(local_n[1][j] / 200, 5)

    #а теперь уже строим
    fig, ax = plt.subplots()
    prev = 0.0
    arr = [0 for i in range(len(x)-1)]
    for i in range(len(x) - 1):
        ax.plot([x[i], x[i+1]], [local_w[i] + prev, local_w[i] + prev], 'b')
        prev += local_w[i]
        arr[i] = round(prev, 6)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.title("Эмпирическая функция распределения")
    plt.grid()
    plt.show()

def hystogram(a, w, step):
    fig, ax = plt.subplots()
    for i in range(len(w)):
        h = round(w[i]/step, 5)
        ax.plot([a[i], a[i]], [0, h], 'b')
        ax.plot([a[i], a[i+1]], [h, h], 'b')
        ax.plot([a[i + 1], a[i+1]], [h, 0], 'b')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.title("Гистограмма относительных частот")
    plt.grid()
    plt.show()


def math_exp(a, x, w):
    M = 0.0
    for i in range(len(x)):
        M += x[i]*w[i]
    print(round(M, 5), " экспериментальное выборочное среднее,", round(a, 5), " теоретическое выборочное среднее,", round(abs(a-M), 5), " абсолютное отклонение,", round(100*abs(a-M)/a, 5), "% относительное отклонение")


def disp(sko, x, w, step):
    M = 0.0
    for i in range(len(x)):
        M += x[i] * w[i]
    M = M**2
    M_2 = 0.0
    for i in range(len(x)):
        M_2 += x[i]**2 * w[i]
    D = M_2 - M - step**2/12
    print(round(D, 5), "экспериментальная выборочная дисперсия с поправкой Шеппарда,", round(sko**2, 5), "теоретическая дисперсия,", round(abs(D-sko**2), 5), " абсолютное отклонение,", round(100*abs(D-sko**2)/sko**2, 5), "% относительное отклонение")


def mean_sq_dev(sko, x, w):
    M = 0.0
    for i in range(len(x)):
        M += x[i] * w[i]
    M = M ** 2
    M_2 = 0.0
    for i in range(len(x)):
        M_2 += x[i] ** 2 * w[i]
    D = M_2 - M
    print(round(D**0.5, 5), "экспериментальное среднеквадратичное отклонение,", round(sko, 5), "теоретическое среднеквадратичное отклонение,", round(abs(D**0.5-sko), 5), " абсолютное отклонение,", round(100*abs(D**0.5-sko)/sko, 5), "% относительное отклонение")

def is_int(x):
    return int(x) == float(x)


def mode(par_a, a, w, step):
    #получаем индекс, соотв-ий модальному интервалу
    k = 0
    for i in range(len(w)):
        if w[i] == max(w):
            k = i
    m = a[k] + step * (w[k] - w[k-1]) / (2*w[k] - w[k-1] - w[k+1])
    print(round(m, 5), "экспериментальная выборочная мода,", round(par_a, 5), " теоретическая мода,", round(abs(par_a-m), 5), " абсолютное отклонение,", round(100*abs(par_a-m)/par_a, 5), "% относительное отклонение")


def median(par_a, a, w, step):
    sum_w = 0
    for i in range(len(w)):
        sum_w += w[i]
        if sum_w == 1/2:
            print(round(a[i+1], 5), " экспериментальная медиана", round(par_a, 5), " теоретическая медиана", round(abs(par_a-a[i+1]), 5), " абсолютное отклонение,", round(100*abs(par_a-a[i+1])/par_a, 5), "% относительное отклонение")
            return
        if sum_w > 1/2:
            answer = a[i]+step*(0.5-sum_w+w[i-1])/w[i]
            print(round(answer, 5), " экспериментальная медиана", round(par_a, 5), " теоретическая медиана", round(abs(par_a-answer), 5), " абсолютное отклонение,", round(100*abs(par_a-answer)/par_a, 5), "% относительное отклонение")
            return


def asymm_exc_coef(x_i, w_i):
    m_p = 0
    for i in range(len(x_i)):
        m_p += x_i[i]*w_i[i]
    mu1 = round(m_p, 6)
    mu2 = 0
    mu3 = 0
    mu4 = 0
    for i in range(len(x_i)):
        mu2 += ((x_i[i]) ** 2) * w_i[i]
        mu3 += ((x_i[i]) ** 3) * w_i[i]
        mu4 += ((x_i[i]) ** 4) * w_i[i]
    mu3_0 = mu3 - 3 * mu2 * mu1 + 2 * mu1 ** 3
    mu4_0 = mu4 - 4 * mu3 * mu1 + 6 * mu2 * mu1 ** 2 - 3 * mu1 ** 4

    m_p = 0.0
    for i in range(len(x_i)):
        m_p += x_i[i] * w_i[i]
    m_p = m_p ** 2
    m_2 = 0.0
    for i in range(len(x_i)):
        m_2 += x_i[i] ** 2 * w_i[i]
    d = m_2 - m_p
    sigma = round(d**0.5, 6)
    vka = mu3_0 / (sigma ** 3)
    vke = mu4_0 / (sigma ** 4) - 3
    print(round(vka, 5), "экспериментальный коэффициент ассиметрии,", 0, "теоретический коэффициент асимметрии,", round(vka, 5), "абсолютное отклонение")
    print(round(vke, 5), "экспериментальный коэффициент эксцесса,", 0, "теоретический коэффициент эксцесса,", round(vke, 5), "абсолютное отклонение")


selection = norm.rvs(par_a, sko, size=200)
for i in range(len(selection)):
    selection[i] = round(selection[i], 5)
print("выборка", selection)

selection.sort()
print("упорядоченная выборка", selection)

x = np.unique(selection)
#print("x_i", x)

m = 1 + round(math.log2(200))
d = selection[len(selection)-1] - selection[0]
step = d/m
a = [i for i in range(m+1)]
a[0] = x[0]
for i in range(1, m+1):
    a[i] = a[i-1]+step
for i in range(m+1):
    a[i] = round(a[i], 5)

n = [0 for i in range(m)]
n[0] = 1
j = 1
for i in range(m):
    while j < len(x) and a[i] < x[j] <= a[i + 1]:
        j += 1
        n[i] += 1

w = [0 for i in range(m)]
for i in range(m):
    w[i] = round(n[i]/200, 5)

# интервальный ряд (группированная выборка)
print("интервальный ряд (группированная выборка)")
for i in range(1, m+1):
    print("[", a[i-1], ",", a[i], "]", n[i-1], w[i-1])
print("sum n_i", sum(n), "sum w_i", round(sum(w), 5))

# ассоциированный ряд
print("ассоциированный ряд")
for i in range(1, m+1):
    print("x*_i", round((a[i]+a[i-1])/2, 5), n[i-1], w[i-1])

print("анализ результатов 1)таблица сравнения относительных частот и теоретических вероятностей")
p = [0 for i in range(len(w))]
for i in range(len(w)):
    p[i] += math.erf((a[i + 1]-par_a)/sko) - math.erf((a[i]-par_a)/sko)
for i in range(len(p)):
    p[i] = round(p[i], 5)
delta = 0
for i in range(1, m+1):
    print("[", a[i-1], ",", a[i], "]", w[i-1], p[i-1], round(abs(w[i-1]-p[i-1]), 5))
    if round(abs(w[i-1]-p[i-1]), 5) > delta:
        delta = round(abs(w[i-1]-p[i-1]), 5)
print("sum w_i", sum(w), "sum p_i", round(sum(p), 5), "max delta", delta)

print("анализ результатов 2) таблица сравнения рассчитанных характеристик с теоретическими значениями")
local_n = np.unique(selection, return_counts=True)
arr_size = local_n[1].size
local_w = [i for i in range(arr_size)]
for j in range(arr_size):
    local_w[j] = round(local_n[1][j] / 200, 5)

math_exp(par_a, x, local_w)
disp(sko, x, local_w, step)
mean_sq_dev(sko, x, local_w)
mode(par_a, a, w, step)
median(par_a, a, w, step)
asymm_exc_coef(x, local_w)

hystogram(a, w, step)
empirical_distr(x, selection)
