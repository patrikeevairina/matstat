import math
import statistics

from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import pandas as pd
from collections import Counter

random.seed(7)
n, p = 11, 0.68


def print_polygons(x_i, w_i):
    plt.figure()
    plt.title("Полигон относительных частот")
    plt.grid()
    plt.plot(x_i, w_i, 'b')

    teor_w_i = np.zeros(len(x_i))
    for i in range(len(x_i)):
        teor_w_i[i] = (math.comb(n, x_i[i]) * (p ** x_i[i]) * ((1 - p) ** (n - x_i[i])))
    plt.plot(x_i, teor_w_i, 'r')
    plt.show()

def empirical_distr(x_i, w_i):
    plt.figure()
    prev = 0.0
    for i in range(len(x_i) - 1):
        plt.plot([x_i[i], x_i[i+1]], [w_i[i] + prev, w_i[i] + prev], 'b')
        prev += w_i[i]
    plt.title("Эмпирическая функция распределения")
    plt.show()

def math_exp(n, p, x_i, w_i):
    print(round(n*p, 6), " экспериментальное мат ожидание")
    M = 0.0
    for i in range(len(x_i)):
        M += x_i[i]*w_i[i]
    print(round(M, 6), " теоретическое мат ожидание")

def disp(n, p, x_i, w_i):
    q = 1.0 - p
    print(round(n*p*q, 6), " экспериментальная дисперсия")
    M = 0.0
    for i in range(len(x_i)):
        M += x_i[i] * w_i[i]
    M = M**2
    M_2 = 0.0
    for i in range(len(x_i)):
        M_2 += x_i[i]**2 * w_i[i]
    D = M_2 - M
    print(round(D, 6), " эталонная дисперсия")

def mean_sq_dev(n, p, x_i, w_i):
    q = 1.0 - p
    print(round((n*p*q)**0.5, 6), " экспериментальное среднеквадратичное отклонение")
    M = 0.0
    for i in range(len(x_i)):
        M += x_i[i] * w_i[i]
    M = M ** 2
    M_2 = 0.0
    for i in range(len(x_i)):
        M_2 += x_i[i] ** 2 * w_i[i]
    D = M_2 - M
    print(round(D**0.5, 6), " эталонное среднеквадратичное отклонение")

def is_int(x):
    return int(x) == float(x)

def mode(n, p, x_i, w_i):
    m = p*(n+1)
    if not is_int(m):
        m -= 0.5
    print(round(m, 6), " экспериментальная мода")
    flag = 0.0
    key = 0
    value = max(w_i)
    for i in range(len(x_i)):
        if w_i[i] == value:
            if flag > 0 and w_i[i] != w_i[i-1]:
                print(" эталонной моды нет")
                return
            key += x_i[i]
            flag += 1.0
    m = float(key) / flag
    print(round(m, 6), " эталонная мода")

def median(n, p, x_i, w_i):
    print(round(n*p, 6), " экспериментальная медиана")
    prev = 0.0
    for i in range(len(x_i) - 1):
        prev += w_i[i]
        if prev == 0.5:
            print(round((x_i[i]+x_i[i+1])*0.5, 6), " эталонная медиана")
            return
        if prev > 0.5:
            print(round(x_i[i], 6), " эталонная медиана")
            return

def assym_coef(n, p, x_i, w_i):
    pass;

def excess_coef(n, p, x_i, w_i):
    pass;

selection = binom.rvs(n, p, size=200)
print(selection)
selection.sort()
print(selection)
count = Counter(selection)
w_i = list(count.values())
for i in range(len(count)):
    w_i[i] = w_i[i] / 200.0
x_i = list(count.keys())
print_polygons(x_i, w_i)
empirical_distr(x_i, w_i)
math_exp(n, p, x_i, w_i)
disp(n, p, x_i, w_i)
mean_sq_dev(n, p, x_i, w_i)
mode(n, p, x_i, w_i)
median(n, p, x_i, w_i)