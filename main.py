import math
import statistics

from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import pandas as pd

random.seed(10)

num, p = 11, 0.68
r = binom.rvs(num, p, size=200)
k = 0
rows, col = 20, 10
without_order = np.zeros((rows, col))
for i in range(rows):
    for j in range(col):
        without_order[i][j] = r[k]
        k += 1
# print(without_order)
r.sort()
with_order = np.zeros((rows, col), dtype=np.int8)
k = 0
for i in range(rows):
    for j in range(col):
        with_order[i][j] = r[k]
        k += 1
print(with_order)

n = 1
for i in range(199):
    if r[i + 1] != r[i]:
        n += 1
print(n)
pol = np.zeros(2 * n)
plt.figure()
k = 1.0
l = 0
for i in range(199):
    if r[i + 1] == r[i]:
        k += 1.0
    else:
        x = r[i]
        y = k / 200.0
        pol[l] = x
        pol[l + n] = y
        l += 1
        k = 1.0
if r[199] != r[198]:
    pol[n - 1] = r[199]
    pol[2 * n - 1] = 1.0 / 200
else:
    pol[n - 1] = r[199]
    pol[2 * n - 1] = k / 200

pol_teor = np.zeros(2 * n)
for i in range(n):
    pol_teor[i] = pol[i]
for i in range(n, 2 * n):
    pol_teor[i] = (math.comb(num, int(pol[i - n])) * (p ** (pol[i - n])) * ((1 - p) ** (num - pol[i - n])))
plt.plot([pol[j] for j in range(n)], [pol[i] for i in range(n, 2 * n)], 'b')
plt.plot([pol_teor[j] for j in range(n)], [pol_teor[i] for i in range(n, 2 * n)], "r")
plt.title("Полигон относительных частот")
plt.show()

plt.figure()
prev = 0.0
for j in range(n - 1):
    plt.plot([pol[j], pol[j + 1]], [pol[j + n] + prev, pol[j + n] + prev], 'b')
    prev += pol[j + n]
plt.title("Эмпирическая функция распределения")
plt.show()

print(statistics.mean(r), " выборочное среднее")
print(statistics.variance(r), " выборочная дисперсия")
print(statistics.stdev(r), " выборочное ско")
print(statistics.mode(r), " выборочная мода")
print(statistics.median(r), "выборочная медиана")
s = pd.Series(r)
print(s.skew(), " выборочная ассиметрия")
print(s.kurt(), " выборочный эксцесс")
