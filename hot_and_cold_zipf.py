import math
import matplotlib.pyplot as plt
import statistics
import numpy as np
import time
import random
import bisect
from functools import reduce
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.stats import norm

# Define constants

N = int(1e6)
k = 2
m = int(8e4)

E = 10000

card_H = 100
card_C = 1000

lambda_h = card_H / m
lambda_c = card_C / m

p_h = 1 / 2
p_c = 1 / 2

C_h = p_h / card_H
C_c = p_c / card_C

V_h = N / m * C_h * (N * C_h + 1)
V_c = N / m * C_c * (N * C_c + 1)

sigma_h = V_h * (card_H - 1) + V_c * card_C
sigma_c = V_h * card_H + V_c * (card_C - 1)

mu_h = C_h * N
mu_c = C_c * N

sigma_c_2 = (C_c * (1 - C_c)) ** 2 * N
sigma_h_2 = (C_h * (1 - C_h)) ** 2 * N

values = [5.0]

mode = "MLE on class with approx"

approx = False

verbose = True

epsilon = 1e-15


class Hot_and_Cold_generator:

    def __init__(self, p_h, card_H, card_C):

        # draw with probability p_h
        self.p_h = p_h
        self.Card_h = card_H
        self.Card_c = card_C

    def next(self):
        # Take a uniform 0-1 pseudo-random value:
        u = np.random.random()

        if u <= p_h:
            return np.random.randint(1, card_H + 1)
        else:
            return card_H + np.random.randint(1, card_C + 1)


class Count_sketch:
    def __init__(self, k, n):
        self.k = k  # number of hash functions
        self.n = n  # number of counters per hash function
        self.array = [[0 for _ in range(n)] for _ in range(k)]
        self.hash_functions = []

        p = 2**31 - 1

        for _ in range(k):
            a = np.random.randint(0, p)
            b = np.random.randint(0, p)
            self.hash_functions.append(lambda x, a=a, b=b: ((a * x + b) % p) % n)

        self.s_functions = []

        for _ in range(k):
            a = np.random.randint(0, p)
            b = np.random.randint(0, p)
            self.s_functions.append(
                lambda x, a=a, b=b: -1 if ((a * x + b) % p) % 2 == 0 else 1
            )

    def add(self, element):
        for i in range(self.k):
            h_i = self.hash_functions[i]
            s_i = self.s_functions[i]
            key = h_i(element)
            sign = s_i(element)
            self.array[i][key] += sign

    def point_query(self, element):
        return statistics.median(
            [
                self.array[i][self.hash_functions[i](element)]
                * self.s_functions[i](element)
                for i in range(self.k)
            ]
        )  # Can we do better in terms of space/time complexity?
        # I could calculate it on the fly like the min


def gaussian(x, mu, sigma):
    return max(
        epsilon,
        norm.pdf(x=x, loc=mu, scale=sigma),
    )


def mle_estimator(values):

    if approx:
        return np.mean(values)

    def f(x):
        return p_h * gaussian(x, mu=0, sigma=sigma_h) + p_c * gaussian(
            x, mu=0, sigma=sigma_c
        )

    res = minimize(
        lambda x: -sum([math.log(f(val - x[0])) for val in values]),
        [np.mean(values) + 1],
        method="Nelder-Mead",
    )

    return res


def mle_class_estimator(values):

    lower_bound_h = max(0, int(mu_h - 5 * math.sqrt(mu_h)))
    upper_bound_h = int(mu_h + 5 * math.sqrt(mu_h))

    lower_bound_c = max(0, int(mu_c - 5 * math.sqrt(mu_c)))
    upper_bound_c = int(mu_c + 5 * math.sqrt(mu_c))

    def g_hot(y):
        if approx:
            return gaussian(x=y - mu_h, mu=0, sigma=sigma_h)
        else:
            return sum