import math
import matplotlib.pyplot as plt
import statistics
import numpy as np
import time
import random
import bisect
from functools import reduce
import pandas as pd


class ZipfGenerator:

    def __init__(self, n, alpha):
        # Calculate Zeta values from 1 to n:
        tmp = [1.0 / (math.pow(float(i), alpha)) for i in range(1, n + 1)]

        # reduce = fold_left (gives the non-normalized cdf)
        zeta = reduce(lambda sums, x: sums + [sums[-1] + x], tmp, [0])

        # Store the translation (i.e., the cdf):
        self.distMap = [x / zeta[-1] for x in zeta]

    def next(self):
        # Take a uniform 0-1 pseudo-random value:
        u = np.random.random()

        # Translate the Zipf variable (works because the probability of being in an interval is exactly the probability of drawing the considered integer):
        return bisect.bisect(self.distMap, u) - 1


class Count_min_sketch:
    def __init__(self, k, n, conservative=False):
        self.k = k  # number of hash functions
        self.n = n  # number of counters per row
        self.array = [[0 for _ in range(n)] for _ in range(k)]
        self.hash_functions = []

        p = 2**31 - 1

        # Works modulo n, so it's sufficient, because p >> n (we have an equivalent)
        for i in range(k):
            a = np.random.randint(0, p)
            b = np.random.randint(0, p)
            self.hash_functions.append(lambda x, a=a, b=b: ((a * x + b) % p) % n)

        self.conservative = conservative

    def add(self, element):
        if self.conservative:
            minimum = float("inf")
            for i, h in enumerate(self.hash_functions):
                minimum = min(minimum, self.array[i][h(element)])

            for i, h in enumerate(self.hash_functions):
                key = h(element)
                if self.array[i][key] == minimum:
                    self.array[i][key] += 1
        else:
            for i, h in enumerate(self.hash_functions):
                key = h(element)
                self.array[i][key] += 1

    def point_query(self, element):
        minimum = float("inf")
        for i, h in enumerate(self.hash_functions):
            minimum = min(minimum, self.array[i][h(element)])
        return minimum


# We perform a count-min on a single row
class Count_min_one_line:
    def __init__(self, k, n, conservative=False):
        self.k = k  # number of hash functions
        self.n = n  # number of counters
        self.array = [0 for _ in range(n)]
        self.hash_functions = []

        p = 2**31 - 1

        # Works modulo n, so it's sufficient, because p >> n (we have an equivalent)
        for i in range(k):
            a = np.random.randint(0, p)
            b = np.random.randint(0, p)
            self.hash_functions.append(lambda x, a=a, b=b: ((a * x + b) % p) % n)

        self.conservative = conservative

    def add(self, element):
        if self.conservative:
            minimum = float("inf")
            for h in self.hash_functions:
                minimum = min(minimum, self.array[h(element)])

            for h in self.hash_functions:
                key = h(element)
                if self.array[key] == minimum:
                    self.array[key] += 1
        else:
            for h in self.hash_functions:
                key = h(element)
                self.array[key] += 1

    def point_query(self, element):
        minimum = float("inf")
        for h in self.hash_functions:
            minimum = min(minimum, self.array[h(element)])
        return minimum


if __name__ == "__main__":

    np.random.seed(42)

    parameter = 0.5
    t = 11
    b = 10000
    N = 1000000
    E = 10000

    zipf = ZipfGenerator(E, parameter)

    count_min_sketch = Count_min_sketch(t, int(b / t))
    count_min_one_line_instance = Count_min_one_line(t, b)

    real_occ = dict()

    for _ in range(N):
        a = zipf.next()
        count_min_sketch.add(a)
        count_min_one_line_instance.add(a)
        real_occ[a] = real_occ.get(a, 0) + 1

    real_items = sorted([(value, key) for key, value in real_occ.items()], reverse=True)

    res_count_min = [
        (i + 1, count_min_sketch.point_query(real_items[i][1]))
        for i in range(len(real_items))
    ]

    res_count_min_one_line = [
        (i + 1, count_min_one_line_instance.point_query(real_items[i][1]))
        for i in range(len(real_items))
    ]

    x_count_min, y_count_min = zip(*res_count_min)
    x_count_min_one_line, y_count_min_one_line = zip(
        *res_count_min_one_line
    )
    x_real, y_real = zip(*[(i + 1, real_items[i][0]) for i in range(len(real_items))])

    plt.figure(figsize=(8, 6))

    plt.scatter(
        x_count_min, y_count_min, color="blue", label="Count-Min Sketch", alpha=0.6
    )

    plt.scatter(
        x_count_min_one_line,
        y_count_min_one_line,
        color="green",
        label="Spectral Bloom Filter",
        alpha=0.6,
    )

    plt.scatter(x_real, y_real, color="red", label="True value", alpha=0.6)

    plt.xlabel("Rank")
    plt.ylabel("Estimate")
    plt.title("Spectral Bloom Filter vs Count-Min Sketch Estimate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.xscale("log")
    plt.yscale("log")

    plt.savefig("experiment_SBF_CM.png")
