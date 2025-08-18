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

likelihoods = []
ranks = []


class ZipfGenerator:
    def __init__(self, n, alpha):
        # Calculate Zeta values from 1 to n:
        tmp = [1.0 / (math.pow(float(i), alpha)) for i in range(1, n + 1)]

        # reduce = fold_left (non-normalized cdf)
        self.zeta = reduce(lambda sums, x: sums + [sums[-1] + x], tmp, [0])

        # Store the translation (i.e., the cdf):
        self.distMap = [x / self.zeta[-1] for x in self.zeta]

    def next(self):
        # Take a uniform 0-1 pseudo-random value:
        u = np.random.random()

        # Translate the Zipf variable (works because the probability to be in an interval is exactly the probability to pick the considered integer):
        return bisect.bisect(self.distMap, u)

class Count_min_sketch:
    def __init__(self, k, n, conservative=False):
        self.k = k  # number of hash functions
        self.n = n  # number of counters per hash function
        self.array = [[0 for _ in range(n)] for _ in range(k)]
        self.hash_functions = []

        p = 2**31 - 1

        # works modulo the modulo n, so works, because p >> n (there is an equivalent)
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
        )


if __name__ == "__main__":

    np.random.seed(42)

    parameters = [1.2]

    for parameter in parameters:

        t = 5
        b = int(8e2)
        N = int(1e6)
        E = int(1e5)
        conservative=True

        zipf = ZipfGenerator(E, parameter)

        count_min_sketch = Count_min_sketch(t, b, conservative=conservative)
        count_sketch_1 = Count_sketch(t, b)
        count_sketch_2 = Count_sketch(t, b)

        real_occ = dict()

        for _ in tqdm(range(N)):
            a = zipf.next()
            count_min_sketch.add(a)
            count_sketch_1.add(a)
            count_sketch_2.add(a)
            real_occ[a] = real_occ.get(a, 0) + 1

        real_items = sorted([(value, key) for key, value in real_occ.items()], reverse=True)

        res_count_min = [
            (i + 1, count_min_sketch.point_query(real_items[i][1]))
            for i in range(len(real_items))
        ]
        occ_count = [
            (
                i + 1,
                (
                    (
                        count_sketch_1.point_query(real_items[i][1])
                        + count_sketch_2.point_query(real_items[i][1])
                    )
                    / 2
                ),
            )
            for i in range(len(real_items))
        ]

        x_count_min, y_count_min = zip(*res_count_min)
        x_count, y_count = zip(*occ_count)
        x_real, y_real = zip(*[(i + 1, real_items[i][0]) for i in range(len(real_items))])

        plt.figure(figsize=(8, 6))
        plt.scatter(x_count, y_count, color="green", label="Count Sketch", alpha=0.6)
        plt.scatter(x_count_min, y_count_min, color="blue", label="Count Min", alpha=0.6)
        plt.scatter(x_real, y_real, color="red", label="True Counts", alpha=0.6)

        plt.xlabel("Rank")
        plt.ylabel("Estimate")
        plt.title("Count Min Vs Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.xscale("log")
        plt.yscale("log")

        plt.savefig("count_min_vs_count_parameter={}_N={}_E={}_hash_functions={}_counters_per_hash={}_conservative_count_min={}.png".format(parameter, N, E, t, b, conservative))
