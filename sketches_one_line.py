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
        return bisect.bisect(self.distMap, u)


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


# On a single row
class Count_sketch_one_line:
    def __init__(self, k, n):
        self.k = k  # number of hash functions
        self.n = n  # number of counters
        self.array = [0 for _ in range(n)]
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
            self.array[key] += sign

    def point_query(self, element):
        return statistics.median(
            [
                self.array[self.hash_functions[i](element)]
                * self.s_functions[i](element)
                for i in range(self.k)
            ]
        )

    def display(self, rank, real_items, data):
        element = real_items[rank][1]
        array = sorted(
            [
                self.array[self.hash_functions[i](element)]
                * self.s_functions[i](element)
                for i in range(self.k)
            ]
        )
        maximum = max(array)
        minimum = min(array)
        median = statistics.median(array)
        median_i = 0
        while array[median_i] != median:
            median_i += 1
        dispersion = sum(
            [
                (
                    abs(
                        median
                        - self.array[self.hash_functions[i](element)]
                        * self.s_functions[i](element)
                    )
                )
                ** 2
                for i in range(self.k)
            ]
        )
        data["Rank"].append(rank)
        data["Values"].append(array)
        data["True value"].append(real_items[rank][0])

    def to_show(self, rank, real_items):
        element = real_items[rank][1]
        maximum = max(
            [
                self.array[self.hash_functions[i](element)]
                * self.s_functions[i](element)
                for i in range(self.k)
            ]
        )
        minimum = min(
            [
                self.array[self.hash_functions[i](element)]
                * self.s_functions[i](element)
                for i in range(self.k)
            ]
        )
        median = statistics.median(
            [
                self.array[self.hash_functions[i](element)]
                * self.s_functions[i](element)
                for i in range(self.k)
            ]
        )
        if median == 0:
            median = 1
        if maximum == 0:
            maximum = 1
        if minimum == 0:
            minimum = 1
        return abs((maximum - minimum))

    def dispersion(self, rank, real_items):
        element = real_items[rank][1]
        array = [
            self.array[self.hash_functions[i](element)] * self.s_functions[i](element)
            for i in range(self.k)
        ]
        median = statistics.median(array)
        dispersion = sum(
            [
                (
                    median
                    - self.array[self.hash_functions[i](element)]
                    * self.s_functions[i](element)
                )
                ** 2
                for i in range(self.k)
            ]
        )

        return dispersion


if __name__ == "__main__":

    np.random.seed(42)

    parameter = 1.2
    t = 11
    b = 10000
    N = 100000
    E = 50000

    zipf = ZipfGenerator(E, parameter)

    count_min_sketch = Count_min_one_line(t, b, conservative=True)
    count_sketch = Count_sketch_one_line(t, b)

    real_occ = dict()

    for _ in range(N):
        a = zipf.next()
        count_min_sketch.add(a)
        count_sketch.add(a)
        real_occ[a] = real_occ.get(a, 0) + 1

    real_items = sorted([(value, key) for key, value in real_occ.items()], reverse=True)

    res_count_min = [
        (i + 1, count_min_sketch.point_query(real_items[i][1]))
        for i in range(len(real_items))
    ]
    occ_count = [
        (i + 1, count_sketch.point_query(real_items[i][1]))
        for i in range(len(real_items))
    ]

    x_count_min, y_count_min = zip(*res_count_min)
    x_count, y_count = zip(*occ_count)
    x_real, y_real = zip(*[(i + 1, real_items[i][0]) for i in range(len(real_items))])

    data = {"Rank": [], "Values": [], "True value": []}

    for i in range(2000):
        count_sketch.display(i, real_items, data)

    df = pd.DataFrame(data)

    df.to_csv("stats_in_the_count_sketch.csv", index=False)

    print("Top 10 elements:")
    print()
    for i in range(10):
        print("#occurences rank {} : {}".format(i + 1, real_items[i][0]))

    plt.figure(figsize=(8, 6))

    plt.scatter(x_count, y_count, color="green", label="Count Sketch", alpha=0.6)

    plt.scatter(
        x_count_min, y_count_min, color="blue", label="Count-Min Sketch", alpha=0.6
    )

    plt.scatter(x_real, y_real, color="red", label="True value", alpha=0.6)

    plt.xlabel("Rank")
    plt.ylabel("Estimate")
    plt.title("Count Sketch vs Count-Min Sketch Estimate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.xscale("log")
    plt.yscale("log")

    plt.savefig("count_sketch_versus_count_one_line.png")
