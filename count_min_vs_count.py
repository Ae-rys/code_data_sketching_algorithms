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


# Zipf, but we remove the element of rank "spiked_rank"
class ZipfGenerator_spiked:
    def __init__(self, E, alpha, spiked_rank):
        self.tmp = [
            1.0 / (math.pow(float(i), alpha)) if i != spiked_rank else 0
            for i in range(1, E + 1)
        ]

        self.zeta = reduce(lambda sums, x: sums + [sums[-1] + x], self.tmp, [0])

        self.distMap = [x / self.zeta[-1] for x in self.zeta]

    def next(self):
        u = np.random.random()

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

    def display(self, rank, real_items, data):
        element = real_items[rank][1]
        array = sorted(
            [
                self.array[i][self.hash_functions[i](element)]
                * self.s_functions[i](element)
                for i in range(self.k)
            ]
        )[1:-1]
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
                        - self.array[i][self.hash_functions[i](element)]
                        * self.s_functions[i](element)
                    )
                )
                ** 2
                for i in range(self.k)
            ]
        )
        # print("   min :", minimum),
        # print("   max :", maximum)
        # print("   delta/max :", (maximum - minimum) / maximum)
        # print("   max/min :", (maximum) / minimum)
        # print("   dispersion:", dispersion)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            [i for i in range(1, self.k-1)], array, color="blue", label="count_sketch", alpha=0.6
        )
        plt.axhline(y=real_items[rank][0], color="red", label="true value")
        plt.scatter([median_i+1], [median], color="green", label="median", alpha=0.6)
        # plt.scatter(x_min, y_min, color="blue", label="min", alpha=0.6)
        plt.xlabel("i-th row")
        plt.ylabel("value found in the i-th row")
        plt.title(
            "Count Sketch (t = {} ; b = {} ; N = {} ; E = {} ; parameter = {})".format(
                t, b, N, E, parameter
            )
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig("dispersion_rank_{}.png".format(rank))

        print()

        data["Rank"].append(rank)
        data["Values"].append(array)
        data["True value"].append(real_items[rank][0])

    def to_show(self, rank, real_items):
        element = real_items[rank][1]
        maximum = max(
            [
                self.array[i][self.hash_functions[i](element)]
                * self.s_functions[i](element)
                for i in range(self.k)
            ]
        )
        minimum = min(
            [
                self.array[i][self.hash_functions[i](element)]
                * self.s_functions[i](element)
                for i in range(self.k)
            ]
        )
        median = statistics.median(
            [
                self.array[i][self.hash_functions[i](element)]
                * self.s_functions[i](element)
                for i in range(self.k)
            ]
        )
        dispersion = self.dispersion(rank, real_items)
        if median == 0:
            median = 1
        if maximum == 0:
            maximum = 1
        if minimum == 0:
            minimum = 1
        if dispersion == 0:
            return 1e-12
        return dispersion/median

    def dispersion(self, rank, real_items):
        element = real_items[rank][1]
        array = [
            self.array[i][self.hash_functions[i](element)]
            * self.s_functions[i](element)
            for i in range(self.k)
        ]
        median = statistics.median(array)
        dispersion = sum(
            [
                (
                    median
                    - self.array[i][self.hash_functions[i](element)]
                    * self.s_functions[i](element)
                )
                ** 2
                for i in range(self.k)
            ]
        )

        return dispersion


# TO REVIEW, I'm close
def monte_carlo(r_q, G, z, N, E, nb_counters, alpha, n=1000):
    print("starting estimation")
    spiked_zipf = ZipfGenerator_spiked(E, alpha, r_q)

    res = 0

    val = alpha / r_q**z

    for s in tqdm(range(n)):
        colliding_values = set()
        for j in range(1, E + 1):
            if j != r_q and random.randint(1, nb_counters) == 1:
                colliding_values.add(j)
        error_q = 0
        for l in range(N - G - 1):
            a = spiked_zipf.next()
            if a in colliding_values:
                error_q += 1
        for l in range(G + 1):
            a = spiked_zipf.next()
            if a in colliding_values:
                error_q += 1

            i = G - l

            # print(colliding_values)

            # print("error_q, l:", error_q, ",", l)

            if error_q == l:
                log_beta = (
                    math.log(math.comb(N, i))
                    + i * (math.log(alpha) - z * math.log(r_q))
                    + (N - i) * math.log(1 - val)
                )
                beta = math.exp(log_beta)

                res += beta / n

    return res


# TO REVIEW - Add a binary search? Complicated
def mle_estimator_count_min(Gains, z, N, E, omega, min_rank, max_rank, precision=1000):

    val_sum = 0

    for i in range(1, N + 1):
        val_sum += 1 / i**z

    alpha = 1 / val_sum

    estimated_rank = -1
    max_log_likelihood = -float("inf")
    for rank in range(max(1, min_rank), min(E, max_rank) + 1):
        ranks.append(rank)
        print("---------------------------------- rank", rank)
        log_likelihood = 0
        for gain in Gains:
            print("gain :", gain)
            prob = monte_carlo(rank, gain, z, N, E, omega, alpha, n=precision)
            if prob == 0:
                log_likelihood = -float("inf")
                print("log_likelihood:", log_likelihood)
                break
            else:
                log_likelihood += math.log(prob)
            print("log_likelihood:", math.log(prob))
        likelihoods.append(math.exp(log_likelihood))
        if log_likelihood > max_log_likelihood:
            max_log_likelihood = log_likelihood
            estimated_rank = rank

    return estimated_rank


if __name__ == "__main__":

    np.random.seed(int(time.time()))

    parameters = [1.5] #[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2]

    for parameter in parameters:

        t = 5
        b = int(8e2)
        N = int(1e6)
        E = int(1e5)
        conservative=False

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



        # data = {"Rank": [], "Values": [], "True value": []}

        # for i in range(10):
        #     count_sketch_1.display(i, real_items, data)

        # for i in range(20,31):
        #     count_sketch_1.display(i, real_items, data)

        # for i in range(40, 51):
        #     count_sketch_1.display(i, real_items, data)

        # for i in range(100, 111):
        #     count_sketch_1.display(i, real_items, data)
        # for i in range(500, 511):
        #     count_sketch_1.display(i, real_items, data)
        # for i in range(1000, 1011):
        #     count_sketch_1.display(i, real_items, data)
        # for i in range(2000, 2011):
        #     count_sketch_1.display(i, real_items, data)

        # df = pd.DataFrame(data)

        # # Apply the median condition to each row
        # df_filtered = df[
        #     df["Values"].apply(lambda v: np.median(v) >= df["True value"][30])
        # ]

        # print(df_filtered)

        # df_filtered.to_csv("stats.csv", index=False)

        # print("Top 10 elements:")
        # for i in range(10):
        #     print("rank {} : {}".format(i, real_items[i][0]))

        # estimated_ranks = []

        # print(count_min_sketch.array)

        # precision = 500
        # N=1000
        # E=100
        # omega=b

        # for _, key in real_items:
        #     Gains = [count_min_sketch.array[i][count_min_sketch.hash_functions[i](key)] for i in range(t)]
        #     estimated_rank = mle_estimator_count_min(
        #                         Gains=Gains,
        #                         z=parameter,
        #                         N=N,
        #                         E=E,
        #                         omega=omega,
        #                         min_rank=1,
        #                         max_rank=100,
        #                         precision=precision,
        #                         )
        #     estimated_ranks.append(estimated_rank)


        # plt.plot([i for i in range(1, len(real_items)+1)], [i for i in range(1, len(real_items)+1)], color="red", label="Real rank (post-sampling)", alpha=0.6)

        # plt.scatter([i for i in range(1, len(real_items)+1)], estimated_ranks, color="green", label="Real rank (pre-sampling)", alpha=0.6)

        # plt.scatter([i for i in range(1, len(real_items)+1)], estimated_ranks, color="blue", label="Estimated ranks", alpha=0.6)


        plt.xlabel("Rank")
        plt.ylabel("Estimate")
        plt.title("Count Min Vs Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.xscale("log")
        plt.yscale("log")

        plt.savefig("count_min_vs_count_parameter={}_N={}_E={}_hash_functions={}_counters_per_hash={}_conservative_count_min={}.png".format(parameter, N, E, t, b, conservative))

        # print(
        #     "estimated rank",
        #     mle_estimator_count_min([76, 76, 2], 0.5, 1000, 10, 15, 20),
        # )

        # to_show_max = []

        # print("number of ranks:", len(real_occ))

        # for rank in range(len(real_occ)):
        #     if  count_sketch.dispersion(rank, real_items) > 0:
        #         to_show_max.append((rank+1, count_sketch.dispersion(rank, real_items)))
        #     else:
        #         print("zero dispersion")
        #         to_show_max.append((rank+1, 1))


        # to_show_max = [
        #     (rank, count_sketch_1.to_show(rank, real_items)) for rank in range(len(real_occ))
        # ]
        # x_max, y_max = zip(*to_show_max)
        # # x_min, y_min = zip(*to_show_min)

        # plt.figure(figsize=(8, 6))
        # plt.scatter(x_max, y_max, color="blue", label="count_sketch", alpha=0.6)
        # # plt.scatter(x_min, y_min, color="blue", label="min", alpha=0.6)
        # plt.xlabel("Rank")
        # plt.ylabel("variance/median")
        # plt.title(
        #     "Count Sketch (t = {} ; b = {} ; N = {} ; E = {} ; parameter = {})".format(
        #         t, b, N, E, parameter
        #     )
        # )
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()

        # plt.xscale("log")
        # plt.yscale("log")

        # plt.savefig("figures/dispersion_over_median/dispersion_over_median_parameter_{}.png".format(parameter))
        # plt.close()