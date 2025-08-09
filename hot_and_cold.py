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
            return sum(
                [
                    poisson.pmf(j, mu=mu_h) * gaussian(x=y - j, mu=0, sigma=sigma_h)
                    for j in range(lower_bound_h, upper_bound_h + 1)
                ]
            )

    def g_cold(y):
        if approx:
            return gaussian(x=y - mu_c, mu=0, sigma=sigma_c)
        else:
            return sum(
                [
                    poisson.pmf(j, mu=mu_c) * gaussian(x=y - j, mu=0, sigma=sigma_c)
                    for j in range(lower_bound_c, upper_bound_c + 1)
                ]
            )

    likelihood_hot = math.exp(sum([math.log(g_hot(val)) for val in values]))

    likelihood_cold = math.exp(sum([math.log(g_cold(val)) for val in values]))

    estimated_class = "unknown"
    likelihood_ratio = 1

    if verbose:
        print("likelihood hot:", likelihood_hot)
        print("likelihood cold:", likelihood_cold)

    if likelihood_hot > likelihood_cold:
        likelihood_ratio = likelihood_hot / likelihood_cold
        estimated_class = "hot"

    elif likelihood_hot < likelihood_cold:
        likelihood_ratio = likelihood_cold / likelihood_hot
        estimated_class = "cold"

    return (estimated_class, likelihood_ratio)


def map_estimator(values):

    def pi(y):
        res = p_h * gaussian(x=y, mu=mu_h, sigma=sigma_h_2) + p_c * gaussian(
            x=y, mu=mu_c, sigma=sigma_c_2
        )  # is very often zero........
        # print("pi_val:", res)
        return res

    def f(x):
        return p_h * gaussian(x, mu=0, sigma=sigma_h) + p_c * gaussian(
            x, mu=0, sigma=sigma_c
        )

    res = minimize(
        lambda x: -math.log(pi(x[0]))
        - sum([math.log(f(val - x[0])) for val in values]),
        [np.mean(values) + 1],
        method="Nelder-Mead",
    )

    return res


def map_class_estimator(values):

    lower_bound_h = max(0, int(mu_h - 5 * math.sqrt(mu_h)))
    upper_bound_h = int(mu_h + 5 * math.sqrt(mu_h))

    lower_bound_c = max(0, int(mu_c - 5 * math.sqrt(mu_c)))
    upper_bound_c = int(mu_c + 5 * math.sqrt(mu_c))

    def pi(y):
        if y == "hot":
            return p_h
        else:
            return p_c

    def g_hot(y):
        if approx:
            return gaussian(x=y - mu_h, mu=0, sigma=sigma_h)
        else:
            return sum(
                [
                    poisson.pmf(j, mu=mu_h) * gaussian(x=y - j, mu=0, sigma=sigma_h)
                    for j in range(lower_bound_h, upper_bound_h + 1)
                ]
            )

    def g_cold(y):
        if approx:
            return gaussian(x=y - mu_c, mu=0, sigma=sigma_c)
        else:
            return sum(
                [
                    poisson.pmf(j, mu=mu_c) * gaussian(x=y - j, mu=0, sigma=sigma_c)
                    for j in range(lower_bound_c, upper_bound_c + 1)
                ]
            )

    likelihood_hot = math.exp(
        math.log(pi("hot")) + sum([math.log(g_hot(val)) for val in values])
    )

    likelihood_cold = math.exp(
        math.log(pi("cold")) + sum([math.log(g_cold(val)) for val in values])
    )

    estimated_class = "unknown"
    likelihood_ratio = 1

    if verbose:

        print("likelihood hot:", likelihood_hot)
        print("likelihood cold:", likelihood_cold)

    if likelihood_hot > likelihood_cold:
        likelihood_ratio = likelihood_hot / likelihood_cold
        estimated_class = "hot"

    elif likelihood_hot < likelihood_cold:
        likelihood_ratio = likelihood_cold / likelihood_hot
        estimated_class = "cold"

    return (estimated_class, likelihood_ratio)


if __name__ == "__main__":

    np.random.seed(int(time.time()))

    def display_variables():
        for var, val in globals().items():
            if not var.startswith("__") and not callable(val):
                print(f"{var} - {val}")

    if verbose:

        print("variables: -----------------------------")
        display_variables()
        print()

        print("MLE estimator with approx (mean): -----------------------------")
        approx = True
        print(mle_estimator(values=values))
        print()

        print("MLE estimator without approx: -----------------------------")
        approx = False
        print(mle_estimator(values=values))
        print()

        print("MLE class estimator with approx: -----------------------------")
        approx = True
        print(mle_class_estimator(values=values))
        print()

        print("MLE class estimator without approx: -----------------------------")
        approx = False
        print(mle_class_estimator(values=values))
        print()

        print("MAP estimator: -----------------------------")
        print(map_estimator(values=values))
        print()

        print("MAP class estimator with approx: -----------------------------")
        approx = True
        print(map_class_estimator(values=values))
        print()

        print("MAP class estimator without approx: -----------------------------")
        approx = False
        print(map_class_estimator(values=values))
        print()

    for mode_v in tqdm(
        [
            "MLE with approx",
            "MLE without approx",
            # "MLE on class with approx",
            # "MLE on class without approx",
            "MAP",
            # "MAP on class with approx",
            # "MAP on class without approx",
        ]
    ):
        for N_v in [10000]:
            for p_h_v, p_c_v in [(1/2, 1/2), (1 / 3, 2 / 3)]:
                for card_H_v, card_C_v in [(100, 1000)]:
                    for m_v in [1000]:
                        for k_v in [1, 3]:

                            # redefine constants
                            N = N_v
                            k = k_v
                            m = m_v

                            card_H = card_H_v
                            card_C = card_C_v

                            lambda_h = card_H / m
                            lambda_c = card_C / m

                            p_h = p_h_v
                            p_c = p_c_v

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

                            mode = mode_v

                            approx = False

                            epsilon = 1e-15

                            # Fill a Count Sketch

                            count_sketch = Count_sketch(k, m)

                            real_occ = dict()

                            Hot_and_Cold = Hot_and_Cold_generator(
                                p_h=p_h, card_H=card_H, card_C=card_C
                            )

                            for _ in tqdm(range(N)):
                                a = Hot_and_Cold.next()
                                count_sketch.add(a)
                                real_occ[a] = real_occ.get(a, 0) + 1

                            real_items = sorted(
                                [(value, key) for key, value in real_occ.items()],
                                reverse=True,
                            )
                            estimates = []

                            # estimate the values
                            loss = 0

                            for value, key in tqdm(real_items):
                                values = [
                                    count_sketch.array[i][
                                        count_sketch.hash_functions[i](key)
                                    ]
                                    * count_sketch.s_functions[i](key)
                                    for i in range(count_sketch.k)
                                ]

                                if mode == "MAP":
                                    estimates.append(
                                        map_estimator(values=values).x[0]
                                    )
                                elif mode == "MLE with approx":
                                    approx = True
                                    estimates.append(mle_estimator(values=values))
                                elif mode == "MLE without approx":
                                    approx = False
                                    estimates.append(
                                        mle_estimator(values=values).x[0]
                                    )
                                elif mode == "MLE on class with approx":
                                    approx = True
                                    estimates.append(
                                        mle_class_estimator(values=values)
                                    )
                                elif mode == "MLE on class without approx":
                                    approx = False
                                    estimates.append(
                                        mle_class_estimator(values=values)
                                    )
                                elif mode == "MAP on class with approx":
                                    approx = True
                                    estimates.append(
                                        map_class_estimator(values=values)
                                    )
                                elif mode == "MAP on class without approx":
                                    approx = False
                                    estimates.append(
                                        map_class_estimator(values=values)
                                    )
                                else:
                                    raise Exception("Unknown mode")

                                if "class" not in mode:
                                    loss += (value - estimates[-1]) ** 2
                                else:
                                    if (estimates[-1][0] == "hot" and key > card_H) or (estimates[-1][0] == "cold" and key <= card_H):
                                        loss += 1
                            
                            loss = loss/len(estimates)

                            if "class" not in mode:
                                # create the plot

                                plt.figure(figsize=(12, 6))

                                colors = [
                                    "green" if key <= card_H else "blue"
                                    for _, key in real_items
                                ]
                                plt.scatter(
                                    [i for i in range(len(real_items))],
                                    [value for (value, key) in real_items],
                                    c=colors,
                                    label="true value",
                                    marker="o",
                                )

                                plt.scatter(
                                    [i for i in range(len(real_items))],
                                    estimates,
                                    color="red",
                                    label="estimates",
                                    marker="x",
                                )

                                plt.title(
                                    "Estimates (" + mode + ") - (p_h = {}, N = {}, m = {}, k= {}, card_H = {}, card_C = {})".format(
                                        p_h, N, m, k, card_H, card_C
                                    )
                                )
                                plt.xlabel(
                                    "rank in stream (loss = {})".format(loss)
                                )
                                plt.grid(True)
                                plt.legend()
                                plt.tight_layout()
                                plt.savefig(
                                    "estimates/count_estimates - ("
                                    + mode
                                    + ", p_h = {}, N = {}, m = {}, k= {}, card_H = {}, card_C = {}).png".format(
                                        p_h, N, m, k, card_H, card_C
                                    )
                                )

                            else:

                                data = {
                                    "Rank": [i for i in range(len(real_items))],
                                    "True Class": [
                                        "hot" if key <= card_H else "cold"
                                        for _, key in real_items
                                    ],
                                    "Estimated Class": [x[0] for x in estimates],
                                    "Likelihood Ratio": [x[1] for x in estimates],
                                }

                                df = pd.DataFrame(data)

                                if verbose:
                                    print(df)

                                filename = (
                                    "estimates/class_estimates - ("
                                    + mode
                                    + ", p_h = {}, N = {}, m = {}, k= {}, card_H = {}, card_C = {}).csv".format(
                                        p_h, N, m, k, card_H, card_C
                                    )
                                )

                                df.to_csv(
                                    filename,
                                    index=False,
                                )

                                with open(filename, "a") as f:
                                    f.write(f"\nLoss,,,,{loss}")

                            if verbose:

                                # display the error
                                print("loss:", loss)