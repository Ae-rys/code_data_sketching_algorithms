import math
import matplotlib.pyplot as plt
import statistics
import numpy as np
import time
import os
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.stats import norm

# Define constants
modes = [
    "MAP",
    "MLE with approx",
    "MLE without approx",
    #"MLE on class with approx",
   # "MLE on class without approx",
    #"MAP on class with approx",
    #"MAP on class without approx",
]

verbose = True
epsilon = 1e-15


class Hot_and_Cold_generator:
    def __init__(self, p_h, card_H, card_C):
        self.p_h = p_h
        self.card_H = card_H
        self.card_C = card_C

    def next(self):
        u = np.random.random()
        if u <= self.p_h:
            return np.random.randint(1, self.card_H + 1)
        else:
            return self.card_H + np.random.randint(1, self.card_C + 1)


class Count_sketch:
    def __init__(self, k, n):
        self.k = k
        self.n = n
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


def gaussian(x, mu, sigma):
    return max(epsilon, norm.pdf(x=x, loc=mu, scale=sigma))


def mle_estimator(values, approx, params):
    if approx:
        return np.mean(values)

    def f(x):
        return params["p_h"] * gaussian(x, mu=0, sigma=params["sigma_h"]) + params[
            "p_c"
        ] * gaussian(x, mu=0, sigma=params["sigma_c"])

    res = minimize(
        lambda x: -sum([math.log(f(val - x[0])) for val in values]),
        [np.mean(values) + 1],
        method="Nelder-Mead",
    )
    return res.x[0]


def mle_class_estimator(values, approx, params):
    mu_h, mu_c = params["mu_h"], params["mu_c"]
    sigma_h, sigma_c = params["sigma_h"], params["sigma_c"]

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

    log_likelihood_hot = sum([math.log(g_hot(val)) for val in values])
    log_likelihood_cold = sum([math.log(g_cold(val)) for val in values])

    estimated_class = "unknown"
    likelihood_ratio = 1

    if log_likelihood_hot > log_likelihood_cold:
        likelihood_ratio = math.exp(log_likelihood_hot - log_likelihood_cold)
        estimated_class = "hot"
    elif log_likelihood_cold > log_likelihood_hot:
        likelihood_ratio = math.exp(log_likelihood_cold - log_likelihood_hot)
        estimated_class = "cold"

    return (estimated_class, likelihood_ratio)


def map_estimator(values, params):
    def pi(y):
        return params["p_h"] * gaussian(
            x=y, mu=params["mu_h"], sigma=params["sigma_h_2"]
        ) + params["p_c"] * gaussian(x=y, mu=params["mu_c"], sigma=params["sigma_c_2"])

    def f(x):
        return params["p_h"] * gaussian(x, mu=0, sigma=params["sigma_h"]) + params[
            "p_c"
        ] * gaussian(x, mu=0, sigma=params["sigma_c"])

    res = minimize(
        lambda x: -math.log(pi(x[0]))
        - sum([math.log(f(val - x[0])) for val in values]),
        [np.mean(values) + 1],
        method="Nelder-Mead",
    )
    return res.x[0]


def map_class_estimator(values, approx, params):
    p_h, p_c = params["p_h"], params["p_c"]
    mu_h, mu_c = params["mu_h"], params["mu_c"]
    sigma_h, sigma_c = params["sigma_h"], params["sigma_c"]

    lower_bound_h = max(0, int(mu_h - 5 * math.sqrt(mu_h)))
    upper_bound_h = int(mu_h + 5 * math.sqrt(mu_h))
    lower_bound_c = max(0, int(mu_c - 5 * math.sqrt(mu_c)))
    upper_bound_c = int(mu_c + 5 * math.sqrt(mu_c))

    def pi(y):
        return p_h if y == "hot" else p_c

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

    log_likelihood_hot = math.log(pi("hot")) + sum(
        [math.log(g_hot(val)) for val in values]
    )
    log_likelihood_cold = math.log(pi("cold")) + sum(
        [math.log(g_cold(val)) for val in values]
    )

    estimated_class, likelihood_ratio = "unknown", 1

    if log_likelihood_hot > log_likelihood_cold:
        likelihood_ratio, estimated_class = (
            math.exp(log_likelihood_hot - log_likelihood_cold),
            "hot",
        )
    elif log_likelihood_cold > log_likelihood_hot:
        likelihood_ratio, estimated_class = (
            math.exp(log_likelihood_cold - log_likelihood_hot),
            "cold",
        )

    return (estimated_class, likelihood_ratio)


if __name__ == "__main__":
    np.random.seed(42)
    output_dir = "estimates"
    os.makedirs(output_dir, exist_ok=True)

    for N_v in [1000000]:
        for p_h_v, p_c_v in [(1 / 2, 1 / 2)]:
            for card_H_v, card_C_v in [(10000, 90000)]:
                for m_v in [100000]:
                    for k_v in [3]:

                        number_of_elements_for_during = min(N_v, 10000)

                        params = {
                            "N": N_v,
                            "k": k_v,
                            "m": m_v,
                            "card_H": card_H_v,
                            "card_C": card_C_v,
                            "p_h": p_h_v,
                            "p_c": p_c_v,
                        }
                        params["C_h"], params["C_c"] = (
                            params["p_h"] / params["card_H"],
                            params["p_c"] / params["card_C"],
                        )
                        params["mu_h"], params["mu_c"] = (
                            params["C_h"] * params["N"],
                            params["C_c"] * params["N"],
                        )
                        V_h = (
                            params["N"]
                            / params["m"]
                            * params["C_h"]
                            * (params["N"] * params["C_h"] + 1)
                        )
                        V_c = (
                            params["N"]
                            / params["m"]
                            * params["C_c"]
                            * (params["N"] * params["C_c"] + 1)
                        )
                        params["sigma_h"] = math.sqrt(
                            V_h * (params["card_H"] - 1) + V_c * params["card_C"]
                        )
                        params["sigma_c"] = math.sqrt(V_h * params["card_H"] + V_c * (
                            params["card_C"] - 1)
                        )
                        params["sigma_c_2"] = math.sqrt(
                            params["C_c"] * params["N"])
                        params["sigma_h_2"] = math.sqrt(
                            params["C_h"] * params["N"])

                        losses = {
                            mode + suffix: []
                            for mode in modes
                            for suffix in ["_final", "_during"]
                        }
                        losses.update({"median_final": [], "median_during": []})

                        for t in range(3):
                            loss_during = {mode: 0 for mode in modes}
                            loss_median_during = 0
                            count_sketch = Count_sketch(params["k"], params["m"])
                            real_occ = {}
                            Hot_and_Cold = Hot_and_Cold_generator(
                                p_h=params["p_h"],
                                card_H=params["card_H"],
                                card_C=params["card_C"],
                            )

                            for N_rank in tqdm(
                                range(params["N"]), desc=f"Simulation {t+1}/3"
                            ):

                                local_params = {
                                    "N": N_rank,
                                    "k": k_v,
                                    "m": m_v,
                                    "card_H": card_H_v,
                                    "card_C": card_C_v,
                                    "p_h": p_h_v,
                                    "p_c": p_c_v,
                                }
                                local_params["C_h"], local_params["C_c"] = (
                                    local_params["p_h"] / local_params["card_H"],
                                    local_params["p_c"] / local_params["card_C"],
                                )
                                local_params["mu_h"], local_params["mu_c"] = (
                                    local_params["C_h"] * local_params["N"],
                                    local_params["C_c"] * local_params["N"],
                                )
                                V_h = (
                                    local_params["N"]
                                    / local_params["m"]
                                    * local_params["C_h"]
                                    * (local_params["N"] * local_params["C_h"] + 1)
                                )
                                V_c = (
                                    local_params["N"]
                                    / local_params["m"]
                                    * local_params["C_c"]
                                    * (local_params["N"] * local_params["C_c"] + 1)
                                )
                                local_params["sigma_h"] = math.sqrt(
                                    V_h * (local_params["card_H"] - 1)
                                    + V_c * local_params["card_C"]
                                )
                                local_params["sigma_c"] = math.sqrt(V_h * local_params[
                                    "card_H"
                                ] + V_c * (local_params["card_C"] - 1))
                                local_params["sigma_c_2"] = math.sqrt(
                                    local_params["C_c"] * local_params["N"])
                                local_params["sigma_h_2"] = math.sqrt(
                                    local_params["C_h"] * local_params["N"])

                                a = Hot_and_Cold.next()
                                count_sketch.add(a)
                                real_occ[a] = real_occ.get(a, 0) + 1

                                if (
                                    params["N"] - N_rank
                                    <= number_of_elements_for_during
                                ):
                                    values = [
                                        count_sketch.array[i][
                                            count_sketch.hash_functions[i](a)
                                        ]
                                        * count_sketch.s_functions[i](a)
                                        for i in range(count_sketch.k)
                                    ]
                                    value = real_occ[a]
                                    estimate_median = np.median(values)
                                    loss_median_during += (value - estimate_median) ** 2

                                    for mode in modes:
                                        approx = "with approx" in mode
                                        if "class" in mode:
                                            estimate = (
                                                map_class_estimator(
                                                    values, approx, local_params
                                                )[0]
                                                if "MAP" in mode
                                                else mle_class_estimator(
                                                    values, approx, local_params
                                                )[0]
                                            )
                                            if (
                                                estimate == "hot"
                                                and a > params["card_H"]
                                            ) or (
                                                estimate == "cold"
                                                and a <= params["card_H"]
                                            ):
                                                loss_during[mode] += 1
                                        else:
                                            if mode == "MAP":

                                                estimate = map_estimator(
                                                    values, local_params
                                                )
                                            else:
                                                estimate = mle_estimator(
                                                    values, approx, local_params
                                                )
                                            loss_during[mode] += (value - estimate) ** 2

                            losses["median_during"].append(
                                loss_median_during / number_of_elements_for_during
                            )
                            for mode in modes:
                                losses[mode + "_during"].append(
                                    loss_during[mode] / number_of_elements_for_during
                                )

                            real_items = sorted(
                                [(v, k) for k, v in real_occ.items()], reverse=True
                            )
                            estimates_median = [
                                count_sketch.point_query(key) for _, key in real_items
                            ]
                            loss_median_final = sum(
                                (val - est) ** 2
                                for (val, _), est in zip(real_items, estimates_median)
                            )
                            losses["median_final"].append(
                                loss_median_final / len(real_items) if real_items else 0
                            )

                            if t == 2:
                                final_reports = {}

                            for mode in modes:
                                approx = "with approx" in mode
                                estimates = []
                                for _, key in real_items:
                                    values = [
                                        count_sketch.array[i][
                                            count_sketch.hash_functions[i](key)
                                        ]
                                        * count_sketch.s_functions[i](key)
                                        for i in range(params["k"])
                                    ]
                                    if "class" in mode:
                                        estimates.append(
                                            map_class_estimator(values, approx, params)
                                            if "MAP" in mode
                                            else mle_class_estimator(
                                                values, approx, params
                                            )
                                        )
                                    else:
                                        if mode == "MAP":
                                            estimates.append(
                                                map_estimator(values, params)
                                            )
                                        else:
                                            estimates.append(
                                                mle_estimator(values, approx, params)
                                            )

                                loss_final = 0
                                if "class" in mode:
                                    for (val, key), est in zip(real_items, estimates):
                                        if (
                                            est[0] == "hot" and key > params["card_H"]
                                        ) or (
                                            est[0] == "cold" and key <= params["card_H"]
                                        ):
                                            loss_final += 1
                                else:
                                    loss_final = sum(
                                        (val - est) ** 2
                                        for (val, _), est in zip(real_items, estimates)
                                    )

                                losses[mode + "_final"].append(
                                    loss_final / len(real_items) if real_items else 0
                                )
                                if t == 2:
                                    final_reports[mode] = {
                                        "estimates": estimates,
                                        "real_items": real_items,
                                        "estimates_median": estimates_median,
                                    }

                        print("\n--- Final results (mean taken on 3 simulations) ---")
                        param_str = f"p_h={params['p_h']}, N={params['N']}, m={params['m']}, k={params['k']}, card_H={params['card_H']}, card_C={params['card_C']}"

                        loss_median_final = round(np.mean(losses["median_final"]), 3)
                        loss_median_during = round(np.mean(losses["median_during"]), 3)

                        for mode in modes:
                            loss_final = round(np.mean(losses[mode + "_final"]), 3)
                            loss_during = round(np.mean(losses[mode + "_during"]), 3)
                            report_data = final_reports[mode]

                            if "class" in mode:
                                data = {
                                    "Rank": range(len(report_data["real_items"])),
                                    "True Class": [
                                        "hot" if key <= params["card_H"] else "cold"
                                        for _, key in report_data["real_items"]
                                    ],
                                    "Estimated Class": [
                                        x[0] for x in report_data["estimates"]
                                    ],
                                    "Likelihood Ratio": [
                                        x[1] for x in report_data["estimates"]
                                    ],
                                }
                                df = pd.DataFrame(data)
                                if verbose:
                                    print(f"\n--- Results for: {mode} ---\n{df.head()}")
                                filename = os.path.join(
                                    output_dir,
                                    f"class_estimates - ({mode}, {param_str}).csv",
                                )
                                df.to_csv(filename, index=False)
                                with open(filename, "a") as f:
                                    f.write(f"\nAverage Loss,,,,{loss_final}")
                            else:
                                plt.figure(figsize=(12, 6))
                                colors = [
                                    "green" if key <= params["card_H"] else "blue"
                                    for _, key in report_data["real_items"]
                                ]
                                plt.scatter(
                                    range(len(report_data["real_items"])),
                                    [v for v, k in report_data["real_items"]],
                                    c=colors,
                                    label="True value",
                                    marker="o",
                                    alpha=0.6,
                                )

                                skip = max(1, len(report_data["real_items"]) // 800)
                                plt.scatter(
                                    range(0, len(report_data["estimates"]), skip),
                                    [
                                        report_data["estimates"][i]
                                        for i in range(
                                            0, len(report_data["estimates"]), skip
                                        )
                                    ],
                                    color="red",
                                    label="Model estimates",
                                    marker="x",
                                )
                                plt.scatter(
                                    range(
                                        0, len(report_data["estimates_median"]), skip
                                    ),
                                    [
                                        report_data["estimates_median"][i]
                                        for i in range(
                                            0,
                                            len(report_data["estimates_median"]),
                                            skip,
                                        )
                                    ],
                                    color="black",
                                    label="Median estimates",
                                    marker="+",
                                )

                                plt.title(f"Frequency Estimates ({mode})\n{param_str}")
                                plt.xlabel(
                                    f"Item Rank\n(Final Loss: Model={loss_final}, Median={loss_median_final} | During Loss: Model={loss_during}, Median={loss_median_during})"
                                )
                                plt.ylabel("Frequency")
                                plt.grid(True)
                                plt.legend()
                                plt.tight_layout()
                                plt.savefig(
                                    os.path.join(
                                        output_dir,
                                        f"freq_estimates - ({mode}, {param_str}).png",
                                    )
                                )
                                plt.close()

                            if verbose:
                                print(
                                    f"\nMean loss for {mode}:\n  - final loss: {loss_final}\n  - 'during' loss: {loss_during}"
                                )

                        if verbose:
                            print(
                                f"\nMean loss for Median:\n  - final loss: {loss_median_final}\n  - 'during' loss: {loss_median_during}"
                            )
