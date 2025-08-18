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


def monte_carlo(r_q, G, z, N, E, nb_counters, alpha, n=1000):
    spiked_zipf = ZipfGenerator_spiked(E, alpha, r_q)

    res = 0

    val = alpha / r_q**z

    for s in range(n):
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

            if error_q == l:
                log_beta = (
                    math.log(math.comb(N, i))
                    + i * (math.log(alpha) - z * math.log(r_q))
                    + (N - i) * math.log(1 - val)
                )
                beta = math.exp(log_beta)

                res += beta / n

    return res


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
    prec = 500
    param_z = 1.2
    N = 1000
    E = 100
    nb_comp = 80
    nb_fun = 3

    np.random.seed(42)
    random.seed(42)

    print("Setting up simulation...")
    zipf_gen = ZipfGenerator(n=E, alpha=param_z)
    cm_sketch = Count_min_sketch(k=nb_fun, n=nb_comp)
    real_occ = {}

    print(f"Generating stream of {N} elements...")
    for _ in tqdm(range(N), desc="Processing Stream"):
        item = zipf_gen.next()
        cm_sketch.add(item)
        real_occ[item] = real_occ.get(item, 0) + 1
    
    true_ranks = []
    estimated_ranks = []
    
    unique_items = sorted(real_occ.keys())
    
    print(f"\nEstimating ranks for {len(unique_items)} unique items...")
    def clean_mle_estimator(Gains, z, N, E, omega, min_rank, max_rank, local_ranks_list, local_likelihoods_list, precision=1000):
        val_sum = sum(1 / (i**z) for i in range(1, E + 1))
        alpha = 1 / val_sum
        
        estimated_rank = -1
        max_log_likelihood = -float('inf')
        
        for rank in tqdm(range(max(1, min_rank), min(E, max_rank) + 1), desc="Testing Ranks", leave=False):
            local_ranks_list.append(rank)
            log_likelihood = 0
            for gain in Gains:
                prob = monte_carlo(rank, gain, z, N, E, omega, alpha, n=precision)
                if prob == 0:
                    log_likelihood = -float('inf')
                    break
                else:
                    log_likelihood += math.log(prob)
            
            current_likelihood = math.exp(log_likelihood) if log_likelihood > -float('inf') else 0
            local_likelihoods_list.append(current_likelihood)

            if log_likelihood > max_log_likelihood:
                max_log_likelihood = log_likelihood
                estimated_rank = rank
        return estimated_rank

    for true_rank in tqdm(unique_items, desc="Estimating Items"):
        gain = cm_sketch.point_query(true_rank)
        
        temp_ranks = []
        temp_likelihoods = []
        
        estimated_rank = clean_mle_estimator(
            Gains=[gain], 
            z=param_z, 
            N=N, 
            E=E, 
            omega=nb_comp,
            min_rank=1, 
            max_rank=E,
            local_ranks_list=temp_ranks,
            local_likelihoods_list=temp_likelihoods,
            precision=prec
        )
        
        true_ranks.append(true_rank)
        estimated_ranks.append(estimated_rank)

    print("\nGenerating plot...")
    plt.figure(figsize=(10, 8))
    
    plt.scatter(true_ranks, true_ranks, color='red', alpha=0.6, label='True rank')
    
    plt.scatter(true_ranks, estimated_ranks, color='blue', alpha=0.6, label='Estimated ranks')

    title_str = f"MLE : prec = {prec}, param = {param_z}, N = {N}, E = {E}, nb_comp = {nb_comp}, nb_fun = {nb_fun}"
    plt.title(title_str)
    plt.xlabel("Rank")
    plt.ylabel("Estimation")
    
    plt.grid(True)
    plt.legend()
    
    plt.show()
