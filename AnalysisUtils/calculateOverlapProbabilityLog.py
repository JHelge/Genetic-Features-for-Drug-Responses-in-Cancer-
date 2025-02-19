import math
from scipy.special import comb, logsumexp

def log_comb(n, k):
    return math.log(comb(n, k))

def overlap_probability_exactly(k, n, m, o):
    if o > min(n, m) or o > k:
        return float('-inf')  # log(0) for impossible events
    if n - o > k - o or m - o > k - n:
        return float('-inf')  # log(0) for impossible distributions
    log_numerator = log_comb(k, o) + log_comb(k - o, n - o) + log_comb(k - n, m - o)
    log_denominator = log_comb(k, n) + log_comb(k, m)
    return log_numerator - log_denominator

def overlap_probability_at_least(k, n, m, o):
    max_overlap = min(n, m)
    log_probabilities = []
    for o_prime in range(o, max_overlap + 1):
        log_probabilities.append(overlap_probability_exactly(k, n, m, o_prime))
    return math.exp(logsumexp(log_probabilities))  # Convert from log space to normal space

# Example usage with large numbers
k = 38977  # Total number of elements in the set
n = 349    # Size of the first subset
m = 437    # Size of the second subset
o = 118    # Minimum number of overlapping elements

probability = overlap_probability_at_least(k, n, m, o)
print("Probability of overlapping at least", o, "elements:", probability)

