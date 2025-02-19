import math
from scipy.special import gammaln, logsumexp
import matplotlib.pyplot as plt

def log_comb(n, k):
    if k > n or k < 0:
        return float('-inf')  # log(0) for invalid combinations
    return gammaln(n + 1) - (gammaln(k + 1) + gammaln(n - k + 1))

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
        log_p = overlap_probability_exactly(k, n, m, o_prime)
        if math.isfinite(log_p):  # Only include valid log probabilities
            log_probabilities.append(log_p)
    if not log_probabilities:  # If no valid probabilities were computed
        return 0
    return math.exp(logsumexp(log_probabilities))  # Convert from log space to normal space

# Example usage with large numbers
#k = 38977  # Total number of elements in the set
#n = 1069    # Size of the first subset
#m = 1069    # Size of the second subset
#o = 118    # Minimum number of overlapping elements

k = 38977  # Total number of elements in the set
n = 437    # Size of the first subset
m = 349    # Size of the second subset
o = 118    # Minimum number of overlapping elements

probability = overlap_probability_at_least(k, n, m, o)
print("Probability of overlapping at least", o, "elements:", probability)


overlap_range = range(1, 101)
probabilities = [overlap_probability_at_least(k, n, m, o) for o in overlap_range]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(overlap_range, probabilities, marker='o')
plt.title('Probability of Overlapping at Least o Elements')
plt.xlabel('Number of Overlapping Elements (o)')
plt.ylabel('Probability')
plt.grid(True)
plt.show()
