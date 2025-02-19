from scipy.special import comb
import math

def overlap_probability_exactly(k, n, m, o):
    if o > min(n, m) or o > k:
        return 0
    if n - o > k - o or m - o > k - n:
        return 0
    numerator = comb(k, o) * comb(k - o, n - o) * comb(k - n, m - o)
    denominator = comb(k, n) * comb(k, m)
    #print(denominator)
    return numerator / denominator

def overlap_probability_at_least(k, n, m, o):
    max_overlap = min(n, m)
    probability_sum = 0
    for o_prime in range(o, max_overlap + 1):
        probability_sum += overlap_probability_exactly(k, n, m, o_prime)
    return probability_sum

# Example usage
k = 38977  # Total number of elements in the set
n = 349 # Size of the first subset
m = 437   # Size of the second subset
o = 118    # Minimum number of overlapping elements

probability = overlap_probability_at_least(k, n, m, o)
print("Probability of overlapping at least", o, "elements:", probability)

