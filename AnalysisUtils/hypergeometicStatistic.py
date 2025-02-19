from scipy.stats import hypergeom
import numpy as np

# Total number of features
N = 38977  # Example: Total number of genes

# Features in Set B
K = 349   # Example: Number of genes involved in a disease

# Features in Set A
n = 223    # Example: Number of genes expressed under certain conditions

# Overlap between Set A and Set B
k = 12     # Example: Observed number of disease genes also expressed

# Perform the hypergeometric test
p_value = hypergeom.sf(k-1, N, K, n)

print(f"P-value: {p_value}")

