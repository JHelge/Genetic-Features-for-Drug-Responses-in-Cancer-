import pandas as pd
import os
import fnmatch
import matplotlib.pyplot as plt
import numpy as np

# Base directory where the subdirectories are located
base_directory = './'

# Find all subdirectories matching "Drug*_analysis"
directories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) 
               if os.path.isdir(os.path.join(base_directory, d)) and fnmatch.fnmatch(d, 'Drug*_analysis')]

# List to hold the number of lines in each features_0.csv
feature_set_sizes = []

# Load data from each directory
for directory in directories:
    csv_path = os.path.join(directory, 'best/features_0.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, header=None)
        feature_set_sizes.append(len(df))
        print(f"Loaded {len(df)} features from {csv_path}")
    else:
        print(f"No CSV file found in {csv_path}")

# Check if any sizes were collected
if feature_set_sizes:
    # Create a histogram of the number of features per CSV
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(feature_set_sizes, bins=20, color='blue', edgecolor='black')

    # Add labels to bars
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        # Label the raw counts
        if count > 0:
            plt.annotate(f'{int(count)}', xy=(x, count), xytext=(0, 3), 
                         textcoords="offset points", ha='center', va='bottom')
    
    plt.title('Histogram of Number of Features in Each features_0.csv')
    plt.xlabel('Number of Features')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
else:
    print("No features_0.csv files found or no lines to count.")

