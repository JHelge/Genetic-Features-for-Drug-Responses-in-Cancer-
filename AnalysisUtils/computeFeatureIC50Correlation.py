import pandas as pd
import os
import fnmatch
import matplotlib.pyplot as plt
import numpy as np

# Base directory where the subdirectories are located
base_directory = './'

# Feature to find in features_0.csv and to analyze the correlation
feature_to_analyze = 'NCOA1_mut'

# Find all subdirectories matching "Drug*_analysis"
directories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) 
               if os.path.isdir(os.path.join(base_directory, d)) and fnmatch.fnmatch(d, 'Drug*_analysis')]

# Dictionary to hold correlations
correlations = {}

# Iterate through each directory
for directory in directories:
    features_path = os.path.join(directory, 'best/features_0.csv')
    data_path = os.path.join(directory, 'best/data_0.csv')
    labels_path = os.path.join(directory, 'best/labels.csv')
    
    if os.path.exists(features_path) and os.path.exists(data_path) and os.path.exists(labels_path):
        # Load data
        features = pd.read_csv(features_path, header=None)
        data = pd.read_csv(data_path, header=None)
        labels = pd.read_csv(labels_path, header=None)
        
        # Find the index of the feature
        feature_index = features[features[0] == feature_to_analyze].index.tolist()
        if feature_index:
            feature_index = feature_index[0]
            feature_data = data[feature_index]
            correlation = feature_data.corr(labels[0])
            correlations[directory] = correlation
            print(f"Correlation in {directory} is {correlation}")
        else:
            print(f"Feature {feature_to_analyze} not found in {features_path}")
    else:
        print(f"Required files not found in {directory}")

# Plot the correlations
if correlations:
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(correlations)), list(correlations.values()), align='center')
    plt.xticks(range(len(correlations)), list(correlations.keys()), rotation=45)
    plt.ylabel('Correlation')
    plt.title('Correlation of Feature with Labels Across Different Datasets')
    plt.show()
else:
    print("No correlations computed, possibly due to missing files or the feature not being found.")

