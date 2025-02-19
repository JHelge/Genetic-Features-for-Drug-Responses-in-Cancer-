import pandas as pd
import os
import matplotlib.pyplot as plt

# Path to the specific subdirectory
specific_directory = './Drug1034_analysis'

# Feature to find in features_0.csv and to analyze the correlation
feature_to_analyze = 'RNY1_exp'

# Paths to CSV files
features_path = os.path.join(specific_directory, 'best/features_0.csv')
data_path = os.path.join(specific_directory, 'best/data_0.csv')
labels_path = os.path.join(specific_directory, 'best/labels.csv')

# Load data
if os.path.exists(features_path) and os.path.exists(data_path) and os.path.exists(labels_path):
    features = pd.read_csv(features_path, header=None)
    data = pd.read_csv(data_path, header=None)
    labels = pd.read_csv(labels_path, header=None)

    # Find the index of the feature
    feature_index = features[features[0] == feature_to_analyze].index.tolist()
    if feature_index:
        feature_index = feature_index[0]
        feature_data = data[feature_index]
        correlation = feature_data.corr(labels[0])

        # Plotting the scatter plot of feature values against labels
        plt.figure(figsize=(10, 6))
        plt.scatter(feature_data, labels[0], alpha=0.5)
        plt.title(f'Scatter Plot of {feature_to_analyze} vs. Labels\nCorrelation: {correlation:.2f}')
        plt.xlabel(f'{feature_to_analyze} Values')
        plt.ylabel('Label Values')
        plt.grid(True)
        plt.show()
    else:
        print(f"Feature {feature_to_analyze} not found in {features_path}")
else:
    print(f"Required files not found in {specific_directory}")

