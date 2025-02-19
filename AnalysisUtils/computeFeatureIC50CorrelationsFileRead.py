import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch
import scipy.stats as stats

def find_feature_indices(features_df, feature_list):
    """Helper function to find indices of desired features in the features DataFrame."""
    indices = []
    for feature in feature_list:
        if feature in features_df[0].values:
            idx = features_df.index[features_df[0] == feature].tolist()[0]
            indices.append(idx)
    return indices

def plot_feature_correlations(base_directory, features_to_analyze_path):
    # Load the features to analyze from a CSV file
    features_to_analyze_df = pd.read_csv(features_to_analyze_path, header=None)
    feature_list = features_to_analyze_df[0].tolist()
    
    directories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) 
                   if os.path.isdir(os.path.join(base_directory, d)) and fnmatch.fnmatch(d, 'Drug*_analysis')]

    all_correlations = pd.DataFrame()

    for directory in directories:
        features_path = os.path.join(directory, 'features_0.csv')
        data_path = os.path.join(directory, 'data_0.csv')
        labels_path = os.path.join(directory, 'labels.csv')
        
        if os.path.exists(features_path) and os.path.exists(data_path) and os.path.exists(labels_path):
            features = pd.read_csv(features_path, header=None)
            data = pd.read_csv(data_path, header=None)
            labels = pd.read_csv(labels_path, header=None)

            feature_indices = find_feature_indices(features, feature_list)

            # Calculate correlation and p-value for each feature
            for index in feature_indices:
                feature_data = data[index]
                correlation, p_value = stats.pearsonr(feature_data, labels.iloc[:, 0])
                new_row = pd.DataFrame({
                    'Feature': [features.iloc[index, 0]],
                    'Correlation': [correlation],
                    'P-value': [p_value]
                })
                all_correlations = pd.concat([all_correlations, new_row], ignore_index=True)

    # Handling if correlations were found
    if not all_correlations.empty:
        # Filter significant correlations based on p-value < 0.05
        significant_correlations = all_correlations[all_correlations['P-value'] < 0.05]

        # Sort by the absolute values of correlations to find highest
        significant_correlations['AbsCorrelation'] = significant_correlations['Correlation'].abs()
        sorted_correlations = significant_correlations.sort_values(by='AbsCorrelation', ascending=False).drop(columns='AbsCorrelation')

        # Print the 10 highest significant correlations
        print(sorted_correlations.head(10))

        # Optionally, you can plot these top correlations
        top_correlations = sorted_correlations.head(10)
        plt.figure(figsize=(12, 8))
        top_correlations.set_index('Feature')['Correlation'].plot(kind='bar', color='skyblue')
        plt.title('Top 10 Significant Feature Correlations with Labels Across All Subdirectories')
        plt.xlabel('Features')
        plt.ylabel('Pearson Correlation Coefficient')
        plt.show()
    else:
        print("No valid or significant correlations were calculated.")

# Example usage
base_directory = './'
features_to_analyze_path = 'possiblyImportantFeatures.csv'  # Path to the CSV file with features to analyze
plot_feature_correlations(base_directory, features_to_analyze_path)

