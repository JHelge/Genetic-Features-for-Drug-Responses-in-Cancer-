import pandas as pd
import os
import fnmatch
import matplotlib.pyplot as plt

# Base directory where the subdirectories are located
base_directory = './'

# Find all subdirectories matching "Drug*_analysis/best"
directories = [os.path.join(base_directory, d, 'best') for d in os.listdir(base_directory) 
               if os.path.isdir(os.path.join(base_directory, d)) and fnmatch.fnmatch(d, 'Drug*_analysis')]

def load_csv(path):
    df = pd.read_csv(path, header=None)
    print(f"Loaded from {path}")
    return df

# List to hold all dataframes
dataframes = []

# Load data from each directory
for directory in directories:
    csv_path = os.path.join(directory, 'features_0.csv')
    if os.path.exists(csv_path):
        df = load_csv(csv_path)
        dataframes.append(df)
    else:
        print(f"No CSV file found in {csv_path}")

# Check if any data was loaded
if dataframes:
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Count occurrences of each feature
    feature_counts = combined_df[0].value_counts()  # Assuming features are in the first column

    # Optionally, print the feature counts
    print("Feature Occurrences")
    print(feature_counts)
    
    # Print the 10 most frequent features
    print("10 Most Frequent Features")
    print(feature_counts.head(10))

    # Plotting the histogram of feature occurrences
    plt.figure(figsize=(10, 6))
    feature_counts.hist(bins=113, edgecolor='black')
    plt.title('Histogram of Feature Occurrences')
    plt.xlabel('Occurrences')
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.show()
    # Save the feature occurrences to a new CSV file
    feature_counts.to_csv('feature_occurrences.csv', header=["Occurrences"], index_label='Feature')
    # Save the 100 rarest feature occurrences to a new CSV file
    feature_counts.head(100).to_csv('feature100_occurrences.csv', header=["Occurrences"], index_label='Feature')
    # Save the 10 rarest feature occurrences to a new CSV file
    feature_counts.head(10).to_csv('feature10_occurrences.csv', header=["Occurrences"], index_label='Feature')
    # Save the feature occurrences to a new CSV file
    #feature_counts.to_csv('feature_occurrences.csv', header=True, index_label='Feature', column_label='Occurrences')
else:
    print("No dataframes were loaded.")

