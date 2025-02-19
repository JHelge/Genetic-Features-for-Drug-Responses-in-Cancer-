import matplotlib.pyplot as plt
import pandas as pd
import sys
# Load the data from the text file
data = pd.read_csv("../Drug"+str(sys.argv[1])+'_analysis/best/results.txt', delimiter='\t', header=None, names=['Algorithm', 'R2_Score', 'Error'])

# Filter out the SGDRegressor
#data = data[~data['Algorithm'].str.contains("RandomForest")]

# Replace R2 scores that are less than zero with zero
data['R2_Score'] = data['R2_Score'].clip(lower=0)

# Calculate the average R2 Score of all algorithms
average_r2 = data['R2_Score'].mean()

# Plot each algorithm as a separate point in the plot

best_score = data['R2_Score'].max()
best_algorithm = data[data['R2_Score'] == best_score]['Algorithm'].values[0]

# Plotting
plt.figure(figsize=(10, 5))

# Plot each algorithm as a separate point in the plot
for index, row in data.iterrows():
    color = 'green' if row['Algorithm'] == best_algorithm else 'blue'
    plt.bar(row['Algorithm'], row['R2_Score'], color=color, label=f'{row["Algorithm"]} R2={row["R2_Score"]:.2f}')

#for index, row in data.iterrows():
#    plt.bar(row['Algorithm'], row['R2_Score'], label=f'{row["Algorithm"]} R2={row["R2_Score"]:.2f}')

# Add the average bar
plt.bar('Average of Algorithms', average_r2, color='red', label=f'Average R2={average_r2:.2f}')


plt.title('R2 Scores of Various Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('R2 Score')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.ylim(0, 1)  # Adjust y-axis limits if needed based on your data range
plt.grid(True)
# Major grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')

# Minor grid
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

plt.tight_layout()
plt.show()

#################################################################################################


# Load the data from the text file
data = pd.read_csv("../Drug"+str(sys.argv[1])+'_analysis/best/results.txt', delimiter='\t', header=None, names=['Algorithm', 'R2_Score', 'Error'])

# Filter out the SGDRegressor
data = data[~data['Algorithm'].str.contains("RandomForest")]
data = data[~data['Algorithm'].str.contains("BaggingRegressor")]
data = data[~data['Algorithm'].str.contains("GradientBoosting")]
# Replace R2 scores that are less than zero with zero
data['R2_Score'] = data['R2_Score'].clip(lower=0)

# Calculate the average R2 Score of all algorithms
average_r2 = data['R2_Score'].mean()

# Plot each algorithm as a separate point in the plot

best_score = data['R2_Score'].max()
best_algorithm = data[data['R2_Score'] == best_score]['Algorithm'].values[0]

# Plotting
plt.figure(figsize=(10, 5))

# Plot each algorithm as a separate point in the plot
for index, row in data.iterrows():
    color = 'green' if row['Algorithm'] == best_algorithm else 'blue'
    plt.bar(row['Algorithm'], row['R2_Score'], color=color, label=f'{row["Algorithm"]} R2={row["R2_Score"]:.2f}')

#for index, row in data.iterrows():
#    plt.bar(row['Algorithm'], row['R2_Score'], label=f'{row["Algorithm"]} R2={row["R2_Score"]:.2f}')

# Add the average bar
plt.bar('Average of Algorithms', average_r2, color='red', label=f'Average R2={average_r2:.2f}')


plt.title('R2 Scores of Various Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('R2 Score')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.ylim(0, 1)  # Adjust y-axis limits if needed based on your data range
plt.grid(True)
# Major grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')

# Minor grid
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

plt.tight_layout()
plt.show()

########################################################################################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Base directory containing all subdirectories
base_dir = '../'

# Prepare to collect R2 scores for each algorithm
algorithm_scores = {}

# Iterate over each directory
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path) and "Drug" in subdir_path:
        file_path = os.path.join(subdir_path, 'best/results.txt')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, delimiter='\t', header=None, names=['Algorithm', 'R2_Score', 'Error'])
            
            # Process each algorithm
            for index, row in data.iterrows():
                algorithm = row['Algorithm']
                r2_score = max(row['R2_Score'], 0)  # Ensure non-negative R2 score
                
                if algorithm not in algorithm_scores:
                    algorithm_scores[algorithm] = []
                
                algorithm_scores[algorithm].append(r2_score)

# Plotting
plt.figure(figsize=(12, 8))
# Creating a list for algorithm names and corresponding scores for boxplot
algorithm_names = list(algorithm_scores.keys())
score_lists = [algorithm_scores[alg] for alg in algorithm_names]

# Create boxplots
plt.boxplot(score_lists, labels=algorithm_names, notch=True, patch_artist=True)
plt.xticks(rotation=45, ha='right')  # Improve label readability
plt.title('R2 Scores of Various Algorithms Across Experiments')
plt.ylabel('R2 Score')
plt.grid(True)
plt.tight_layout()
plt.show()
########################################################################################


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Base directory containing all subdirectories
base_dir = '../'

# Prepare to collect R2 scores for each algorithm, along with subdir paths
algorithm_scores = {}

# Iterate over each directory
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path) and "Drug" in subdir_path:
        file_path = os.path.join(subdir_path, 'best/results.txt')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, delimiter='\t', header=None, names=['Algorithm', 'R2_Score', 'Error'])
            
            # Process each algorithm
            for index, row in data.iterrows():
                algorithm = row['Algorithm']
                r2_score = max(row['R2_Score'], 0)  # Ensure non-negative R2 score
                
                if algorithm not in algorithm_scores:
                    algorithm_scores[algorithm] = []
                
                # Append both score and subdir path
                algorithm_scores[algorithm].append((r2_score, subdir_path))

# Plotting
plt.figure(figsize=(12, 8))
colors = plt.cm.Paired(np.arange(len(algorithm_scores)))

# Creating a list for algorithm names
algorithm_names = list(algorithm_scores.keys())

# Plot each algorithm's scores
for idx, alg in enumerate(algorithm_names):
    scores, paths = zip(*algorithm_scores[alg])
    plt.boxplot(scores, positions=[idx+1], widths=0.6, patch_artist=True, manage_ticks=False)
    
    # Plot individual scores and label them
    for score, path in zip(scores, paths):
        plt.plot(idx+1, score, 'o', color=colors[idx])  # Mark the individual scores
        plt.text(idx+1, score, path.split('/')[-1], fontsize=8, ha='center', va='bottom')  # Annotate with subdir path

plt.xticks(np.arange(1, len(algorithm_names) + 1), algorithm_names, rotation=45, ha='right')  # Set x-ticks properly
plt.title('R2 Scores of Various Algorithms Across Experiments with Subdirectory Labels')
plt.ylabel('R2 Score')
plt.grid(True)
plt.tight_layout()  # Adjust layout to make room for labels
plt.show()
####################################################################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Base directory containing all subdirectories
base_dir = '../'

# Prepare to collect R2 scores for each algorithm, along with subdir paths
algorithm_scores = {}

# Iterate over each directory
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path) and "Drug" in subdir_path:
        file_path = os.path.join(subdir_path, 'best/results.txt')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, delimiter='\t', header=None, names=['Algorithm', 'R2_Score', 'Error'])
            
            # Process each algorithm
            for index, row in data.iterrows():
                algorithm = row['Algorithm']
                r2_score = max(row['R2_Score'], 0)  # Ensure non-negative R2 score
                
                if algorithm not in algorithm_scores:
                    algorithm_scores[algorithm] = []
                
                # Append both score and subdir path
                algorithm_scores[algorithm].append((r2_score, subdir_path))

# Plotting
plt.figure(figsize=(12, 8))
colors = plt.cm.Paired(np.arange(len(algorithm_scores)))

# Creating a list for algorithm names
algorithm_names = list(algorithm_scores.keys())

# Plot each algorithm's scores
for idx, alg in enumerate(algorithm_names):
    scores, paths = zip(*algorithm_scores[alg])
    scores = np.array(scores)
    result = plt.boxplot(scores, positions=[idx+1], widths=0.6, patch_artist=True, manage_ticks=False)
    
    # Determine outliers
    whiskers = [whisker.get_ydata() for whisker in result['whiskers']]
    lower_whisker = whiskers[0][1]
    upper_whisker = whiskers[1][1]

    # Plot individual scores and label outliers
    for score, path in zip(scores, paths):
        if score < lower_whisker or score > upper_whisker:
            plt.plot(idx+1, score, 'o', color=colors[idx])  # Mark the outlier
            plt.text(idx+1, score, path.split('/')[-1], fontsize=8, ha='center', va='bottom')  # Annotate with subdir path

plt.xticks(np.arange(1, len(algorithm_names) + 1), algorithm_names, rotation=45, ha='right')  # Set x-ticks properly
plt.title('R2 Scores of Various Algorithms Across Experiments with Outlier Labels')
plt.ylabel('R2 Score')
plt.grid(True)
plt.tight_layout()  # Adjust layout to make room for labels
plt.show()
################################################################################################




