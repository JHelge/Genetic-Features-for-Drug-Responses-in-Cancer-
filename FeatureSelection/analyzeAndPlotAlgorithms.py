import pandas as pd
import matplotlib.pyplot as plt

import sys

dataPath="../Drug"+str(sys.argv[1])+"_analysis/run0/results.txt"


# Reading the data using StringIO to simulate file reading
df = pd.read_csv(dataPath,sep="\t", header=None, names=["Model", "R2_Score", "Metric2"])
df = df[~df['Model'].str.contains('SGDRegressor')]

# Group by model to collect all entries for each model type
grouped = df.groupby("Model")

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
for name, group in grouped:
    group.reset_index().plot(kind='line', x='index', y='R2_Score', ax=ax, label=name)

plt.title("Comparison of R² Scores Across Different Models")
plt.xlabel("Experiment Number")
plt.ylabel("R² Score")
plt.legend(title="Model")
plt.grid(True)
plt.savefig('algorithm_r2_scores.png')
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample data loading method; replace this with your actual data file
data = pd.read_csv(dataPath, delimiter='\t', header=None, names=['Algorithm', 'R2_Score', 'Error'])

# Define the subplot grid dimensions
nrows, ncols = 2, 4  # For 8 algorithms
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Flatten the axes for easier handling
axes = axes.flatten()

# Extract unique algorithms (assumed exactly 8 for this setup)
algorithms = data['Algorithm'].unique()

# We're assuming exactly 8 algorithms, if there are more or fewer, adjust accordingly.
assert len(algorithms) == 8, "The number of unique algorithms should be exactly 8."

# Define maximum number of experiments (x-axis limit)
max_experiments = 40

for i, algorithm in enumerate(algorithms):
    # Filter the data for the current algorithm
    algorithm_data = data[data['Algorithm'] == algorithm]

    # Get the scores and errors
    r2_scores = algorithm_data['R2_Score'].tolist()
    #errors = algorithm_data['Error'].tolist()
    
    # Generate x values based on actual experiments conducted
    x_ticks = list(range(1, len(r2_scores) + 1))
    
    # Plotting
    axes[i].plot(x_ticks, r2_scores, label='R2 Score', marker='o')
  #  axes[i].plot(x_ticks, errors, label='Error', marker='x')
    axes[i].set_title(algorithm)
    axes[i].set_xlabel('Experiment Number')
    axes[i].set_ylabel('Value')
    axes[i].set_xlim(1, max_experiments)  # Setting x-axis limits
    axes[i].legend()

plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample data loading method; replace this with your actual data file
data = pd.read_csv(dataPath, delimiter='\t', header=None, names=['Algorithm', 'R2_Score', 'Error'])

# Define the subplot grid dimensions
nrows, ncols = 2, 4  # For 8 algorithms
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Flatten the axes for easier handling
axes = axes.flatten()

# Extract unique algorithms (assumed exactly 8 for this setup)
algorithms = data['Algorithm'].unique()

# We're assuming exactly 8 algorithms, if there are more or fewer, adjust accordingly.
assert len(algorithms) == 8, "The number of unique algorithms should be exactly 8."

# Maximum number of experiments to display per plot
max_display_experiments = 24

for i, algorithm in enumerate(algorithms):
    # Filter the data for the current algorithm
    algorithm_data = data[data['Algorithm'] == algorithm]

    # Get the scores and errors
    r2_scores = algorithm_data['R2_Score'].tolist()
    errors = algorithm_data['Error'].tolist()
    
    # If the number of experiments is more than 30, slice to get the last 30
    if len(r2_scores) > max_display_experiments:
        r2_scores = r2_scores[-max_display_experiments:]
        errors = errors[-max_display_experiments:]

    # Generate x values for the last 30 experiments
    x_ticks = list(range(len(r2_scores) - max_display_experiments + 1, len(r2_scores) + 1))
    
    # Plotting
    axes[i].plot(x_ticks, r2_scores, label='R2 Score', marker='o')
   # axes[i].plot(x_ticks, errors, label='Error', marker='x')
    axes[i].set_title(algorithm)
    axes[i].set_xlabel('Experiment Number')
    axes[i].set_ylabel('Value')
    axes[i].set_xlim(x_ticks[0], x_ticks[-1])  # Setting x-axis limits specifically for last 30 entries
    axes[i].legend()

plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv(dataPath, delimiter='\t', header=None, names=['Algorithm', 'R2_Score', 'Error'])
data = data[~data['Algorithm'].str.contains('SGDRegressor')]

# Extract unique algorithms and verify their count
algorithms = data['Algorithm'].unique()
assert len(algorithms) == 7, "The number of unique algorithms should be exactly 7."

# Create lists to store R2 scores and errors for aggregation
r2_scores_all = []
errors_all = []

# Collect all R2 scores and errors
for algorithm in algorithms:
    algorithm_data = data[data['Algorithm'] == algorithm]

    r2_scores = algorithm_data['R2_Score'].tolist()
    errors = algorithm_data['Error'].tolist()

    # If the number of experiments exceeds 30, take the last 30
    if len(r2_scores) > 30:
        r2_scores = r2_scores[-30:]
        errors = errors[-30:]

    r2_scores_all.append(r2_scores)
    errors_all.append(errors)

# Convert lists to NumPy arrays for easier average calculation
r2_scores_all = np.array(r2_scores_all)
#errors_all = np.array(errors_all)


# Calculate the mean across all algorithms for each experiment index
average_r2_scores = np.mean(r2_scores_all, axis=0)
#average_errors = np.mean(errors_all, axis=0)

# Generate x values for the experiments
x_ticks = np.arange(1, len(average_r2_scores) + 1)


# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x_ticks, average_r2_scores, label='Average R2 Score', marker='o', color='blue')
#plt.plot(x_ticks, average_errors, label='Average Error', marker='x', color='red')
plt.title('Average Performance of All Algorithms Over Last 30 Experiments')
plt.xlabel('Experiment Number')
plt.ylabel('Average Value')
plt.xticks(x_ticks)
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv(dataPath, delimiter='\t', header=None, names=['Algorithm', 'R2_Score', 'Error'])
data = data[~data['Algorithm'].str.contains('SGDRegressor')]

# Extract unique algorithms and verify their count
algorithms = data['Algorithm'].unique()
assert len(algorithms) == 7, "The number of unique algorithms should be exactly 7."

# Create lists to store R2 scores and errors for aggregation
r2_scores_all = []
errors_all = []

# Collect all R2 scores and errors
for algorithm in algorithms:
    algorithm_data = data[data['Algorithm'] == algorithm]

    r2_scores = algorithm_data['R2_Score'].tolist()
    errors = algorithm_data['Error'].tolist()



    r2_scores_all.append(r2_scores)
    errors_all.append(errors)

# Convert lists to NumPy arrays for easier average calculation
r2_scores_all = np.array(r2_scores_all)
#errors_all = np.array(errors_all)

r2_scores_all = np.maximum(r2_scores_all, 0)

# Calculate the mean across all algorithms for each experiment index
average_r2_scores = np.mean(r2_scores_all, axis=0)
#average_errors = np.mean(errors_all, axis=0)

# Generate x values for the experiments
x_ticks = np.arange(1, len(average_r2_scores) + 1)


# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x_ticks, average_r2_scores, label='Average R2 Score', marker='o', color='blue')
#plt.plot(x_ticks, average_errors, label='Average Error', marker='x', color='red')
plt.title('Average Performance of All Algorithms Over Last 30 Experiments')
plt.xlabel('Experiment Number')
plt.ylabel('Average Value')
plt.xticks(x_ticks)
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Base directory containing all subdirectories
base_dir = '../'

# Prepare to collect average R2 scores and errors for each algorithm
aggregate_scores = {}

best_algo_score = 0.0
worst_algo_score = 1.0
# Iterate over each directory
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    print(subdir_path)
    if os.path.isdir(subdir_path) and "Drug" in subdir_path:
        print(subdir_path)
        # Assuming the same file name in each directory
        #file_path = os.path.join(subdir_path, str(subdir_path+'/run0/results.txt'))
        file_path = str(subdir_path+'/best/results.txt')
        print(file_path)
        data = pd.read_csv(file_path, delimiter='\t', header=None, names=['Algorithm', 'R2_Score', 'Error'])

# Process each algorithm
        for algorithm in data['Algorithm'].unique():
            if algorithm not in aggregate_scores:
                aggregate_scores[algorithm] = []  # Initialize if not already present

            algorithm_data = data[data['Algorithm'] == algorithm]

            # Ensure non-negative R2 scores safely using .loc
            algorithm_data.loc[:, 'R2_Score'] = np.maximum(algorithm_data['R2_Score'], 0)

            # Calculate average scores
            mean_r2_score = algorithm_data['R2_Score'].mean()
            mean_error = algorithm_data['Error'].mean()
            if mean_r2_score > best_algo_score:
                best_algo_score = mean_r2_score
            if mean_r2_score < worst_algo_score:
                worst_algo_score = mean_r2_score
            # Append averages to the respective lists
            aggregate_scores[algorithm].append((mean_r2_score, mean_error))

# Average the averages for each algorithm
#print("Best Score: ", best_algo_score)
#print("Worst Score: ", worst_algo_score)
final_averages = {alg: (np.mean([t[0] for t in scores]), np.mean([t[1] for t in scores])) for alg, scores in aggregate_scores.items()}




# Plotting
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(final_averages)))

for idx, (alg, (avg_r2_score, avg_error)) in enumerate(final_averages.items()):
    print(alg)
    print(avg_r2_score)
    plt.bar(alg, avg_r2_score, color=colors[idx], label=f'{alg} R2')
    #plt.scatter(alg, avg_error, color='red', s=100, label=f'Algorithm {alg} Error' if idx == 0 else "")

plt.title('Final Average R2 Scores Across all Drugs')
plt.xlabel('Algorithms')
plt.ylabel('Average Score')
#plt.legend()
plt.xticks(rotation=90)
plt.grid(True)
plt.show()





import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Base directory containing all subdirectories
base_dir = '../'

# Initialize a dictionary to collect scores and errors for each algorithm
aggregate_scores = {}
aggregate_errors = {}

# Iterate over each directory
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path) and "Drug" in subdir_path:
        # Assuming the same file name in each directory
        file_path = str(subdir_path+'/best/results.txt')
        data = pd.read_csv(file_path, delimiter='\t', header=None, names=['Algorithm', 'R2_Score', 'Error'])

        # Process each algorithm
        for algorithm in data['Algorithm'].unique():
            if algorithm not in aggregate_scores:
                aggregate_scores[algorithm] = []
                aggregate_errors[algorithm] = []

            algorithm_data = data[data['Algorithm'] == algorithm]
            algorithm_data['R2_Score'] = np.maximum(algorithm_data['R2_Score'], 0)  # Ensure non-negative R2 scores

            # Aggregate data
            mean_r2_score = algorithm_data['R2_Score'].mean()

            #mean_error = algorithm_data['Error'].mean()

            aggregate_scores[algorithm].append(mean_r2_score)
          #  aggregate_errors[algorithm].append(mean_error)

# Compute the average of averages for each algorithm

experiment_numbers = range(1, max(len(scores) for scores in aggregate_scores.values()) + 1)
average_scores = {alg: np.mean(scores) for alg, scores in aggregate_scores.items()}
#average_errors = {alg: np.mean(errors) for alg, errors in aggregate_errors.items()}

# Plotting
plt.figure(figsize=(20, 10))
colors = plt.cm.viridis(np.linspace(0, 1, len(aggregate_scores)))

for idx, (algorithm, scores) in enumerate(aggregate_scores.items()):
    plt.plot(experiment_numbers, scores, label=f'{algorithm} R2 Score', color=colors[idx], marker='o')
    #plt.plot(experiment_numbers, aggregate_errors[algorithm], label=f'{algorithm} Error', linestyle='--', color=colors[idx])

plt.title('Average R2 Scores Across All Experiments for Each Algorithm')
plt.xlabel('Experiment Number')
plt.ylabel('Average Value')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Base directory containing all subdirectories
base_dir = '../'

# Initialize a dictionary to collect scores and errors for each algorithm
aggregate_scores = {}
aggregate_errors = {}

# Iterate over each directory
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path) and "Drug" in subdir_path:
        # Assuming the same file name in each directory
        file_path = str(subdir_path+'/run0/results.txt')
        data = pd.read_csv(file_path, delimiter='\t', header=None, names=['Algorithm', 'R2_Score', 'Error'])

        # Process each algorithm
        for algorithm in data['Algorithm'].unique():
            if algorithm not in aggregate_scores:
                aggregate_scores[algorithm] = []
                aggregate_errors[algorithm] = []

            algorithm_data = data[data['Algorithm'] == algorithm]
            algorithm_data['R2_Score'] = np.maximum(algorithm_data['R2_Score'], 0)  # Ensure non-negative R2 scores

            # Aggregate data
            mean_r2_score = algorithm_data['R2_Score'].mean()
            mean_error = algorithm_data['Error'].mean()

            aggregate_scores[algorithm].append(mean_r2_score)
            aggregate_errors[algorithm].append(mean_error)

# Calculate the number of experiments based on the assumption of a fixed 8 algorithms
num_experiments = min(len(scores) for scores in aggregate_scores.values())

# Ensuring we plot the data only for the first 40 experiments if more are available
num_experiments = min(num_experiments, 40)

# Plotting
plt.figure(figsize=(20, 10))
colors = plt.cm.viridis(np.linspace(0, 1, len(aggregate_scores)))

for idx, (algorithm, scores) in enumerate(aggregate_scores.items()):
    plt.plot(range(1, num_experiments + 1), scores[:num_experiments], label=f'{algorithm} R2 Score', color=colors[idx], marker='o')
    #plt.plot(range(1, num_experiments + 1), aggregate_errors[algorithm][:num_experiments], label=f'{algorithm} Error', linestyle='--', color=colors[idx])

plt.title('Average R2 Scores Across Algorithms Per Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Average Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


