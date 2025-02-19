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


