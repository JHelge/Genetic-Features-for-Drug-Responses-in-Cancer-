import pandas as pd
import matplotlib.pyplot as plt

# Load the data
feature_counts = pd.read_csv('feature_occurrences.csv', index_col='Feature')

# Create masks for each suffix
mut_mask = feature_counts.index.str.endswith('_mut')
exp_mask = feature_counts.index.str.endswith('_exp')
cnv_mask = feature_counts.index.str.endswith('_cnv')

# Divide the DataFrame into three separate DataFrames based on suffix
features_mut = feature_counts[mut_mask]
features_exp = feature_counts[exp_mask]
features_cnv = feature_counts[cnv_mask]

print("MUT SHAPE")
print(features_mut.shape)
print("EXP SHAPE")
print(features_exp.shape)
print("CNV SHAPE")
print(features_cnv.shape)

# Plotting each feature group
fig, axes = plt.subplots(3, 1, figsize=(10, 9))

# Plot for "_mut" features
axes[0].bar(features_mut.index, features_mut['Occurrences'], color='b')
axes[0].set_title('Mutation Features (_mut)')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Occurrences')
axes[0].tick_params(axis='x', rotation=0)

# Plot for "_exp" features
axes[1].bar(features_exp.index, features_exp['Occurrences'], color='r')
axes[1].set_title('Expression Features (_exp)')
axes[1].set_xlabel('Features')
axes[1].set_ylabel('Occurrences')
axes[1].tick_params(axis='x', rotation=90)

# Plot for "_cnv" features
axes[2].bar(features_cnv.index, features_cnv['Occurrences'], color='g')
axes[2].set_title('Copy Number Variation Features (_cnv)')
axes[2].set_xlabel('Features')
axes[2].set_ylabel('Occurrences')
axes[2].tick_params(axis='x', rotation=90)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

