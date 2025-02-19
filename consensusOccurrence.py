import pandas as pd

# Load the CSV files
#consensus_df = pd.read_csv('driverGenes.csv', header=None, names=['Feature'])
consensus_df = pd.read_csv('consensusCOSMIC_Large.csv', header=None, names=['Feature'])
occurrences_df = pd.read_csv('possiblyImportantFeatures.csv', names=['Feature', 'Occurrences'])

# Remove the last four characters from the 'Feature' column in the occurrences DataFrame
occurrences_df['Feature'] = occurrences_df['Feature'].str[:-4]
# Remove duplicates from occurrences_df based on the 'Feature' column
occurrences_df = occurrences_df.drop_duplicates(subset=['Feature'])
# Merge the dataframes to find matching features and their occurrences
result_df = pd.merge(consensus_df, occurrences_df, on='Feature', how='left')

# Display the results
print(result_df)

# Count the number of NaNs in the 'Occurrences' column
nan_count = result_df['Occurrences'].isna().sum()
print("Number of NaNs in 'Occurrences':", nan_count)


# Filter out rows where 'Occurrences' is NaN
non_nan_features = result_df.dropna(subset=['Occurrences'])

# Display the non-NaN features and their occurrences
print(non_nan_features)
print(len(non_nan_features['Feature']))
