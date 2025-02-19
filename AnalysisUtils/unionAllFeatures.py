import pandas as pd
import os
import fnmatch

# Base directory where the subdirectories are located
base_directory = './'

# Find all subdirectories matching "Drug*_analysis/best"
directories = [os.path.join(base_directory, d, 'best') for d in os.listdir(base_directory) 
               if os.path.isdir(os.path.join(base_directory, d)) and fnmatch.fnmatch(d, 'Drug*_analysis')]

def load_and_print_csv(path):
    df = pd.read_csv(path, header=None)
    #print(f"Loaded from {path}")
    #print(df)
    return df

# Check if any directories were found
if directories:
    # Load the initial DataFrame from the first directory
    initial_path = os.path.join(directories[0], 'features_0.csv')
    if os.path.exists(initial_path):
        union_df = load_and_print_csv(initial_path)
    else:
        print(f"No CSV file found in {initial_path}")
        exit()

    # Iterate over the remaining directories and compute the union
    for directory in directories[1:]:
        csv_path = os.path.join(directory, 'features_0.csv')
        if os.path.exists(csv_path):
            df = load_and_print_csv(csv_path)
            union_df = pd.merge(union_df, df, how='outer')
        else:
            print(f"No CSV file found in {csv_path}")

    # Optionally, print the union
    #print("union")
    #print(union_df)
    print(union_df.values.ravel().tolist())
    # Save the union to a new CSV file
    # union_df.to_csv('union.csv', index=False, header=False)
else:
    print("No matching directories found.")

