import os

# Path to the directory containing Cluster_* subdirectories
base_dir = '../'

# Iterate over each subdirectory in the base directory
for subdir in os.listdir(base_dir):
    if subdir.startswith('Cluster_'):
        cluster_path = os.path.join(base_dir, subdir)
        
        if os.path.isdir(cluster_path):
            intersection_file = os.path.join(cluster_path, 'directories.csv')
            
            # Check if the intersection.csv file exists
            if os.path.isfile(intersection_file):
                with open(intersection_file, 'r') as file:
                    # Count the number of lines in the file
                    line_count = sum(1 for line in file)
                    
                print(f"{subdir}: {line_count} lines in intersection.csv")
            else:
                print(f"{subdir}: intersection.csv not found")

