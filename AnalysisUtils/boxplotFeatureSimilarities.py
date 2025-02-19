import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sys import argv
def normalize_matrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix

def calculate_sorensen_dice(file1, file2):
    set1 = set()
    set2 = set()

    with open(file1, 'r') as csv_file1, open(file2, 'r') as csv_file2:
        csv_reader1 = csv.reader(csv_file1)
        csv_reader2 = csv.reader(csv_file2)

        for row in csv_reader1:
            set1.update(row)

        for row in csv_reader2:
            set2.update(row)
    intersection = len(set1.intersection(set2))
    dice_coefficient = (2 * intersection) / (len(set1) + len(set2))
    return dice_coefficient

def calculate_szymkiewicz_simpson(file1, file2):
    set1 = set()
    set2 = set()

    with open(file1, 'r') as csv_file1, open(file2, 'r') as csv_file2:
        csv_reader1 = csv.reader(csv_file1)
        csv_reader2 = csv.reader(csv_file2)

        for row in csv_reader1:
            set1.update(row)

        for row in csv_reader2:
            set2.update(row)
    intersection = len(set1.intersection(set2))
    smaller_set_size = min(len(set1), len(set2))
    szymkiewicz_simpson = intersection / smaller_set_size
    return szymkiewicz_simpson


def calculate_jaccard_similarity(file1, file2):
    set1 = set()
    set2 = set()

    with open(file1, 'r') as csv_file1, open(file2, 'r') as csv_file2:
        csv_reader1 = csv.reader(csv_file1)
        csv_reader2 = csv.reader(csv_file2)

        for row in csv_reader1:
            set1.update(row)

        for row in csv_reader2:
            set2.update(row)

    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard_similarity = len(intersection) / len(union)
    return jaccard_similarity

def compare_csv_files(folder_path, drug_id1, drug_id2):
    file1 = os.path.join(folder_path, f"Drug{drug_id1}_analysis/best/features_0.csv")
    file2 = os.path.join(folder_path, f"Drug{drug_id2}_analysis/best/features_0.csv")
    
    if os.path.isfile(file1) and os.path.isfile(file2):
        similarity = calculate_szymkiewicz_simpson(file1, file2)
        return similarity
    return None

# Dictionary to store similarity scores by drug
similarity_scores = {}

# Assuming argv[1] and argv[2] are the range of drugs to compare
start_id = int(argv[1])
end_id = int(argv[2])
threshold = float(argv[3])

for drug_id1 in range(start_id, end_id + 1):
    for drug_id2 in range(drug_id1 + 1, end_id + 1):  # Compare each pair once
        score = compare_csv_files('./', drug_id1, drug_id2)
        if score is not None and score > threshold:
            if drug_id1 not in similarity_scores:
                similarity_scores[drug_id1] = []
            if drug_id2 not in similarity_scores:
                similarity_scores[drug_id2] = []
            
            similarity_scores[drug_id1].append(score)
            similarity_scores[drug_id2].append(score)

# Convert dictionary to a DataFrame for easier plotting
df_scores = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in similarity_scores.items()]))

# Plotting
plt.figure(figsize=(12, 6))
df_scores.boxplot()
plt.title('Similarity Scores of Each Drug')
plt.ylabel('Similarity Score')
plt.xlabel('Drug ID')
plt.xticks(rotation=90)  # Rotate x-axis labels by 90 degrees
plt.show()

