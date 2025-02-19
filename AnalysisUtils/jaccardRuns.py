import csv
import os

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

def compare_feature_files(folder_path):
    similarity_matrix = []

    for i in range(10):
        run_folder1 = os.path.join(folder_path, f"run{i}")
        feature_file1 = os.path.join(run_folder1, "features_0.csv")

        row_similarities = []
        for j in range(i + 1, 10):
            run_folder2 = os.path.join(folder_path, f"run{j}")
            feature_file2 = os.path.join(run_folder2, "features_0.csv")

            similarity = calculate_jaccard_similarity(feature_file1, feature_file2)
            row_similarities.append(similarity)

        similarity_matrix.append(row_similarities)

    return similarity_matrix

folder_path = './Drug2003_analysis/'
similarity_matrix = compare_feature_files(folder_path)

for i, row in enumerate(similarity_matrix):
    for j, similarity in enumerate(row):
        print(f"Jaccard Similarity between run{i} and run{j+i+1}: {similarity}")

