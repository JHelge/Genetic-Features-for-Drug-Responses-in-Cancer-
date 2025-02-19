import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy.stats as stats
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
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

def compare_pathway_names(file_path, id1, id2):
    # Öffne die XLSX-Datei
    data = pd.read_excel(file_path)

    # Filtere die Daten basierend auf den IDs
    filtered_data = data[data['DRUG_ID'].isin([id1, id2])]

    # Überprüfe, ob die Werte in der Spalte "PATHWAY_NAME" identisch sind
    are_identical = filtered_data['PATHWAY_NAME'].nunique() == 1

    # Rückgabe des Ergebnisses
    return are_identical



def compare_csv_files(folder_path, file_1, file_2, threshold):
    file_list = []
    ID_list = []
    #print("compare_csv_files")

    file1 = os.path.join(folder_path, f"Drug{file_1}_analysis/best/features_0.csv")
    #print(os.path.isfile(file_path))
    

    file2 = os.path.join(folder_path, f"Drug{file_2}_analysis/best/features_0.csv")

    #similarity = calculate_sorensen_dice(file1, file2)
    similarity = calculate_szymkiewicz_simpson(file1, file2)
    #similarity = calculate_jaccard_similarity(file1, file2)
    result = f"Similarity {file1} and {file2}: {similarity:.5f}"
    if similarity >threshold:
        print(result)
        

        set1 = set()
        set2 = set()

        with open(file1, 'r') as file1, open(file2, 'r') as file2:
            csv_reader1 = csv.reader(file1)
            csv_reader2 = csv.reader(file2)

            for row in csv_reader1:
                set1.update(row)

            for row in csv_reader2:
                set2.update(row)

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        print("intersection")
        print(len(intersection))
    
    return result 






folder_path = './'
file_1 =1803
file_2= 1804
print("compare_csv_files")
print(argv)
result = compare_csv_files(folder_path, argv[1], argv[2],0)

for i in range(1003, 2500):

    file_1 =i
    file_2= argv[1]
    #print(f"Drug{file_2}_analysis/run0/features_0.csv")
    #print(os.path.isfile(f"Drug{file_2}_analysis/run0/features_0.csv"))
    if os.path.isfile(f"Drug{file_2}_analysis/best/features_0.csv") and os.path.isfile(f"Drug{file_1}_analysis/best/features_0.csv") :
        result = compare_csv_files(folder_path, file_1, file_2,float(argv[3]))


for i in range(1003, 2500):

    file_1 =i
    file_2= argv[2]
    #print(f"Drug{file_2}_analysis/run0/features_0.csv")
    #print(os.path.isfile(f"Drug{file_2}_analysis/run0/features_0.csv"))
    if os.path.isfile(f"Drug{file_2}_analysis/run0/features_0.csv") and os.path.isfile(f"Drug{file_1}_analysis/run0/features_0.csv") :
        result = compare_csv_files(folder_path, file_1, file_2, float(argv[3]))

