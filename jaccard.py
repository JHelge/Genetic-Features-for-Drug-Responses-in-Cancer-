import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy.stats as stats
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram

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



def compare_csv_files(folder_path):
    file_list = []
    ID_list = []
    print("compare_csv_files")
    for i in range(1003, 2501):
        file_path = os.path.join(folder_path, f"Drug{i}_analysis/best/features_0.csv")
        #print(os.path.isfile(file_path))
        if os.path.isfile(file_path):
            file_list.append(file_path)
            ID_list.append(i)
    num_files = len(file_list)
    results = []
    similarity_matrix = np.zeros((num_files, num_files))
    for i in range(num_files):
        file1 = file_list[i]
        ID1 = ID_list[i]
        for j in range(i + 1, num_files):
            file2 = file_list[j]
            ID2 = ID_list[j]
            #similarity = calculate_sorensen_dice(file1, file2)
            similarity = calculate_szymkiewicz_simpson(file1, file2)
            #similarity = calculate_jaccard_similarity(file1, file2)
            result = f"Similarity between {file1} and {file2}: {similarity:.5f}"
            if similarity>0.299:
               print(result)
               # Beispielaufruf der Funktion
               #result = compare_pathway_names("GDSC2_fitted_dose_response_24Jul22.xlsx", ID1, ID2)

               # Ausgabe des Ergebnisses
               #if result:
               #   print("Die Werte für die Spalte PATHWAY_NAME sind identisch.")
               #else:
               #   print("Die Werte für die Spalte PATHWAY_NAME sind nicht identisch.")
            #print(result)
            results.append(result)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    return results, similarity_matrix, file_list



def plot_heatmap(similarity_matrix, file_list):
#    filtered_matrix = np.where(similarity_matrix > 0.1, similarity_matrix, np.nan)
#    num_files = len(file_list)
#    filtered_file_list = [file_list[i] for i in range(num_files) if    np.sum(filtered_matrix[i, :]) > 0]
#    print(filtered_file_list)
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest', vmin=0.1)
    plt.colorbar()
    plt.xticks(np.arange(len(file_list)), file_list, rotation='vertical')
    plt.yticks(np.arange(len(file_list)), file_list)
    plt.title('Jaccard Similarity Heatmap')
    plt.tight_layout()
    plt.show()




folder_path = './'
print("compare_csv_files")
similarity_results, similarity_matrix, file_list = compare_csv_files(folder_path)
i=0
for result in similarity_results:
    #print(result)
    i+=1

from sklearn.cluster import KMeans

# Nehmen wir an, similarity_matrix ist deine Ähnlichkeitsmatrix

# Anzahl der Cluster festlegen
k = 10

# k-means-Algorithmus auf die Ähnlichkeitsmatrix anwenden
#kmeans = KMeans(n_clusters=k, random_state=0).fit(similarity_matrix)

# Clusterzuordnung für jede Dateninstanz erhalten
#cluster_labels = kmeans.labels_

# Clusterzentren erhalten
#cluster_centers = kmeans.cluster_centers_



# Assume cluster_labels contains the cluster assignments for each data point

# Create a scatter plot of the clusters
#plt.scatter(similarity_matrix[:, 0], similarity_matrix[:, 1], c=cluster_labels, cmap='viridis')
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.title('Clustering Results')
#plt.colorbar(label='Cluster')
#plt.show()

normalized_matrix = normalize_matrix(similarity_matrix)
values = similarity_matrix.flatten()
normalized_values = normalized_matrix.flatten()

sorted_values = np.sort(values)
#sorted_values = np.sort(normalized_values)
percentiles = np.percentile(sorted_values, [10, 25, 50, 75, 90, 95, 97, 98, 99])

print("10. Perzentil:", percentiles[0])
print("25. Perzentil:", percentiles[1])
print("50. Perzentil (Median):", percentiles[2])
print("75. Perzentil:", percentiles[3])
print("90. Perzentil:", percentiles[4])
print("95. Perzentil:", percentiles[5])
print("97. Perzentil:", percentiles[6])
print("98. Perzentil:", percentiles[7])
print("99. Perzentil:", percentiles[8])
# Erstellen des x-Achsen-Arrays
x = np.linspace(0, 1, len(sorted_values))





# Exponentialverteilung
exponential_distribution = stats.expon(scale=1)

# Kolmogorov-Smirnov-Test
ks_statistic, ks_pvalue = stats.kstest(values, 'gamma', args=(2,))

print(str(stats.kstest(values, 'gamma', args=(2,))))
print('KS-Test: Statistik = {}, p-Wert = {}'.format(ks_statistic, ks_pvalue))

# Anderson-Darling-Test
ad_statistic, ad_critical_values, ad_significance_levels = stats.anderson(sorted_values, 'expon')
print('Anderson-Darling-Test: Statistik = {}, Kritische Werte = {}, Signifikanzniveau = {}'.format(ad_statistic, ad_critical_values, ad_significance_levels))

# Histogramm der Daten
plt.hist(values, bins=100, density=True, alpha=0.5, label='Data')

# Theoretische Verteilungen zum Vergleich
distributions = ['norm', 'expon', 'gamma', 'lognorm']
colors = ['red', 'green', 'blue', 'purple']

for dist, color in zip(distributions, colors):
    # Fit der Daten an die theoretische Verteilung
    params = getattr(stats, dist).fit(values)
    
    # Generiere eine PDF der theoretischen Verteilung
    x = np.linspace(stats.__dict__[dist].ppf(0.001, *params[:-2], loc=params[-2], scale=params[-1]),
                    stats.__dict__[dist].ppf(0.999, *params[:-2], loc=params[-2], scale=params[-1]), len(values))
    pdf = stats.__dict__[dist].pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
    
    # Plotten der theoretischen Verteilung
    plt.plot(x, pdf, color=color, label=dist)

# Legende und Achsenbeschriftungen
plt.legend()
plt.xlabel('Werte')
plt.ylabel('Dichte')

# Anzeigen des Plots
#plt.show()

# Liste der Verteilungen zum Testen
distributions = ['norm', 'expon', 'gamma', 'lognorm']

# Schleife über verschiedene Verteilungen
for distribution in distributions:
    # Anpassung der Daten an die Verteilung
    dist = getattr(stats, distribution)
    params = dist.fit(values)
    # Berechnung der Log-Likelihood
    log_likelihood = dist.logpdf(values, *params).sum()
    # Berechnung des AIC (Akaike Information Criterion)
    k = len(params)
    n = len(values)
    aic = -2 * log_likelihood + 2 * k
    # Berechnung des BIC (Bayesian Information Criterion)
    bic = -2 * log_likelihood + k * np.log(n)

    print(f"Verteilung: {distribution}")
    print(f"AIC: {aic}")
    print(f"BIC: {bic}")
    print("-----------------------------------")


# Boxplot
plt.boxplot(sorted_values)
plt.xlabel('Data')
plt.ylabel('Values')
plt.title('Boxplot')
#plt.show()

# Anpassung der Daten an eine Gamma-Verteilung
params = stats.gamma.fit(sorted_values)
dist = stats.gamma(*params)

# Generiere erwartete Quantile aus der Gamma-Verteilung
expected_quantiles = np.linspace(0, 1, len(values))
expected_values = dist.ppf(expected_quantiles)

# Berechne beobachtete Quantile der Daten
sorted_values = np.sort(values)
observed_quantiles = np.arange(1, len(values) + 1) / len(values)

# Plot des Quantil-Quantil-Diagramms
plt.scatter(expected_values, sorted_values, color='blue', alpha=0.5)
plt.plot([np.min(sorted_values), np.max(sorted_values)], [np.min(sorted_values), np.max(sorted_values)], color='red')
plt.xlabel('Theoretische Quantile')
plt.ylabel('Beobachtete Quantile')
plt.title('Quantil-Quantil-Diagramm - Gamma-Verteilung')
plt.show()

# Quantil-Quantil-Diagramm
stats.probplot(sorted_values, dist='expon', plot=plt)
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Ordered Values')
plt.title('Quantile-Quantile Plot')
plt.show()

# Plotten der Verteilung
plt.plot( sorted_values,x)
plt.xlabel('Index')
plt.ylabel('Werte')
plt.title('Kontinuierliche Verteilung der Werte')
plt.show()

# Plotten des Histogramms
plt.hist(values, bins=100, range=(0.0, 1))
plt.xlabel('Werte')
plt.ylabel('Häufigkeit')
plt.title('Histogramm der Matrix-Werte')
plt.show()    
normalized_matrix = normalize_matrix(similarity_matrix)

#plot_heatmap(normalized_matrix, file_list)
plot_heatmap(similarity_matrix, file_list)
print(i)
