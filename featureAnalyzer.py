# Script that makes use of more advanced feature selection techniques
# by Alberto Tonda, 2017

import copy
import datetime
import graphviz
import logging
import numpy as np
import os
import sys
from joblib import Parallel, delayed
import multiprocessing
import scipy.stats as stats

import time

import matplotlib.pyplot as plt
import csv
from collections import Counter



from sklearn.multiclass import OneVsOneClassifier 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

# used for normalization
from sklearn.preprocessing import StandardScaler

# used for cross-validation
from sklearn.model_selection import StratifiedKFold


# this is an incredibly useful function
from pandas import read_csv
import pandas as pd 


def compare_csv_files_percent(folder_path):
    folder_numbers=[]
    for i in range(1003, 2501):
        print("Drug"+str(i)+"_analysis/best/features_0.csv")
        print(os.path.isfile("Drug"+str(i)+"_analysis/best/features_0.csv"))
        if os.path.isfile("Drug"+str(i)+"_analysis/best/features_0.csv"):
        
            folder_numbers.append(i)
    print("folder_numbers")        
    print(folder_numbers)
    num_rows = 0
    num_diffs = 0
    for i in range(0, len(folder_numbers)):
        file1 = os.path.join(folder_path, f"Drug{folder_numbers[i]}_analysis/best/features_0.csv")
        file2 = os.path.join(folder_path, f"Drug{folder_numbers[i-1]}_analysis/best/features_0.csv")
        if os.path.isfile(file1) and os.path.isfile(file2):
            with open(file1, 'r') as csv_file1, open(file2, 'r') as csv_file2:
                csv_reader1 = csv.reader(csv_file1)
                csv_reader2 = csv.reader(csv_file2)
                for row1, row2 in zip(csv_reader1, csv_reader2):
                    num_rows += 1
                    if row1 != row2:
                        num_diffs += 1
    percent_same = (1 - (num_diffs / num_rows)) * 100
    return percent_same

def compare_csv_files(folder_path):
	results = []
	for i in range(1003, 2501):
		#print(i)
		file1 = os.path.join(folder_path, f"Drug{i}_analysis/best/features_0.csv")
		#print(file1)
		file2 = os.path.join(folder_path, f"Drug{i-1}_analysis/best/features_0.csv")
		if os.path.isfile(file1) and os.path.isfile(file2):
			with open(file1, 'r') as csv_file1, open(file2, 'r') as csv_file2:
				csv_reader1 = csv.reader(csv_file1)
				csv_reader2 = csv.reader(csv_file2)
				for row1, row2 in zip(csv_reader1, csv_reader2):
					if row1 != row2:
						results.append(f"Files {file1} and {file2} differ in row: {row1}")
						#print(i)
	return results


def converter(i) :
	dataDirectory="../Drug"+str(i)+"_analysis"
	#dataDirectory="../data"
	if not os.path.isdir(dataDirectory):
		return
	start_time = time.time()

	# data used for the predictions
	dfData = read_csv(dataDirectory+"/tmp_rows.csv", header=None, sep=',', dtype='unicode', low_memory=False)
	biomarkers = dfData.transpose()[0]
	totalRow = len(dfData.index)
	dfData= dfData.iloc[1: totalRow]
	dfLabels = read_csv(dataDirectory+"/results.csv", header=None)
	totalRow = len(dfLabels.index)
	dfLabels= dfLabels.iloc[1: totalRow]
	dfLabels= dfLabels[1].astype('float')
	#dfData.fillna(value=dfData.mean, inplace=True)
	dfDataWithoutNaN=dfData.fillna(0)
	#print(dfLabels)


	# p-Werte berechnen
	#p_values = [stats.ttest_1samp(dfLabels, i).pvalue for i in dfLabels]

	#print("P_VALUES: "+str(p_values))


	#t, p = stats.ttest_1samp(dfLabels, 0)

	#print("T_VALUES: "+str(t))
	#print("P_VALUES: "+str(p))

	# p-Werte als neue Spalte hinzufügen
	#dfLabels['p_values'] = p_values

	#print(dfLabels)
	#sns.kdeplot(dfLabels, fill=True)

	# Diagramm-Titel und Achsenbeschriftungen hinzufügen
	#plt.title('Verteilung der Werte in Spalte1')
	#plt.xlabel('Wert')
	#plt.ylabel('Dichte')

	# Diagramm anzeigen

	#import pandas as pd
	#import numpy as np


	# Percentil berechnen (hier für das 90. Percentil)
	percentile_90 = np.percentile(dfLabels, 95)
	percentile_10 = np.percentile(dfLabels, 5)

	print("Das 90. Percentil ist:", percentile_90)
	print("Das 10. Percentil ist:", percentile_10)

	# Schwellwerte berechnen
	#cluster_boundaries = []
	#for i in range(3):
	#	cluster = dfLabels[predictions == i]
	#	print("cluster "+str(i)+": "+str(cluster))
	#	#cluster_mean = np.mean(cluster)
	#	#cluster_std = np.std(cluster)
	#	#lower_boundary = cluster_mean - cluster_std
	#	#upper_boundary = cluster_mean + cluster_std
	#	lower_boundary = np.min(cluster)
	#	upper_boundary = np.max(cluster)
	#	cluster_boundaries.append((lower_boundary, upper_boundary))

	# Schwellwerte ausgeben
	print(dfLabels)
	#print(cluster_boundaries)
	print(dfLabels.mean())
	print(np.mean(dfLabels))

	#plt.show()
	dfLabels.mask(dfLabels >= 1.645 , 2, inplace=True) #1.645 corresponds t 90%
	dfLabels.mask((dfLabels < 1.645) & (dfLabels > -1.645)  ,1, inplace=True)
	dfLabels.mask(dfLabels <= -1.645 ,0, inplace=True)


	#print(dfLabels)

	print(dfDataWithoutNaN)
	print(biomarkers)
	print(dfLabels)

	pd.DataFrame(biomarkers.values.ravel()).to_csv(dataDirectory+"/features_0.csv", header=None, index =None)
	print("features_0 written")
	pd.DataFrame(dfLabels.values.ravel()).to_csv(dataDirectory+"/labels.csv", header=None, index =None)
	print("labels written")
	pd.DataFrame(dfDataWithoutNaN.values).to_csv(dataDirectory+"/data_0.csv", header=None, index =None)
	print("data_0 written")

	elapsed_time = time.time() - start_time
	print("time")
	print(elapsed_time)
	return

def main():
	seen = set()
	words=[]
	uniq=[]
	blacklist=['./Drug1016_analysis/best/features_0.csv','./Drug1019_analysis/best/features_0.csv','./Drug1020_analysis/best/features_0.csv','./Drug1030_analysis/best/features_0.csv','./Drug1032_analysis/best/features_0.csv','./Drug1033_analysis/best/features_0.csv','./Drug1037_analysis/best/features_0.csv','./Drug1038_analysis/best/features_0.csv','./Drug1039_analysis/best/features_0.csv','./Drug1042_analysis/best/features_0.csv','./Drug1043_analysis/best/features_0.csv','./Drug1046_analysis/best/features_0.csv','./Drug1066_analysis/best/features_0.csv']
	drugList=[]
	drugCounter=0
	for i in range(1000,2500):
		dataDirectory='./Drug'+str(i)+'_analysis/'

		#print(i)
		#print(dataDirectory)
		if os.path.exists(dataDirectory) and os.path.exists(dataDirectory+'best/features_0.csv') and dataDirectory+'best/features_0.csv' not in blacklist:
			drugCounter+=1
			#print(dataDirectory)
			#print(os.path.exists(dataDirectory) and os.path.exists(dataDirectory+'best/features_0.csv'))
			#print(i)
			#print(dataDirectory+'best/features_0.csv')
			drugFeatureNumber=[]
			with open(dataDirectory+'best/features_0.csv') as input_file:
				#print(dataDirectory+'best/features_0.csv')
				for row in csv.reader(input_file, delimiter=','):
					#print(row[0])
					#print(len(words))
					
					if row[0] not in seen:
						uniq.append(row[0])
						seen.add(row[0])
						
					if (row[0]=="NUP210L" or row[0]=="RN7SKP104" or row[0]=="PPIAP11" or row[0]=="EEF1A1P13"  or row[0]=="RN7SKP147" or row[0]=="SCG2" or row[0]=="KRT33B" or row[0]=="CHIC2" or row[0]=="ITGA3" or row[0]=="ZSCAN5C" or row[0]=='GALNT9' or row[0]=='PLN' or row[0]=='SNORA23' or row[0]=='NGEF' or row[0]=='ZACN' or row[0]=='SCUBE2' or row[0]=='TRIM80P' or row[0]=='LGR6' or row[0]=='SLCO4A1-AS1'  or row[0]=='PFN3' or row[0]=='FLG' or row[0]=='IL11' or row[0]=='RPSAP65' or row[0]=='SNORD15A' or row[0]=='TNFSF18' or row[0]=='OCSTAMP' or row[0]=='IL20' or row[0]=='LINC02385' or row[0]=='C15orf62' or row[0]=='TBL1XR1-AS1'  or row[0]=='PRSS57' or row[0]=='MTND4P12' or row[0]=='SDC1' or row[0]=='ARL5AP3' or row[0]=='KCNIP1' or row[0]=='LINC00460' or row[0]=='FGF7P3' or row[0]=='C1orf109' or row[0]=='RNU4-6P' or row[0]=='NMUR2' or row[0]=='RPL13AP20' or row[0]=='SIGLEC6' or row[0]=='B3GNT6' or row[0]=='DMBT1L1' or row[0]=='MED28P4' or row[0]=='SNORA53' or row[0]=='DEFB4B' or row[0]=='RPS10P3' or row[0]=='SYNE1' or row[0]=='THBD' or row[0]=='EPX'or row[0]=='DKK3'or row[0]=='DEFB4A'or row[0]=='CFAP126' or row[0]=='LINC02415' or row[0]=='PLA2R1' or row[0]=='KL' or row[0]=='NORD53' or row[0]=='MUCL3' or row[0]=='GYG1P1' or row[0]=='IGHG4' or row[0]=='CTSH' or row[0]=='PPDPF' or row[0]=='OR6C74' or row[0]=='HMOX1' or row[0]=='KRTAP3-4P' or row[0]=='CDRT15P2' or row[0]=='RNU6-1120P' or row[0]=='RTN4IP1' or row[0]=='CYP4F11') and dataDirectory not in drugList:
							drugList.append(dataDirectory)
					
					words.append(row[0])
					drugFeatureNumber.append(row[0])
					#print(Counter(words).keys())
					#print(Counter(words).values())
				#print(len(drugFeatureNumber))
				if len(drugFeatureNumber) >1000:
					print(len(drugFeatureNumber))
					print(dataDirectory+'best/features_0.csv')
			#print 'Number of A grades: %s' % grades['A']
		#print(Counter(words).keys())
		#print(Counter(words).values())
	#print(Counter(uniq).keys())
	#print(Counter(uniq).values())
	print(len(uniq))
	print(Counter(words).keys())
	print(Counter(words).values())
	print(Counter(words).most_common(400))
	print(drugList)
	print(len(drugList))
	print(drugCounter)


	
	with open('./Drug1003_analysis/best/features_0.csv', 'r') as t1, open('./Drug1005_analysis/best/features_0.csv', 'r') as t2:
		fileone = t1.readlines()
		filetwo = t2.readlines()
	
	fileoneList=[]
	filetwoList=[]
	notinfileoneList=[]
	notinfiletwoList=[]
	with open('compare.csv', 'w') as outFile:
		for line in filetwo:
			if line not in fileone:
				notinfileoneList.append(line)

				outFile.write(line)
			elif line in fileone:
				fileoneList.append(line)
				
	print(notinfileoneList)
	print(len(notinfileoneList))
	print(fileoneList)
	print(len(fileoneList))
	with open('compare.csv', 'w') as outFile:
		for line in fileone:
			if line not in filetwo:
				notinfiletwoList.append(line)

				outFile.write(line)
			if line in filetwo:
				filetwoList.append(line)
	print(notinfiletwoList)
	print(len(notinfiletwoList))	
	print(filetwoList)
	print(len(filetwoList))		
			
	# An "interface" to matplotlib.axes.Axes.hist() method
	plt.hist(x=Counter(words).values(), bins=[x for x in range(1,30)])
	plt.xlabel('Value')
	plt.ylabel('Frequency')
	plt.title('My Very Own Histogram')
	plt.text(23, 45, r'$\mu=15, b=3$')

	plt.show()
	folder_path = './'
	#comparison_results = compare_csv_files(folder_path)
	#print(comparison_results)
	
	comparison_percent = compare_csv_files_percent('./')
	print(comparison_percent)
main()


#converter(0)
def createDataFile() :
	
	# data used for the predictions
	dfData = read_csv(dataDirectory+"/tmp_rows.csv", header=None, sep=',')
	biomarkers = dfData.transpose()[0]

	totalRow = len(dfData.index)
	dfData= dfData.iloc[1: totalRow]
	dfData.fillna(value=dfData.mean(), inplace=True)
	dfDataWithoutNaN=dfData.fillna(0)


	print(dfDataWithoutNaN)

	#biomarkers = read_csv(dataDirectory+"/features_0.csv", header=None)

	# create folder
	#folderName =dataDirectory+"/run"+str(run)+"/"
	#if not os.path.exists(folderName) : os.makedirs(folderName)
	pd.DataFrame(biomarkers.values.ravel()).to_csv(dataDirectory+"/features_0.csv", header=None, index =None)

	pd.DataFrame(dfDataWithoutNaN.values).to_csv(dataDirectory+"/data_0.csv", header=None, index =None)
	return dfDataWithoutNaN.values, biomarkers.values.ravel()

#createDataFile()

