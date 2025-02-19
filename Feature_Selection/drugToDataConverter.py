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
import seaborn as sns
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier

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
	threads=-1
	totalRuns=2500
	Parallel(n_jobs=threads, verbose=5, backend="multiprocessing")(delayed(converter)(i) for i in range(1531,totalRuns))
	return
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

