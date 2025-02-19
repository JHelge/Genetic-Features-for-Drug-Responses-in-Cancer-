# Script that makes use of more advanced feature selection techniques
# by Alberto Tonda, 2017

import copy
import datetime
import graphviz
import logging
import numpy as np
import os
import sys
import pandas as pd 

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import Ridge

from sklearn.linear_model import SGDRegressor


from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV


from sklearn.svm import SVR

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
# used for normalization
from sklearn.preprocessing import  Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# used for cross-validation
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
# this is an incredibly useful function
from pandas import read_csv


from scipy import stats

def find_feature_indices(features_df, feature_list):
	"""Helper function to find indices of desired features in the features DataFrame."""
	indices = []
	for feature in feature_list:
		if feature in features_df[0].values:
			idx = features_df.index[features_df[0] == feature].tolist()[0]
			indices.append(idx)
	return indices


def loadDataset(drugIndex,features_to_analyze_path) :
	dataDirectory="../"+drugIndex
	# create folder
	folderName =dataDirectory+"/clusterUnion/"
	if os.path.exists(dataDirectory):
		if not os.path.exists(folderName) : os.makedirs(folderName)
	features_to_analyze_df = pd.read_csv(features_to_analyze_path, header=None)
	feature_list = features_to_analyze_df[0].tolist()

	features_path = os.path.join(dataDirectory, 'features_0.csv')
	data_path = os.path.join(dataDirectory, 'data_0.csv')
	labels_path = os.path.join(dataDirectory, 'labels.csv')
	if os.path.exists(features_path) and os.path.exists(data_path) and os.path.exists(labels_path):
		features = pd.read_csv(features_path, header=None)
		data = pd.read_csv(data_path, header=None)
		labels = pd.read_csv(labels_path, header=None)
		feature_indices = find_feature_indices(features, feature_list)
		
		
# Debugging information
		print("Feature indices found:", feature_indices)
		print("Data shape:", data.shape)
		print("Labels shape:", labels.shape)
		# Select only the rows corresponding to the specified feature indices
		# Select only the rows corresponding to the specified feature indices
		dfData = data.iloc[:, feature_indices]
		#dfLabels = labels.iloc[:, feature_indices]
		dfLabels = labels
		#dfData = data.iloc[feature_indices, :]
		#dfLabels = labels.iloc[feature_indices, :]
		#dfData = data.iloc[feature_indices]
		#dfLabels = labels.iloc[feature_indices]
		# Convert the selected rows to a numpy array

		
	# data used for the predictions
	#dfData = read_csv(dataDirectory+"/possiblyImportant/data_0.csv", header=None, sep=',')
	#dfLabels = read_csv(dataDirectory+"/possiblyImportant/labels.csv", header=None)
	print (dfData.to_numpy(), dfLabels.to_numpy().ravel())
	return dfData.to_numpy(), dfLabels.to_numpy().ravel() # to have it in the format that the classifiers like
	#return dfData.as_matrix(), dfLabels.as_matrix().ravel() # to have it in the format that the classifiers like


def runFeatureReduce(drugIndex,features_to_analyze_path) :
	dataDirectory="../"+drugIndex
	
	# a few hard-coded values
	numberOfFolds = 10
	
	# list of classifiers, selected on the basis of our previous paper "
	classifierList = [
		
			
			[GradientBoostingRegressor(n_estimators=300), "GradientBoostingRegressor(n_estimators=300)"],
			[RandomForestRegressor(n_estimators=300), "RandomForestRegressor(n_estimators=300)"],
			[LinearRegression(), "LinearRegression"],
			[PassiveAggressiveRegressor(),"PassiveAggressiveRegressor"],
			[SGDRegressor(penalty='elasticnet'), "SGDRegressor(elasticnet)"],
			[SVR(kernel='linear'), "SVR(linear)"],
			[Ridge(), "Ridge"],
			[BaggingRegressor(n_estimators=300), "BaggingRegressor(n_estimators=300)"],


			]
	


	print("Loading dataset...")
	X, y = loadDataset(drugIndex,features_to_analyze_path)
	
	print(len(X))
	print(len(X[0]))
	print(len(y))

	
	labels=np.max(y)+1
	# prepare folds
	kf = KFold(n_splits=numberOfFolds, shuffle=True)
	indexes = [ (training, test) for training, test in kf.split(X, y) ]
	
	# this will be used for the top features
	topFeatures = dict()
	
	# iterate over all classifiers
	classifierIndex = 0
	
	
	
	
	for originalClassifier, classifierName in classifierList :
		
		print("\nClassifier " + classifierName)
		classifierPerformance = []
		classifierRMSE = []

		#cMatrix=np.zeros((labels, labels))
		# iterate over all folds
		
		indexFold = 0

		yTest=[]
		yNew=[]

		for train_index, test_index in indexes :
			
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			# let's normalize, anyway
			# MinMaxScaler StandardScaler Normalizer
			scaler = StandardScaler()
			X_train = scaler.fit_transform(X_train)
			X_test = scaler.transform(X_test)

		
			
			classifier = copy.deepcopy(originalClassifier)
			classifier.fit(X_train, y_train)
			predictionsTraining = classifier.predict(X_train)
			scoreTraining = r2_score(y_train,predictionsTraining)
			predictionsTest = classifier.predict(X_test)
			scoreTest = r2_score(y_test, predictionsTest)
			
			print("\tR2 training: %.4f, test: %.4f" % (scoreTraining, scoreTest))
			classifierPerformance.append( scoreTest )
			
			#scoreTraining = adjusted_r_squared(y_train,predictionsTraining)
			#scoreTest = adjusted_r_squared(y_test, predictionsTest)

			#print("\tAdjustedR2training: %.4f, test: %.4f" % (scoreTraining, scoreTest))
			
			scoreTraining = mean_absolute_error(y_train,predictionsTraining)
			scoreTest = mean_absolute_error(y_test, predictionsTest)

			print("\tMAEtraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))
			scoreTraining = mean_squared_error(y_train,predictionsTraining)
			scoreTest = mean_squared_error(y_test, predictionsTest)

			print("\tMSEtraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))

			
			scoreTraining = mean_squared_error(y_train,predictionsTraining, squared=False)
			scoreTest = mean_squared_error(y_test, predictionsTest, squared=False)
			classifierRMSE.append( scoreTest )
			print("\tRMSEtraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))
			
			#scoreTraining = calculate_mape(y_train,predictionsTraining)
			#scoreTest = calculate_mape(y_test, predictionsTest)
			
			#print("\tMAPEtraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))

			
			y_new = classifier.predict(X_test)
			
			
			
			yNew.append(y_new)
			yTest.append(y_test)
			
			#for i in range(0,len(y_new)):
			#	cMatrix[y_test[i]][y_new[i]]+=1



		#pd.DataFrame(cMatrix).to_csv(dataDirectory+"/best/cMatrix"+str(classifierIndex)+".csv", header=None, index =None)
		classifierIndex+=1
		line ="%s \t %.4f \t %.4f \n" % (classifierName, np.mean(classifierPerformance), np.std(classifierPerformance))
		rmseline ="%s \t %.4f \t %.4f \n" % (classifierName, np.mean(classifierRMSE), np.std(classifierRMSE))
		
		print(line)
		fo = open(dataDirectory+"/clusterUnion/results.txt", 'a')
		fo.write( line )
		fo.close()
		rmseo = open(dataDirectory+"/clusterUnion/rmseresults.txt", 'a')
		rmseo.write( rmseline )
		rmseo.close()
	
	return

def run_all_feature_reduction(base_dir):
    cluster_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and 'Cluster_' in d]
    
    for cluster_folder in cluster_folders:
        union_features_path = os.path.join(cluster_folder, 'union.csv')
        directories_path = os.path.join(cluster_folder, 'directories.csv')
        
        if os.path.exists(directories_path) and os.path.exists(union_features_path):
            directories_df = pd.read_csv(directories_path)
            for index, row in directories_df.iterrows():
                drug_directory = row[0]

                print(drug_directory)
                print(f"Running feature reduction for {drug_directory} with features from {union_features_path}")
                # Assuming drug_index and features_to_analyze_path are the inputs needed
                runFeatureReduce(drug_directory, union_features_path)
        else:
            print(f"Missing files for {cluster_folder}, skipping...")

# The runFeatureReduce function from the code snippet above should be defined here, or imported if it is in a separate module.

if __name__ == "__main__":
    base_directory = "../"  # Change to your base directory
    run_all_feature_reduction(base_directory)
