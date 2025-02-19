# Script that makes use of more advanced feature selection techniques
# by Alberto Tonda, 2017
import joblib

import copy
import datetime
import graphviz
import logging
import numpy as np
import os
import sys
import pandas as pd 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import Ridge
#from sklearn.linear_model import RidgeRegressorCV
from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

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

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
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

def loadDataset(drugIndex) :
	dataDirectory="../Drug"+str(drugIndex)+"_analysis"

	# data used for the predictions
	dfData = read_csv(dataDirectory+"/best/data_0.csv", header=None, sep=',')
	dfLabels = read_csv(dataDirectory+"/best/labels.csv", header=None)
	print (dfData.to_numpy(), dfLabels.to_numpy().ravel())
	return dfData.to_numpy(), dfLabels.to_numpy().ravel() # to have it in the format that the classifiers like
	#return dfData.as_matrix(), dfLabels.as_matrix().ravel() # to have it in the format that the classifiers like


def runFeatureReduce(drugIndex) :
	dataDirectory="../Drug"+str(drugIndex)+"_analysis"
	
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
			# ensemble
			#[AdaBoostClassifier(), "AdaBoostClassifier"],
			#[AdaBoostClassifier(n_estimators=300), "AdaBoostClassifier(n_estimators=300)"],
			#[AdaBoostClassifier(n_estimators=1500), "AdaBoostClassifier(n_estimators=1500)"],
			#[BaggingClassifier(), "BaggingClassifier"],
			
			#[ExtraTreesClassifier(), "ExtraTreesClassifier"],
			#[ExtraTreesClassifier(n_estimators=300), "ExtraTreesClassifier(n_estimators=300)"],
			 # features_importances_
			#[GradientBoostingClassifier(n_estimators=300), "GradientBoostingClassifier(n_estimators=300)"],
			#[GradientBoostingClassifier(n_estimators=1000), "GradientBoostingClassifier(n_estimators=1000)"],
			
			#[RandomForestClassifier(n_estimators=300), "RandomForestClassifier(n_estimators=300)"],
			#[RandomForestClassifier(n_estimators=1000), "RandomForestClassifier(n_estimators=1000)"], # features_importances_

			# linear
			#[ElasticNet(), "ElasticNet"],
			#[ElasticNetCV(), "ElasticNetCV"],
			#[Lasso(), "Lasso"],
			#[LassoCV(), "LassoCV"],
			 # coef_
			#[LogisticRegressionCV(), "LogisticRegressionCV"],
			  # coef_
			 # coef_
			#[RidgeClassifierCV(), "RidgeClassifierCV"],
			 # coef_
			 # coef_, but only if the kernel is linear...the default is 'rbf', which is NOT linear
			
			# naive Bayes
			#[BernoulliNB(), "BernoulliNB"],
			#[GaussianNB(), "GaussianNB"],
			#[MultinomialNB(), "MultinomialNB"],
			
			# neighbors
			#[KNeighborsClassifier(), "KNeighborsClassifier"], # no way to return feature importance
			# TODO this one creates issues
			#[NearestCentroid(), "NearestCentroid"], # it does not have some necessary methods, apparently
			#[RadiusNeighborsClassifier(), "RadiusNeighborsClassifier"],
			
			# tree
			#[DecisionTreeClassifier(), "DecisionTreeClassifier"],
			#[ExtraTreeClassifier(), "ExtraTreeClassifier"],

			]
	
	# this is just a hack to check a few things
	#classifierList = [
	#		[RandomForestClassifier(), "RandomForestClassifier"]
	#		]

	print("Loading dataset...")
	X, y = loadDataset(drugIndex)
	
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
			model_filename = f"{dataDirectory}/best/model_{classifierName}_fold{fold}.joblib"
			joblib.dump(classifier, model_filename)
			print(f"Model saved to {model_filename}")
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
		fo = open(dataDirectory+"/best/results.txt", 'a')
		fo.write( line )
		fo.close()
		rmseo = open(dataDirectory+"/best/rmseresults.txt", 'a')
		rmseo.write( rmseline )
		rmseo.close()
	
	return

if __name__ == "__main__" :
	drugIndex=sys.argv[1]
	sys.exit( runFeatureReduce(drugIndex) )
