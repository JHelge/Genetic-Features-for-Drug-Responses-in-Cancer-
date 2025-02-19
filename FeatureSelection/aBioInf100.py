from features import *
from reduceData import *
import sys
from joblib import Parallel, delayed
import multiprocessing
#from runClassifiers import *
import time

def mainRun(drugIndex,indexRun) :

	if os.path.isdir("../Drug"+str(drugIndex)+"_analysis"):
		print("In mainRun: "+str(drugIndex))
		run=indexRun
		start_time = time.time()
		globalAnt=0.0
		globalIndex=0
		globalAccuracy=0.0

		X, y, biomarkerNames = loadDatasetOriginal(drugIndex, run)

		
		if (int(len(X[0]))>1000):
			numberOfTopFeatures = int(len(X[0])*0.50)
		else :
			numberOfTopFeatures = int(len(X[0])*0.80)

		variableSize=numberOfTopFeatures;
		while True:
			globalAnt=globalAccuracy
			globalAccuracy=featureSelection(drugIndex, globalIndex,variableSize, run)
			print(globalAccuracy)
			print(globalIndex)
			print(variableSize)
			size,sizereduced=reduceDataset(drugIndex, globalIndex, run)
			
			if((globalAccuracy<(0.0)) and (globalIndex!=0)):
				break
			if(variableSize==0):
				break
			variableSize=int(variableSize*0.80)
			
			globalIndex=globalIndex+1
		elapsed_time = time.time() - start_time
		print("time")
		print(elapsed_time)
		print("endtime")
		print(str(time.time()))
		return

def main():
	threads=10
	totalRuns=10
	print("Starting Job in aBioInf: "+sys.argv[1])
	for i in range(0,totalRuns):
		print(str(int(sys.argv[1])+i))

	Parallel(n_jobs=threads, verbose=5, backend="multiprocessing")(delayed(mainRun)(str(int(sys.argv[1])+i), 0) for i in range(0,totalRuns))
	return

if __name__ == "__main__" :
	sys.exit( main() )
