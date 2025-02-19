# Load CSV using Pandas
import csv
import pandas
import os
from sklearn.ensemble import RandomForestRegressor

from joblib import parallel_backend, Parallel, delayed


def convert_cnvTable():
    # read in X
    filename = 'WES_pureCN_CNV_genes_20220623.csv'
    X = pandas.read_csv(filename, usecols=["model_id", "symbol", "gene_mean"])
    # remove duplicates
    X = X.drop_duplicates(subset=["model_id", "symbol"], keep='first')
    # cell lines as rows, genes as columns
    X = X.pivot(index='model_id', columns='symbol', values='gene_mean')
    # fill NULL values
    X.fillna(X.mean(), inplace=True)
    # write into file
    X.to_csv('./cnv.csv')
    return X



# COPY NUMBER
convert_cnvTable()


def convert_expressionTable():
    # read in X
    filename = 'rnaseq_tpm_20220624.csv'
    X = pandas.read_csv(filename)
    print(X.shape)
    X = X.transpose()
    # X.drop(["Unnamed: 1"], axis=0, inplace=True)
    X.columns = X.iloc[0]
    X = X.drop_duplicates(subset=["model_name"])
    X.columns = X.iloc[1]
    X.drop(["model_id", "Unnamed: 1"], axis=0, inplace=True)
    X.set_axis(X.iloc[:, 0], axis=0, )
    # X.drop(["model_name"], axis=0, inplace=True)
    X.drop(["symbol"], axis=1, inplace=True)
    X = X.loc[:, X.columns.notna()]
  
    print(X)
    print(X.shape)
    X.to_csv('exp.csv')
    return X

convert_expressionTable()



def convert_mutTable():
    X = pandas.read_csv('mutations_summary_20221018.csv', usecols=["gene_symbol", "coding", "model_id"])
    X.drop_duplicates(inplace=True)
   # X = X.pivot(index='model_name', columns='gene_symbol', values='cancer_driver')
    print(X.shape)

    p = X.duplicated(subset=["gene_symbol", "model_id"])
    p = p.to_frame(name="is_dup")
    print((p.loc[p['is_dup'] == True]).shape)
    pX = pandas.merge(p, X, left_index=True, right_index=True)
    pX = pX.loc[pX['is_dup'] == True]
    print(pX.shape)
    pX['coding'] = True
    print(pX.shape)
    X.drop_duplicates(subset=["gene_symbol", "model_id"], keep=False, inplace=True)
    pX = pandas.merge(pX, X, how='outer')
    print(pX.shape)
    X = pX[["gene_symbol", "coding", "model_id"]]
    print(X["coding"])
    X["coding"] = X["coding"].astype(float)
    X.to_csv('CODINGmut.csv')
    

    print(X["coding"])
    X = X.pivot(index='model_id', columns='gene_symbol', values='coding')

    X.fillna(float(0), inplace=True)
    
    X.to_csv('mut.csv')
    return X


#MUTATION
convert_mutTable()

def convert_results(y, DRUG_ID):
    y = y.loc[y['DRUG_ID'] == DRUG_ID]
    y.drop(['DRUG_ID'], axis=1, inplace=True)
    print(y)
   # y.to_csv('auc_results.csv')
    return y


#FORMAT TABLES
#make tables have same cell lines

result_csv = pandas.read_excel('GDSC2_fitted_dose_response_24Jul22.xlsx', index_col=0, usecols=["SANGER_MODEL_ID", "LN_IC50", "DRUG_ID"])
cnv_csv = pandas.read_csv('cnv.csv', index_col=0).add_suffix("_cnv")
exp_csv = pandas.read_csv('exp.csv', index_col=0).add_suffix("_exp")
mut_csv = pandas.read_csv('mut.csv', index_col=0).add_suffix("_mut")




def my_function(i):
    print("DRUG_ID: "+str(i))
    res = convert_results(result_csv, i)
    print("SHAPE RES: "+str(res.shape))
    #res = pandas.read_csv('auc_results.csv', index_col=0)
    print("ResIndeX: "+str(len(res.index)))
    if len(res.index) == 0:
        print("NOTHING")
    else:
        if not os.path.exists('Drug'+str(i)+'_analysis') : os.mkdir('Drug'+str(i)+'_analysis')
        

        X=cnv_csv.copy(deep=True)
        print("SHAPE CNV: "+str(cnv_csv.shape))


        Y=exp_csv.copy(deep=True)
        print("SHAPE EXPRESSION: "+str(exp_csv.shape))


        Z=mut_csv.copy(deep=True)
        print("SHAPE MUTATIONS: "+str(mut_csv.shape))



        XY = pandas.merge(cnv_csv, exp_csv, left_index=True, right_index=True, how='outer').fillna(0.0)


        print("SHAPE XY: "+str(XY.shape))
        XYZ = pandas.merge(XY, mut_csv, left_index=True, right_index=True, how='outer').fillna(0.0)
        print("SHAPE XYZ: "+str(XYZ.shape))



	

	
        XYZr = pandas.merge(XYZ, res, left_index=True, right_index=True)

        XYZr.to_csv('merged.csv', index=False)




       







        res = XYZr.filter(items=["LN_IC50"])
        res.fillna(res.mean(), inplace=True)  # fill missing values with column mean




        XYZr.drop(["LN_IC50"], axis=1).to_csv('Drug'+str(i)+'_analysis/X_data.csv', index=False)
    

        res.to_csv('Drug'+str(i)+'_analysis/results.csv')
        print(i)


with parallel_backend('threading', n_jobs=-1):
    Parallel()(delayed(my_function)(i) for i in range(1003,2500))

