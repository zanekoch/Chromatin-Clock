from collections import defaultdict
import pandas as pd
import os
import sys
import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import pickle
import glob
import math 



def make_feature_matrix(data_fns):
    feature_matrix_df = pd.DataFrame()
    n = 0
    for fn in data_fns:
        cur_file = pd.read_csv(fn)
        cur_file = cur_file.set_index("SAMPLE_NAME")
        feature_matrix_df = feature_matrix_df.merge(cur_file, how='outer', left_index=True, right_index=True)
    
    return feature_matrix_df

def test_clock(X, y, feat_coef_dict, test_feat_dir):
    # get the features-coefs into a pandas df
    feat_coef_dict = dict(feat_coef_dict)
    coef_intercept_df = pd.DataFrame(feat_coef_dict, index=[0])

    intercept = coef_intercept_df['intercept'][0]
    coef_df = coef_intercept_df.drop("intercept", axis = 1)

    # sort coef_df and X features in the same order
    X = X.reindex(sorted(X.columns), axis=1)
    coef_df = coef_df.reindex(sorted(coef_df.columns), axis=1)

    b = coef_df.to_numpy().T
    a = X.to_numpy()
    predicted_ages = (a.dot(b)) + intercept 


    # test performance
    y = y.to_numpy()
    predicted_ages = predicted_ages.reshape(63,)
    
    acc =  (np.corrcoef(y, predicted_ages)[1][0])**2
    

    y = y.tolist()
    predicted_ages = predicted_ages.tolist()
    # convert to years 
    y = [math.exp(x)-1 for x in y]
    predicted_ages = [math.exp(x)-1 for x in predicted_ages]
    diffs = [x-z for x,z in zip(predicted_ages, y)]
    mae = 0
    for d in diffs:
        mae += abs(d)
    mae = mae / len(diffs)
    with open(os.path.join(test_feat_dir, "performance.txt"), "w+") as f:
        f.write("r^2: {}, mae: {} years".format(acc, mae))
        #f.write(diffs)
        #f.write(predicted_ages)
    return 

"""
test_clock.py: take the features from the testing data that were chosen by using chromatinClock.py to train a model and get_model_features_from_test_data.py to extract the features from test data, and test the model on these features
@ test_feat_dir: directory of .csv files, filled with sample_num rows and feature num columns. One file for each chromosome with a feature in it
@ age_fn: path to file of ages for the samples in data_dir
@ feat_coef_fn: pickled dict created by get_model_features_from_test_data.py of al keys=features with nonzero coef in trained model, and vals = the coefs (also one key is "intercept" and the value is the learned intercept)
"""


def main():
    test_feat_dir = sys.argv[1]
    age_fn = sys.argv[2]
    feat_coef_fn = sys.argv[3]
    #out_dir = sys.argv[4]
    #os.makedirs(out_dir, exist_ok=True)

    fn_expression = os.path.join(test_feat_dir, '*.csv')
    data_fns = glob.glob(fn_expression)

    # read in the test_feat files and combine into a feature_matrix
    feature_matrix_df = make_feature_matrix(data_fns)
    
    # double check that the number of nonzero features matches the # of feature keys in feat_coef_fn (one less than len bc of intercept feature)
    # read in the feat_coef_dict
    f = open(feat_coef_fn, "rb")
    feat_coef_dict = pickle.load(f)
    if len(feat_coef_dict) - 1 != len(feature_matrix_df.columns):
        print("number of features in test_feat_dir is not the same as the number of features in feat_coef_fn")
        quit()
    
    # if they are equal, read in age_fn and keep the relevant ages to the tesing data
    age_df = pd.read_csv(age_fn, sep='\t')
    age_df = age_df.set_index('SAMPLE_NAME')
    X = feature_matrix_df.merge(age_df, how='outer', left_index=True, right_index=True)
    X = X.dropna()
    # log transform age
    y = np.log(X['DONOR_AGE']+1) 
    # drop ages from X
    X = X.drop(['DONOR_AGE'], axis = 1)
    
    test_clock(X, y, feat_coef_dict, test_feat_dir)

    return

main()