import pandas as pd
import os
import sys
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
import pickle
import math
from scipy import stats


def train_clock(X, y, out_dir):
    print("training clock ...")
    # don't train using the age
    X = X.drop(['DONOR_AGE'], axis = 1)
    model = ElasticNetCV(max_iter = 50000, random_state=0)
    model.fit(X.to_numpy(), y)
    with open(os.path.join(out_dir, "log_top_.1%_wholegenome_no0age_trained_model.pickle"), "wb+") as f:
        pickle.dump(model, f)



def train_clock_cv(X, y, out_dir):
    # don't train using the age
    X = X.drop(['DONOR_AGE'], axis = 1)
    # do 5-fold CV to find accuracy of model
    corr_list = []
    mae_list = []
    cv = KFold(n_splits = 5, shuffle = True, random_state = 9)
    n = 0
    print("Starting CVs ...")
    for train_indices, test_indices in cv.split(X.values):
        X_train = X.iloc[train_indices, :]
        X_test = X.iloc[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]
        
        model = ElasticNetCV(max_iter = 50000, random_state=0)
        model.fit(X_train.to_numpy(), y_train)
        #print(model.intercept_)
        y_pred = model.predict(X_test.to_numpy())
        #acc =  (np.corrcoef(y_test, y_pred)[1][0])**2

        y_test = y_test.tolist()
        y_pred = y_pred.tolist()
        y_test_yrs = [math.exp(x)-1 for x in y_test]
        y_pred_yrs = [math.exp(x)-1 for x in y_pred]
        abs_diffs = [abs(x-z) for x,z in zip(y_test_yrs, y_pred_yrs)]
        mae = sum(abs_diffs) / len(abs_diffs)

        correlation, _ = stats.pearsonr(y_test_yrs, y_pred_yrs)

        # add to lists to get average values
        corr_list.append(correlation)
        mae_list.append(mae)

        coef = pd.Series(model.coef_, index = X_train.columns)
        coef_zero = coef[coef!=0]
        coef_zero.to_csv(os.path.join(out_dir, "fixed_log_coef_noState3" + str(n) + ".tsv"), sep = '\t')
        with open(os.path.join(out_dir, 'fixed_log_hyperCV_5fold_wholegenome_top.1%variance_noState3.txt'), 'a+') as f:
            f.write("Elastic Net picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables for a correlation of "+ str(correlation)+ ", a MAE of " +str(mae) + " years, and found a supposedly optimal alpha of " + str(model.alpha_) + '\n')
        print("done this CV iteration")
        n += 1

    with open(os.path.join(out_dir, 'fixed_log_hyperCV_5fold_wholegenome_top.1%variance_noState3.txt'), 'a+') as f:
        f.write("r^2: "+ str(np.mean(corr_list))+'\n')
        f.write("mae: "+str(np.mean(mae_list))+'\n')


""" 
chromatinClock.py
@ given the file of features chosen (e.g. top variance locaitons) and file of ages of samples, do a hyperparameter search and train an ElasticNet
"""
def main():
    # read command line args 
    top_variance_locs_fn = sys.argv[1]
    age_fn = sys.argv[2]
    out_dir = sys.argv[3]
    mode = sys.argv[4]
    os.makedirs(out_dir, exist_ok=True)

    # read and combine top variance locations and ages into training df X
    top_var_df = pd.read_csv(top_variance_locs_fn)
    top_var_df = top_var_df.set_index('Unnamed: 0')
    top_var_df.index.rename('SAMPLE_NAME', inplace=True)
    # remove state 3
    cols = [c for c in top_var_df.columns if c.split('_')[-1] != 'E3']
    top_var_df = top_var_df[cols]
    age_df = pd.read_csv(age_fn, sep='\t')
    age_df = age_df.set_index('SAMPLE_NAME')

    # create X, columns: top1% variance positions of posterior values + DONOR_AGE
    X = top_var_df.merge(age_df, how='outer', left_index=True, right_index=True)
    # get rid of disease samples
    X = X.dropna()
    #X = X[X['DONOR_AGE'] != 0] # TODO
    # y is column of donor ages
    # take log, add a small value to make 0's log-able
    y = np.log(X['DONOR_AGE']+1) 
    #y = X['DONOR_AGE']

    if mode == "train_cv":
        train_clock_cv(X,y, out_dir)
    elif mode == "train":
        train_clock(X,y, out_dir)
    else:
        print("Mode supplied in wrong format or not at all")
        quit()

main()