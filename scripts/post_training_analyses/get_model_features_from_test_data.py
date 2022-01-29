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
import time


def get_nonzero_coef_features(model_fn, model_features_fn):
    # read the trained model
    with open(model_fn, 'rb') as f:
        model = pickle.load(f)
    # read the model features
    file_to_read = open(model_features_fn , "rb")
    feature_dict = pickle.load(file_to_read)
    # if top .1% selection method keep only those
    l = len(feature_dict)
    keep_num = int(l * .02)
    feature_dict = dict(sorted(feature_dict.items(), key = itemgetter(1), reverse = True)[:keep_num])

    # match the non-zero coefficients with their feature names
    feat_coef_dict = defaultdict(float)
    coefs = model.coef_
    feature_keys = list(feature_dict)
    for i in range(len(coefs)):
        if coefs[i] != 0:
            feat_coef_dict[feature_keys[i]] = coefs[i]
    # add intercept to feat_coef_dict also
    feat_coef_dict["intercept"] = model.intercept_
    return feat_coef_dict
            
def which_file(feature_name):
    WIN_PER_F = 50000
    chrom = feature_name.split('_')[0]
    window = int(feature_name.split('_')[1])
    # based on WIN_PER_F return the file that this feature will be found in
    partition_num = int(window / WIN_PER_F)
    fn = "{}_{}.tsv".format(chrom, partition_num)
    return fn

def make_feature_matrix_one_chrom(feat_coef_dict, data_dir, chrom_str, samples_names_df_fn):
    # make dictionary with keys of partition fns to go into and values the features to get from that f
    partition_fn_dict = defaultdict(list)
    for feat in feat_coef_dict.keys():
        # only get partition files for this chromosome
        if feat.split('_')[0] != chrom_str:
            continue
        fn = which_file(feat)
        partition_fn_dict[fn].append(feat)

    # check if any features found on this chromosmoe    
    if len(partition_fn_dict) == 0:
        print("No features in this chromosome")
        quit()

    start = time.process_time()
    print("starting to get features now {}".format(start), flush=True)
    n = 0
    # build up a pandas df with samples as row and features from feat_coef_dict as columns
    feat_matrix_df = pd.DataFrame()
    for partition_fn in partition_fn_dict.keys():
        features_to_get = partition_fn_dict[partition_fn]
        print(partition_fn, features_to_get)
        partition_df = pd.read_csv(os.path.join(data_dir, partition_fn), sep='\t', skiprows=0, na_filter=False, low_memory=False)
        keep_df = partition_df[features_to_get]
        feat_matrix_df = feat_matrix_df.merge(keep_df, how='outer', left_index=True, right_index=True)
        n+=1
        print("done getting variance from file {}/{} {}".format(n, len(partition_fn_dict), time.process_time() - start), flush=True)
    
    # make index_col the sample names
    samples_names_df = pd.read_csv(samples_names_df_fn, header = None)
    feat_matrix_df['SAMPLE_NAME'] = samples_names_df[0]
    feat_matrix_df = feat_matrix_df.set_index("SAMPLE_NAME")
    feat_matrix_df = feat_matrix_df.drop('Unnamed: 0', axis = 1)
    
    return feat_matrix_df


'''
get_model_features_from_test_data.py: takes in an already trained model, a dataset to make predictions about using the model, and outputs the features that model wants to use from one chromosome in that dataset
@ model_fn: path to pickled model file
@ data_dir: path to dir of the flattened and partitioned data being used as input
@ out_dir: where to output the selected feature for this chromosome and the feat_coef_dict (inside of directory for data used to train model e.g. "raw_data_var_above_.1_flattened_partitioned/prediction_results/blueprint_disease/disease_features_this_model")
@ model_features_fn: path to file containing the features selected for use in the model (e.g. /selected_featres/smaller_top1%_variance_dataset.csv)
@ sample_names_df_fn: path to file coontaining samples names for htese samples (e.g. blueprint_data/blueprint_disease_sample_list.txt)
@ chrom_num: number of chromosome to get features from in this instance
'''

def main():
    model_fn = sys.argv[1]
    data_dir = sys.argv[2]
    out_dir = sys.argv[3]
    model_features_fn = sys.argv[4]
    samples_names_df_fn =sys.argv[5]
    os.makedirs(out_dir, exist_ok=True)
    # which chromosome
    chrom_num = sys.argv[6]
    chrom_str = "chr{}".format(chrom_num)


    # get the names and coefs of features with nonzero coefs in the trained model
    feat_coef_dict = get_nonzero_coef_features(model_fn, model_features_fn)
    # get the posterior values at each of these features, from the data_dir dataset, and turn into feature matrix
    feature_matrix_df = make_feature_matrix_one_chrom(feat_coef_dict, data_dir, chrom_str, samples_names_df_fn)

    # write out files
    feature_matrix_df.to_csv(os.path.join(out_dir, "{}_disease_feature_matrix.csv".format(chrom_str)))
    f = open(os.path.join(out_dir, "feat_coef_dict.pkl"), "wb+")
    pickle.dump(feat_coef_dict, f)
    f.close()

main()