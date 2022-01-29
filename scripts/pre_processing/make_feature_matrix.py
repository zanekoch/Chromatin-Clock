import pandas as pd
import os
import sys
import numpy as np
import glob
import pickle
from operator import itemgetter
import time
from sklearn.feature_selection import VarianceThreshold

FEAT_NUM = 180000000
NO_ZERO_AGE = True

def calculate_variances(data_fns, out_dir):
    start = time.process_time()
    variance_dict = {}
    # read in each flattened data file and combine them 
    n = 0
    num_files = len(data_fns)
    for fn in data_fns:
        print("starting next file", flush=True)
        # read in raw-ish file
        cur_f_df = pd.read_csv(fn, sep='\t', skiprows=0, na_filter=False, low_memory=False)
        cur_f_df = cur_f_df.set_index('Unnamed: 0')
        
        if NO_ZERO_AGE:
            # drop the rows corresponding to samples with 0 age
            age_df = pd.read_csv("/cellar/users/zkoch/cs_clock_proj/blueprint_data/all_samples_age_pairs.tsv", sep='\t')
            age_df = age_df.set_index('SAMPLE_NAME')
            not_zero_age_df = age_df[age_df['DONOR_AGE'] != 0]
            not_zero_age_list = not_zero_age_df['SAMPLE_NAME'].to_list()
            cur_f_df = cur_f_df[cur_f_df.index.isin(not_zero_age_list)]
            print(len(cur_f_df.index), flush=True)
        print("done dropping 0s{}".format( time.process_time() - start), flush=True)
        # filter out features with variance below .1
        try:
            constant_filter = VarianceThreshold(threshold=0.1)
            constant_filter.fit(cur_f_df)
            kept_feat_df = cur_f_df[cur_f_df.columns[constant_filter.get_support(indices=True)]]
            kept_feat_df.to_csv(os.path.join(out_dir, fn.split('/')[-1]), sep = '\t')
            print("done filtering {}".format( time.process_time() - start), flush=True)
            # for each feature in the current file, write the feature name and variance to variance_dict
            test_list = kept_feat_df.columns.tolist()
            for (columnName, columnData) in cur_f_df[test_list].iteritems():
                variance_dict[columnName] = np.var(columnData.values)
            print("done getting variance dict items{}".format( time.process_time() - start), flush=True)
        except Exception as e: # happens when no features above .1 in that partition
            print("error {} occured in iteration {}".format(e, n))
            # don't add anything to variance_dict bc no features above .1
            continue

        print("done getting variance from file {}/{} {}".format(n, num_files, time.process_time() - start), flush=True)
        n+=1
    # choose keep num to keep highest .1% of variances
    l = len(variance_dict)
    if FEAT_NUM * .001 > l:
        keep_num = l
    else:
        keep_num = FEAT_NUM * .001

    # keep only top .1% variance positions
    highest_variance_dict = dict(sorted(variance_dict.items(), key = itemgetter(1), reverse = True)[:keep_num])
    variance_dict = dict(sorted(variance_dict.items(), key = itemgetter(1), reverse = True))
    return highest_variance_dict, variance_dict

def select_features_to_keep(data_fns, features_to_keep_dict):
    start = time.process_time()
    out_df = pd.DataFrame()
    # select the features from data files that are in the features_to_keep_dict and write them to an output dataframe
    n = 0 
    num_files = len(data_fns)
    for fn in data_fns:
        cur_file = pd.read_csv(fn, sep='\t', skiprows=0, na_filter=False, low_memory=False)
        cur_file = cur_file.set_index('Unnamed: 0')
        # keep only the features present in the keys of features_to_keep_dict
        features_to_keep = set(cur_file.columns) & set(features_to_keep_dict.keys())
        keep_df = cur_file[features_to_keep]
        out_df = out_df.merge(keep_df, how='outer', left_index=True, right_index=True)
        print("done getting features from file {}/{} {}".format(n, num_files, time.process_time() - start), flush=True)
        n+=1
    return out_df

def from_file_selection(data_fns, out_dir, features_fn):
    # read in all the included features
    file_to_read = open(features_fn , "rb")
    feature_dict = pickle.load(file_to_read)
    # create a df with these features
    out_df = select_features_to_keep(data_fns, feature_dict)
    out_df.to_csv(os.path.join(out_dir, 'selected_features', 'disease_features_from_log_top.1%_variance_healthy_dataset.csv'))
    return 

def variance_selection(data_fns, out_dir):
    # get top 1% variances across all features
    highest_variance_dict, all_variance_dict = calculate_variances(data_fns, out_dir)
    # write these both sets of variances out for further analysis
    with open(os.path.join(out_dir, 'selected_features', 'top.1%_variance_dict.pkl'), 'wb') as f:
        pickle.dump(highest_variance_dict, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(out_dir, 'selected_features', 'all_variance_dict.pkl'), 'wb') as f:
        pickle.dump(all_variance_dict, f, pickle.HIGHEST_PROTOCOL)

    # write out highest variance features as .csv for use in training model 
    out_df = select_features_to_keep(data_fns, highest_variance_dict)
    out_df.to_csv(os.path.join(out_dir, 'selected_features', 'no0age_top.1%_variance_dataset.csv'))

    return

"""
make_feature_matrix.py: given a set of flattened files, raw or partially processed files (e.g. only_gene_bodies), do some feature selection on them and write out the resulting features to a format that ChromatinClock.py can take as input (e.g. .csv file with columns=features, rows=samples)
@ feature_selection_method: variance or from_file
@ data_dir: path to directory containing flattened and partitioned files to get features frmo
@ out_dir: path to directory to write intermediate files to (resulting feature matrix is in out_dir/selected_features)
@ features_fn: [optional] if from_file then this is the file of features to select. Should be a pickled dictionary
"""
def main():
    # current possible inputs: variance, from_file
    feature_selection_method = sys.argv[1]
    # path to directory of files to do the feature selection on
    data_dir = sys.argv[2]
    # where to write output feature file and intermediates
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "selected_features"), exist_ok=True)
    
    # get list of input file names
    fn_expression = os.path.join(data_dir, '*.tsv')
    data_fns = glob.glob(fn_expression)
    if feature_selection_method == 'variance':
        variance_selection(data_fns, out_dir)
    elif feature_selection_method == "from_file":
        features_fn = sys.argv[4]
        from_file_selection(data_fns, out_dir, features_fn)
    
main()