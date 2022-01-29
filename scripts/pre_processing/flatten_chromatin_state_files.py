import sys
import numpy as np
import re
import pandas as pd
import os
import glob
import time
NUM_STATES = 12
NUM_WINDOWS_PER_FILE = 5000

"""
flatten_chromosome_state_files.py
@ this script takes all the posterior files for a given chromosome, breaks them into smaller 10k window pieces, and flattens them so that each row is a sample and the columns are the 10k*12 states
"""


def flatten(sample_list, data_dir, chrom_name, chrom_num, state_order, chrom_partition_list, out_dir):
    start = time.process_time()
    out_fn = os.path.join(out_dir, "{}.tsv".format(chrom_name))
    first = True
    for sample in sample_list:
        fn = os.path.join(data_dir, sample + '*' + chrom_name + '_*.tsv')
        print(fn, flush=True)
        chrom_file = glob.glob(fn)
        try:
            chrom_fn = chrom_file[0]
        except:
            sys.exit('no file {} for this sample and chromosome found in data_dir'.format(fn))
        if data_dir.split('/')[2] == 'processed_data':
            whole_chrom_df = pd.read_csv(chrom_fn, sep='\t', index_col=0)
        else:
            whole_chrom_df = pd.read_csv(chrom_fn, sep='\t', skiprows=1)
        to_flatten_df = whole_chrom_df
        this_flat_row_df = flatten_one_partition(to_flatten_df, state_order, sample, chrom_name)
        if first:
            chrom_df = this_flat_row_df
            first = False
        else:
            chrom_df = chrom_df.append(this_flat_row_df.loc[sample], ignore_index=False)
        print(chrom_df, flush=True)
        print("1",time.process_time() - start, flush=True)
    chrom_df.to_csv(out_fn, sep='\t')

def flatten_with_partition(sample_list, data_dir, chrom_name, chrom_num, state_order, chrom_partition_list, out_dir):
    start = time.process_time()
    for partition_num in range(len(chrom_partition_list)):
        partition_tup = chrom_partition_list[partition_num]
        partition_fn = os.path.join(out_dir, "{}_{}.tsv".format(chrom_name, partition_num))
        first = True
        for sample in sample_list:
            if data_dir.split('/')[-2] == 'processed_data':
                fn = os.path.join(data_dir, sample + '*' + chrom_name + '_*.tsv')
            else:
                fn = os.path.join(data_dir, sample + '*' + chrom_name + '_*.txt')
            chrom_file = glob.glob(fn)

            try:
                chrom_fn = chrom_file[0]
            except:
                sys.exit('no file {} for this sample and chromosome found in data_dir'.format(fn))
            # read in one chromosome data file
            # the way we read it in depends on if it is processed or not
            if data_dir.split('/')[-2] == 'processed_data':
                whole_chrom_df = pd.read_csv(chrom_fn, sep='\t', index_col=0)
            else:
                whole_chrom_df = pd.read_csv(chrom_fn, sep='\t', skiprows=1)
            to_flatten_df = whole_chrom_df.iloc[partition_tup[0]:partition_tup[1]+1]
            this_flat_row_df = flatten_one_partition(to_flatten_df, state_order, sample, chrom_name)
            if first:
                partition_df = this_flat_row_df
                first = False
            else:
                partition_df = partition_df.append(this_flat_row_df.loc[sample], ignore_index=False)
            print(partition_df, flush=True)
            print("1",time.process_time() - start, flush=True)
        partition_df.to_csv(partition_fn, sep='\t')


def flatten_one_partition(to_flatten_df, state_order, sample, chrom_name):
    # flatten the matrix
    chrom_df = to_flatten_df.rename_axis('window').reset_index().melt('window', value_name = sample, var_name='state')
    chrom_df['state'] = chrom_df['state'].astype(state_order)
    chrom_df = chrom_df.sort_values(['window', 'state'])
    chrom_df['chr_window_state'] = chrom_name + '_' + chrom_df['window'].astype(str) + '_' + chrom_df['state'].astype(str)
    chrom_df.drop(['window','state'], inplace = True, axis = 1)
    # transpose so first row is window_state and second is posterior value
    chrom_df = chrom_df.set_index('chr_window_state').transpose()
    return chrom_df


def split_chrom_len(chrom_name, data_dir):
    # get length of this chrom by looking at length of a arbitrary sample
    sample = "S01PV8H1"
    if data_dir.split('/')[2] == 'processed_data':
        fn = os.path.join(data_dir, sample + '*' + chrom_name + '_*.tsv')
        chrom_fn= glob.glob(fn)[0]
        whole_chrom_df = pd.read_csv(chrom_fn, sep='\t', index_col=0)
    else:
        fn = os.path.join(data_dir, sample + '*' + chrom_name + '_*.tsv')
        chrom_fn= glob.glob(fn)[0]
        whole_chrom_df = pd.read_csv(chrom_fn, sep='\t', index_col=0)
    chrom_len = len(whole_chrom_df)
    # return a list of tuples of window indeces for each new file
    # the tuples are (num of first window in file, num of last window in file)
    num_files = chrom_len // NUM_WINDOWS_PER_FILE
    if num_files % NUM_WINDOWS_PER_FILE != 0:
        leftover = chrom_len % NUM_WINDOWS_PER_FILE
        num_files += 1
    chrom_partition_list = []
    for file_num in range(num_files):
        if file_num == num_files - 1:
            tup = (file_num * NUM_WINDOWS_PER_FILE, file_num * NUM_WINDOWS_PER_FILE + leftover)
            chrom_partition_list.append(tup)
        else:
            tup = (file_num * NUM_WINDOWS_PER_FILE, (file_num + 1) * NUM_WINDOWS_PER_FILE - 1)
            chrom_partition_list.append(tup)
    return chrom_partition_list



def main():
    # read command line args
    chrom_num = str(sys.argv[1])
    sample_list_fn = str(sys.argv[2])
    data_dir = str(sys.argv[3] ) #"../blueprint_data/raw_data"
    out_dir = str(sys.argv[4]) #"../blueprint_data/processed_data"

    chrom_name = 'chr' + str(chrom_num)
    # create out_dir if it does not exist
    os.makedirs(out_dir, exist_ok=True)

    # create an ordering of states to later use in sorting
    states_list = ['E' + str(i) for i in range(1,13)]
    state_order = pd.CategoricalDtype(states_list, ordered = True)

    # create a list of all the blueprint sample names
    with open(sample_list_fn) as f:
        sample_list = f.readlines()
        sample_list = [sample.rstrip() for sample in sample_list]
    print(sample_list)
    # break into sections
    chrom_partition_list = split_chrom_len(chrom_name, data_dir)
    print(chrom_partition_list, flush=True)
    flatten_with_partition(sample_list, data_dir, chrom_name, chrom_num, state_order, chrom_partition_list, out_dir)
    #flatten(sample_list, data_dir, chrom_name, chrom_num, state_order, chrom_partition_list, out_dir)


main()
