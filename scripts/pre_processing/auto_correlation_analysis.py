import pandas as pd
import sys
import os
import argparse
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
"""
auto_correlation_analysis.py
@ this script takes all the posterior files for a given sample, and calculates the correlation coefficient of the posterior prob values for each state with themselves shifted by some between 1 and max_windows_to_shift
"""


def shift_df(unshifted_df, num_shift_windows):
    if num_shift_windows == 0:
        return unshifted_df, unshifted_df
    shifted_df = unshifted_df.copy(deep = True)
    # we are shifting the shifted df 'down' so need to remove its last row and the first row of the unshifted bc they will not have a corresponding row in the other
    shifted_df = shifted_df[:-num_shift_windows]
    unshifted_df = unshifted_df[num_shift_windows:]
    shifted_df.reset_index(drop = True, inplace = True)
    unshifted_df.reset_index(drop = True, inplace = True)
    return shifted_df, unshifted_df


def correlate_one_state(shifted_df, unshifted_df, state_num):
    state_name = 'E' + str(state_num)
    #print(unshifted_df[state_name])
    #print(shifted_df[state_name])
    correlation = unshifted_df[state_name].corr(shifted_df[state_name])
    return correlation

def auto_correlation(max_windows_to_shift, post_fn_root):
    # create a dict with window offsets as keys and pandas series of mean auto-correlations by state
    corr_by_shift_dict = {}
    for num_shift_windows in range(1, max_windows_to_shift + 1,5):
        # create a dict with states as keys and a list of corr values, one for each chromosome, to make into a df
        correlations = {key:[] for key in range(1,13)}
        # ignores sex and mitochondrial chomosomes
        for chromosome in range(1,23):
            this_chrom_post_fn = post_fn_root + '_chr' + str(chromosome) + '_posterior.txt'
            post_df = pd.read_csv(this_chrom_post_fn, sep = '\t', skiprows=[0])
            shifted_df, unshifted_df = shift_df(post_df, num_shift_windows)
            for state in range(1,13):
                correlation = correlate_one_state(shifted_df, unshifted_df, state)
                correlations[state].append(correlation)
        # turn the dict we built up into a df
        corr_df = pd.DataFrame.from_dict(correlations)
        mean_series = corr_df.mean()
        mean_series = mean_series.rename(str(num_shift_windows))
        corr_by_shift_dict['offset' + str(num_shift_windows)] = mean_series
    corr_by_shift_df = pd.DataFrame.from_dict(corr_by_shift_dict, orient='index')
    print(corr_by_shift_df)

    # create a scatterplot with x-axis as offset amount, y axis is autocorrelation, and each state's points are a different color
    figure(figsize=(5, 5), dpi=80)
    corr_by_shift_df.plot.bar()
    plt.title("Autocorrelation by offset amount")
    plt.ylabel("Autocorrelation (pearson)")
    plt.xlabel("Offset amount (200bp windows)")
    plt.xticks(rotation=0)
    plt.savefig("./output/" + post_fn_root.split('/')[-1] + '_barplot.pdf')



def main():
    # read arguemnts
    parser = argparse.ArgumentParser()
    # it does not matter which chromosome, all will be looked at
    parser.add_argument('-p' ,'--post_fn', help='path to posterior prob file for one chromosome (it does not matter which one, all chromosomes from that sample will be used) of one sample')
    parser.add_argument('-w' ,'--max_windows_to_shift', type= int, help='maximum number of windows to shift by. windows will be shifted by 1 through this number, by 5')
    args = parser.parse_args()
    post_fn = args.post_fn
    max_windows_to_shift = args.max_windows_to_shift
    post_fn_root = '_'.join(post_fn.split('_')[:-2])
    auto_correlation(max_windows_to_shift, post_fn_root)


main()
