import sys
import pandas as pd
import os
import glob

"""
filter_by_functional_element.py
@ given a file of positions of functional elements (e.g. gene bodies, promoters, enhancers) subset the raw posterior data to only include windows overlapping these positions
@ do this for one chromosome
"""

def subset(raw_dir, chrom_name, pos_to_keep_l, out_dir):
    fns = glob.glob(os.path.join(raw_dir, '*' + chrom_name + '_posterior.txt'))
    for sample_fn in fns:
        sample_df = pd.read_csv(sample_fn, sep = '\t', skiprows=1)
        sample_df = sample_df.iloc[pos_to_keep_l]
        new_name = sample_fn.split('/')[-1].split('.')[0] + '_only_gene_bodies.tsv'
        sample_df.to_csv(os.path.join(out_dir, new_name), sep='\t')

def get_positions(func_el_fn, chrom_name):
    # process the func_df to get only the coding regions
    func_df = pd.read_csv(func_el_fn, sep = '\t')
    # keep only this chrom
    chrom_df = func_df[func_df.seqname == chrom_name]
    chrom_len = len(chrom_df)
    # select only exons that are protein coding
    exon_df = chrom_df[chrom_df.feature == 'exon']
    coding_df = exon_df[exon_df.gene_type == 'protein_coding']
    # convert each start and end bp into window, rounding down for start
    start_bp_l = coding_df['start'].to_list()
    end_bp_l = coding_df['end'].to_list()
    # only include exons that are not super small
    start_window_l = []
    end_window_l = []
    for start, end in zip(start_bp_l, end_bp_l):
        if end - start < 200:
            continue
        else:
            start_window_l.append(start//200)
            end_window_l.append(end//200)
    # create a list of all numbers between each start_window_l[i] and end_window_l[i]
    all_indices_l = []
    for start, end in zip(start_window_l, end_window_l):
        all_indices_l.extend([pos for pos in range(start, end + 1)])
    print(len(all_indices_l))
    # drop duplicates by converting the list to a set
    all_indices = set(all_indices_l)
    all_indices = list(all_indices)
    all_indices.sort()
    print(len(all_indices))
    return all_indices

def main():
    raw_dir = sys.argv[1]
    func_el_fn = sys.argv[2]
    chrom_num = sys.argv[3]
    out_dir = sys.argv[4]
    chrom_name = 'chr' + str(chrom_num)
    pos_to_keep_l = get_positions(func_el_fn, chrom_name)
    subset(raw_dir, chrom_name, pos_to_keep_l, out_dir)
# make list of tuples where the tuples are start and end locations of the fucntional element, converted into 200bp window (rounding up to include more than necessary where rounding is needed)
# turn this list into a list of every position we want to keep, then use .iloc to select these rows from the raw_df


main()
