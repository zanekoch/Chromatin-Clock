import sys
import numpy as np
import glob
import pandas as pd
import os

'''
average_windows.py
@ script to take the raw files (a file for each chrom and sample) and average num_windows_to_combine adjacent windows into one window
@ one chromosome at a time
'''
def average_files(files_to_average, num_windows_to_combine, out_dir):
    # loop over the chrom file for each sample
    for fn in files_to_average:
        df = pd.read_csv(fn, sep = '\t', skiprows=1)
        out_df = df.groupby(np.arange(len(df))//num_windows_to_combine).mean()
        out_fn = os.path.join(out_dir, fn.split('/')[-1].split('.')[0] + '_' + str(num_windows_to_combine) + 'winAveraged.tsv')
        out_df.to_csv(out_fn, sep='\t')


def main():
    data_dir = sys.argv[1]
    chrom_num = int(sys.argv[2])
    num_windows_to_combine = int(sys.argv[3])
    out_dir = sys.argv[4]

    chrom_name = 'chr' + str(chrom_num)
    files_to_average =  glob.glob(os.path.join(data_dir, '*' + chrom_name + '_*.txt'))
    print(files_to_average)
    average_files(files_to_average, num_windows_to_combine, out_dir)


main()