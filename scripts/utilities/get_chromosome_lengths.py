import sys
import numpy as np
import pandas as pd
import os
import glob

data_dir = "/cellar/users/zkoch/cs_clock_proj/blueprint_data/POSTERIOR_Blueprint_release_201608/POSTERIOR_healthy"

random_sample = "S00FWHH1"

# open each chromosome file and get the number of windows
len_dict = {}
for chrom_num in range(1,23):
    df = pd.read_csv(os.path.join(data_dir, random_sample + '_12_12_Blueprint_release_201608_chr' + str(chrom_num) + '_posterior.txt'), sep = '\t', skiprows = 1)
    chrom_tag = 'chr' + str(chrom_num)
    len_dict[chrom_tag] =  len(df)
print(len_dict)
len_df = pd.DataFrame.from_dict(len_dict, orient='index')
len_df = len_df.rename(columns={"index" : "chr", 0 : "Length"})
len_df.reset_index(level=0, inplace = True)
len_df.to_csv("../blueprint_data/chromosome_lengths.tsv", sep = '\t')
