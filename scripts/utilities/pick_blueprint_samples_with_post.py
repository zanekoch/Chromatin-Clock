'''
This script reads in all the data from the blueprint samples, selects on the samples we want (those with posterior and age values), writes out a .tsv file with sample-transformedAge pairs, and creates a age distr. histogram
'''
import pandas as pd
import sys
import os
import glob
import matplotlib
import math
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt

# get each of the sets of blueprint data, healthy and healthy_model
# dir = '../blueprint_data/POSTERIOR_Blueprint_release_201608/POSTERIOR_healthy_model'
sample_name_list = []
# for f in glob.glob(os.path.join(dir, '*.txt')):
#     sample_name = f.split('/')[-1].split('_')[0]
#     if sample_name not in sample_name_list:
#         sample_name_list.append(sample_name)
# dir = '../blueprint_data/POSTERIOR_Blueprint_release_201608/POSTERIOR_healthy'
# for f in glob.glob(os.path.join(dir, '*.txt')):
#     sample_name = f.split('/')[-1].split('_')[0]
#     if sample_name not in sample_name_list:
#         sample_name_list.append(sample_name)
dir = '../../blueprint_data/POSTERIOR_Blueprint_release_201608/POSTERIOR_disease'
for f in glob.glob(os.path.join(dir, '*.txt')):
    sample_name = f.split('/')[-1].split('_')[0]
    if sample_name not in sample_name_list:
        sample_name_list.append(sample_name)
# with open("../../blueprint_data/blueprint_disease_sample_list.txt", "w+") as f:
#     for name in sample_name_list:
#         f.write(name+'\n')
print(len(sample_name_list))


# select only the metadata for above samples
df = pd.read_csv('../blueprint_data/20160816.data.index_metadata', sep='\t')
boolean_series = df.SAMPLE_NAME.isin(sample_name_list)
df = df[boolean_series]

# now we have df with just the lines for samples that have posterior prob data
# need to just select lines about chip-seq
df = df[(df['STUDY_NAME'] == 'BLUEPRINT ChIP-seq data for cells in the haematopoietic lineages, from adult and cord blood samples.') | (df['STUDY_NAME'] == 'ChIP-seq data for cell lines in the haematopoietic lineages, hematological malignancy') | (df['STUDY_NAME'] == 'BLUEPRINT ChIP-seq of Epigenetic programming during monocyte to macrophage differentiation and trained innate immunity')]


# need to get the age,tossue type, cell type, and biomaterial for each unique sample name
out_ages = []
out_tt = []
out_ct =[]
out_bio = []
for sample_name in sample_name_list:
    one_sample_df = df[df['SAMPLE_NAME'] == sample_name]
    # double check all of the samples with same ID have same age
    all_expt = one_sample_df.EXPERIMENT_TYPE.unique()
    all_ls = one_sample_df.LIBRARY_STRATEGY.unique()
    all_ages = one_sample_df.DONOR_AGE.unique()
    all_bio = one_sample_df.BIOMATERIAL_TYPE.unique()
    all_ct = one_sample_df.CELL_TYPE.unique()
    all_tt = one_sample_df.TISSUE_TYPE.unique()
    if len(all_ages) != 1 or len(all_bio) != 1 or len(all_ct)!=1 or len(all_tt)!=1 or len(all_ls) != 1:
        print("error!",all_ages,all_bio,all_ct,all_tt, all_ls, all_expt)
        print(sample_name)
        quit()
    out_ages.append(all_ages[0])
    out_bio.append(all_ct[0])
    out_tt.append(all_tt[0])
    out_ct.append(all_bio[0])


# create df with all the info and make variable types correct
out_dict = {'SAMPLE_NAME':sample_name_list, 'DONOR_AGE':out_ages, 'BIOMATERIAL_TYPE':out_bio, 'CELL_TYPE':out_ct, 'TISSUE_TYPE':out_tt}
print(out_dict)
out_df = pd.DataFrame.from_dict(out_dict)
print(out_df)

# drop samples with ages that are nan
out_df = out_df.dropna()
print(out_df)

# get ages in correct format, taking the younger age of the range
out_df['DONOR_AGE'] = out_df['DONOR_AGE'].apply(lambda x: int(x.split(' ')[0]))
print(out_df)
# age transformation funciton for pd.apply
def transform_age(age, adult_age):
    if age <= adult_age:
        new_age = math.log(age + 1) - math.log(adult_age + 1)
    else:
        new_age = (age - adult_age)/(adult_age + 1)
    return new_age

sample_age_df = pd.concat([out_df['SAMPLE_NAME'],out_df['DONOR_AGE']],axis=1 )
sample_age_df.to_csv('../blueprint_data/healthy_samples_age_pairs.tsv', sep='\t', index=False)
# transform age
out_df['DONOR_AGE'] = out_df['DONOR_AGE'].apply(lambda x : transform_age(x, 20))
# wirte out sample-transormed age file
sample_age_df = pd.concat([out_df['SAMPLE_NAME'],out_df['DONOR_AGE']],axis=1 )
sample_age_df.to_csv('../blueprint_data/healthy_samples_age_pairs_transformed.tsv', sep='\t', index=False)
print()

print(out_df)
print(out_df.DONOR_AGE.value_counts())
print(out_df.BIOMATERIAL_TYPE.value_counts())
print(out_df.CELL_TYPE.value_counts())
print(out_df.TISSUE_TYPE.value_counts())

# create age distribution plot
'''
d = out_df[out_df['BIOMATERIAL_TYPE'] == 'Primary Cell Culture']
print(d)
out_df.hist(column='DONOR_AGE', bins=80)
plt.xlabel("Age")
plt.ylabel("Number of individuals")
#plt.savefig('blueprint_age_distr_hist.pdf')
with open('sample_list.txt', 'w+') as out_f:
    for sample in sample_name_list:
        out_f.write("%s\n" % sample)


print(sample_name_list)
print(len(sample_name_list))
'''
