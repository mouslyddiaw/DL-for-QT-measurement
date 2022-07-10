import torch
import random 
from pathlib import Path
import pandas as pd
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader 
from sklearn.preprocessing import  StandardScaler
import matplotlib.pyplot as plt
from model.data_loader import *

# class ECGDatasetStratified2(Dataset):
#     def __init__(self, shift=False):
#         self.directory = 'data/private-database/templates/' 
#         self.idx_fname = associate_idx_to_fname()
#         self.shift  = shift
        
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(idx_fname)
    
#     def __getitem__(self, index):
#         'Generates one sample of data' 
#         # Select sample
#         fname = self.idx_fname[index]  
#         # Load data and get label
        
#         templates = pd.read_csv(self.directory+str(fname)+'.csv').transpose().values 
#         templates_normalized = [torch.tensor(normalize(template)).float() for template in templates] 
#         qt_interval, mask  = get_markers_w_fname(fname)
         
#         sample = {'templates': templates_normalized, 
#                   'interval': torch.tensor(qt_interval).float(), 
#                   'class': torch.tensor(compute_class_qt(qt_interval)).float(), 
#                   'mask': torch.tensor(mask).float()}
#         plt.figure(figsize=(5,5))         
#         for template in templates:
#             plt.plot(normalize(template), color = 'silver' )
#         plt.plot(mask, color ='k')   
#         plt.savefig(f'plot_{index}.png')

#         if self.shift:
#             value_shift = random.randint(-180, 180) 
#             transform = Shift(value_shift)  
#             sample = transform(sample)
#         return sample

def get_first_row_csv(csv_file):
    with open(csv_file) as csvfile:
        for index, row in enumerate(csv.reader(csvfile)):
            return row

def get_ecgs(csv_file): 
    ecgs = []
    with open(csv_file) as csvfile:
        for index, row in enumerate(csv.reader(csvfile) ):
            if index<5:
                pass
            else:
                ecgs.append(np.array([float(val) for val in row])) 
    df = pd.DataFrame(np.array(ecgs), columns = ['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6'])
    return df



if __name__ == '__main__':
    # data = ECGDatasetStratified2()
    # idx = np.random.randint(1,8855)
    # print(idx, data.idx_fname[idx] )
    # y = data[idx]
    labels  = pd.read_csv('data/labels.csv')
    pathlist = Path('data/templates-with-pid').glob('**/*.csv') 
    compt=0
    for csv_file in pathlist:
        #print(csv_file)
        try:
            fname = get_first_row_csv(csv_file)[0]  
        except TypeError:
            pass
        else: 
            #print(fname)
            assert(fname in list(labels.filename))
            ecgs = get_ecgs(csv_file)  
            ecgs.to_csv(f'data/private-database/templates/{fname}.csv', index=False)