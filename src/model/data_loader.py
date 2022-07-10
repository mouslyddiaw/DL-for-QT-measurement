import torch
import random 
from pathlib import Path
import pandas as pd
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader 
from sklearn.preprocessing import  StandardScaler


class ECGDatasetStratified(Dataset):
    def __init__(self, shift=False):
        self.directory = 'data/private-database/templates/' 
        self.idx_fname = associate_idx_to_fname()
        self.shift  = shift
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(idx_fname)
    
    def __getitem__(self, index):
        'Generates one sample of data' 
        # Select sample
        fname = self.idx_fname[index]  
        # Load data and get label
        
        templates = pd.read_csv(self.directory+str(fname)+'.csv').transpose().values 
        templates_normalized = [torch.tensor(normalize(template)).float() for template in templates] 
        qt_interval, mask  = get_markers_w_fname(fname)
         
        sample = {'templates': templates_normalized, 
                  'interval': torch.tensor(qt_interval).float(), 
                  'class': torch.tensor(compute_class_qt(qt_interval)).float(), 
                  'mask': torch.tensor(mask).float()}

        if self.shift:
            value_shift = random.randint(-180, 180) 
            transform = Shift(value_shift)  
            sample = transform(sample)
        return sample
    
def associate_idx_to_fname(directory='data/templates-with-pid'):
    idx_fname = {}
    index=1
    pathlist = list(Path(directory).glob('**/*.csv')) 
    for csv_file in pathlist: 
        try:
            idx_fname[index] = get_first_row_csv(csv_file)[0]
            index+=1 
        except TypeError:
            pass 
    return idx_fname

def get_markers_w_fname(fname, directory='data/', file_labels='labels.csv'):
    '''
    Get Q and T positions
    '''
    mask = [0 for i in range(600)]
    with open(directory+file_labels) as csvfile:
        reader =  csv.reader(csvfile, delimiter=',') 
        for row in reader: 
            if str(fname) in row[1]:
                qonset = int(row[2])
                toffset = int(row[3])
                qt_interval = toffset - qonset
                lower = qonset//2
                upper = toffset//2 + 1
                mask[lower:upper] = [1 for i in range (lower, upper)] 
                return  qt_interval, mask 

def get_first_row_csv(csv_file):
    with open(csv_file) as csvfile:
        for index, row in enumerate(csv.reader(csvfile)):
            return row

################ 

class ECGDataset(Dataset):
    def __init__(self, shift=False):
        self.directory = 'data/' 
        self.list_IDs = list(range(1, 28000))
        self.shift  = shift
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        'Generates one sample of data' 
        # Select sample
        ID = self.list_IDs[index]  
        # Load data and get label
        if index>8853: 
            templates = pd.read_csv(self.directory+'templates_ptb_xl/id-'+str(ID)+'.csv').transpose().values 
            templates_normalized = [torch.tensor(template).float() for template in templates]  
            qt_interval, mask  = get_markers(ID, self.directory, file_labels='qt_templates_ptb_xl.csv')
        else:
            templates = pd.read_csv(self.directory+'templates/id-'+str(ID)+'.csv').transpose().values 
            templates_normalized = [torch.tensor(normalize(template)).float() for template in templates] 
            qt_interval, mask  = get_markers(ID, self.directory)
         
        sample = {'templates': templates_normalized, 
                  'interval': torch.tensor(qt_interval).float(), 
                  'class': torch.tensor(compute_class_qt(qt_interval)).float(), 
                  'mask': torch.tensor(mask).float()}

        if self.shift:
            value_shift = random.randint(-180, 180) 
            transform = Shift(value_shift)  
            sample = transform(sample)
        return sample

class Shift(object):
    """Shift signals
    """

    def __init__(self, shift):
        self.shift = shift

    def __call__(self, sample):
        templates, qt_interval, class_qt, mask = sample['templates'], sample['interval'], sample['class'], sample['mask']
        shift = self.shift
        if shift<0:
            mask_shifted = np.array(list(mask[abs(shift):]) + [0 for i in range(abs(shift))])
        else:
            mask_shifted = np.array([0 for i in range(abs(shift))] + list(mask[:len(mask)-abs(shift)]) )
                                                                  
        shifted_templates = [] 
        for index, template in enumerate(templates):
            if shift<0:
                template_shifted = np.array(list(template[abs(shift):]) + [0 for i in range(abs(shift))])
            else:
                template_shifted = np.array([0 for i in range(abs(shift))] + list(template[:len(template)-abs(shift)]) ) 
            shifted_templates.append(torch.tensor(template_shifted).float())
          
        sample_shifted =  {'templates': shifted_templates, 
                           'interval': qt_interval, 
                           'class': class_qt, 
                           'mask': torch.tensor(mask_shifted).float()}
 
        return sample_shifted 

def get_markers(file_id, directory, file_labels='labels.csv'):
    '''
    Get Q and T positions
    '''
    mask = [0 for i in range(600)]
    with open(directory+file_labels) as csvfile:
        reader =  csv.reader(csvfile, delimiter=',') 
        for row in reader: 
            if str(file_id) in row[0]:
                qonset = int(row[2])
                toffset = int(row[3])
                qt_interval = toffset - qonset
                lower = qonset//2
                upper = toffset//2 + 1
                mask[lower:upper] = [1 for i in range (lower, upper)] 
                return  qt_interval, mask 
            
def compute_class_qt(qt):
    intervals = range(150, 752, 2) 
    qt_class = 0 
    for index, (lower, upper) in enumerate(zip(intervals[:-1],intervals[1:] )):   
        if qt>=lower and qt<upper: 
            qt_class=index
            break
    return qt_class

def normalize(signal):
    signal_normalized = StandardScaler().fit_transform(np.array(signal).reshape(-1, 1) ).reshape(len(signal))  
    return signal_normalized
