import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import OneHotEncoder
from .utils import threshold_adjacency_matrices


class AdhdROIDataset(Dataset):
    def __init__(self, csv, data_root, atlas, transforms=None, filter='Yes', cp="", cnp="", task='DX'):
        super().__init__()
        self.filter = filter
        if filter == 'both':
            self.csv = csv
        elif filter == 'Yes':
            self.csv = csv[csv['Filename'].str.startswith('sf')]
        elif filter == 'No':
            self.csv = csv[csv['Filename'].str.startswith('sn')]
        else:
            raise ValueError(f'Invalid filter: {filter}')
        self.csv = self.csv.fillna(-999)
        self.csv['Subject_ID'] = self.csv['Subject_ID'].apply(lambda x: f'{x:07d}')
        self.data_root = data_root
        self.atlas = atlas
        self.data_path = os.path.join(data_root, atlas)
        self.transforms = transforms
        self.task = task

        self.labels = self.csv['DX'].values
        enc = OneHotEncoder(handle_unknown='ignore')
        self.onehot = enc.fit_transform(self.labels.reshape(-1, 1)).toarray()
        self.onehot = self.onehot.astype(int)

        if self.task == 'DX': # binary classification
            self.labels[self.labels > 0] = 1 
        self.n_classes = len(np.unique(self.labels))

        if cp:
            self.cp_columns = cp.replace(' ', '').split(',')
            self.cp_fea = self.csv[self.cp_columns].fillna(-1).values.astype(int)
            self.cp_fea[self.cp_fea < 0] = -1
            self.num_cp = len(self.cp_columns)
        else:
            self.cp_columns = None
            self.cp_fea = None
            self.num_cp = 0
        if cnp:
            self.cnp_columns = cnp.replace(' ', '').split(',')
            self.cnp_fea = self.csv[self.cnp_columns].fillna(-1).values.astype(float)
            self.cnp_fea[self.cnp_fea < 0] = -1
            self.num_cnp = len(self.cnp_columns)
        else:
            self.cnp_columns = None
            self.cnp_fea = None
            self.num_cnp = 0
        
        self.num_phe = self.num_cp + self.num_cnp

        if cnp and cp:
            self.phenotypes = np.concatenate((self.cp_fea, self.cnp_fea), axis=1)
        elif cnp:
            self.phenotypes = self.cnp_fea
        elif cp:
            self.phenotypes = self.cp_fea
        else:
            raise ValueError("No phenotypes provided")

    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        subject = row['Subject_ID']
        filename = row['Filename']
        label = self.labels[idx]
        onehot = self.onehot[idx]
        phi = self.phenotypes[idx]
        file_path = os.path.join(self.data_path, subject, filename)
        timeseries = pd.read_csv(file_path, sep='\t').values[:, 2:].astype(float) # drop the first two columns
        measure = ConnectivityMeasure(kind='correlation')
        corr = measure.fit_transform([timeseries])[0]
        corr[corr == float('inf')] = 0

        if self.transforms:
            timeseries = self.transforms(timeseries)
        timeseries = timeseries[:100]
        timeseries = torch.from_numpy(timeseries).float()
        corr = torch.from_numpy(corr).float()
        label = torch.from_numpy(np.array(label)).long()
        onehot = torch.from_numpy(onehot).long()
        phenotypes = torch.from_numpy(phi).float()

        sparse_connection = corr.clone()
        sparse_connection.fill_diagonal_(1)

        return {'timeseries': timeseries, 'corr': corr, 'label': label, 'onehot': onehot,
                'sparse_connection': sparse_connection, 'phenotypes': phenotypes}

