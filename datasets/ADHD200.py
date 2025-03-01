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


        self.cp_columns = cp.split(', ')
        self.cnp_columns = cnp.split(', ')

        self.cp_labels = self.csv[self.cp_columns].values.astype(int)
        self.cnp_labels = self.csv[self.cnp_columns].values.astype(float)

        self.cp_labels[self.cp_labels < 0] = -1
        self.cnp_labels[self.cnp_labels < 0] = -1

        self.num_cp = len(self.cp_columns) + 1
        self.num_cnp = len(self.cnp_columns)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        subject = row['Subject_ID']
        filename = row['Filename']
        label = self.labels[idx]
        onehot = self.onehot[idx]
        cp_label = self.cp_labels[idx]
        cnp_label = self.cnp_labels[idx]
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
        cnp_label = torch.from_numpy(cnp_label).float()
        cp_label = torch.from_numpy(cp_label).float()

        sparse_connection = corr.clone()
        sparse_connection.fill_diagonal_(1)

        return {'timeseries': timeseries, 'corr': corr, 'label': label, 'onehot': onehot,
                'sparse_connection': sparse_connection, 'cnp_label': cnp_label, 'cp_label': cp_label}

