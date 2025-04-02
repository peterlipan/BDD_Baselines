import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import OneHotEncoder
from .utils import threshold_adjacency_matrices


class String2Index:
    def __init__(self, unknown_token='unk'):
        self.unknown_token = unknown_token
        # Store mappings for each column: {column_idx: {value: index}}
        self.column_value2idx = {}
        # Reverse mappings: {column_idx: {index: value}}
        self.column_idx2value = {}
    
    def fit(self, data):
        """Learn mappings for each column independently."""
        n_columns = data.shape[1]
        for col in range(n_columns):
            # Get unique values in this column
            unique_values = np.unique(data[:, col])
            
            # Create value-to-index mapping (0, 1, 2...)
            value2idx = {val: idx for idx, val in enumerate(unique_values)}
            # Explicitly map unknown_token to -1 (overwrite if present)
            value2idx[self.unknown_token] = -1
            
            # Create reverse mapping (index-to-value)
            idx2value = {idx: val for val, idx in value2idx.items()}
            # Ensure -1 maps back to unknown_token
            idx2value[-1] = self.unknown_token
            
            # Store both mappings
            self.column_value2idx[col] = value2idx
            self.column_idx2value[col] = idx2value
    
    def transform(self, data):
        """Convert string data to indices using learned mappings."""
        transformed = np.empty_like(data, dtype=int)
        for col in range(data.shape[1]):
            col_mapping = self.column_value2idx.get(col, {})
            for row in range(data.shape[0]):
                # Use -1 as default for unseen values
                transformed[row, col] = col_mapping.get(data[row, col], -1)
        return transformed
    
    def reverse_transform(self, data):
        """Convert indices back to original strings."""
        reversed_data = np.empty_like(data, dtype=object)
        for col in range(data.shape[1]):
            col_mapping = self.column_idx2value.get(col, {})
            for row in range(data.shape[0]):
                # Default to unknown_token for invalid indices
                reversed_data[row, col] = col_mapping.get(data[row, col], self.unknown_token)
        return reversed_data


class AbideROIDataset(Dataset):
    def __init__(self, csv, data_root, atlas='cc400', task='DX', 
                 transforms=None, cp="", cnp="", string2index=None):
        self.csv = csv
        # keep consistent with the nan filling strategy
        csv = csv.fillna(-9999)

        self.filenames = csv['FILE_ID'].values

        self.labels = csv['DX_GROUP'].values
        self.sub_idx = csv['SUB_ID'].values
        enc = OneHotEncoder(handle_unknown='ignore')
        self.onehot = enc.fit_transform(self.labels.reshape(-1, 1)).toarray()
        self.onehot = self.onehot.astype(int)
        
        self.suffix = f"_rois_{atlas}.1D"
        self.data_root = data_root
        self.transforms = transforms
        self.n_classes = len(np.unique(self.labels))

        if cp:
            self.category_phenotype_names = cp.replace(' ', '').split(',')
            self.cp_fea = csv[self.category_phenotype_names].values.astype(str)
            self.cp_fea[self.cp_fea == -9999] = 'unk'
            self.cp_fea[self.cp_fea == '-9999'] = 'unk'
            self.cp_fea[self.cp_fea == '`'] = 'unk'
            if string2index:
                self.cp_fea = string2index.transform(self.cp_fea)
            else:
                self.string2index = String2Index()
                self.string2index.fit(self.cp_fea)
                self.cp_fea = self.string2index.transform(self.cp_fea)
            self.num_cp = len(self.category_phenotype_names)
        else:
            self.category_phenotype_names = None
            self.cp_fea = None
            self.num_cp = 0

        if cnp:
            self.continuous_phenotype_names = cnp.replace(' ', '').split(',')
            self.cnp_fea = csv[self.continuous_phenotype_names].values.astype(float)
            self.cnp_fea[self.cnp_fea == -9999] = -1
            self.cnp_fea[self.cnp_fea == '-9999'] = -1
            self.num_cnp = len(self.continuous_phenotype_names)
        else:
            self.continuous_phenotype_names = None
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
        label = self.labels[idx]
        onehot = self.onehot[idx]
        file_id = self.filenames[idx]
        phi = self.phenotypes[idx]
        cp = self.cp_fea[idx]
        cnp = self.cnp_fea[idx]
        sub = self.sub_idx[idx]

        file_path = os.path.join(self.data_root, self.filenames[idx] + self.suffix)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        timeseries = np.loadtxt(file_path, skiprows=0) # [T, N]
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
        phenotpyes = torch.from_numpy(phi).float()   
        cp = torch.from_numpy(cp).long()
        cnp = torch.from_numpy(cnp).float()    
        sub = torch.from_numpy(np.array(sub)).long()
        

        sparse_connection = corr.clone()
        sparse_connection.fill_diagonal_(1)
        
        return {'timeseries': timeseries, 'corr': corr, 'label': label, 'onehot': onehot,
                'sparse_connection': sparse_connection, 'phenotypes': phenotpyes,
                'cp': cp, 'cnp': cnp, 'sub_id': sub}
