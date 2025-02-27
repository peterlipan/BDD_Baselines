import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import OneHotEncoder


class AbideROIDataset(Dataset):
    def __init__(self, csv, data_root, atlas='cc400', task='DX', transforms=None, cp="", cnp=""):
        self.csv = csv
        # keep consistent with the nan filling strategy
        csv = csv.fillna(-9999)

        self.filenames = csv['FILE_ID'].values

        self.labels = csv['DX_GROUP'].values
        enc = OneHotEncoder(handle_unknown='ignore')
        self.onehot = enc.fit_transform(self.labels.reshape(-1, 1)).toarray()
        self.onehot = self.onehot.astype(int)
        
        self.suffix = f"_rois_{atlas}.1D"
        self.data_root = data_root
        self.transforms = transforms
        self.n_classes = len(np.unique(self.labels))

        self.category_phenotype_names = cp.split(', ')
        self.continuous_phenotype_names = cnp.split(', ')
        
        self.cp_fea = csv[self.category_phenotype_names].values
        self.cnp_fea = csv[self.continuous_phenotype_names].values

        # TODO: deal with the missing values
        self.cp_fea[self.cp_fea == -9999] = 'unk'
        self.cp_fea[self.cp_fea == '-9999'] = 'unk'
        # special case. The real-world data!
        self.cp_fea[self.cp_fea == '`'] = 'unk'
        self.cp_fea = self._string2index(self.cp_fea)

        self.cnp_fea[self.cnp_fea == -9999] = -1
        self.cnp_fea[self.cnp_fea == '-9999'] = -1

        self.num_cp = len(self.category_phenotype_names) + 1 # add label information
        self.num_cnp = len(self.continuous_phenotype_names)


    @staticmethod
    def _string2index(data):
        transformed_data = np.empty(data.shape, dtype=int)
        for col in range(data.shape[1]):
            unique_values = {value: idx for idx, value in enumerate(set(data[:, col]))}
            # Map 'unk' to -1
            unique_values['unk'] = -1
        
            for row in range(data.shape[0]):
                transformed_data[row, col] = unique_values.get(data[row, col], -1)
    
        return transformed_data

    
    def __len__(self):
        return len(self.labels)
        
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        onehot = self.onehot[idx]
        file_id = self.filenames[idx]
        cnp_label = self.cnp_fea[idx]
        cp_label = self.cp_fea[idx]
        file_path = os.path.join(self.data_root, self.filenames[idx] + self.suffix)
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
        cnp_label = torch.from_numpy(cnp_label).float()
        cp_label = torch.from_numpy(cp_label).float()

        return {'timeseries': timeseries, 'corr': corr, 'label': label, 'onehot': onehot,
                'cnp_label': cnp_label, 'cp_label': cp_label}
