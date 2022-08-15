import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random


class GrIS:
    def __init__(self, local_path=None):
        self.X = None
        self.y = None
        self.data = None
        self.local_path = local_path

    def load(self):
        if not os.path.isfile(self.local_path) or self.local_path is None:
            SSPs = [119, 126, 245, 370, 585, ]  # took out NDC
            data = pd.DataFrame()
            for SSP in SSPs:
                dir = f'https://raw.githubusercontent.com/tamsinedwards/emulandice/master/results/proj_MAIN_TIMESERIES/projections_FAIR_SSP{SSP}.csv'
                temp = pd.read_csv(dir)
                temp['ssp'] = SSP
                data = pd.concat([data, temp])
            data.to_csv(r'C:/Users/Peter/Downloads/climate_time_data.csv', index=False)
        else:
            data = pd.read_csv(self.local_path)
        self.data = data
        return self

    def format(self, lag=None, filters=None, drop_columns=None):
        self.data = self.data.sort_values(by=['ssp', 'sample', 'year'])
        if filters:
            for key in filters:
                self.data = self.data[self.data[key] == filters[key]]
                drop_columns.extend([key])
        if drop_columns:
            self.data = self.data.drop(columns=drop_columns)
        if lag:
            for shift in range(1, lag + 1):
                self.data[f"SLE.lag{shift}"] = self.data.SLE.shift(shift)
            self.data = self.data[self.data.year >= 2016 + lag]

        self.X = torch.tensor(np.array(self.data.drop(columns=['SLE'])), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.data.SLE), dtype=torch.float32)
        return self

    def split(self, type='random', divide_year=None, num_samples=5):

        if type == 'random':
            random_samples = random.sample(range(1, max(self.data['sample'])), num_samples)
            train = self.data[~self.data['sample'].isin(random_samples)]
            test = self.data[self.data['sample'].isin(random_samples)]
        elif type == 'temporal':
            assert divide_year is not None, "divide_year cannot be None"
            train = self.data[self.data.year <= divide_year]
            test = self.data[self.data.year > divide_year]


        return train, test


class GrISDataset(Dataset):
    def __init__(self, X, y, sequence_length=5):
        super().__init__()
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]
