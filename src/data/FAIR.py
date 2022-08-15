import os
import numpy as np
import pandas as pd
from src.utils import check_input

def load_all_data(dir='./data/', write=False):
    files = os.listdir(dir)

    df = pd.DataFrame()
    for file in files:
        if "projections_FAIR_" in file:
            temp = pd.read_csv(dir + '/' + file)
            temp['ssp'] = file[-10:-4]
            df = pd.concat([df, temp])
    df['ice_source'] = df['ice_source'].apply(lambda x: x.lower())
    if write:
        df.to_csv(dir + '/' + 'all_ssp.csv')
    return df


def get_FAIR_data(dir=f'../data/', ice_source='gris_data', ssp='all', region=None, ssp_as_feature=False):
    check_input(ice_source, ['gris_data', 'glaciers', 'ais'])
    check_input(ssp, ['all', 'ssp370', 'ssp245', 'ssp126', 'ssp119', 'ssp585', 'sspndc'])

    data = load_all_data(dir=dir)
    data = data[data['ice_source'] == ice_source]

    # Determine what to do with SSP variable (subset, use as feature, etc.)
    ssp_columns = np.zeros(len(data))
    if ssp != 'all':
        if ssp_as_feature is False:
            data = data[data.ssp == ssp.upper()]
        else:
            raise AttributeError('ssp parameter must be false to use ssp as a feature')

    if ssp == 'all':
        if ssp_as_feature is True:
            ssp_columns = np.array(pd.get_dummies(data.ssp))
        else:
            pass

    if ice_source == "glaciers":
        if region is not None:
            ssp_columns = ssp_columns[data.region == 'region_' + str(region)]
            data = data[data.region == 'region_' + str(region)]

        x = np.array(data.GSAT)
        y = np.array(data.SLE)

    if ice_source == "gris_data":
        x = np.zeros((len(data), 2))
        x[:, 0] = np.array(data.GSAT)
        x[:, 1] = np.array(data['melt'])
        y = np.array(data.SLE)

    if ice_source == "ais":
        if region is not None and region != 'all':
            # ssp_columns = ssp_columns[data.region == str(region), :]
            data = data[data.region == str(region)]

        x = np.zeros((len(data), 3))
        x[:, 0] = np.array(data.GSAT)
        x[:, 1] = np.array(data['melt'])
        x[:, 2] = np.array(data['collapse'])
        y = np.array(data.SLE)

    if ssp_as_feature:
        x = np.append(x, ssp_columns, 1)

    return x, y, data