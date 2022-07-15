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


def get_FAIR_data(dir=f'../data/', ice_source='gris', ssp='all', region=None):
    check_input(ice_source, ['gris', 'glaciers', 'ais'])
    check_input(ssp, ['all', 'ssp370', 'ssp245', 'ssp126', 'ssp119', 'ssp585', 'sspndc'])

    data = load_all_data(dir=dir)
    data = data[data['ice_source'] == ice_source]
    if ssp != 'all':
        data = data[data.ssp == ssp]
    if ice_source == "glaciers":
        if region is not None:
            data = data[data.region == 'region_' + str(region)]

        x = np.array(data.GSAT)
        y = np.array(data.SLE)

    if ice_source == "gris":
        x = np.zeros((len(data), 2))
        x[:, 0] = np.array(data.GSAT)
        x[:, 1] = np.array(data['melt'])
        y = np.array(data.SLE)

    if ice_source == "ais":
        x = np.zeros((len(data), 3))
        x[:, 0] = np.array(data.GSAT)
        x[:, 1] = np.array(data['melt'])
        x[:, 2] = np.array(data['collapse'])
        y = np.array(data.SLE)

    return x, y, data