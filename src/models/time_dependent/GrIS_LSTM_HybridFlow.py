import pandas as pd
import numpy as np

original_data = pd.read_csv(r'C:/Users/Peter/Downloads/sample_1.csv')

data = pd.get_dummies(original_data, columns=['ssp'])

data = data.drop(columns=['year', 'ice_source', 'region', 'sample', 'collapse'])

def split_sequence(sequence, look_back, forecast_horizon):
    X, y = list(), list()
    for i in range(len(sequence)):
        lag_end = i + look_back
        forecast_end = lag_end + forecast_horizon
        if forecast_end > len(sequence):
            break
        seq_x, seq_y = sequence[i:lag_end], sequence[lag_end:forecast_end]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


data.index = data.year
data = data.drop(columns=['year', 'ice_source', 'region', 'sample', 'collapse'])

target_column = "SLE"
features = list(data.columns.difference([target_column]))

forecast_lead = 1  # years
target = f"{target_column}_+{forecast_lead}years"
data[target] = data[target_column].shift(-forecast_lead)
df = data.iloc[:-forecast_lead]

stop = ''
