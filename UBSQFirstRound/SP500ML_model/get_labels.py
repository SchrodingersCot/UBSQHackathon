import pandas as pd
import numpy as np
from alphien.utils import get_all_data, _read_output


def add_features(data):
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data['adjClose'] = pd.to_numeric(data['adjClose'], errors='coerce')
    data['Forward Return Daily'] = data['adjClose'].shift(-1)/data['adjClose']
    data['Forward Return Quarterly'] = data['adjClose'].shift(-63)/data['adjClose']
    data = data.dropna()
    return data


datalist = list(_read_output().values())
datalist = [data for data in datalist if data.shape[0] > 1]
datalist = [add_features(data) for data in datalist]

df = pd.concat(datalist)
df = df.reset_index()
df = df.sort_values(by=['date', 'ticker'])
df = df.set_index(['date', 'ticker'])


"""Get labels"""
labels = df.groupby(['date'])['Forward Return Quarterly'].nlargest(60)
labels = labels.droplevel(0).reset_index()
labels = labels.drop('Forward Return Quarterly', axis=1)
labels.to_csv('labels60.csv')
