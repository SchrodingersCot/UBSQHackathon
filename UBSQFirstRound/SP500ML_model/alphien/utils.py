import pandas as pd
import os


def _read_output(output_dir='output'):
    all_tickers = []
    datadict = {}

    for f in os.listdir(output_dir):
        ticker = f.split('.')[0]
        data = pd.read_csv(os.path.join(output_dir, f))
        data = data.rename(columns={'Unnamed: 0': 'ticker'})
        data['ticker'] = ticker

        all_tickers.append(ticker)
        datadict.update({ticker: data})
    return datadict


def _consolidate_data(datadict, save=False):
    master_data = pd.concat(datadict.values(), axis=0)
    master_data['date'] = pd.to_datetime(master_data['date'])
    return master_data


def get_all_data():
    """Read all data in output folder and store it in a dataframe

    """
    datadict = _read_output()
    data = _consolidate_data(datadict, False)
    return data
