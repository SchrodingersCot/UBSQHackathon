import pandas as pd
import talib as ta


def add_features(data):
    """Add features to raw data

    Parameters
    ----------
    data: pd.DataFrame
        Raw data

    Returns
    -------
    pd.DataFrame
        Results
    
    NOTE: MACD and SMA are adjusted to cum. corp. actions effect
    
    """
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data['tradevol'] = (data['volume']) * ((data['high'] + data['low'])/2)
    data['spread'] = data['high'] - data['low']
    data['returns'] = data['adjClose'].pct_change(1)
    data['vol'] = data['returns'].rolling(126).std()
    


    data['MOM5'] = data['adjClose'].pct_change(5)
    data['MOM22'] = data['adjClose'].pct_change(22)
    data['MOM252'] = data['adjClose'].pct_change(252)

    data['MACD'], data['MACDSignal'], data['MACDHist'] = ta.MACD(data['adjClose'], 12, 26, 9)
    #Scaling
    data['MACD'] = data['MACD']/data['adjClose']
    data['MACDSignal'] = data['MACDHist']/data['adjClose']
    data['MACDSignal'] = data['MACDHist']/data['adjClose']

    data['RSI14'] = ta.RSI(data['adjClose'], 14)
    data['RSI28'] = ta.RSI(data['adjClose'], 28)
    data['RSI56'] = ta.RSI(data['adjClose'], 56)

    data['ADX14'] = ta.ADX(data['adjHigh'], data['adjLow'], data['adjClose'], 14)
    data['ADX56'] = ta.ADX(data['adjHigh'], data['adjLow'], data['adjClose'], 56)

    data['SMA14'] = ta.SMA(data['adjClose'], 14)/data['adjClose']
    data['SMA42'] = ta.SMA(data['adjClose'], 42)/data['adjClose']



    for lag in [3,4,14,56,112,224]:
        col = 'C' + str(lag)
        data[col] = data['close'].shift(lag)




    data = data.drop(['adjHigh', 'adjLow', 'adjOpen', 'adjClose','adjVolume','divCash','splitFactor'], axis = 1)


    # data['Forward Return Daily'] = data['adjClose'].shift(-1)/data['adjClose']
    # data['Forward Return Quarterly'] = data['adjClose'].shift(-63)/data['adjClose']
    data = data.replace(np.inf,np.nan)
    data = data.dropna()
    return data

def rebase(arr, idx = -1):
    return arr/arr[idx]