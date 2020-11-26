import talib as ta
import pandas as pd

def TechnicalIndicators(df, k):
    adj = df.iloc[:,0]
    opening = df.iloc[:,1]
    high = df.iloc[:,2]
    low = df.iloc[:,3]
    close = df.iloc[:,4]
    
    indicators = pd.DataFrame()
    indicators['SMA'] = ta.SMA(adj,k)  #simple moving average
    indicators['EMA'] = ta.EMA(adj, k)     #EMA
    indicators['RSI'] = ta.RSI(adj, 14)     #RSI 14days
    indicators['ADX'] = ta.ADX(high, low, close, k)     #Average Directional Movement Index timeperiod 20  #strong trend if >25
    indicators['up_band'], indicators['mid_band'], indicators['low_band']=ta.BBANDS(adj, timeperiod =k)         #Bollinger Bands
    indicators['macd'], indicators['macdsignal'], indicators['macdhist'] = ta.MACD(adj, fastperiod=12, slowperiod=26, signalperiod=9)     #MACD - Moving Average Convergence/Divergence
    
    return indicators
