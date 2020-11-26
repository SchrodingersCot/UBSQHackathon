def VolatilityTimedMomentum(features, select=50, x=1):
    #Slice
    adj = features.subset(fields='bb_live', asDataFeatures=True)
    # O = features.subset(fields=['open_price'], asDataFeatures=True)
    # H = features.subset(fields=['high_price'], asDataFeatures=True)
    # L = features.subset(fields=['low_price'], asDataFeatures=True)
    # C = features.subset(fields=['close_price'], asDataFeatures=True)
    # V = features.subset(fields=['volume'], asDataFeatures=True)

    #Add extras
    inclusionMatrix = getTickersSP500(ticker=features.tickers, asMatrix=True)
    inclusionMatrix = inclusionMatrix.loc[:,~inclusionMatrix.columns.duplicated()]
    inclusion_stacked = inclusionMatrix.stack()
    inclusion_stacked = inclusion_stacked[inclusion_stacked > 0]
    inclusion_stacked.name = 'inclusion'

    # idx_price = getTickersSP500Data(ticker = ["SPX Index", 'SPTR Index', 'SPW Index', 'VIX Index'], asPrice = True)

    momentum = adj.pxs.pct_change(252)
    momentum.columns = features.tickers

    momentum_stacked = momentum.stack()
    momentum_stacked.name = 'momentum'
    momentum_stacked.index.names = inclusion_stacked.index.names

    momentum_incl = pd.merge(inclusion_stacked, momentum_stacked, left_index = True, right_index=True, how='left')
    momentum_incl = momentum_incl[~momentum_incl['momentum'].isnull()]
    momentum_incl = momentum_incl.drop('inclusion', axis = 1)

    returns = adj.pxs.pct_change().copy()
    returns.columns = features.tickers
    returns_stacked = returns.stack()
    returns_stacked.name = '1D Forward Returns'
    returns_stacked.index.names = momentum_incl.index.names

    momentum_returns = pd.merge(momentum_incl, returns_stacked, left_index = True, right_index=True, how='left')
    top50 = momentum_returns.groupby(level=0)['momentum'].nlargest(50)
    top50 = top50.droplevel(1)
    top50returns = momentum_returns.loc[top50.index]
    returns = top50returns.groupby(level=0)['1D Forward Returns'].mean()

    #Trailing realized volatility
    trailing_window = 126
    volatility = np.sqrt((returns**2).rolling(trailing_window).sum()*2)

    volatility_scaled = 0.27/volatility
    volatility_threshold = volatility_scaled.apply(lambda x: np.where(x < 1, -1, 1))
    returns_timed = returns * volatility_threshold
    
    timed_momentum = momentum.mul(volatility_threshold, axis = 0)

    selection = adj.changeFreq('quarterly').copy()
    selection = selection.loc[timed_momentum.index[252+126+1]:,:]
    selection.columns = features.tickers
    selection[:] = 0
    rebalDates = selection.index.tolist()

    for date in rebalDates:
        includedAtDate = (inclusionMatrix.loc[date,:]>0.0).values
        tkrsAtDate = [tk for tk, incl in zip(inclusionMatrix.loc[date,:].index.tolist(), includedAtDate) if incl]

        latest_momentum = timed_momentum.loc[:date, tkrsAtDate].iloc[-1]
        selected = latest_momentum.sort_values(ascending=False).iloc[:50].index.tolist()
        selection.loc[date,selected] =  float(1 / 50)
    
    return selection.round(4)