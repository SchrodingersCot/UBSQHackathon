import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import pandas as pd

def feature_engineering_carry(bb, carry12):
    targets = []
    for col in bb.columns:
        bb[f"{col}_Targets"] = np.log(bb[col][0]) - np.log(bb[col].shift(3))   #future return T+2
        targets.append(f"{col}_Targets")
    returns = bb[targets]

    returns_mean = returns.shift(3).rolling(5).mean().add_suffix("_mean")
    returns_max = returns.shift(3).rolling(5).max().add_suffix("_max")
    returns_min = returns.shift(3).rolling(5).min().add_suffix("_min")
    returns_std = returns.shift(3).rolling(5).std().add_suffix("_std")
    returns_kurt = returns.shift(3).rolling(5).kurt().add_suffix("_kurt")
    returns_median = returns.shift(3).rolling(5).median().add_suffix("_median")
    returns_lag1  = returns.shift(3).add_suffix("_lagg1")
    returns_lag2  = returns.shift(4).add_suffix("_lagg2")
    returns_lag3  = returns.shift(5).add_suffix("_lagg3")
    returns_lag4  = returns.shift(6).add_suffix("_lagg4")
    returns_lag5  = returns.shift(7).add_suffix("_lagg5")
    returns_lag6  = returns.shift(8).add_suffix("_lagg6")
    returns_lag7  = returns.shift(9).add_suffix("_lagg7")

    returns["market_std"] = returns.shift(1).std(axis = 1)
    returns["market_mean"] = returns.shift(1).mean(axis = 1)
    returns["market_max"] = returns.shift(1).max(axis = 1)
    returns["market_min"] = returns.shift(1).min(axis = 1)
    returns["market_median"] = returns.shift(1).median(axis = 1)
    returns["market_kurt"] = returns.shift(1).kurt(axis = 1)

    return_features = [returns, 
                       returns_mean, 
                       returns_max, 
                       returns_min, 
                       returns_std, 
                       returns_kurt, 
                       returns_median,
                       returns_lag1,
                       returns_lag2,
                       returns_lag3,
                       returns_lag4,
                       returns_lag5,
                       returns_lag6,
                       returns_lag7]

    return_concat = pd.concat(return_features, axis=1)
    feature_list = (
        ["date", "market_std", "market_mean", "market_max", "market_min", "market_median", "market_kurt"]
        + list(returns_median.columns)  
        + list(returns_mean.columns) 
        + list(returns_max.columns)
        + list(returns_min.columns)
        + list(returns_std.columns)
        + list(returns_kurt.columns)
        + list(returns_lag1.columns)
        + list(returns_lag2.columns)
        + list(returns_lag3.columns)
        + list(returns_lag4.columns)
        + list(returns_lag5.columns)
        + list(returns_lag6.columns)
        + list(returns_lag7.columns)
    )
    returns_indexed = return_concat.reset_index()


    returns_data = pd.melt(returns_indexed, 
                           id_vars=feature_list, 
                           var_name="currency", 
                           value_name="return")
    returns_data["binary_target"] = np.where(returns_data["return"]>0, 1, 0)
    returns_data["currency"] = returns_data["currency"].str.replace("_Targets", "")
    returns_data = returns_data.fillna(0)
    returns_data = returns_data.drop("return", axis= 1)


    carry12_indexed = carry12.reset_index()

    carry12_indexed = carry12_indexed.fillna(0)

    carry12_melted = pd.melt(carry12_indexed, id_vars=["date"], var_name="currency", value_name="carry12")

    carry12_7mean = pd.melt(carry12_indexed.rolling(5).mean().fillna(0), var_name="variable", value_name="value_7mean").drop("variable", 1)
    carry12_7max = pd.melt(carry12_indexed.rolling(5).max().fillna(0), var_name="variable", value_name="value_7max").drop("variable", 1)
    carry12_7min = pd.melt(carry12_indexed.rolling(5).min().fillna(0), var_name="variable", value_name="value_min").drop("variable", 1)
    carry12_7median = pd.melt(carry12_indexed.rolling(5).median().fillna(0), var_name="variable", value_name="value_7median").drop("variable", 1)
    carry12_7std = pd.melt(carry12_indexed.rolling(5).std().fillna(0), var_name="variable", value_name="value_std").drop("variable", 1)
    carry12_7kurt = pd.melt(carry12_indexed.rolling(5).kurt().fillna(0), var_name="variable", value_name="value_7kurt").drop("variable", 1)

    carry12_25mean = pd.melt(carry12_indexed.rolling(10).mean().fillna(0), var_name="variable", value_name="value_25mean").drop("variable", 1)
    carry12_25max = pd.melt(carry12_indexed.rolling(10).max().fillna(0), var_name="variable", value_name="value_25max").drop("variable", 1)
    carry12_25min = pd.melt(carry12_indexed.rolling(10).min().fillna(0), var_name="variable", value_name="value_25min").drop("variable", 1)
    carry12_25median = pd.melt(carry12_indexed.rolling(10).median().fillna(0), var_name="variable", value_name="value_25median").drop("variable", 1)
    carry12_25std = pd.melt(carry12_indexed.rolling(10).std().fillna(0), var_name="variable", value_name="value_25std").drop("variable", 1)
    carry12_25kurt = pd.melt(carry12_indexed.rolling(10).kurt().fillna(0), var_name="variable", value_name="value_25kurt").drop("variable", 1)

    list_carry_12_ = [
        carry12_melted,
        carry12_7mean,
        carry12_7max,
        carry12_7min, 
        carry12_7median, 
        carry12_7std, 
        carry12_7kurt, 
        carry12_25mean, 
        carry12_25max, 
        carry12_25min, 
        carry12_25median, 
        carry12_25std, 
        carry12_25kurt
    ]

    carry_12_features = pd.concat(list_carry_12_, axis=1)


    modelling_set = pd.merge(returns_data, carry_12_features,  how='left', left_on=["date", "currency"], right_on = ["date", "currency"])
    
    return modelling_set


def payout(features):

    #Prepare data
    bb = features.subset(fields="bb_live").pxs
    carry12 = features.subset(fields="carry12").pxs
    modelling_set = feature_engineering_carry(bb.copy(), carry12.copy())
    
    #Adjustment
    mom = bb / bb.shift(60).values - 1
    logReturns = np.log(bb) - np.log(bb.shift(25))
    vol = logReturns.rolling(60).std() * np.sqrt(60)
    signWithNa = lambda x: np.nan if np.isnan(x) else np.sign(x)
    momSign = mom.applymap(signWithNa)
    carrySign = carry12.applymap(signWithNa)
    
    signal = (momSign + carrySign.values)/(2*vol.values)
    sumOfSignals = np.abs(signal).apply(lambda x: x.sum() if x.isna().sum()==0.0 else np.nan, axis=1)[:,None] #explicitly remove days with some NaN momentums.
    adjustment = (signal / sumOfSignals).fillna(method='pad')
    
    #Define training loop
    def trainingloop(modelling_set, year_value):
        training_set = modelling_set[pd.DatetimeIndex(modelling_set["date"]).year == year_value]
        test_set = modelling_set[pd.DatetimeIndex(modelling_set["date"]).year == year_value + 1]
        modelling_set_fx = training_set.drop(["date", "currency"], axis = 1).fillna(0)
        returns_target = modelling_set_fx["binary_target"]
        features = modelling_set_fx.drop("binary_target", axis = 1)
        test_set_fx = test_set.drop(["date", "currency"], axis = 1).fillna(0)
        returns_target_test = test_set_fx["binary_target"]
        features_test = test_set_fx.drop("binary_target", axis = 1)
        return features, returns_target, features_test, returns_target_test

    firstyear = pd.DatetimeIndex(modelling_set["date"]).year[0]
    lastyear = pd.DatetimeIndex(modelling_set["date"]).year[-1]
    
    trial = modelling_set[pd.DatetimeIndex(modelling_set.date).year > firstyear]
    trial = trial.drop(["date", "currency", "binary_target"], axis = 1)
    
    predictions = list()
    
    firstyear = pd.DatetimeIndex(modelling_set["date"]).year[0]
    lastyear = pd.DatetimeIndex(modelling_set["date"]).year[-1]
    
    for year_value in range(firstyear, lastyear):
        features, returns_target, features_test, returns_target_test = trainingloop(modelling_set, year_value)
        rf_clf = RandomForestClassifier(n_estimators=1000)
        RandomForest_Model = rf_clf.fit(features, returns_target)
        year_predictions = RandomForest_Model.predict(features_test).tolist()
        predictions.extend(year_predictions)
    
    trial['predictions'] = predictions

    trial['date'] = modelling_set.date
    trial['currency'] = modelling_set.currency
    trial = trial[['date','currency','predictions']].sort_values(by = 'date')
    
    scaled_predicted_target = list()
    for x in trial.predictions:
        if x == 0:
            scaled_predicted_target.append(-1)
        else:
            scaled_predicted_target.append(1)
            
    trial['scaled_predicted_target'] = scaled_predicted_target
    trial = trial[['date','currency','scaled_predicted_target']]
    
    trial = trial.set_index(['date','currency']).unstack()
    trial.columns = trial.columns.droplevel()
    

    weights = (trial.shift(0))
    X = weights.diff(1).abs().sum(axis=1).apply(lambda x: 1 if x < 0.05 else 1)
    weights = weights.div(weights.abs().sum(axis=1), axis=0)
    weights = weights.div(X, axis=0)
    
    #Adjust exposure per pair
    weights = (3 * adjustment + weights)/2
    weights = weights.applymap(lambda x: 0.3 if x > 0.3 else x)
    weights = weights.applymap(lambda x: -0.3 if x < -0.3 else x)
    
    #Adjust overall portfolio L/S exposure
    exposure_adj = weights.sum(axis=1).abs().apply(lambda x: 1 if x < 0.9 else (x/0.9))
    weights = weights.div(exposure_adj, axis=0)
    
    #Adjust total leverage
    leverage_adj = weights.abs().sum(axis=1).apply(lambda x: 1 if x > 2 else (x/2))
    weights = weights.div(leverage_adj, axis=0)
    
    return weights.dropna() * 0.8 * 0.89
