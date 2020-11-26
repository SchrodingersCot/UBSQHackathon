import alphien
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
import os
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

import gc
import pandas as pd

def compile_feature_extractor(X_train_cnn):
    model = Sequential()

    model.add(Conv2D(8, (2,5), padding="same", activation="relu", input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2], X_train_cnn.shape[3])))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (2,5), padding="same", activation="relu", ))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (2,5), padding="same", activation="relu", ))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(128, activation="relu", ))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(50, name ="intermediate_layer"))
    model.add(BatchNormalization())

    model.add(Dense(1, activation="tanh"))
    
    model.compile(loss="mse", metrics=["mae", "mse"], optimizer="adam")
    
    
    return model 
    



def Reshape_TDD(DF):
    Undl0 = ['Undl0volatm_6M', 'Undl0skew95.105_6M', 'Undl0skew70.130_6M',
       'Undl1skew70.130_6M' ] 
    Undl1 = ['Undl0svc', 'Undl1svc', 'Undl2svc', 'Undl0mr']
    Undl2 = ['ContractFeature_AutocallBarrierLevelLevelInitial',
       'ContractFeature_ExpiryPaymentBarrierLevel',
       'ContractFeature_ExpiryPaymentKickInPaymentGearing_ENCODED',
       'ContractFeature_ScheduleEndDate_ENCODED']
       
    stock_attr = []

    for index, row in DF.iterrows():
        Undl0_Var = row[Undl0]
        Undl1_Var  = row[Undl1]
        Undl2_Var  = row[Undl2]
        vstack_variables = np.vstack((Undl0_Var, Undl1_Var, Undl2_Var))
        vstack_variables = vstack_variables.tolist()
        stock_attr.append(vstack_variables)
    stock_attr_array = np.array(stock_attr)
    
    stock_attr_array_2 = stock_attr_array.reshape(stock_attr_array.shape[0], stock_attr_array.shape[1], stock_attr_array.shape[2], -1)
    return stock_attr_array_2


def FeatureEngineering(data, train_fe_model = False, model_file =  "/home/lyadr/tau/fe_model14features.h5"):

    data.columns=data.columns.str.replace(",","")
    data = data.reset_index(drop=True)
    data["ContractFeature_ScheduleEndDate_ENCODED"] = pd.to_numeric(data["ContractFeature_ScheduleEndDate_ENCODED"], errors='ignore')
    
    data = data[['Undl0volatm_6M', 'Undl0skew95.105_6M', 'Undl0skew70.130_6M',
       'Undl1skew70.130_6M', 'Undl0svc', 'Undl1svc', 'Undl2svc', 'Undl0mr',
       'ContractFeature_AutocallBarrierLevelLevelInitial',
       'ContractFeature_ExpiryPaymentBarrierLevel',
       'ContractFeature_ExpiryPaymentKickInPaymentGearing_ENCODED',
       'ContractFeature_ScheduleEndDate_ENCODED',
       'ContractFeature_SchedulePeriodFrequency_ENCODED', 'val_lvsvcharge']]
       
    
    target = np.array(data["val_lvsvcharge"])
    predictors = data.drop("val_lvsvcharge", axis = 1)
    gc.collect()
    X_train_cnn = Reshape_TDD(predictors)
    gc.collect()
    if train_fe_model == False:
        fe_model = tf.keras.models.load_model(model_file)
    else:
        fe_model = compile_feature_extractor(X_train_cnn)
        fe_model.fit(X_train_cnn, target, validation_split=0.1, epochs = 20, batch_size = 2000)
    intermediate_layer_model = tf.keras.models.Model(inputs=fe_model.input, outputs=fe_model.get_layer('intermediate_layer').output)
    gc.collect()
    cnn_train_features = intermediate_layer_model.predict(X_train_cnn)
    gc.collect()
    cnn_train_features = pd.DataFrame(cnn_train_features)
    gc.collect()
    X_features = pd.concat([predictors.reset_index(drop=True), cnn_train_features.reset_index(drop=True)], axis = 1, ignore_index=True)

    if train_fe_model == True:
        return X_features, target, fe_model
    else:
        return X_features, target


def RNNFeatureEngineering(data, train_fe_model = False, model_file = "/home/lyadr/tau/fe_model14features.h5"):

    data.columns=data.columns.str.replace(",","")
    data = data.reset_index(drop=True)
    data["ContractFeature_ScheduleEndDate_ENCODED"] = pd.to_numeric(data["ContractFeature_ScheduleEndDate_ENCODED"], errors='ignore')
    
    data = data[['Undl0volatm_6M', 'Undl0skew95.105_6M', 'Undl0skew70.130_6M',
       'Undl1skew70.130_6M', 'Undl0svc', 'Undl1svc', 'Undl2svc', 'Undl0mr',
       'ContractFeature_AutocallBarrierLevelLevelInitial',
       'ContractFeature_ExpiryPaymentBarrierLevel',
       'ContractFeature_ExpiryPaymentKickInPaymentGearing_ENCODED',
       'ContractFeature_ScheduleEndDate_ENCODED',
       'ContractFeature_SchedulePeriodFrequency_ENCODED', 'val_lvsvcharge']]
       
    target = np.array(data["val_lvsvcharge"])
    predictors = data.drop("val_lvsvcharge", axis = 1)
    gc.collect()
    X_train_cnn = Reshape_TDD(predictors)
    gc.collect()
    if train_fe_model == False:
        fe_model = tf.keras.models.load_model(model_file)
    else:
        fe_model = compile_feature_extractor(X_train_cnn)
        fe_model.fit(X_train_cnn, target, validation_split=0.1, epochs = 20, batch_size = 2000)
    intermediate_layer_model = tf.keras.models.Model(inputs=fe_model.input, outputs=fe_model.get_layer('intermediate_layer').output)
    gc.collect()
    cnn_train_features = intermediate_layer_model.predict(X_train_cnn)
    gc.collect()
    cnn_train_features = pd.DataFrame(cnn_train_features)
    gc.collect()
    X_features = pd.concat([predictors.reset_index(drop=True), cnn_train_features.reset_index(drop=True)], axis = 1, ignore_index=True)
    scaler = MinMaxScaler()
    gc.collect()
    X_features = scaler.fit_transform(X_features)
    gc.collect()
    X_features = X_features.reshape((X_features.shape[0], X_features.shape[1], 1))    
    
    if train_fe_model == True:
        return X_features, target, intermediate_layer_model
    else:
        return X_features, target
