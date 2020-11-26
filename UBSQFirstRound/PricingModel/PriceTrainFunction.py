import alphien
import numpy as np
from tensorflow.keras.layers import Conv2D, LSTM, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Activation, Bidirectional
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
import os
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

import gc
import pandas as pd


def TrainModel(X_train, y_train):
    Hist_GBM = HistGradientBoostingRegressor(max_iter=500, learning_rate= 0.01, validation_fraction=0.1, verbose=1)
    HistBoosting = Hist_GBM.fit(X_train, y_train)
    return HistBoosting


def TrainStackingModel(X_train, y_train):
    estimators = [
        ('lr', RidgeCV()), 
        ('svr', LinearSVR(random_state=42)),
        ('hist_model', HistGradientBoostingRegressor(max_iter=500, learning_rate= 0.01, validation_fraction=0.1))
    ]
    
    StackingReg = StackingRegressor(
        estimators=estimators,
        final_estimator=RandomForestRegressor()
    )
    
    StackingRegressionModel = StackingReg.fit(X_train, y_train)
    return StackingRegressionModel
    
    

def TrainTensorflowModel(features, labels):
    model = Sequential()

    model.add(LSTM(32, input_shape=(features.shape[1], features.shape[2]), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(LSTM(16))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mae", metrics=["mae", "mse"])

    gc.collect()

    lstm_history = model.fit(features, 
                             labels, 
                             validation_split = 0.1, 
                             batch_size =4000, 
                             epochs = 100,
                             verbose = 1)
    
    return model
    
    
