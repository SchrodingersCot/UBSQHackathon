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

import gc
import pandas as pd

def LoadModel(file_path = '/home/lyadr/tau/HistBoosting.pickle'):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model
    
def LoadTensorflowModel(file_path = "/home/lyadr/tau/rnn_prediction_submission.h5"):
    return tf.keras.models.load_model(file_path)

def PredictionFunction(model, DF):
    return model.predict(DF)
    
