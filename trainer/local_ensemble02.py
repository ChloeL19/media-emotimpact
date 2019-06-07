#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 20:56:58 2018

training a boosting ensemble with LSTMs

@author: chloeloughridge
"""

import tensorflow as tf
import numpy as np
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Conv1D, Input
from keras.layers import LSTM
from keras.models import Model
import keras
from sklearn.ensemble import *
from tensorflow.python.keras._impl.keras.wrappers.scikit_learn import *

# loading the data

X_train = np.load('../VGG16_feature_data.npy')[:5, :, :]
X_test = np.load('../VGG16_feature_data.npy')[5:, :, :]

Y_train = np.load('../labels.npy')[:5, :]
Y_test = X_test = np.load('../labels.npy')[5:, :]

timesteps = X_train.shape[1]
data_dim = X_train.shape[2]


print("I've loaded the data!")

# a new metric of evaluation! the F-score!
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def build_LSTM(timesteps, data_dim):
    # constructing a many-to-one LSTM model in keras 
    #model = Sequential()
    
    inputs = Input(shape=(timesteps, data_dim))
    
    # normalization
    X = BatchNormalization()(inputs)
    
    X = Conv1D(15, input_shape=(timesteps, data_dim), kernel_size=3, activation="sigmoid")(X)
    
    X = LSTM(timesteps, input_shape=(timesteps, data_dim), return_sequences=True)(X)
    #input_shape=(timesteps, data_dim)
    
    X = Flatten()(X)
    # add the final dense layer and then softmax
    X = Dense(timesteps, activation='softmax')(X)
    
    return X

indices = np.zeros([15, ])
for baby_class in range(15):
    

print("starting training now!")


#checkpoint = LambdaCallback(on_epoch_end=lambda epoch, logs: save_modelToCloud(epoch, 'gs://mediaeval_data_storage/models01'))

model = Model(inputs=inputs, outputs=X)

# compiling the model
model.compile(loss='binary_crossentropy',
          optimizer='Adam',
          metrics=['binary_accuracy', f1])


# train the ensemble
boosted_LSTM.fit(X_train, Y_train)

# save the model to the cloud storage bucket
filename = 'VGG_ensemble'
path = 'gs://mediaeval_data_storage/models01'
#model.save(filename)
#with file_io.FileIO(filename, mode='r') as inputFile:
        #with file_io.FileIO(path + '/' + filename, mode='w+') as outFile:
           # outFile.write(inputFile.read())


print("finished training!")

out = boosted_LSTM.predict(X_test)

movie_index = 247

print("the feature data:")
print(X_test[movie_index])

print("target:")
print(Y_test[movie_index])
print(Y_test[movie_index].shape)

print("rounded")
print(np.round(out)[movie_index])
print(np.round(out)[movie_index].shape)