#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 20:56:58 2018

@author: chloeloughridge
"""

import tensorflow as tf
import numpy as np
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers import LSTM

from StringIO import StringIO
from tensorflow.python.lib.io import file_io

# loading data in a way that will interface with google cloud storage buckets

# Create a variable initialized to the value of a serialized numpy array
X_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet01/VGG16_data01.npy'))

X_input_long = np.load(X_data)

Y_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet01/labels01.npy'))

y_data_input_long = np.load(Y_data)

#reshape the feature and labels data so that it contains more 

# first reshape the labels
y_data_input = np.reshape(y_data_input_long, [868, 101])

# then reshape the feature data
X_input = np.zeros([868, 101, X_input_long.shape[2]])
mov_count = 0
for mov in X_input_long:
    # we will take 62 101-second worth chunks out of the VGG matrix
    for i in range(62):
        new_mov = mov[i*101:(i+1)*101, :]
        X_input[i + (62*mov_count), :, :] = new_mov  
    mov_count +=1

# important variables for the LSTM
timesteps = X_input.shape[1]
data_dim = X_input.shape[2]

X_train = X_input[:(10*62),:timesteps, :]
Y_train = y_data_input[:(10*62), :timesteps]

X_test = X_input[(10*62):, :timesteps, :]
Y_test = y_data_input[(10*62):, :timesteps]

print("I've loaded the data!")

# constructing a many-to-one LSTM model in keras 
model = Sequential()

# normalization
model.add(BatchNormalization(input_shape=(timesteps, data_dim)))

model.add(LSTM(timesteps, return_sequences=True))
#input_shape=(timesteps, data_dim)

model.add(Flatten()) 
# add the final dense layer and then softmax
model.add(Dense(timesteps, activation='sigmoid'))
# going to add a softmax activation to this
#model.add(Activation('softmax'))

# a new metric of evaluation! the F-score!
def FScore2(y_true, y_pred):
    '''
    The F score, beta=2
    '''
    B2 = K.variable(4)
    OnePlusB2 = K.variable(5)
    pred = K.round(y_pred)
    tp = K.sum(K.cast(K.less(K.abs(pred - K.clip(y_true, .5, 1.)), 0.01), 'float32'), -1)
    fp = K.sum(K.cast(K.greater(pred - y_true, 0.1), 'float32'), -1)
    fn = K.sum(K.cast(K.less(pred - y_true, -0.1), 'float32'), -1)

    f2 = OnePlusB2 * tp / (OnePlusB2 * tp + B2 * fn + fp)

    return K.mean(f2)
print("I've loaded the model!")

# compiling LSTM model
# note that Ng used an Adam optimizer and categorical cross-entropy loss
# but this is a binary classification problem so I think the parameters below should suffice
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['binary_accuracy', FScore2])

print("starting training now!")

# running the LSTM model
model.fit(X_train, Y_train, epochs = 20, batch_size = 128, validation_data=(X_test, Y_test))
print("finished training!")