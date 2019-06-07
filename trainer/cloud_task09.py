#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 20:56:58 2018

training and saving models to google cloud storage buckets
models in this file include convolutional layers

make sure to load the correct data and save the models with the correct name

@author: chloeloughridge
"""

import tensorflow as tf
import numpy as np
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Conv1D
from keras.layers import LSTM

from StringIO import StringIO
from tensorflow.python.lib.io import file_io

# loading data in a way that will interface with google cloud storage buckets

# Create a variable initialized to the value of a serialized numpy array
X_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet01/VGG16_data01.npy'))
#X_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet01/audio_data01.npy'))

X_input_long = np.load(X_data)

Y_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet01/labels01.npy'))

y_data_input_long = np.load(Y_data)

#reshape the feature and labels data so that it contains more 

# first reshape the labels
y_reshape = np.reshape(y_data_input_long, [868, 101])

# then reshape the feature data
X_reshape = np.zeros([868, 101, X_input_long.shape[2]])
mov_count = 0
for mov in X_input_long:
    # we will take 62 101-second worth chunks out of the VGG matrix
    for i in range(62):
        new_mov = mov[i*101:(i+1)*101, :]
        X_reshape[i + (62*mov_count), :, :] = new_mov  
    mov_count +=1
    

# important variables for the LSTM
timesteps = X_reshape.shape[1]
data_dim = X_reshape.shape[2]
    
# sort into training and validation sets first
X_reshaped = X_reshape[:(10*62),:timesteps, :]
Y_reshaped = y_reshape[:(10*62), :timesteps]

X_test = X_reshape[(10*62):, :timesteps, :]
Y_test = y_reshape[(10*62):, :timesteps]
 
# selecting only the matrices with fear-inducing seconds to send in
X_sendin = []
Y_sendin = []
count = 0
for seg in Y_reshaped:
    if seg.any() == 1:
        X_sendin.append(X_reshaped[count])
        Y_sendin.append(seg)
    count += 1
    
X_train = np.asarray(X_sendin)
Y_train = np.asarray(Y_sendin)

balance = np.mean(Y_train)
print(balance)


print("I've loaded the data!")

# constructing a many-to-one LSTM model in keras 
model = Sequential()

# normalization
model.add(BatchNormalization(input_shape=(timesteps, data_dim)))

model.add(Conv1D(15, kernel_size=3, activation="sigmoid"))

model.add(LSTM(timesteps, return_sequences=True))
#input_shape=(timesteps, data_dim)

model.add(Flatten()) 
# add the final dense layer and then softmax
model.add(Dense(timesteps, activation='sigmoid'))
# going to add a softmax activation to this
#model.add(Activation('softmax'))

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

# compiling LSTM model
# note that Ng used an Adam optimizer and categorical cross-entropy loss
# but this is a binary classification problem so I think the parameters below should suffice
model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['binary_accuracy', f1])

print("starting training now!")

#checkpoint = LambdaCallback(on_epoch_end=lambda epoch, logs: save_modelToCloud(epoch, 'gs://mediaeval_data_storage/models01'))

# running the LSTM model
model.fit(X_train, Y_train, epochs = 50, batch_size = 128, validation_data=(X_test, Y_test))

# save the model to the cloud storage bucket
filename = 'modelVGG_CNN'
path = 'gs://mediaeval_data_storage/models01'
#model.save(filename)
#with file_io.FileIO(filename, mode='r') as inputFile:
        #with file_io.FileIO(path + '/' + filename, mode='w+') as outFile:
            #outFile.write(inputFile.read())


print("finished training!")

out = model.predict(X_test)

movie_index = 247

print("the feature data:")
print(X_test[movie_index])

print("target:")
print(Y_test[movie_index])
print(Y_test[movie_index].shape)

print("rounded")
print(np.round(out)[movie_index])
print(np.round(out)[movie_index].shape)