#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 13:58:20 2018

@author: chloeloughridge
"""
# libraries and functions we need
from tensorflow.python.lib.io import file_io
import keras
import numpy as np
from StringIO import StringIO
import keras.backend as K
from keras.models import Model
from keras.layers import Average, Input


# load all of the models --> let's load all of the VGG models
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


# load the ensemble model
model_file = file_io.FileIO('gs://mediaeval_data_storage/ensembles/ensemble_VGG', mode='rb')

temp_model_location = './temp_model'
temp_model_file = open(temp_model_location, 'wb')
temp_model_file.write(model_file.read())
temp_model_file.close()
model_file.close()

ensemble_VGG_trained = keras.models.load_model(temp_model_location, custom_objects={'f1': f1})


# load the training/validation data
# Create a variable initialized to the value of a serialized numpy array
X_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet02/VGG16_data02_2.npy'))
#X_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet01/audio_data01.npy'))

X_input_long = np.load(X_data)

Y_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet02/labels02_2.npy'))

y_data_input_long = np.load(Y_data)

#reshape the feature and labels data so that it contains more 

# first reshape the labels
y_reshape = np.reshape(y_data_input_long, [62*X_input_long.shape[0], 101])

# then reshape the feature data
X_reshape = np.zeros([62*X_input_long.shape[0], 101, X_input_long.shape[2]])
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
X_reshaped = X_reshape[:(6*62),:timesteps, :]
Y_reshaped = y_reshape[:(6*62), :timesteps]

X_test = X_reshape[(6*62):, :timesteps, :]
Y_test = y_reshape[(6*62):, :timesteps]
 
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

# train the model
ensemble_VGG_trained.fit([X_train, X_train, X_train, X_train], Y_train, epochs = 50, batch_size = 1, validation_data=([X_test, X_test, X_test, X_test], Y_test))

# save the model with _trained appended to name
filename = 'ensemble_VGG_trained'
path = 'gs://mediaeval_data_storage/ensembles'
model.save(filename)
with file_io.FileIO(filename, mode='r') as inputFile:
        with file_io.FileIO(path + '/' + filename, mode='w+') as outFile:
            outFile.write(inputFile.read())