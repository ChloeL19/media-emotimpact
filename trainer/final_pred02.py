#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:08:33 2018

@author: chloeloughridge
"""
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


# load the model of interest
model_file = file_io.FileIO('gs://mediaeval_data_storage/models01/modelVGG', mode='rb')

temp_model_location = './temp_model'
temp_model_file = open(temp_model_location, 'wb')
temp_model_file.write(model_file.read())
temp_model_file.close()
model_file.close()

run_model = keras.models.load_model(temp_model_location, custom_objects={'f1': f1})

# load the feature data
#X_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet02/VGG16_data02_2.npy'))
X_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/TestSet/VGG16_test.npy'))

X_input_long = np.load(X_data)

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

X_predict = X_reshape[:, :timesteps, :]


# predict on the feature data
out = run_model.predict(X_predict)

# round the predictions
round_out = np.round(out)

# reshape the predictions
round_out = np.reshape(round_out, (X_input_long.shape[0], 6262))

stored_out = np.zeros(round_out.shape)
    
# find the starting and ending times of all the consecutive chunks
# go through round_out
mov_num = 0
for mov in round_out:
    print("mov: {}".format(mov_num))
    count = 0
    count2 = 0
    for num in mov:
        # if the value of the current index minus prev index OR value of next index minus this index value are 
        # not equal to one
        if (num == 1) and ((count+1) != 6262) and ((num - mov[count-1] != 0) or (mov[count+1] - num != 0)):
            #print the number
            stored_out[mov_num, count2] = count
            #print(count)
            count2 += 1
        count +=1
    mov_num += 1
    
np.save(file_io.FileIO('gs://mediaeval_data_storage/prediction_run4', 'w'), stored_out)