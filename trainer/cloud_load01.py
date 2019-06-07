#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:39:02 2018

for loading saved models from google buckets

@author: chloeloughridge
"""

from tensorflow.python.lib.io import file_io
import keras
import numpy as np
from StringIO import StringIO
import keras.backend as K

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

# loading the model
model_file = file_io.FileIO('gs://mediaeval_data_storage/models01/VGG16_norm_lstm', mode='rb')

temp_model_location = './temp_model'
temp_model_file = open(temp_model_location, 'wb')
temp_model_file.write(model_file.read())
temp_model_file.close()
model_file.close()

model = keras.models.load_model(temp_model_location, custom_objects={'FScore2': FScore2})

# loading the data
# loading data in a way that will interface with google cloud storage buckets

# Create a variable initialized to the value of a serialized numpy array
X_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet01/VGG16_data01.npy'))
#X_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet01/audio_data01.npy'))

X_input_long = np.load(X_data)

Y_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet01/labels01.npy'))

y_data_input_long = np.load(Y_data)

#reshape the feature and labels data so that it contains more 

X_test = X_input_long[:13, :212, :]
Y_test = y_data_input_long[:13, :212]

balance=np.mean(Y_test)
print(balance)

# evaluate the model on the test data
X, Y, Fscore = model.evaluate(X_test, Y_test)
print(X, Y, Fscore)


