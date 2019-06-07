#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:13:51 2018

creating ensembles for the official runs

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

# loading the model

#VGG_02_all model
model_file = file_io.FileIO('gs://mediaeval_data_storage/models02/modelVGG02_all', mode='rb')

temp_model_location = './temp_model'
temp_model_file = open(temp_model_location, 'wb')
temp_model_file.write(model_file.read())
temp_model_file.close()
model_file.close()

modelVGG02_all = keras.models.load_model(temp_model_location, custom_objects={'f1': f1})

#ModelVGG 
model_file = file_io.FileIO('gs://mediaeval_data_storage/models01/modelVGG', mode='rb')

temp_model_location = './temp_model'
temp_model_file = open(temp_model_location, 'wb')
temp_model_file.write(model_file.read())
temp_model_file.close()
model_file.close()

modelVGG = keras.models.load_model(temp_model_location, custom_objects={'f1': f1})

#modelVGG02
model_file = file_io.FileIO('gs://mediaeval_data_storage/models02/modelVGG02', mode='rb')

temp_model_location = './temp_model'
temp_model_file = open(temp_model_location, 'wb')
temp_model_file.write(model_file.read())
temp_model_file.close()
model_file.close()

modelVGG02 = keras.models.load_model(temp_model_location, custom_objects={'f1': f1})

#modelVGG_CNN
model_file = file_io.FileIO('gs://mediaeval_data_storage/models01/modelVGG_CNN', mode='rb')

temp_model_location = './temp_model'
temp_model_file = open(temp_model_location, 'wb')
temp_model_file.write(model_file.read())
temp_model_file.close()
model_file.close()

modelVGG_CNN = keras.models.load_model(temp_model_location, custom_objects={'f1': f1})

# load the evaluation data
# Create a variable initialized to the value of a serialized numpy array
X_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet02/VGG16_data02_2.npy'))
#X_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet02/audio_data02_2.npy'))

X_input_long = np.load(X_data)

Y_data = StringIO(file_io.read_file_to_string('gs://mediaeval_data_storage/DevSet02/labels02_2.npy'))

y_data_input_long = np.load(Y_data)
print(y_data_input_long.shape)

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
X_test = X_reshape[(3*62):,:timesteps, :]
Y_test = y_reshape[(3*62):, :timesteps]

print(modelVGG02.summary())

# change all the names of the layers in each model
count = 0
for layer in modelVGG02.layers:
    layer.name='VGG02_'+ str(count)
    count = count + 1
modelVGG02.get_layer(name='batch_normalization_1_input').name='VGG02_norm'

count = 0
for layer in modelVGG.layers:
    layer.name='VGG_'+ str(count)
    count = count + 1
modelVGG.get_layer(name='batch_normalization_1_input').name='VGG_norm'
    
count = 0
for layer in modelVGG02_all.layers:
    layer.name='VGG02_all_'+ str(count)
    count = count + 1
modelVGG02_all.get_layer(name='batch_normalization_1_input').name='VGG02_all_norm'
    
count = 0
for layer in modelVGG_CNN.layers:
    layer.name='VGG_CNN_'+ str(count)
    count = count + 1
modelVGG_CNN.get_layer(name='batch_normalization_1_input').name='VGG_CNN_norm'

# join the models with an average layer
#inputs = Input(shape=(timesteps, data_dim))
X_1 = modelVGG02_all.output
X_2 = modelVGG.output
X_3 = modelVGG02.output
X_4 = modelVGG_CNN.output
out = Average()([X_1, X_2, X_3, X_4])

ensemble_VGG = Model(inputs=[modelVGG02_all.input, modelVGG.input, modelVGG02.input, modelVGG_CNN.input], outputs=out)

ensemble_VGG.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['binary_accuracy', f1])

# evaluate on the mock test set
X, Y, Fscore = ensemble_VGG.evaluate([X_test, X_test, X_test, X_test], Y_test)
print(X, Y, Fscore)

# save the ensemble model
# save the model to the cloud storage bucket
filename = 'ensemble_VGG'
path = 'gs://mediaeval_data_storage/ensembles'
ensemble_VGG.save(filename)
with file_io.FileIO(filename, mode='r') as inputFile:
        with file_io.FileIO(path + '/' + filename, mode='w+') as outFile:
            outFile.write(inputFile.read())