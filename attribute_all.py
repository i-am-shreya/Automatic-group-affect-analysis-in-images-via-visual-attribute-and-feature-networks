#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 19:59:32 2017

@author: shreya
"""
from keras.utils import np_utils
import scipy.io as sio
import h5py
import keras
from keras import backend as K
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D,Activation
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

with h5py.File('vgg_face.h5')as hf:
        x_train=hf['train'][:].transpose()
        y_train=hf['train_label'][:].transpose()
        x_test=hf['test'][:].transpose()
        y_test=hf['test_label'][:].transpose()       

y_train = np_utils.to_categorical(y_train, 40)
y_test  = np_utils.to_categorical(y_test, 40)


model = Sequential()
model.add(Dense(1024, input_shape=(4096,)))
model.add(Activation(custom_activation))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation(custom_activation))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Activation(custom_activation))
model.add(Dense(40, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.SGD( lr=0.01, decay=1e-6 , momentum=0.9, nesterov=True), metrics=["accuracy"])
filepath="weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping=keras.callbacks.EarlyStopping(monitor='val_acc', patience=50, verbose=0, mode='max')
#callbacks_list = [checkpoint,early_stopping]

model.fit(x_train, y_train, batch_size=64, nb_epoch=5000, callbacks=[checkpoint,early_stopping],verbose=1, validation_data=(x_test, y_test))
model.load_weights("weights.h5")
model.save('celebA.h5')   
    
scores=np.zeros([40,1])



    
    