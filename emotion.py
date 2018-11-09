#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:23:08 2018

@author:shreya
"""

import h5py
from keras import layers, models
from keras import backend as K
from keras.layers import Activation
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.utils.generic_utils import get_custom_objects


def custom_activation(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

def CapsNet(input_shape, n_class, num_routing):
    
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=48, kernel_size=3, strides=1, padding='valid', name='conv1')(x)
    conv1 = layers.Activation(custom_activation)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', name='conv2')(pool1)
    conv2 = layers.Activation(custom_activation)(conv2)
    conv3 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', name='conv3')(conv2)
    conv3 = layers.Activation(custom_activation)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', name='conv4')(pool3)
    primarycaps = PrimaryCap(conv4, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    primarycaps1 = PrimaryCap(conv4, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    primarycaps_out=layers.Add()([primarycaps, primarycaps1])   
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps_out)
    out_caps = Length(name='out_caps')(digitcaps)

    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y]) 
    x_recon = layers.Dense(512)(masked)
    x_recon = layers.Activation(custom_activation)(x_recon)    
    x_recon = layers.Dense(1024)(x_recon)
    x_recon = layers.Activation(custom_activation)(x_recon) 
    x_recon = layers.Dense(4096)(x_recon)
    x_recon = layers.Activation(custom_activation)(x_recon) 
    x_recon = layers.Dense(30000, activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=[100, 100,3], name='out_recon')(x_recon)
    
    return models.Model([x, y], [out_caps, x_recon])


def margin_loss(y_true, y_pred):
    
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data):
    
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger('/log.csv')
    tb = callbacks.TensorBoard('/tensorboard-logs',
                               batch_size=64, histogram_freq=0)
    checkpoint = callbacks.ModelCheckpoint('/weights_best.h5',monitor='val_out_caps_acc',
                                           save_best_only=True, save_weights_only=True,verbose=1,mode='max')
#    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))
    early_stopping=callbacks.EarlyStopping(monitor='val_out_caps_acc', patience=20, verbose=0, mode='max')
    # compile the model
    model.compile(optimizer='adam',
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.0005],
                  metrics={'out_caps': 'accuracy'})

    
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=64, epochs=500,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint,early_stopping])
    
    model.load_weights('emotion.h5')
    model.save('/emotion.h5')
    print('Trained model saved ')

    from utils import plot_log
    plot_log('/log.csv', show=True)

    return model



def load_data():

    with h5py.File('D:/Shreya/dataset.h5')as hf:
        x_train=hf['train'][:].transpose()
        y_train=hf['train_label'][:].transpose()
        x_test=hf['test'][:].transpose()
        y_test=hf['test_label'][:].transpose()       
        y_train = to_categorical(y_train.astype('float32'))
        y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    from keras import callbacks  

    # load data
    (x_train, y_train), (x_test, y_test) = load_data()

    # define model
    model = CapsNet(input_shape=[100,100,3],
                    n_class=3,
                    num_routing=3)
    model.summary()
  
    train(model=model, data=((x_train, y_train), (x_test, y_test)))
    

