# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 18:03:41 2017

@author: Shreya
"""
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense,Activation, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import to_categorical
import h5py
from keras.optimizers import SGD
from keras.layers import Input
import keras
from keras import callbacks
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

with h5py.File('dataset.h5')as hf:
        x_train=hf['train'][:].transpose()
        y_train=hf['train_label'][:].transpose()
        x_test=hf['test'][:].transpose()
        y_test=hf['test_label'][:].transpose()       
        y_train = to_categorical(y_train.astype('float32'))
        y_test = to_categorical(y_test.astype('float32'))

input_tensor = Input(shape=(224,224, 3))
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(4096)(x)
x= Activation(custom_activation)(x)
x = Dense(4096)(x)
x= Activation(custom_activation)(x)
x = Dense(4096)(x)
x= Activation(custom_activation)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

for layer in model.layers[:249]:
   layer.trainable = True
for layer in model.layers[249:]:
   layer.trainable = False



model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
early_stopping=keras.callbacks.EarlyStopping(monitor='val_acc', patience=50, verbose=0, mode='max')
checkpoint = callbacks.ModelCheckpoint( 'best_weight.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1,mode='max')
model.fit(x_train, y_train,batch_size=10, epochs=1000,
              validation_data=[x_test, y_test],verbose=1, callbacks=[early_stopping,checkpoint])

model.load_weights('best_weight.h5')
val_acc=model.evaluate(x_test, y_test,verbose=1)
model.save('holistic.h5')

