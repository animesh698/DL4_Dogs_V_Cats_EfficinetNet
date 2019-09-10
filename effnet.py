#EfficientNet
import numpy as np
import keras
from keras.models import Model
from keras import backend as K
from keras.layers.core import Dense
from keras.models import Sequential
from keras.layers.convolutional import *
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from keras_efficientnets import EfficientNetB0

train_path='C:/Users/legend698/Desktop/train'
test_path='C:/Users/legend698/Desktop/ANIMESH/python/Cats-and-Dogs/test'
valid_path='C:/Users/legend698/Desktop/valid'

train_batches=ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes=['dog','cat'],batch_size=10)
test_batches =ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224),classes=['dog','cat'],batch_size=10)
valid_batches=ImageDataGenerator().flow_from_directory(valid_path,target_size=(224,224),classes=['dog','cat'],batch_size=10)

eff_model=EfficientNetB0()
eff_model.summary()
x=eff_model.layers[-3].output
prediction=Dense(2,activation='softmax')(x)
model=Model(inputs=eff_model.input,outputs=prediction)
model.summary()	


# model.add(Dense(2,activation='softmax')) 	

#model=EfficientNetB0((224,224,3),classes=['cat','dog'])
model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_batches,steps_per_epoch=2500,validation_data=valid_batches,validation_steps=40,epochs=10,verbose=2)

model.save('effnet.h5')