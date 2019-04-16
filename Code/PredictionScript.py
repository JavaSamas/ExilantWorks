#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 12:23:17 2019

@author: samas
"""

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import Sequential,load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import cv2
import os
import time
dp_start=time.time()

classes=['EM-QRCRI','Others','ZZ-QRCRI','ZZ-VDLAY','ZZ-VDSHP','ZZ-VDZZZ','ZZ-VRZZZ']

# step 1: load data

img_width = 150
img_height = 450
train_data_dir = '/Users/samas/Desktop/Model/data/train'
valid_data_dir = '/Users/samas/Desktop/Model/data/validation'

datagen = ImageDataGenerator(rescale = 1./255)

train_generator = datagen.flow_from_directory(directory=train_data_dir,
											   target_size=(img_width,img_height),
											   classes=['EM-QRCRI','Others','ZZ-QRCRI','ZZ-VDLAY','ZZ-VDSHP','ZZ-VDZZZ','ZZ-VRZZZ'],
											   class_mode='categorical',
											   batch_size=16)



validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
											   target_size=(img_width,img_height),
											   classes=['EM-QRCRI','Others','ZZ-QRCRI','ZZ-VDLAY','ZZ-VDSHP','ZZ-VDZZZ','ZZ-VRZZZ'],
											   class_mode='categorical',
											   batch_size=32)


# step-2 : build model

model =Sequential()

model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

print('model complied!!')

print('starting training....')
training = model.fit_generator(generator=train_generator, steps_per_epoch=2048 // 16,epochs=3,validation_data=validation_generator,validation_steps=832//16)

print('training finished!!')

print('saving weights to docclassifier_CNN.h5')

model.save_weights('/Users/samas/Desktop/Model/models/docclassifier_CNN.h5')

#model=model.load_weights('/Users/samas/Desktop/Model/models/docclassifier_CNN.h5')


#model=load_weights('/Users/samas/Desktop/Model/models/docclassifier_CNN.h5')

#pred=model.predict_generator(validation_generator,2000)
#print(pred)

path='/Users/samas/Desktop/Model/data/train/ZZ-VDSHP'
list=[]

#Reading Images and Predicting The Output
for img in os.listdir(path):
    img = os.path.join(path,img)
    if img.endswith('.jpg'):    
        img = cv2.imread(img)
        #img=img * (1.255)
        img = cv2.resize(img,(150,450))
        im2arr = np.reshape(img,[1,150,450,3])    
        # Predicting the Test set results
        y_pred = model.predict_classes(im2arr)
        print(y_pred)
        prediction=classes[int(y_pred)]
        print(prediction)
        list.append(prediction)
print(list)


