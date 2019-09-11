#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:33:44 2019

@author: shandilya
"""
import os
os.chdir('/kaggle/input')
print(os.listdir("/kaggle/input/kermany2018"))
print(os.getcwd())
"""
    Google colab import to access data from disk on google colab
"""
#from google.colab import drive
#drive.mount('/home/shandilya')

"""
    imageio is used to read images with extension .dcm
    Images of type DICOM have the extension .dcm
"""
import imageio
#image1 = '~/Shandilya/Padhai/ED6001/sample-DICOM/xr_chicken1.dcm'
#image2 = '~/Shandilya/Padhai/ED6001/sample-DICOM/xr_chicken2.dcm'
#image3 = '~/Shandilya/Padhai/ED6001/sample-DICOM/E1154S7I.dcm'
TRAIN_PATH = '/kaggle/input/kermany2018/oct2017/OCT2017 /train'
VAL_PATH = '/kaggle/input/kermany2018/oct2017/OCT2017 /val'
TEST_PATH = '/kaggle/input/kermany2018/oct2017/OCT2017 /test'
#im1 = imageio.imread(image1)
#im2 = imageio.imread(image2) 
#im3 = imageio.imread(image3)
"""
    imread() reads images as an Image object which is a standard numoy array
    Often medical images also come with metadata that is very useful for image analysis
    Metadata can be quite rich in medical images and can include:
        -Patient demographics: name, age, sex, clinical information
        -Acquisition information: image shape, sampling rates, data type, modality (such as X-Ray, CT or MRI)
"""
## Prints all the keys of the metadata available with an image
#print(im.meta.keys())
## Prints all the metadata available with an image. Metadata comes with only a certain types of images like DICOM images
#print(im.meta) 
import matplotlib.pyplot as plt

"""
    Images are plotted using matplotlib.pyplot's imshow() function.
    Matplotlib's imshow() function provides a simple way to do visualize and plot images. 
    Knowing a few simple arguments will help:
        -cmap controls the color mappings for each value. The "gray" colormap is common, but many others are available.
        -vmin and vmax control the color contrast between values. Changing these can reduce the influence of extreme values.
        -plt.axis('off') removes axis and tick labels from the image.
    Changing the values of cmap, vmin and vmax only affect the plot and not the data
"""

#plt.imshow(im3, cmap = 'gray')
#plt.show()

"""
    Handling 3d images
"""
#vol = np.stack([im1, im2, im3], axis = 0) # This statement stacks im1, im2, im3 along the axis 0
"""
    The above statement is useful is plotting 3-D images
"""
#vol = imageio.volread(<folder-containing-all-slices>) # This statement is used to reas 3-d images 
"""
    To get useful insights from a 3-D image it is recommended to plot slices of the images. This can be done as follows
"""
#fig, axes = plt.subplot(nrows = 1, ncols = 3) # This creates subplots to plot three images side by side
#axes[0].imshow(vol[0], cmap = 'gray')
#axes[1].imshow(vol[1], cmap = 'gray')
#axes[2].imshow(vol[2], cmap = 'gray')
#for ax in axes: 
#    ax.axis('off')
#plt.show()

"""
    3-D images can be sliced along any axes
    The direction of the slice depends on the indexing
    The indexing to slice along a desired axes is the same as extracting data along a desired axis in a 3-D array or list
    
    The aspect ratio of an image can be varied using the aspect parameter in imageio.imshow()
    For DICOM images the aspect ratio can be calculated using the sampling rates along the three axes
     gives the sampl
"""
#d0,d1,d2 = vol.meta['sampling'] 
#asp = d0/d1 # Calculating the aspect ratio of a DICOM image using the metadata available with the image
#asp = 4/9
#plt.imshow(im3, cmap = 'gray', aspect = asp)

import numpy as np 
import pandas as pd
import os

"""
    Keras is a model-level library, providing high-level building blocks for developing deep learning models
    -- INTRO TO KERAS -- https://www.learnopencv.com/deep-learning-using-keras-the-basics/
    It does not handle low-level operations such as tensor products, convolutions and so on itself
    The Sequential model is a linear stack of layers
    The first layer in a Sequential model needs to receive information about its input shape
"""
import keras
from keras.models import Sequential
model = Sequential()

"""
    Layers are added to the neural network according to the achitecture desired
    Conv1D imports 1D convolutional layer
    The layer takes the following arguments -- filters -- kernel_size -- strides -- padding
                                            -- activation -- bias_initializer
    Input shape - 3D tensor with shape: (batch, steps, channels)
    Output shape - 3D tensor with shape: (batch, new_steps, filters)
    
    Conv2D imports 2D convolution layer (e.g. spatial convolution over images)
    The layer takes the same arguments as Conv1D
    When using this layer as the first layer in a model, provide the keyword argument input_shape 
    Input Shape - 4D tensor with shape: (batch, channels, rows, cols) if data_format is "channels_first"
                - 4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last"
    Output Shape - 4D tensor with shape: (batch, filters, new_rows, new_cols) if data_format is "channels_first"
                 - 4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format is "channels_last"
"""

from keras.layers import Conv1D
#layer = Conv1D(filters, 2
#               kernel_size,
#               strides=1,
#               padding='valid',
#               data_format='channels_last',
#               dilation_rate=1, activation=None,
#               use_bias=True,
#               kernel_initializer='glorot_uniform',
#               bias_initializer='zeros',
#               kernel_regularizer=None,
#               bias_regularizer=None,
#               activity_regularizer=None,
#               kernel_constraint=None,
#               bias_constraint=None)

from keras.layers import Conv2D
layer_conv2d = Conv2D(filters = 3,
                      kernel_size = 10, # -- (2,8)
                      strides = (2,3), # -- 2
                      padding = 'same', # -- 'valid' -- NOT VALID FOR -- strides = 1
                      activation = 'linear', # -- 'elu' -- 'selu' -- relu -- tanh -- sigmoid 
                                  # -- 'exponential' 
                      use_bias = True,
                      bias_initializer = 'zeros', # -- 'ones' -- 'constant'
                      input_shape = (256,256,3),
                      )
"""
    model.add() is used to add a given layer to a model
    Either pass the list of layers as an argument to the constructor or 
    add the layers sequentially using the model.add() function
"""
model.add(layer_conv2d) 

# For a multi-class classification problem

"""
    Dense module imports a dense connection of neurons with unit neurons
    Takes the following parameters are input:
        - units
        - activation function
        - use_bias -- boolean input
        - bias_initializer -- default:integer
        - input_shape
"""
from keras.layers import Dense
model.add(Dense(units = 32,
                activation = 'linear')
         )
"""
    Flatten() flattens the input. Does not affect the batch size.
    
"""
from keras.layers import Flatten
model.add(Flatten())
model.add(Dense(4,
               activation = 'linear'))

"""
    Compilation
    Before training a model the learning process needs to be configured
    It receives three arguments:
        - optimizer -- 'rmsprop' -- 'adagrad' -- 'sgd' -- 'adadelta' -- 'adam' -- 'adamax' 
        - loss -- 'mean_squared_error' -- 'mean_absolute' -- 'mean_absolute_percentage_error' 
               -- 'mean_squared_logarithmic_error' -- FOR MORE LOSSES -- https://keras.io/losses/ 
        - metrics -- ['accuracy', 'mae', 'acc', 'categorical_accuracy', 'binary_accuracy',] 
                  -- FOR MORE METRICS -- https://keras.io/metrics/
"""

## For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

## For a binary classification problem
#model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#
## For a mean squared error regression problem
#model.compile(optimizer='rmsprop',
#              loss='mse')

"""
    -- FOR BACKEND FUNCTIONS -- https://keras.io/backend/
"""
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
#model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy', mean_pred])

"""
    The functions available with keras.backend can be used to create custom function for metrics, optimizer and loss
    These functions created can be used given as an argument to the function call, model.compile()
"""
#inputs = K.placeholder(shape=(2, 4, 5))
## also works:
#inputs = K.placeholder(shape=(None, 4, 5))
## also works:
#inputs = K.placeholder(ndim=3)
#moving_average_update = keras.backend.moving_average_update(x, value, momentum)
#z = keras.backend.dot(x, y) # Multiplies 2 tensors (and/or variables) and returns a tensor



"""
    A callback is a set of functions to be applied at given stages of the training procedure
    Creating call back for early stopping
"""
from keras.callbacks import EarlyStopping# -- ReduceLROnPlateau -- ModelCheckpoint -- Callback 
early_stop=EarlyStopping(monitor='val_loss',
                         min_delta=0, patience=5,
                         verbose=0, mode='auto',
                         baseline=None,
                         restore_best_weights=False
                         )

"""
"""
import warnings
warnings.filterwarnings("ignore")
 
"""
    ImageDataGenerator generates batches of tensor image data with real-time data augmentation
    flow_from_directory is used to create an object to generate tuples (X,y) for training
    -- FOR MORE ABOUT IMAGEDATAGENERATOR ARGUMENTS -- https://keras.io/preprocessing/image/
"""
from keras.preprocessing.image import ImageDataGenerator
def model_trainer(model):
    train_datagen = ImageDataGenerator(rescale = 1./150, 
                                   shear_range = 0.01, 
                                   zoom_range =[0.9, 1.25],
                                   rotation_range=20,
                                   zca_whitening=True,
                                   vertical_flip=True,
                                   fill_mode='nearest',
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   brightness_range=[0.5, 1.5],
                                   horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./160)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(256,256),
        batch_size=32,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(256,256),
        batch_size=32,
        class_mode='categorical')# multiclass then  categorical
    print(train_generator)
    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=2000, # no of images in training set
        epochs=10,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=968,
        callbacks=[early_stop]) # no of  images in test
    return hist,train_generator

hist,train_generator=model_trainer(model)

print(hist)




    
from keras.layers import MaxPooling2D

from keras.layers import BatchNormalization
from keras.layers import Dropout


from sklearn.model_selection import train_test_split, cross_val_score
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten, Masking
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,  Callback, EarlyStopping, ReduceLROnPlateau