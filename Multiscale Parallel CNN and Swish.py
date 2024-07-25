
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Activation
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, GlobalAveragePooling2D, Flatten, Dense
def convolutional_block(X, f, filters, s=2):
    F1, F2, F3 = filters
    X_shortcut = X
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('swish')(X)
    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('swish')(X)
    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = BatchNormalization(axis=3)(X)
    # Shortcut path
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    # Adding the shortcut to the main path
    X = Add()([X, X_shortcut])
    X = Activation('swish')(X)
    return X
###################################################################################################################################################
###################################################################################################################################################

#############################################################Dense Block From DenseNet201#########################################################
def transition_block(x, reduction):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Conv2D(int(tf.keras.backend.int_shape(x)[-1] * reduction), (1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x
def conv_block(x, growth_rate):
    x1 = layers.BatchNormalization()(x)
    x1 = layers.Activation('swish')(x1)
    x1 = layers.Conv2D(4 * growth_rate, (1, 1), padding='same', kernel_initializer='he_normal')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('swish')(x1)
    x1 = layers.Conv2D(growth_rate, (3, 3), padding='same', kernel_initializer='he_normal')(x1)
    x = layers.Concatenate()([x, x1])
    return x
def dense_block(x, blocks, growth_rate):
    for i in range(blocks):
        x = conv_block(x, growth_rate)
    return x
###################################################################################################################################################

# Parallel Feature Extraction Stem (PFES)
def PFES(input_layer):
    # Conv1 with 9x9 kernel
    conv1 = Conv2D(32, (9, 9), padding='same', activation='swish')(input_layer)
    conv1 = Conv2D(32, (9, 9), padding='same', activation='swish')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = Conv2D(64, (9, 9), padding='same', activation='swish')(pool1)
    conv1 = Conv2D(64, (9, 9), padding='same', activation='swish')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = Conv2D(128, (9, 9), padding='same', activation='swish')(pool1)
    conv1 = Conv2D(128, (9, 9), padding='same', activation='swish')(conv1)

    # Conv2 with 7x7 kernel
    conv2 = Conv2D(32, (7, 7), padding='same', activation='swish')(input_layer)
    conv2 = Conv2D(32, (7, 7), padding='same', activation='swish')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = Conv2D(64, (7, 7), padding='same', activation='swish')(pool2)
    conv2 = Conv2D(64, (7, 7), padding='same', activation='swish')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = Conv2D(128, (7, 7), padding='same', activation='swish')(pool2)
    conv2 = Conv2D(128, (7, 7), padding='same', activation='swish')(conv2)

    # Conv3 with 5x5 kernel
    conv3 = Conv2D(32, (5, 5), padding='same', activation='swish')(input_layer)
    conv3 = Conv2D(32, (5, 5), padding='same', activation='swish')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3 = Conv2D(64, (5, 5), padding='same', activation='swish')(pool3)
    conv3 = Conv2D(64, (5, 5), padding='same', activation='swish')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3 = Conv2D(128, (5, 5), padding='same', activation='swish')(pool3)
    conv3 = Conv2D(128, (5, 5), padding='same', activation='swish')(conv3)

    # Conv4 with 3x3 kernel
    conv4 = Conv2D(32, (3, 3), padding='same', activation='swish')(input_layer)
    conv4 = Conv2D(32, (3, 3), padding='same', activation='swish')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4 = Conv2D(64, (3, 3), padding='same', activation='swish')(pool4)
    conv4 = Conv2D(64, (3, 3), padding='same', activation='swish')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4 = Conv2D(128, (3, 3), padding='same', activation='swish')(pool4)
    conv4 = Conv2D(128, (3, 3), padding='same', activation='swish')(conv4)

    # Concatenate the outputs of the four parallel layers
    concatenated = Concatenate()([conv1, conv2, conv3, conv4])

    # Apply a 1x1 convolution to reduce the number of channels to 256
    conv1x1 = Conv2D(256, (1, 1), padding='same', activation='swish')(concatenated)

    # Apply MaxPooling to reduce the sample dimensions to 52x52
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv1x1)
    return pool5

# Example usage
input_shape = (416, 416, 3)
input_layer = Input(shape=input_shape)

# PFES
pfes_output = PFES(input_layer)
x=pfes_output
# Inception module1 ######################################################################################################
inception1_1 = Conv2D(128, (1, 1), activation='swish')(x)
inception1_3 = Conv2D(128, (1, 1), activation='swish')(x)
inception1_3 = Conv2D(192, (3, 3), activation='swish', padding='same')(inception1_3)
inception1_5 = Conv2D(32, (1, 1), activation='swish')(x)
inception1_5 = Conv2D(96, (5, 5), activation='swish', padding='same')(inception1_5)
inception1_7 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
inception1_7 = Conv2D(64, (1, 1), activation='swish')(inception1_7)
Inception_Block_output = layers.Concatenate()([inception1_1, inception1_3, inception1_5, inception1_7])
x=pfes_output
# Dense Connection Block##################################################################################################
growth_rate = 64
blocks = [6]
reduction = 0.5

for i, block in enumerate(blocks):
    x = dense_block(x, block, growth_rate)
    if i != len(blocks) - 1:
        x = transition_block(x, reduction)

x = layers.BatchNormalization()(x)
Dense_Block_Output = layers.Activation('swish')(x)

##############################################################
IncepDense= layers.Concatenate()([Inception_Block_output, Dense_Block_Output ])
# Apply a 1x1 convolution to reduce the number of channels to 256
conv1x1 = Conv2D(512, (1, 1), padding='same', activation='relu')(IncepDense)

# Apply MaxPooling to reduce the sample dimensions to 52x52
pool5 = MaxPooling2D(pool_size=(2, 2))(conv1x1)


x=pool5
# Inception module1 ######################################################################################################
inception1_1 = Conv2D(128, (1, 1), activation='swish')(x)
inception1_3 = Conv2D(128, (1, 1), activation='swish')(x)
inception1_3 = Conv2D(192, (3, 3), activation='swish', padding='same')(inception1_3)
inception1_5 = Conv2D(32, (1, 1), activation='swish')(x)
inception1_5 = Conv2D(96, (5, 5), activation='swish', padding='same')(inception1_5)
inception1_7 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
inception1_7 = Conv2D(64, (1, 1), activation='swish')(inception1_7)
Inception_Block_output = layers.Concatenate()([inception1_1, inception1_3, inception1_5, inception1_7])
x=pool5
# Dense Connection Block##################################################################################################
growth_rate = 64
blocks = [6]
reduction = 0.5

for i, block in enumerate(blocks):
    x = dense_block(x, block, growth_rate)
    if i != len(blocks) - 1:
        x = transition_block(x, reduction)

x = layers.BatchNormalization()(x)
Dense_Block_Output = layers.Activation('swish')(x)

##############################################################
IncepDense= layers.Concatenate()([Inception_Block_output, Dense_Block_Output ])
# Apply a 1x1 convolution to reduce the number of channels to 256
conv1x1 = Conv2D(1024, (1, 1), padding='same', activation='swish')(IncepDense)

# Apply MaxPooling to reduce the sample dimensions to 52x52
pool5 = MaxPooling2D(pool_size=(2, 2))(conv1x1)
# Create the model


model = Model(inputs=input_layer, outputs=pool5)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summarize the model
model.summary()