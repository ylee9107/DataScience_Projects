"""
File name: ResNet_model.py
Author: YSLee
Date created: 29.10.2020
Date last modified:29.10.2020
Python Version: "3.7"
"""

#=========================== Import the Libraries ===========================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import glob
import numpy as np
import functools
# Import the required Libraries:
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D,
                                     GlobalAveragePooling2D, AveragePooling2D,  BatchNormalization, add)
import tensorflow.keras.regularizers as regulisers

#=========================== Constant Definitions ===========================v


#=========================== Defined Functions/Methods ===========================

# ==== (Subblock 1) Residual Branch:
def _res_conv(filters, kernel_size=3, padding='same', strides=1, use_relu=True, use_bias=False, name='conv_bn_Relu',
              kernel_initialiser='he_normal', kernel_regulariser=regulisers.l2(1e-5)):
    """ This builds the Residual Branch, consisting of the Convolutions, batch norm and ReLU activation.
    Parameters:
        - filters, is the number of filters.
        - kernel_size, is the Kernel Size.
        - padding, is the convolution padding.
        - strides, is the convolution stride.
        - use_rele, is a Flag to apply the ReLU activation function at the end.
        - use_bias, is a Flag to use or not the bias in the Convolutional layer.
        - name, is the Name Suffix for the layers.
        - kernel_initialiser, is the Kernel initialisattion method name.
        - kernel_regulariser, is the kernel regulariser.
    Returns:
        - returns a Callable Layer Block.
    """

    def layer_func(x):
        # Convolution:
        conv = Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_initializer=kernel_initialiser,
                      kernel_regularizer=kernel_regulariser,
                      name=name + '_c'
                      )(x)

        # Batch Norm:
        residual_branch = BatchNormalization(axis=-1,
                                             name=name + '_bn'
                                             )(conv)

        # ReLU activation:
        if use_relu:
            residual_branch = Activation(activation='relu',
                                         name=name + '_r'
                                         )(residual_branch)
        return residual_branch

    return layer_func


# ==== (Subblock 2) Residual Shortcut Branch: A.K.A the Identity Block.
def _merge_with_shortcut(kernel_initialiser='he_normal', kernel_regulariser=regulisers.l2(1e-5), name='block'):
    """ This builds the merging layer block for the input tensor and the residual branch output tensor.
    Parameters:
        - kernel_initialiser, is the Kernel initialisation method name.
        - kernel_regulariser, is the Kernel regulariser.
        - name, is the Name Suffix of this layer.
    Returns:
        - returns a Callable Layer Block.
    """

    def layer_func(x, x_residual):
        # Check if there are changes made to the residual branch output "x_residual":
        # if changed, apply 1x1 convolutions.
        x_shape = tf.keras.backend.int_shape(x)
        x_residual_shape = tf.keras.backend.int_shape(x_residual)

        if (x_shape == x_residual_shape):
            shortcut = x
        else:
            strides = (
                int(round(x_shape[1] / x_residual_shape[1])),  # Vertical Stride.
                int(round(x_shape[2] / x_residual_shape[2]))  # horizontal Stride.
            )
            x_residual_chnls = x_residual_shape[3]

            shortcut = Conv2D(filters=x_residual_chnls,
                              kernel_size=(1, 1),
                              strides=strides,
                              padding="valid",
                              kernel_initializer=kernel_initialiser,
                              kernel_regularizer=kernel_regulariser,
                              name=name + '_shortcut_c'
                              )(x)

        # Merge the shortcut and residual:
        merge = add([shortcut, x_residual])

        # Return the merged output:
        return merge

    return layer_func

# ==== Complete Residual Block:
def _residual_block_basic(filters, kernel_size=3, strides=1, use_bias=False, kernel_initialiser='he_normal',
                          kernel_regulariser=regulisers.l2(1e-5), name='res_basic'):
    """ This builds the Basic Residual Layer Block.
    Parameters:
        - filters, is the number of filters.
        - kernel_size, is the kernel size.
        - strides, is the convolutional strides.
        - use_bias, is a Flag to use or not the bias in the convolution layer.
        - kernel_initialiser, is the kernel initialisation method name.
        - kernel_regulariser, is the kernel regulariser.
    Returns:
        - returns a Callable Layer Block.
    """

    def layer_func(x):
        # For the Residual Branch - the First Convolution block:
        x_conv1 = _res_conv(filters=filters,
                            kernel_size=kernel_size,
                            padding='same',
                            strides=strides,
                            use_relu=True,
                            use_bias=use_bias,
                            kernel_initialiser=kernel_initialiser,
                            kernel_regulariser=kernel_regulariser,
                            name=name + '_conv_bn_Relu1'
                            )(x)

        # For the Residual Branch - the Second Convolution block:
        x_residual = _res_conv(filters=filters,
                               kernel_size=kernel_size,
                               padding='same',
                               strides=1,
                               use_relu=False,
                               use_bias=use_bias,
                               kernel_initialiser=kernel_initialiser,
                               kernel_regulariser=kernel_regulariser,
                               name=name + '_conv_bn_Relu2'
                               )(x_conv1)

        # For the Merging (Identity/shortcut) Branch:
        merge = _merge_with_shortcut(kernel_initialiser=kernel_initialiser,
                                     kernel_regulariser=kernel_regulariser,
                                     name=name)(x, x_residual)

        merge = Activation('relu')(merge)

        # Return the merged output:
        return merge

    return layer_func

# ==== Complete Bottleneck Residual Block:
def _residual_bottleneck_blocks(filters, kernel_size=3, strides=1, use_bias=False,
                                kernel_initialiser='he_normal', kernel_regulariser=regulisers.l2(1e-5),
                                name='res_bottleneck'):
    """ This builds the Bottlenecked Residual Blocks, for a ResNet model deeper than 34 layers.
    Parameters:
        - filters, is the number of filters.
        - kernel_size, is the kernel size.
        - strides, is the convolutional strides.
        - use_bias, is a Flag to use or not the bias in the convolution layer.
        - kernel_initialiser, is the kernel initialisation method name.
        - kernel_regulariser, is the kernel regulariser.
    Returns:
        - returns a Callable Layer Block.
    """

    def layer_func(x):
        # For the Bottleneck Residual Branch - the First (bottlenecked) Convolution block:
        x_bottleneck = _res_conv(filters=filters,
                                 kernel_size=1,
                                 padding='valid',
                                 strides=strides,
                                 use_relu=True,
                                 use_bias=use_bias,
                                 kernel_initialiser=kernel_initialiser,
                                 kernel_regulariser=kernel_regulariser,
                                 name=name + '_conv_bn_Relu1'
                                 )(x)

        # For the Residual Branch - the first (non-bottlenecked) Convolution block:
        x_conv = _res_conv(filters=filters,
                           kernel_size=kernel_size,
                           padding='same',
                           strides=1,
                           use_relu=True,
                           use_bias=use_bias,
                           kernel_initialiser=kernel_initialiser,
                           kernel_regulariser=kernel_regulariser,
                           name=name + '_conv_bn_Relu2'
                           )(x_bottleneck)

        # For the Residual Branch - the Second (non-bottlenecked) Convolution block:
        x_residual = _res_conv(filters=filters * 4,
                               kernel_size=1,
                               padding='valid',
                               strides=1,
                               use_relu=False,
                               use_bias=use_bias,
                               kernel_initialiser=kernel_initialiser,
                               kernel_regulariser=kernel_regulariser,
                               name=name + '_conv_bn_Relu3'
                               )(x_conv)

        # For the Merging (Identity/shortcut) Branch:
        merge = _merge_with_shortcut(kernel_initialiser=kernel_initialiser,
                                     kernel_regulariser=kernel_regulariser,
                                     name=name)(x, x_residual)

        merge = Activation('relu')(merge)

        # Return the merged output:
        return merge

    return layer_func

# ==== Combining by chaining the blocks together forming the Modular Network:
def _residual_macroblock(block_func, filters, repetitions=3, kernel_size=3, strides_1stBlock=1, use_bias=False,
                         kernel_initialiser='he_normal', kernel_regulariser=regulisers.l2(1e-5), name='res_macroblock'):
    """ This builds the a Layer block that is composed of a repetition of 'N' number of residual blocks.
    Parameters:
        - block_fun, is the Block Layer method.
        - repetitions, is the number of times the block func is repeated inside.
        - filters, is the number of filters.
        - kernel_size, is the kernel size.
        - strides_1st_block, is the convolutional strides for the 1st block.
        - use_bias, is a Flag to use or not the bias in the convolution layer.
        - kernel_initialiser, is the kernel initialisation method name.
        - kernel_regulariser, is the kernel regulariser.
    Returns:
        - returns a Callable Layer Block.
    """

    def layer_func(x):
        # Loop through the specified number of repetitions:
        for i in range(repetitions):
            block_name = "{}_{}".format(name, i)
            strides = strides_1stBlock if i == 0 else 1
            x = block_func(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           use_bias=use_bias,
                           kernel_initialiser=kernel_initialiser,
                           kernel_regulariser=kernel_regulariser,
                           name=block_name)(x)

        # Return the output from this loop:
        return x

    return layer_func

# ==== Complete the ResNet Model: includes the initial and final Layers.
def ResNet(input_shape, nb_classes=1000, block_func=_residual_block_basic, repetitions=(2, 2, 2, 2),
           use_bias=False, use_dropout=False, kernel_initialiser='he_normal', kernel_regulariser=regulisers.l2(1e-5)):
    """ This builds the ResNet Model for the Classification task.
    Parameters:
        - input_shape, is the input data shape such as (224, 224, 3).
        - nb_classes, is the number of classes to be predicted.
        - block_func, is the Block Layer method to be used.
        - repetitions, is the List of repetitions for each macro-blocks the network should build/contain.
        - use_bias, is a Flag to use or not the bias in the convolution layer.
        - use_dropout, is a Flag to use or not the dropout in the Fully-connected layer.
        - kernel_initialiser, is the kernel initialisation method name.
        - kernel_regulariser, is the kernel regulariser.
    Returns:
        - returns the ResNet model.
    """
    # ResNet's Input layer:
    inputs = Input(shape=input_shape)

    conv = _res_conv(filters=64,
                     kernel_size=7,
                     strides=2,
                     use_relu=True,
                     use_bias=use_bias,
                     kernel_initialiser=kernel_initialiser,
                     kernel_regulariser=kernel_regulariser
                     )(inputs)

    maxpool = MaxPooling2D(pool_size=3,
                           strides=2,
                           padding='same')(conv)

    # Resnet's chain of Residual Blocks (Repetition Layers):
    filters = 64
    strides = 2
    res_block = maxpool

    for i, repet in enumerate(repetitions):
        # NOTE: no further input size reduction for the 1st block, as max-pooling was applied prior.
        block_strides = strides if i != 0 else 1
        macroblock_name = "block_{}".format(i)

        res_block = _residual_macroblock(block_func=block_func,
                                         filters=filters,
                                         repetitions=repet,
                                         strides_1stBlock=block_strides,
                                         use_bias=use_bias,
                                         kernel_initialiser=kernel_initialiser,
                                         kernel_regulariser=kernel_regulariser,
                                         name=macroblock_name
                                         )(res_block)

        # Limit the number of filters to 1024 as the maximum:
        filters = min(filters * 2, 1024)

        # Resnet's Final/Output Layer:
    res_spatial_dimen = tf.keras.backend.int_shape(res_block)[1:3]

    avg_pool = AveragePooling2D(pool_size=res_spatial_dimen,
                                strides=1
                                )(res_block)

    flatten = Flatten()(avg_pool)

    # # Adding a fully connected layer having 1024 neurons
    # FC_layer = Dense(units=1024,
    #                  activation='relu',
    #                  use_bias=True,
    #                  kernel_initializer='glorot_uniform')(flatten)

    # Adding the 1st hidden fully connected layer having 256 neurons:
    FC_layer_1 = Dense(units=256,
                       activation='relu',
                       use_bias=True,
                       kernel_initializer='glorot_uniform')(flatten)

    # Adding dropout, if True:
    if use_dropout:
        FC_layer_1 = Dropout(rate=0.3)(FC_layer_1)

    # Adding the 2nd hidden fully connected layer having 256 neurons:
    FC_layer_2 = Dense(units=128,
                       activation='relu',
                       use_bias=True,
                       kernel_initializer='glorot_uniform')(FC_layer_1)

    # Adding dropout, if True:
    if use_dropout:
        FC_layer_2 = Dropout(rate=0.2)(FC_layer_2)



    predictions = Dense(units=nb_classes,
                        activation='softmax',
                        kernel_initializer=kernel_initialiser
                        )(FC_layer_2)

    # Model (Keras API):
    model = Model(inputs=inputs,
                  outputs=predictions)

    # Return the model configurations:
    return model

# ==== Define the ResNet Models: ResNet-18 all the way to ResNet-152
def ResNet18(input_shape, nb_classes = 1000, use_bias = True, use_dropout = True,
             kernel_initialiser = 'he_normal', kernel_regulariser = None):
    return ResNet(input_shape,
                  nb_classes= nb_classes,
                  block_func= _residual_block_basic,
                  repetitions = (2, 2, 2, 2),
                  use_bias = use_bias,
                  use_dropout = use_dropout,
                  kernel_initialiser = kernel_initialiser,
                  kernel_regulariser = kernel_regulariser)

def ResNet34(input_shape, nb_classes = 1000, use_bias = True, use_dropout = True,
             kernel_initialiser = 'he_normal', kernel_regulariser = None):
    return ResNet(input_shape,
                  nb_classes= nb_classes,
                  block_func= _residual_block_basic,
                  repetitions = (3, 4, 6, 3),
                  use_bias = use_bias,
                  use_dropout = use_dropout,
                  kernel_initialiser = kernel_initialiser,
                  kernel_regulariser = kernel_regulariser)


# For ResNet50, the basic residual blocks are replaced with bottleneck residual blocks instead:
def ResNet50(input_shape, nb_classes = 1000, use_bias = True, use_dropout = True,
             kernel_initialiser = 'he_normal', kernel_regulariser = None):
    return ResNet(input_shape,
                  nb_classes= nb_classes,
                  block_func= _residual_bottleneck_blocks,
                  repetitions = (3, 4, 6, 3),
                  use_bias = use_bias,
                  use_dropout = use_dropout,
                  kernel_initialiser = kernel_initialiser,
                  kernel_regulariser = kernel_regulariser)

def ResNet101(input_shape, nb_classes = 1000, use_bias = True, use_dropout = True,
              kernel_initialiser = 'he_normal', kernel_regulariser = None):
    return ResNet(input_shape,
                  nb_classes= nb_classes,
                  block_func= _residual_bottleneck_blocks,
                  repetitions = (3, 4, 23, 3),
                  use_bias = use_bias,
                  use_dropout = use_dropout,
                  kernel_initialiser = kernel_initialiser,
                  kernel_regulariser = kernel_regulariser)

def ResNet152(input_shape, nb_classes = 1000, use_bias = True, use_dropout = True,
              kernel_initialiser = 'he_normal', kernel_regulariser = None):
    return ResNet(input_shape,
                  nb_classes= nb_classes,
                  block_func= _residual_bottleneck_blocks,
                  repetitions = (3, 4, 36, 3),
                  use_bias = use_bias,
                  use_dropout = use_dropout,
                  kernel_initialiser = kernel_initialiser,
                  kernel_regulariser = kernel_regulariser)







