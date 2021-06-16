"""
File name: AutoEncoder_model.py
Author: YSLee
Date created: 6.11.2020
Date last modified: 6.11.2020
Python Version: "3.7"
"""
#=========================== Import the Libraries ===========================
import os
import timeit
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import glorot_uniform
from scipy.stats import zscore

#=========================== Constant Definitions ===========================
# N/A

#=========================== Defined Functions/Methods ===========================
def decoupled_7layer_AutoEncoder(dataset, units_in_layers, code_layer_units):
    """ This will build the AutoEncoder model where the nodes in each layer can be defined.
        The AutoEncoder will also be decoupled to return the AutoEncoder, Encoder and Decoder Models.
        For Training, train the "autoEncoder" as a whole.
        For Feature reduction, use the "encoder" to predict on the dataset.
        For Data restoration, use the "decoder" to predict from the codes_layer.
    Parameters:
        - dataset, is the Input Dataset.
        - units_in_layers, is the list to define the nodes in Encoder layers down to the Codes layer.
            (e.g. [500, 500, 2000] ).
        - code_layer_units, is the code_size (no. of units) in the codes layer.
    Returns:
        - return AutoEncoder, Encoder and Decoder models.
    Notes:
        - Use the Keras Functional API.
    """
    # Define the number of nodes in the Input Layer and the Output Layer:
    input_shape = dataset.shape[1]
    output_shape = dataset.shape[1]

    # Design the AutoEncoder model:
    inputs = Input(shape=input_shape, name='input')

    # Encoder - encoding layers:
    encoder_1 = Dense(units=units_in_layers[0], activation='relu',
                      kernel_initializer='glorot_uniform', name='enc_dense1')(inputs)
    encoder_2 = Dense(units=units_in_layers[1], activation='relu',
                      kernel_initializer='glorot_uniform', name='enc_dense2')(encoder_1)
    encoder_3 = Dense(units=units_in_layers[2], activation='relu',
                      kernel_initializer='glorot_uniform', name='enc_dense3')(encoder_2)

    # Code Layer:
    encoded = Dense(units=code_layer_units, activation='relu',
                    kernel_initializer='glorot_uniform', name='enc_dense4')(encoder_3)

    # Decoder - decoding layers:
    decoder_1 = Dense(units=units_in_layers[2], activation='relu',
                      kernel_initializer='glorot_uniform', name='dec_dense1')(encoded)
    decoder_2 = Dense(units=units_in_layers[1], activation='relu',
                      kernel_initializer='glorot_uniform', name='dec_dense2')(decoder_1)
    decoder_3 = Dense(units=units_in_layers[0], activation='relu',
                      kernel_initializer='glorot_uniform', name='dec_dense3')(decoder_2)
    decoded = Dense(units=output_shape, name='dec_dense4')(decoder_3)

    # Instantiate the AutoEncoder Model:
    autoEncoder = Model(inputs, decoded)


    # ============================ Encoder Segment of the model ==============================
    # Define the wrapped encoder model:
    encoder = Model(inputs, encoded)

    # ============================ Decoder Segment of the model ==============================
    # Define the NEW Input: code layer.
    input_code = Input(shape=(code_layer_units,),
                       name='input_code')

    # Rebuild a New Decoder that is based off the AutoEncoder's decoding layers:

    # Count the number of decoding layers: Should be 3 in this case.
    nb_decoder_layer = 0

    for layer in autoEncoder.layers:
        if 'dec_dense' in layer.name:
            nb_decoder_layer += 1

    # Apply each layer to the new data to construct a new graph:
    dec_i = input_code

    # Loop to iterate from the 2 to 0:
    for i in range(nb_decoder_layer, 0, -1):
        # Get the decoder layers from the AutoEncoder model, one at a time:
        decoder_layer = autoEncoder.layers[-i]

        # Construct the new graph, with same parameters:
        dec_i = decoder_layer(dec_i)

    # Instantiate the Decoder model based on the newly graphed layers:
    decoder = Model(input_code, dec_i)

    return autoEncoder, encoder, decoder










