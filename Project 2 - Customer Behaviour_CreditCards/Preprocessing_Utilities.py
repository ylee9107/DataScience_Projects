"""
File name: Preprocessing_Utilities.py
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
class CustomImputer(TransformerMixin):
    """ This builds the CustomImputer, that inherits the TransformerMixin class.
        It essentially imputes the missing values within the listed columns.
        The Strategy can be
        The inheritance should have a .fit_transform method to call with .fit and .transform methods.
    Parameters:
        - col, is the list of columns to impute.
        - impute_strategy, is the impute strategy (choose between 'mean' or 'median').
        - print_log, is an Optional Flag to print out the log to check that thee values were imputed.
    Returns:
        - returns the Inputed Dataframe.
    """

    # Initialise the instance attributes:
    def __init__(self, col, impute_strategy, print_log=False):
        self.col = col
        self.impute_strategy = impute_strategy
        self.print_log = print_log

    # Transform the dataset:
    def transform(self, dataFrame):
        X = dataFrame.copy()

        # Impute the values:
        if self.impute_strategy == 'mean':
            X[self.col] = X[self.col].fillna(X[self.col].mean())
        else:
            X[self.col] = X[self.col].fillna(X[self.col].median())

        # Perform a quick check:
        if self.print_log == True:
            print(X.isnull().sum())
            print("Checking DataFrame and there should be -- no missing values --.")
            print(" ")

        return X

    # Fit method, that follows the fit method from scikit-learn:
    def fit(self, *_):
        return self


class CustomDropUnwantedColumns(TransformerMixin):
    """ This builds the CustomDropUnwantedColumns, that inherits the TransformerMixin class.
        It essentially removes the listed unwanted columns.
        The inheritance should have a .fit_transform method to call with .fit and .transform methods.
    Parameters:
        - col, is the list of columns to impute.
    Returns:
        - returns the transformed dataframe.
    """

    # Initialise the instance attributes, the columns:
    def __init__(self, col):
        self.col = col

    # Transform the dataset by dropping irrelevant columns:
    def transform(self, dataFrame):
        X = dataFrame.copy()

        X_dropped = X.drop(labels=self.col, axis=1)

        return X_dropped

    # Fit method, that follows the fit method from scikit-learn:
    def fit(self, *_):
        return self


def search_categorical_variables(dataset, print_logs=False):
    ''' This function will search through the Dataset and determine which
        column is a categorical one or a numerical one.
    Parameters:
        - dataset, is the input dataset to search through.
        - print_logs, is an Optional Flag to print out the logs, where here
            it is the list of columns found.
    Returns:
        - returns lists_categorical_columns_found, lists_numerical_columns_found.
    '''
    lists_categorical_columns_found = []
    lists_numerical_columns_found = []

    for col in dataset.columns:
        for unique_row_value in dataset[col].unique():
            current_col_categorical = None

            if isinstance(unique_row_value, str):
                current_col_categorical = True
            else:
                ## Set as Numerical type:
                current_col_categorical = False

        # At the end of column check: Update list with the column name.
        if current_col_categorical:
            lists_categorical_columns_found.append(col)
        else:
            lists_numerical_columns_found.append(col)

    if print_logs == True:
        # Shows the Categorical Columns found:
        print('Categorical Columns found are:\t {}'.format(lists_categorical_columns_found))
        print('\t')

        # Shows the Numerical Columns found:
        print('Numerical Columns found are:\t {}'.format(lists_numerical_columns_found))
    else:
        pass

    return lists_categorical_columns_found, lists_numerical_columns_found

# Define the Custom Outlier Removal Class:
class CustomOutlierRemoval(TransformerMixin):
    """ This builds the Custom Outlier Removal, that inherits the TransformerMixin class.
        It will remove outliers based on Z-scores.
        The inheritance should have a .fit_transform method to call with .fit and
        .transform methods.
    Notes:
        - Requires the scipy.stats "zscore" module.
    """

    # Initialise one instance attribute, the columns:
    def __init__(self):
        pass

    # Remove the Outliers:
    def transform(self, dataFrame):
        X = dataFrame.copy()

        z_scores = zscore(X)
        outlier_filter = (np.abs(z_scores) < 3).all(axis=1)
        X = X[outlier_filter]

        return X

    # Fit method, that follows the fit method from scikit-learn:
    def fit(self, *_):
        return self


class scale_features(TransformerMixin):
    """ This builds the Custom Scaler for the dataset, that inherits the TransformerMixin class.
        The inheritance should have a .fit_transform method to call with .fit and .transform methods.
    Parameters:
        - dataFrame, is the input Dataset (DataFrame)
        - scaler_type, is the Flag to choose between MinMaxScaler (set as 'MinMax')
            or StandardScaler (set as 'Standard') to use.
        - set_numpy_array, is an Optional Flag to output in Numpy Array
            (if False, DataFrame will be the output).
    Returns:
        - returns the Scaled DataFrame (rather than np.array).
    """

    # Initialise the instance attributes, the columns and ordering:
    def __init__(self, scaler_type, set_numpy_array):
        self.scaler_type = scaler_type
        self.set_numpy_array = set_numpy_array

    # Transform the dataset's ordinal values:
    def transform(self, dataFrame):
        X = dataFrame.copy()

        if self.scaler_type == 'MinMax':
            scaler = MinMaxScaler()

            # Scale the data: outputs Numpy Array
            X_array = scaler.fit_transform(X)

            if self.set_numpy_array == False:
                # Convert back to DataFrame:
                X_df = pd.DataFrame(X_array, index=X.index, columns=X.columns)
        else:
            scaler = StandardScaler()

            # Scale the data: outputs Numpy Array
            X_array = scaler.fit_transform(X)

            if self.set_numpy_array == False:
                # Convert back to DataFrame:
                X_df = pd.DataFrame(X_array, index=X.index, columns=X.columns)

        if self.set_numpy_array == True:
            X_output = X_array

        else:
            X_output = X_df

        # Outputs the scaled and non-scaled Dataframes/Numpy Arrays:
        return X_output, dataFrame

    # Fit method, that follows the fit method from scikit-learn:
    def fit(self, *_):
        return self





