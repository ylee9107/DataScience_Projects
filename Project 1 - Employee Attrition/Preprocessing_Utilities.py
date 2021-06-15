"""
File name: Preprocessing_Utilities.py
Author: YSLee
Date created: 6.11.2020
Date last modified: 6.11.2020
Python Version: "3.7"
"""
#=========================== Import the Libraries ===========================
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from numpy import mean

#=========================== Constant Definitions ===========================
# N/A

#=========================== Defined Functions/Methods ===========================

class CustomDropUnwantedColumns(TransformerMixin):
    """ This builds the CustomDropUnwantedColumns, that inherits the TransformerMixin class.
        It essentially removes the listed unwanted columns.
        The inheritance should have a .fit_transform method to call with .fit and .transform methods.

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


def search_categorical_variables(dataset):
    ''' This function will search through the Dataset and determine which
        column is a categorical one or a numerical one.
    Parameters:
        - dataset, is the input dataset to search through.
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

    return lists_categorical_columns_found, lists_numerical_columns_found

def show_values_from_categorical_columns(dataset, list_of_columns):
    """ This function will print the unique values from the list of columns defined
        for the dataset.
    Parameters:
        - dataset, is the input dataset (DataFrame)
        - list_of_columns, is the specified list of columns to search for.
    Returns:
        - won't return anything, but it will print the results.
    """
    for col in list_of_columns:
        unique_values = list( dataset[col].unique() )
        print("Column -> {} | Unique values are: {}".format(col, unique_values))
        print('\t')
    return


class CustomCategoryEncoder_nominal(TransformerMixin):
    """ This builds the Custom Category Encoder for nominal columns, that inherits the TransformerMixin class.
        It will essentially dummify the values, this function is similar to one-hot-encoding (from sklearn).
        The inheritance should have a .fit_transform method to call with .fit and .transform methods.
    Parameters:
        - dataFrame, is the input Dataset (DataFrame)
        - cols, are the List of (Nominal) columns of interest for this transformation.
    Returns:
        - returns the Encoded Nominal Columns.

    """

    # Initialise one instance attribute, the columns
    def __init__(self, cols=None):
        self.cols = cols

    # Transform the dataset with the dummy variables:
    def transform(self, X):
        return pd.get_dummies(data=X,
                              columns=self.cols)

    # Fit method, that follows the fit method from scikit-learn:
    def fit(self, *_):
        return self


class CustomCategoryEncoder_ordinal(TransformerMixin):
    """ This builds the Custom Category Encoder for ordinal columns, that inherits the TransformerMixin class.
        The inheritance should have a .fit_transform method to call with .fit and .transform methods.
    Parameters:
        - dataFrame, is the input Dataset (DataFrame)
        - print_cols, is an Optional Flag to print the columns of interest to encode.
        - cols, are the List of (Ordinal) columns of interest for this transformation.
        - ordering, is the ordering of the Ordinal Values.
    Returns:
        - returns the Encoded Ordinal Columns.
    """

    # Initialise the instance attributes, the columns and ordering:
    def __init__(self, print_cols=False, cols=None, ordering=None):
        self.cols = cols
        self.ordering = ordering
        self.print_cols = print_cols

    # Transform the dataset's ordinal values:
    def transform(self, dataFrame):
        X = dataFrame.copy()

        for col in self.cols:

            # Find the ordering of the Ordinal Values:
            if self.ordering is not None:
                ordering_to_encode_with = self.ordering
            else:
                ordering_to_encode_with = list(X[col].unique())

            if self.print_cols == True:
                print("Column to Encode -> {} | Unique values are: {}".format(col, ordering_to_encode_with))

            # Encode:
            X[col] = X[col].map(lambda x: float(ordering_to_encode_with.index(x)))

        return X

    # Fit method, that follows the fit method from scikit-learn:
    def fit(self, *_):
        return self


class scale_features_dataFrame(TransformerMixin):
    """ This builds the Custom Scaler for the dataset, that inherits the TransformerMixin class.
        The inheritance should have a .fit_transform method to call with .fit and .transform methods.
    Parameters:
        - dataFrame, is the input Dataset (DataFrame)
        -
    Returns:
        - returns the Scaled DataFrame (rather than np.array).
    """

    # Initialise the instance attributes, the columns and ordering:
    def __init__(self):
        pass

        # Transform the dataset's ordinal values:

    def transform(self, dataFrame):
        X = dataFrame.copy()
        scaler = MinMaxScaler()

        # Scale the data:
        X_array = scaler.fit_transform(X)

        # Convert back to DataFrame:
        X_df = pd.DataFrame(X_array,
                            index=X.index,
                            columns=X.columns)

        return X_df

    # Fit method, that follows the fit method from scikit-learn:
    def fit(self, *_):
        return self


class CustomStatsCorrSelector(TransformerMixin, BaseEstimator):
    """ This builds a custom Pearson Correlation feature selector that seleects the most highly correlated
        features from the dataset. It inherits the attributes of the TransformerMixin and BaseEstimator classes.
    Parameters:
        - response, is the Target Variable of the dataset.
        - cols_to_keep, is the list columns of the highly correlated features.
        - threshold, is the assigned threshold value for correlation (e.g. -/+ 0.20 correlation).
    Return:
        - returns the transformed dataset.
    Note:
        - Made to be compatible with scikit-learn's fit and transform methods.
    """

    def __init__(self, response, cols_to_keep=[], threshold=None):
        # Store the response variable (Series):
        self.response = response

        # Store the threshold:
        self.threshold = threshold

        # Feature Column names to Keep:
        self.cols_to_keep = cols_to_keep

    def transform(self, dataFrame):
        # Selects the columns to keep from the dataset:
        return dataFrame[self.cols_to_keep]

    def fit(self, dataFrame, *_):
        # Create a new DataFrame by concatenating the Feature + Target(response):
        df = pd.concat([dataFrame, self.response], axis=1)

        # Select and store the correlations (features) that meet the threshold requirement:
        self.cols_to_keep = df.columns[df.corr()[df.columns[-1]].abs() > self.threshold]

        # Remove the Target Variable and Keep ony the feature columns:
        self.cols_to_keep = [feature_cols for feature_cols in self.cols_to_keep if feature_cols in dataFrame.columns]

        return self











