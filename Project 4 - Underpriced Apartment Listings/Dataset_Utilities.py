"""
File name: Dataset_Utilities.py
Author: YSLee
Date created: 6.11.2020
Date last modified: 6.11.2020
Python Version: "3.7"
"""
#=========================== Import the Libraries ===========================
import numpy as np
import pandas as pd
import requests
import os
from bs4 import BeautifulSoup
import googlemaps
import re
import folium
import statsmodels.api as sm
import patsy

import matplotlib.pyplot as plt
# %matplotlib inline

#=========================== Constant Definitions ===========================
# N/A

#=========================== Defined Functions/Methods ===========================
# Data Cleaning:
def clean_data(dataset):
    """ This function will ingest the Dataset (apartment_NY_df) and perform all
        of the data cleaning.
    Parameters:
        - dataset, is the input dataset for cleaning. (DataFrame)
    Returns:
        - returns the cleaned dataset.
    Notes:
        - N/A
    """
    # Fix the 'leading underscore' character in the data: checks whether the element
    # begins with an underscore and removes it.
    dataset['beds'] = dataset['beds'].map(lambda x: x[1:] if x.startswith('_') else x)
    dataset['baths'] = dataset['baths'].map(lambda x: x[1:] if x.startswith('_') else x)

    # Rent Column: remove '$', ',' and change type to integer.
    dataset['rent'] = dataset['rent'].map(lambda x: str(x).replace('$', '').replace(',', '')).astype('int')

    # Bed Column: remove the strings like '_bed', 'Studio' etc. and change it to integer or blanks.
    dataset['beds'] = dataset['beds'].map(lambda x: x.replace('_Bed', ''))
    dataset['beds'] = dataset['beds'].map(lambda x: x.replace('Studio', '0'))
    dataset['beds'] = dataset['beds'].map(lambda x: x.replace('Loft', '0'))
    dataset['beds'] = dataset['beds'].map(lambda x: x.replace('Room', '0'))

    # Apply fix to malformed strings: Loft and Room
    dataset['beds'] = dataset['beds'].values.astype('int')

    # Baths Column: remove the strings like '_Bath' and change it to float
    dataset['baths'] = dataset['baths'].map(lambda x: x.replace('_Bath', '')).astype('float')

    # Fix trailing spaces:
    dataset['neighbourhood'] = dataset['neighbourhood'].map(lambda x: x.strip())

    return dataset

# Predictions:
def predict_zipcode_price(zipcode_1, zipcode_2, nb_beds_1, nb_beds_2, stats_input_func, dataset, print_results=False):
    """ This function will perform a price prediction based on the Apartment's Zipcode.
        It utilises the Statsmodel and Patsy APIs to compute the Ordinary Least Squares (OLS)
        on the dataset.
    Parameters:
        - zipcode_1, is the input Zipcode to check the prices on. (e.g. 10069)
        - zipcode_2, is the input Zipcode to compare with Zipcode_1.
        - nb_beds_1, is the number of beds in the 1st apartment of interest.
        - nb_beds_2, is the number of beds in the 2nd apartment of interest.
        - stats_input_func, is the input statistic function to compute with
            the statistic model API. (e.g. func = 'rent ~ zip + beds')
        - dataset, is the input dataset.
        - print_results, is the optional Flag to print out the OLS results.
    Returns:
        - returns the print_results and Price predictions and Price Comparison/difference.
            pred_results_1, pred_results_2, pred_results_diff, OLS_summary.
    Notes:
        - Requires the packages "statsmodels.api" and "patsy".
        - For OLS Theory -> https://en.wikipedia.org/wiki/Ordinary_least_squares
    """
    # Compute the Ordinary Least Squares:
    y, x = patsy.dmatrices(stats_input_func,
                           dataset,
                           return_type='dataframe')
    # Fit:
    results = sm.OLS(y, x).fit()

    if print_results:
        OLS_summary = results.summary()
    else:
        OLS_summary = None

    # Price Forecasting:
    pred_df = pd.DataFrame(data=np.zeros(len(x.iloc[0].index)),
                           index=x.iloc[0].index,
                           columns=['value'],
                           dtype=None,
                           copy=False)

    # Set the Search parameters of the Apartments:
    if (zipcode_2 and nb_beds_2) == None:
        #     pred_df['value'] = 0
        pred_df.loc['Intercept'] = 1
        pred_df.loc['beds'] = nb_beds_1
        pred_df.loc['zip[T.{}]'.format(zipcode_1)] = 1

        # Price Results:
        pred_results_1 = results.predict(pred_df['value'].to_frame().T)
        pred_results_2 = 0
        pred_results_diff = 0

    else:  # Perform Price Comparison:
        pred_df.loc['Intercept'] = 1
        pred_df.loc['beds'] = nb_beds_1
        pred_df.loc['zip[T.{}]'.format(zipcode_1)] = 1

        # Price Results:
        pred_results_1 = results.predict(pred_df['value'].to_frame().T)

        pred_df.loc['Intercept'] = 1
        pred_df.loc['beds'] = nb_beds_2
        pred_df.loc['zip[T.{}]'.format(zipcode_2)] = 1

        # Price Results:
        pred_results_2 = results.predict(pred_df['value'].to_frame().T)

        # Compute the Difference:
        pred_results_diff = pred_results_2 - pred_results_1

    return round(pred_results_1, 3), round(pred_results_2, 3), round(pred_results_diff, 3), OLS_summary




















