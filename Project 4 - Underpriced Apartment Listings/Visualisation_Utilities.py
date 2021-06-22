"""
File name: Visualisation_Utilities.py
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
# Defined Function to extract the ZIP codes from the addresses:
def extract_zip(row):
    """ This function will piece together the address to call with Google Maps Geolocation API.
        To limit the calls, that only starts with street number. Then iterate over
        the JSON response to parse out the ZIP code. Once the ZIP code is found,
        it will be returned.
    Parameters:
        - row, is the row data.
    Returns:
        - returns piece_geo.
    Notes:
        - Function may take a long time to run depending on the number of addresses.
    """
    try:
        count = 0

        address = row['address'] + ' ' + row['neighbourhood'].split(', ')[-1]

        if re.match('^\d+\s\w', address):
            geocode_result = gmaps.geocode(address)

            for piece_geo in geocode_result[0]['address_components']:
                if ('postal_code' in piece_geo['types']):
                    count += 1
                    return piece_geo['short_name']
                else:
                    pass
        else:
            return np.nan
    except:
        return np.nan































