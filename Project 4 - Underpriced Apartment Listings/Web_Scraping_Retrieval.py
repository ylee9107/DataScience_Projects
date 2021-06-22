"""
File name: Web_Scraping_Retrieval.py
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
# Defined Function to parse and call each webpages:
def parse_data(listing_divs):
    """ This function builds the Parse Function to retive the relevant Apartment Listing Data.
    """
    listing_list = []
    for idx in range(len(listing_divs)):
        indv_listing = []
        current_listing = listing_divs[idx]

        href = current_listing.select('a[id*=title]')[0]['href']
        address = current_listing.select('a[id*=title]')[0].string
        hood = current_listing.select('div[id*=hood]')[0].string.replace('\n', '')
        indv_listing.append(href)
        indv_listing.append(address)
        indv_listing.append(hood)

        listings_specifications = current_listing.select('div[id*=info]')
        for specs in listings_specifications:
            try:
                values = specs.text.strip().replace(' ', '_').replace('|', '').split()
                clean_values = [x for x in values if x != '_']
                if len(clean_values) != 3:
                    clean_values.remove(clean_values[2])
                indv_listing.extend(clean_values)
            except:
                indv_listing.extend(np.nan)
        listing_list.append(indv_listing)
    return listing_list


def extract_apartment_data(nb_webpages_search, url_prefix):
    """ This function will perform a loop over the defined number of webpages to search for.
    Parameters:
        - nb_webpages_search, is the specified number of webpages to search through, to
            gather the data on Apartment Listings.
        - url_prefix, is the Input URL to search with.
            (e.g. "https://www.renthop.com/search/nyc?min_price=0&max_price=50000&q=&sort=hopscore&search=0&page=",
             Notice the missing page number at the end.)
    Returns:
        - returns all_pages_parsed, data_df.
    Notes:
        - N/A
    """
    number_of_webpages = nb_webpages_search
    page_nb = 1

    all_pages_parsed = []
    for idx in range(number_of_webpages):
        # Define the URL to Search with:
        # target_page = url_prefix + str(page_nb) + url_suffix
        target_page = url_prefix + str(page_nb)

        # Request the webpage:
        r = requests.get(target_page)
        soup = BeautifulSoup(r.content, 'html.parser')

        # Select the Container of information:
        listing_divs = soup.select('div[class*=search-info]')

        # Parse the Data:
        one_page_parsed = parse_data(listing_divs)

        # Update thhe List and page count:
        all_pages_parsed.extend(one_page_parsed)
        page_nb += 1

    # Save the Data into DataFrame:
    data_df = pd.DataFrame(all_pages_parsed, columns=['url', 'address',
                                                      'neighbourhood',   'rent',
                                                      'beds', 'baths'])

    return all_pages_parsed, data_df

# ==================== Google API - Geolocation and Codes =====================
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

















