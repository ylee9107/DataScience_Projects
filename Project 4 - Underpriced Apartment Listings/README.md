# Project 4 – App to find Under-priced Apartments

# App to find Under-priced Apartments:

# Introduction:

Searching for apartments that falls within your budget and location choice can be highly time consuming and frustrating. Even when a suitable apartment was found, knowing whether it is the right one is another challenge. During an apartment hunt, there are trade-off to consider such as having situated close to public transport in a city against having an elevator in the building, or time taken to walk to the nearest public transport. 

The __goal__ here is to make an application that will help the apartment hunting process that also explores these kinds of questions.

## Breakdown of the Showcase Files found in this Github Repository (seen above):
1. __MLApp_underpricedApartments.ipynb__ -  Shows my complete step-by-step approach to reaching the final model. [Link]()
2. __UnderpricedApartments_Main.ipynb__ - Shows the “Streamlined” version where it scrapes the website and accesses Google’s Geolocation and Codes API in the background and outputs the results. There is a handy function to input the search options (Zip code and number of beds) and it will generate the prices. This can be considered closer to the “Deployable” version of this project. [Link]()

## Breakdown of this Project:
1. Inspect through the Source Apartment Listing Website with Browser.
2. Explore through HTML Data and prepare the data.
3. Visualise the Data.
4. Modelling with Regression for Pricing.
5. Forecast the Prices according to Location.

## Dataset:

Link of chosen website: https://www.renthop.com/nyc/apartments-for-rent

#### Below shows a sample of the website used for this project:

<img src="Description Images/RentHop_Sample.PNG" width="550">

Image Ref -> https://www.renthop.com/nyc/apartments-for-rent

Website Description (From Source): Apartment hunting can be overwhelming, but we realized that finding a new home isn't about looking at every apartment listing, it's about finding the best ones.

## Requirements:
- NumPy
- Pandas
- Requests
- Matplotlib
- Google Map API
- Regular Expressions
- Folium
- Statsmodels API
- patsy

## Visualisation of the Apartment Prices from the model:

The diagram below shows a Choropleth (heatmap) of the Apartment prices by zip code in New York City. 

<img src="Description Images/Rental_Prices_NY_Choropleths_1.PNG
" width="850">


## Summary

From this project, I have learnt how to acquire data online from a real estate listings webpage; to utilise pandas to manipulate and clean the data obtained. I was able to visualise the data with Folium choropleths. I was able to build a regression model to predict the rental prices on the types of apartments available in NYC. 

