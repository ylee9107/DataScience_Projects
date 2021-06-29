# A GitHub Account Recommendation Engine

## Introduction:

A recommendation engine works by using filtering methods/algorithms on user data and based on the past behaviour of a user, it is able to recommend the most relevant items. Interestingly, while most people have known that a recommendation engine are tools that will find closely related products, topics, music, or movies that the user would appreciate, the original recommendation engine was built for finding potential partners. This project aims to explore the ways and the processes that a recommendation engine can be built on. This project will also implement a recommendation engine for finding related GitHub repositories.

Source: https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/

## Breakdown of this Project:
- Collaborative Filtering
- Content-based Filtering
- Hybrid Systems
- Building the GitHub Account Recommendation Engine

## Breakdown of the Showcase Files found in this GitHub Repository (seen above):

1.	 __ GitHub Account Recommendation Engine.ipynb__ - Shows my complete step-by-step approach to reaching the final model. [https://nbviewer.jupyter.org/github/ylee9107/DataScience_Projects/blob/main/Project%205%20-%20Recommendation%20Engine_GithubProfiles/GitHub%20Account%20Recommendation%20Engine.ipynb](Link)

## Dataset:

The data for this project was obtained through my own personal GitHub starred repositories and the repositories from these users. To setup with your own GitHub handle, you can follow the data collection setup method as follows:
1) A file is provided to enter your own Github Handle and Token, called ‘GitHub_Handle_Token.txt’
2) The GitHub Token can be found in this link: https://github.com/settings/tokens
3) Once the required information is provided in the file, it should run and search for your own starred repositories and collect the data needed.
## Snapshot:

The following diagram shows the GitHub accounts recommended to be starred based on what I have liked so far.

<img src="Description Images/RecommendedAccounts.PNG" width="550">

Image Ref -> Output from the recommendation engine.

## Required Libraries:

1. Pandas
2. Numpy
3. IPython.display
4. OS
5. Requests
6. JSON
7. Re - Regular Expression
8. Scipy
9. Sklearn

## Summary:

I learnt more about recommendation engines and that it also has two primary types that are Collaborative Filtering and Content-based Filtering. Additionally, these systems can be combined to form a hybrid version that would counterbalance each of the system’s flaws. This project also built a recommendation engine that uses the GitHub API. Overall, the project was remarkably interesting and has introduced more than 5 repositories that was of interest. These will prove particularly useful for my future learning process.
