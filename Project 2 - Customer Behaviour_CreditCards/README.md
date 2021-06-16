
# Project 2 – Customer Behaviour (Credit Card)
## Introduction:

In every business, an effective Marketing department plays a crucial role in developing the company's growth and sustainability. The department is able to build the company's brand and engage customers leading towards increased sales for greater revenue growth.

The diagram below summarises the key roles of Marketing:

<img src="Description Images/role_of_marketing.jpg" width="350">

Image Ref -> https://courses.lumenlearning.com/wmopen-introbusiness/chapter/the-role-of-customers-in-marketing/

In this project, the goal is to tackle the challenge of launching a __targeted marketing campaign__ and this campaign will be based on 6 months worth of customer data (description below). To do this, marketers are required to understand their customers, their behaviours and drive, so that their needs can be identified. Having developed an understanding of their needs, the marketing campaign can be specifically tailored to each of the individual customer's needs. This should drive up sales for the company's products (may even discover new opportunities). The availability of customer data can provide data scientists the opportunity to perform __Market Segmentation or rather Customer Segmentation__. This means that, based on the customer's behaviour and needs, they can be grouped separately for different marketing purposes, where the main question to be answered is: __How many distinctive groups can be found for the data to streamline the campaign?__
### Breakdown of the Showcase Files found in this Github Repository (seen above):
1. __ Customer Behaviour.ipynb__ - Shows my complete step-by-step approach to reaching the final model. 
2. __ Customer Behaviour_Main.ipynb__ - Shows the “Streamlined” version where it removes all the experiments from the notebook mentioned above, and focuses on loading in the dataset, precprocessing in and outputting the final results. This can be considered closer to the “Deployable” version of this project.

## Breakdown of this Project:
1. Loading in the Dataset.
2. Exploratory Data Analysis (Visualise the data).
3. Examining the Outliers.
4. Dataset preparation (Data cleaning, training and testing splits)
5. K-Means Clustering Techniques with all the data
6. Visualise the Customer Groups/Segmentation with PCA and t-SNE.
7. Refine the number of clusters with AutoEncoders.
8. Visualise the refined Customer Groups/Segmentation with PCA and t-SNE.
9. Model Comparison, Evaluation and Conclusion (Marketing strategy suggestion).

## Dataset:

Link: https://www.kaggle.com/arjunbhasin2013/ccdata

As quoted from the link, the description is:

This case requires to develop a customer segmentation to define marketing strategy. The
sample Dataset summarizes the usage behaviour of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioural variables.

The following shows the attributes of the Credit Card dataset :

- __CUSTID__ : Identification of Credit Card holder (Categorical) 
- __BALANCE__ : Balance amount left in their account to make purchases 
- __BALANCEFREQUENCY__ : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated) 
- __PURCHASES__ : Amount of purchases made from account 
- __ONEOFFPURCHASES__ : Maximum purchase amount done in one-go 
- __INSTALLMENTSPURCHASES__ : Amount of purchase done in instalment 
- __CASHADVANCE__ : Cash in advance given by the user 
- __PURCHASESFREQUENCY__ : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased) 
- __ONEOFFPURCHASESFREQUENCY__ : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased) 
- __PURCHASESINSTALLMENTSFREQUENCY__ : How frequently purchases in instalments are being done (1 = frequently done, 0 = not frequently done) 
- __CASHADVANCEFREQUENCY__ : How frequently the cash in advance being paid 
- __CASHADVANCETRX__ : Number of Transactions made with "Cash in Advanced" 
- __PURCHASESTRX__ : Number of purchase transactions made 
- __CREDITLIMIT__ : Limit of Credit Card for user 
- __PAYMENTS__ : Amount of Payment done by user 
- __MINIMUM_PAYMENTS__ : Minimum amount of payments made by user 
- __PRCFULLPAYMENT__ : Percent of full payment paid by user 
- __TENURE__ : Tenure of credit card service for user 

## Requirements:
- Numpy
- Pandas
- Seaborn
- Matplotlib
- scikit-learn (sklearn)
- os
- timeit
- tensorflow (or Keras)
- scipy

### Example Results from the Clustering model:
The following shows the cluster separation based on customer’s purchasing behaviour.
<img src="Description Images/ PCA_cluster_separation.PNG" width="650">

The following shows a “cut-down” version of the Clustering output results that details of each customer group behaviour.
<img src="Description Images/ Output_results_Clusters.PNG" width="650">

## Summary:
For this project, I was able to build a Clustering model that groups different types of customers together based on their spending (credit card) behaviour. I waws able to design a pipeline to pre-process the dataset such as removal of outliers, dealing with missing values and scaling them to be ready for modelling. I was also able to dimensionally reduce the size of the dataset to retain the important features with a deep learning AutoEncoder model. Further, to validate that the customer’s behaviour was unique to each group, I had visualised each cluster with PCA or t-SNE to visually validate that the groups were sufficiently separated. Overall, this allow me to aid the (fictitious) marketing team in building a targeted marketing scheme. All the details will be included in the notebook “Customer Behaviour.ipynb”. I also included a streamlined notebook along with .py utility files to process and model the dataset, where this will by-pass all the theory and demonstrate my scripting skills.
