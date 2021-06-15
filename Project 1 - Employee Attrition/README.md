# Data Science Showcase Project 1 – Employee Attrition

## Introduction:
The goal of this project is to develop a model that lowers the costs associated with hiring and training employees, this is done by focusing on the predicting which employees might leave the company. Within any organisation/company, the approach towards spending decisions every day plays a major role in company success and one of these decisions would be the important investment in people. The hiring process takes up a lot of skills, patience, time and money. 

The following outlines the most common hiring costs (https://toggl.com/blog/cost-of-hiring-an-employee):
1. External Hiring Teams.
2. Internal HR Teams.
3. Career Events.
4. Job boards fees.
5. Background Checks.
6. Onboarding and training.
7. Careers page.
8. Salary and extras.

As it can be seen from the lists above, it can be exceedingly difficult to pinpoint precisely the costs that are associated with hiring an employee. With all these rigorous processes already set up for a given company, perhaps there are better questions to ask, such as:
1. Which employee will stay, and which will leave?
2. What are the factors that leads to an employee leaving the company and how it can be predicted?

## Breakdown of the Showcase Files found in this Github Repository(seen above):
1.	Employee Attrition.ipynb – Shows my complete step-by-step approach to reaching the final model. This will include all the theory behind each Classification models that I tried and implements, my preprocessing (data cleaning and feature engineering) pipeline testing and configurations, and GridSearch setup to fine-tune the models.  [Link](https://nbviewer.jupyter.org/github/ylee9107/DataScience_Projects/blob/main/Project%201%20-%20Employee%20Attrition/Employee%20Attrition.ipynb)
2.	Employee Attrition_Main.ipynb – Shows the “Streamlined” version where it removes all the experiments from the notebook mentioned above, and focuses loading in the dataset, precprocessing in and outputting the final results. This can be considered closer to the “Deployable” version of this project. [Link](https://nbviewer.jupyter.org/github/ylee9107/DataScience_Projects/blob/main/Project%201%20-%20Employee%20Attrition/Employee%20Attrition_Main.ipynb)

## Breakdown of this Project:
1. Loading in the Dataset.
2. Visualise the data.
3. Dataset preparation (Data cleaning, training and testing splits)
4. Classifier models (Logistic Regression, Neural Networks, Random Forest)
5. Evaluation methodologies (Accuracy, Precision, Recall and F1-Scores)
6. Classifier Model training and its evaluation.
7. Fine-tune the Hyperparameters to obtain the Optimum Model.
8. Utilise final model to make predictions on Employee data.

## Dataset:
Link: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
Dataset Description (from source): Uncover the factors that lead to employee attrition and explore important questions such as ‘show me a breakdown of distance from home by job role and attrition’ or ‘compare average monthly income by education and attrition’. This is a fictional data set created by IBM data scientists. 

## Requirements:
- Numpy
- Pandas
- Seaborn
- Matplotlib
- sk-learn
- os
- tensorflow (or Keras)

## Summary of all Feature Engineering Trials from the Experimental Notebook:
Feature engineering techniques that were considered and implemented:
1. Statistical-Based selection method on the existing features.
    - Pearson Correlations.
    - Hypothesis Testing. (P-values).
2. Model-based selection method on the existing features.
    - Tree-based models.
3. Principal Component Analysis to transform the features (Parametric assumption).
4. Restricted Boltzmann Machine (RBM) to create more features (non-parametric assumption).

#### Below shows the summary results:

| Technique | Class | Accuracy (%) | F1-Score (%) | Precision (%) | Recall (%) | GridSearch (fine-tune) |
|  --- | --- | --- | --- | --- | --- | --- |
| Baseline Weighted LogReg | 0 (No, stayed) | 86 | 92 | 87 | 97 | None |
|  | 1 (Yes, left) |  | 47 | 72 | 35 |
|  --- | --- | --- | --- | --- | --- | --- |
| Pearson Correlations | 0 (No, stayed) | 88 | 93 | 89 | 98 | YES |
|  | 1 (Yes, left) |  | 48 | 80 | 34 |
|  --- | --- | --- | --- | --- | --- | --- |
| Hypothesis Testing (P-values) | 0 (No, stayed) | 87 | 93 | 87 | 99 | YES |
|  | 1 (Yes, left) |  | 22 | 78 | 13 |
|  --- | --- | --- | --- | --- | --- | --- |
| Tree-based model selection | 0 (No, stayed) | 83 | 91 | 83 | 1 | NO |
|  | 1 (Yes, left) |  | 3 | 100 | 2 |
|  --- | --- | --- | --- | --- | --- | --- |
| PCA | 0 (No, stayed) | 83 | 90 | 83 | 100 | YES |
|  | 1 (Yes, left) |  | 3 | 5 | 2 |
|  --- | --- | --- | --- | --- | --- | --- |
| RBM | 0 (No, stayed) | 84 | 91 | 86 | 96) | YES |
|  | 1 (Yes, left) |  | 27 | 50 | 19 |

## Optimum model and its Parameters:
Best Accuracy: __87.687__% \
Best Parameters: 
{'P_featureEng__featureEngineering__k_best__k': 'all', 'P_featureEng__featureEngineering__pca__n_components': 30, 'P_featureEng__featureEngineering__rbm__n_components': 300, 'P_featureEng__featureEngineering__rbm__n_iter': 100, 'P_featureEng__preprocessing__corr__threshold': 0.02, 
'classifier__C': 0.1}
Average Time to Fit (s): 1.712
Average Time to Score (s): 0.006

## Example Visualisation of the Trained model’s Output Predictions:
<img src="Description Images/LogReg_model_final.png" width="550">
Image Ref -> Self-Made
The above demonstrates an example how the trained Logistic Regression model outputs the predictions based on the input data. At the top-right, it shows the data points that represents what the model believes to be Employees that wishes to leave the company based on their attributes such as Salary, Distance from home etc. At the bottom-left shows the Employees that wishes to stay with the company. Note that in reality, the Sigmoid-Curve (although retaining this shape/curve) of the Logistic regression function can vary slightly to the one seen in the diagram as it will be trained and be fitted for this dataset. 

## Summary:
For this project, I was able to get a good understanding of the characteristics of the dataset by performing exploratory data analysis beforehand. With this, I was able to design a pipeline to pre-process the dataset to perform tasks such as encoding the categorical data points, deal with missing data (impute), remove any irrelevant columns (features) and scale the dataset, so that it was made ready for ML model training. As mentioned in the introduction, several models (Logistic Regression, Random Forest and ANN) were utilised to make predictions, where the model’s performance was also evaluated. From the evaluation results, further tuning was made to ensure that the model was more robust to the inherent problem of the dataset, that is the class imbalance. This issue can be seen in real-world datasets in industry. All the details will be included in the notebook “Employee Attrition.ipynb”. I also included a streamlined notebook along with .py utility files to process and model the dataset, where this will by-pass all the theory and demonstrate my scripting skills.
