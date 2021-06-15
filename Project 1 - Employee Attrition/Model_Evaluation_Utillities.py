"""
File name: Model_Evaluation_Utilities.py
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
def train_and_eval_Model(clf_model, X_train, y_train,
                         X_test, y_test, confusion_mat_labels=None, model_name=None):
    """ This function will train (fit) the specified model and print out the evaluation results.
    Parameters:
        - clf_model, is the input Model (class, e.g. LogisticRegression()).
        - X_train, is the input training set (features).
        - y_train, is the input training set (labels).
        - X_test, is the input testing set (features).
        - y_test, is the input testing set (labels).
        - confusion_mat_labels, is the Target Column to create labels for the confusion matrix (np.Array).
        - model_name, is the (String) model's name for tracking.
    Returns:
        - returns y_pred, model_confusion_matrix, model_classification_report
    Notes:
        - N/A
    """
    # Instantiate the model:
    model = clf_model

    # Traing the model -> Fit the model to the dataset:
    model.fit(X_train, y_train)

    # Make predictions with Unseen data:
    y_pred = model.predict(X_test)

    # Accuracy Metric:
    print("{}'s Accuracy: {}%".format( model_name ,
                                       round( (100 * accuracy_score(y_true=y_test, y_pred=y_pred)), ndigits=2) ))

    # F1-score Metric:
    print("{}'s F1-Score: {}%".format( model_name ,
                                       round( (100 * f1_score(y_true=y_test, y_pred=y_pred)), ndigits=2) ))

    print(" ")

    # Confusion Matrix Plot:
    print(" === Plotting Confusion Matrix === ")

    # output in this order: tn, fp, fn, tp
    model_confusion_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

    ax= plt.subplot()
    sns.heatmap(data=model_confusion_matrix, annot=True)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix');

    true_pos_label = str(confusion_mat_labels.unique()[0])
    true_neg_label = str(confusion_mat_labels.unique()[1])
    false_pos_label = str(confusion_mat_labels.unique()[0])
    false_neg_label = str(confusion_mat_labels.unique()[1])

    ax.yaxis.set_ticklabels([true_neg_label, true_pos_label]);
    ax.xaxis.set_ticklabels([false_neg_label, false_pos_label]);
    plt.show()
    print(" ")

    # Classification Report:
    # Save the report:
    model_classification_report = classification_report(y_true=y_test, y_pred=y_pred)
    print(model_classification_report)

    return y_pred, model_confusion_matrix, model_classification_report


def get_best_model_and_accuracy(model, params, X, y):
    """ This function will run the model with its set parameters, with the dataset.
        It does so by using sklearn's GridSearchCV.
    Parameters:
        - model, is the input model to be used.
        - params, is the Dict of parameters for the model
        - X, is the training dataset.
        - y, is the target variable.
    Returns:
        - returns the Best Accuracy, Best Parameters used, Average time to fit and Average time to Score.
    """
    # Define the GridSearchCV:
    grid = GridSearchCV(estimator=model, param_grid=params, error_score=0., n_jobs=-1)

    # Fit the model to the dataset:
    grid.fit(X, y)

    # Classical metric for performance:
    print("Best Accuracy: {:.3f}%".format((100 * grid.best_score_)))

    # Best parameters that caused the best accuracy
    print("Best Parameters: {}".format(grid.best_params_))

    # Average time it took a model to fit to the data (in seconds)
    print("Average Time to Fit (s): {}".format(round(grid.cv_results_['mean_fit_time'].mean(),
                                                     ndigits=3)))

    # Average time it took a model to predict out of sample data (in seconds)
    # this metric gives us insight into how this model will perform in real-time analysis
    print("Average Time to Score (s): {}".format(round(grid.cv_results_['mean_score_time'].mean(),
                                                       ndigits=3)))

    return grid



def predict_stay_leave(x_data, grid_model, output_col_name):
    """ This function will take in a DataFrame of Employee data (Columns same as original data),
        and used the Trained Model to predict the outcome.
    Parameters:
        - x_data, is the input test data.
        - grid_model, is the trained Grid searched model.
        - output_col_name, is the target column's name.
    Returns:
        - returns a DataFrame of the predictions.
    Notes:
        - N/A
    """
    # Make a copy of the input data:
    X_sample_employee = x_data

    # Transform data with preprocessing + feature engineering pipeline:
    X_test_for_prediction = grid_model.best_estimator_.named_steps['P_featureEng'].transform(X_sample_employee)

    # Compute the predictions:
    predictions_on_employees = grid_model.best_estimator_.named_steps['classifier'].predict(X_test_for_prediction)

    # Convert back to dataFrame with the correct Labels:
    predictions_on_employees_df = pd.DataFrame(data=predictions_on_employees, index=X_sample_employee.index,
                                               columns=[output_col_name])
    predictions_on_employees_df = predictions_on_employees_df[output_col_name].apply(
        lambda x: "Yes" if x == 1 else "No")

    predictions_on_employees_df = predictions_on_employees_df.to_frame()

    employees_stay = predictions_on_employees_df[ predictions_on_employees_df[output_col_name] == "No"]
    employees_leave = predictions_on_employees_df[predictions_on_employees_df[output_col_name] == "Yes"]

    return predictions_on_employees_df, employees_stay, employees_leave