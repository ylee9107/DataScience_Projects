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

def k_means_clustering_silhouette(dataset_processed, dataset_processed_Scaled_df, dataset_processed_nonScaled_df,
                                  k_cluster_range, plot_charts=False):
    """ This builds the K-means Clustering model based on the optimum k-number of clusters.
        The optimum k-means clusters is obtained from the silhouette score.
    Parameters:
        - dataset_processed, is the input NumPy Array dataset.
        - dataset_processed_nonScaled_df, is the Pandas DataFrame Dataset that is non-scaled.
        - k_cluster_range, is the list of ranges to compute for the silhouette score with K-Meeans.
            (e.g. [2, 10], means compute over range(2,10) )
        - plot_charts, is the optional Flag to plot the charts during computation.
    Returns:
        - return marketing_data_labeled_df, labels_Clusters
    Notes:
        -
    """
    # Define the list of silhouette scores to fill:
    silhouette_scores_list_encoder = []

    # Compute the scores for each of the number of clusters: number of cluster will be 5 to 15.
    for nb_cluster in range(k_cluster_range[0], k_cluster_range[1]):
        silhouette_scores_list_encoder.append(
            silhouette_score(dataset_processed, KMeans(n_clusters=nb_cluster,
                                                         init='k-means++',
                                                         max_iter=300,
                                                         n_init=10,
                                                         random_state=101,
                                                         algorithm='auto').fit_predict(dataset_processed)
                             ))

    if plot_charts:
        # Plot: Compare the results.
        k = list( range(k_cluster_range[0], k_cluster_range[1]) )
        plt.bar(k, silhouette_scores_list_encoder, color='dodgerblue')
        plt.xlabel('Number of clusters', fontsize=10)
        plt.ylabel('Silhouette Score', fontsize=10)
        plt.show()
    else:
        k = list(range(k_cluster_range[0], k_cluster_range[1]))

    # Save the scores as DataFrame:
    silhouette_scores_df = pd.DataFrame({'k_clusters': k,
                                         'silhouette_scores': silhouette_scores_list_encoder})

    # Save the Optimum Cluster Number:
    optimum_cluster_number = silhouette_scores_df.k_clusters[silhouette_scores_df.silhouette_scores.idxmax()]
    print("The optimum number of clusters that should be used is: {}".format(optimum_cluster_number))
    print(" ")

    # ======================== Apply the K-Means Model =========================
    # Instantiate The K-means Clustering Model:
    kmeans_model_Clusters = KMeans(n_clusters=optimum_cluster_number,
                                    init='k-means++',
                                    max_iter=300,
                                    n_init=10,
                                    random_state=101,
                                    algorithm='auto')

    # fit the model to the dataset:
    kmeans_model_Clusters.fit(X=dataset_processed)

    # Extract the label data from the model:
    labels_Clusters = kmeans_model_Clusters.labels_

    # Update the Dataset(s) with a new Column (labels): concatenate.
    # Note: the "CUST_ID" column must be dropped here.
    marketing_data_scaled_labeled_df = pd.concat([dataset_processed_Scaled_df.reset_index(drop=True),
                                           pd.DataFrame({'cluster': labels_Clusters})],
                                          axis=1)

    marketing_data_nonScaled_labeled_df = pd.concat([dataset_processed_nonScaled_df.reset_index(drop=True),
                                           pd.DataFrame({'cluster': labels_Clusters})],
                                          axis=1)

    # if plot_charts:
    #     # Inspect:
    #     marketing_data_labeled_df.head()

    print(" ")
    print("K-Means algorithm has been fitted with {}-cluster, output dataset it ready for use.".format(optimum_cluster_number))
    print("Proceed to Visualising the output results and Cluster Separation check.")
    return marketing_data_scaled_labeled_df, marketing_data_nonScaled_labeled_df, labels_Clusters, optimum_cluster_number


# ===================================== Visualisation Utilities ======================================
def visualise_PCA_plots(dataset_reduced, labels_Clusters):
    """ This function will plot the visualisations for the K-Means model results, based on the optimum
        number of clusters computed. The plot will be a 2-Dimensional PCA plot.
    Parameters:
        - dataset_reduced, is the input k-means computed dataset to plot/visualise from.
        - labels_Clusters, is list of clusters labels.
    Returns:
        - returns pca_df and PCA plot.
    Notes:
        - N/A
    """
    dataset_labeled = pd.concat([dataset_reduced,
                                 pd.DataFrame({'cluster': labels_Clusters})],
                                axis=1)

    # Instantiate the Principal Component Analysis:
    pca = PCA(n_components=2)

    # Fit the PCA to the Dataset, to get the Principal Components:
    principal_components = pca.fit_transform(dataset_labeled)

    # Examine the change in shape:
    print("Original Dataset shape:", dataset_labeled.shape)
    print("PCA Transformed shape:", principal_components.shape)

    # Create a DataFrame for the 2 Principal Components and Add the Labels:
    pca_df = pd.concat([pd.DataFrame(data=principal_components, columns=['pca1', 'pca2']),
                        pd.DataFrame({'cluster': labels_Clusters})],
                       axis=1)

    # Plot:
    plt.figure(figsize=(10, 10))

    ax = sns.scatterplot(x='pca1',
                         y='pca2',
                         hue="cluster",
                         data=pca_df,
                         palette="tab10")
    plt.show()

    return pca_df


def visualise_tSNE_plots(dataset_processed, labels_Clusters):
    """ This function will plot the visualisations for the K-Means model results, based on the optimum
        number of clusters computed. The plot will be a 2-Dimensional t-SNE plot.
    Parameters:
        - dataset_processed, is the input dimension reduced dataset to plot/visualise from.
        - labels_Clusters, is an Optional Flag to visualise the dataset with PCA method.
    Returns:
        - returns either or both visualisations (PCA, t-SNE or both).
    Notes:
        - N/A
    """
    # Instantiate a t-SNE object:
    tsne = TSNE(n_components=2, perplexity=30.0, verbose=1)

    # Fit the PCA to the Dataset, to get the Principal Components:
    tsne_AE_df = tsne.fit_transform(dataset_processed)

    # Convert to DataFrame and Add the Labels:
    tSNE_result_df = pd.concat([pd.DataFrame(tsne_AE_df, columns=['tSNE1', 'tSNE2']),
                                pd.DataFrame({'cluster': labels_Clusters})],
                                  axis=1)

    # Plot:
    plt.figure(figsize=(10, 10))

    ax = sns.scatterplot(x='tSNE1',
                         y='tSNE2',
                         hue="cluster",
                         data=tSNE_result_df,
                         palette="tab10")
    plt.show()

    return tSNE_result_df

def visualise_kMeans_results(dataset_original, labels_Clusters, optimum_cluster_number,
                             cluster_number=None, plot_overall=True, plot_cluster=False):
    """ This function will plot out the results from the k-Means clustering model.
        The results here will be used to find insights from the clusters.
    Parameters:
        - dataset_original, is the input (original) Dataset without any processing.
        - labels_Clusters, is the labels data.
        - optimum_cluster_number, is the optimum cluster number computed with Silhouette Score.
        - cluster_number, is the individual cluster number to plot out for a closer view.
        - plot_overall, is the optional Flag that plot all of the cluster distributions (histograms).
        - plot_cluster, is the optional Flag that plot the individual cluster histogram and
            statistical information table according to the cluster specified by "cluster_number" parameter.
    Returns;
        - returns data_labeled_df, and the plots.
    Notes:
        - N/A
    """
    # Update the ORGINAL Dataset with a new Column (labels): concatenate.
    # Note: the "CUST_ID" column must be dropped here.
    # data_labeled_df = pd.concat([dataset_original.drop(labels='CUST_ID', axis=1).reset_index(drop=True),
    #                              pd.DataFrame({'cluster': labels_Clusters})],
    #                             axis=1)

    # data_labeled_df = pd.concat([dataset_original.reset_index(drop=True),
    #                              pd.DataFrame({'cluster': labels_Clusters})],
    #                             axis=1)
    data_labeled_df = dataset_original

    if plot_overall:
        # Plot histograms on the various clusters within the dataset:
        for i in data_labeled_df.columns:
            plt.figure(figsize=(35, 5))
            for j in range(optimum_cluster_number):
                plt.subplot(1, optimum_cluster_number, j + 1)
                cluster = data_labeled_df[data_labeled_df.cluster == j]
                cluster[i].hist(bins=20)
                plt.title("{} \nCluster {}".format(i, j))

        plt.show()

    if plot_cluster:
        # Define which cluster to plot:
        cluster_nb_interest = cluster_number

        # # Plot histograms on the specified clusters of interest:
        for i in data_labeled_df.columns:
            plt.figure(figsize=(35, 5))
            cluster = data_labeled_df[data_labeled_df.cluster == cluster_nb_interest]
            cluster[i].hist(bins=20)
            plt.title("{} \nCluster {}".format(i, cluster_nb_interest))

        plt.show()

        # Print out the Statistical information for this cluster:
        stats_df = data_labeled_df[data_labeled_df.cluster == cluster_nb_interest].describe()

    else:
        stats_df = pd.DataFrame()

    return data_labeled_df, stats_df








