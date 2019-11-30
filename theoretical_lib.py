import time
import sklearn
import warnings
import numpy as np
import pandas as pd
import random as rd
from math import sqrt, floor
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

def get_dataframe():
    dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt', sep = ',',
                                names = ['Osm ID', 'Longitude', 'Latitude', 'Altitude'])
    return dataframe

def scatterplot(dataframe):
    plot = dataframe.plot(kind = 'scatter', x = 'Longitude', y = 'Latitude', c = 'darkblue', figsize = (15, 10), s = 1)
    plt.xlabel('Longitude', fontsize = 20)
    plt.ylabel('Latitude', fontsize = 20)
    plt.title('Longitude vs. Latitude scatterplot', fontsize = 25, pad = 15)
    plt.show()

def get_scaled_reduced_dataframe(dataframe):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    dataframe_scaled = pd.DataFrame(minmax_scaler.fit_transform(dataframe), columns = ['Osm ID', 'Longitude', 'Latitude', 'Altitude'])
    dataframe_scaled_reduced = dataframe_scaled[['Longitude', 'Latitude']]
    return dataframe_scaled_reduced

def k_means_random_init(dataframe_scaled_reduced):
    model = KMeans(n_clusters = 10, init = 'random')
    model.fit(dataframe_scaled_reduced)
    clusters = model.predict(dataframe_scaled_reduced)
    distinct_clusters = np.unique(clusters) + 1
    centroids = model.cluster_centers_
    dataframe_scaled_reduced['Clusters'] = clusters + 1
    return clusters, distinct_clusters, centroids, dataframe_scaled_reduced

def random_init_scatterplot(dataframe_scaled_reduced_updated, distinct_clusters, centroids):
    fig = plt.figure(figsize = (15, 10))
    for cluster in distinct_clusters:
        plt.scatter(dataframe_scaled_reduced_updated[dataframe_scaled_reduced_updated.Clusters == cluster]['Longitude'],
                    dataframe_scaled_reduced_updated[dataframe_scaled_reduced_updated.Clusters == cluster]['Latitude'],
                    marker = '.', alpha = 0.5, s = 200)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker = '*', c = 'gold', edgecolor = 'black', s = 1000)
    plt.xlabel('Longitude', fontsize = 20)
    plt.ylabel('Latitude', fontsize = 20)
    plt.title('K-means with random initialization clustering', fontsize = 25)
    plt.legend(distinct_clusters, title = 'Clusters', title_fontsize = 'xx-large', 
               fancybox = True, shadow = True, fontsize = 'x-large')
    plt.show()

def simulation_random_init(dataframe_scaled_reduced):
    times = {}
    inertias = {}
    iterations = {}
    model = KMeans(n_clusters = 10, init = 'random')
    for i in range(1, 11):
        start_time = time.time()
        model.fit(dataframe_scaled_reduced)
        elapsed_time = time.time() - start_time
        times[i] = elapsed_time
        inertias[i] = model.inertia_
        iterations[i] = model.n_iter_
    return times, inertias, iterations

def k_means_plus_plus(dataframe_scaled_reduced):
    model = KMeans(n_clusters = 10, init = 'k-means++')
    model.fit(dataframe_scaled_reduced)
    clusters = model.predict(dataframe_scaled_reduced)
    distinct_clusters = np.unique(clusters) + 1
    centroids = model.cluster_centers_
    dataframe_scaled_reduced['Clusters'] = clusters + 1
    return clusters, distinct_clusters, centroids, dataframe_scaled_reduced

def plus_plus_scatterplot(dataframe_scaled_reduced_updated, distinct_clusters, centroids):
    fig = plt.figure(figsize = (15, 10))
    for cluster in distinct_clusters:
        plt.scatter(dataframe_scaled_reduced_updated[dataframe_scaled_reduced_updated.Clusters == cluster]['Longitude'],
                    dataframe_scaled_reduced_updated[dataframe_scaled_reduced_updated.Clusters == cluster]['Latitude'],
                    marker = '.', alpha = 0.5, s = 200)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker = '*', c = 'gold', edgecolor = 'black', s = 1000)
    plt.xlabel('Longitude', fontsize = 20)
    plt.ylabel('Latitude', fontsize = 20)
    plt.title('K-means++ clustering', fontsize = 25)
    plt.legend(distinct_clusters, title = 'Clusters', title_fontsize = 'xx-large', 
               fancybox = True, shadow = True, fontsize = 'x-large')
    plt.show()

def simulation_plus_plus(dataframe_scaled_reduced):
    times = {}
    inertias = {}
    iterations = {}
    model = KMeans(n_clusters = 10, init = 'k-means++')
    for i in range(1, 11):
        start_time = time.time()
        model.fit(dataframe_scaled_reduced)
        elapsed_time = time.time() - start_time
        times[i] = elapsed_time
        inertias[i] = model.inertia_
        iterations[i] = model.n_iter_
    return times, inertias, iterations

def performance_plot(times_random_init, times_plus_plus):
    plt.figure(figsize = (15, 10))
    plt.plot(times_random_init.values(), c = 'khaki', linewidth = 5)
    plt.plot(times_plus_plus.values(), c = 'cyan', linewidth = 5)
    plt.xlabel('Simulation iterations', fontsize = 20)
    plt.ylabel('Time', fontsize = 20)
    plt.title('Performances', fontsize = 25, pad = 15)
    plt.legend(['Random', 'K-means++'], title = 'Initialization', title_fontsize = 'xx-large', 
               fancybox = True, shadow = True, fontsize = 'x-large')
    plt.ylim(bottom = 0, top = 10)

def inertias_plot(inertias_random_init, inertias_plus_plus):
    plt.figure(figsize = (15, 10))
    plt.plot(inertias_random_init.values(), c = 'khaki', linewidth = 5)
    plt.plot(inertias_plus_plus.values(), c = 'cyan', linewidth = 5)
    plt.xlabel('Simulation iterations', fontsize = 20)
    plt.ylabel('Inertia', fontsize = 20)
    plt.title('K-means inertias', fontsize = 25, pad = 15)
    plt.legend(['Random', 'K-means++'], title = 'Initialization', title_fontsize = 'xx-large', 
               fancybox = True, shadow = True, fontsize = 'x-large')
    plt.ylim(bottom = 0, top = 50000)
    plt.show()

def iterations_plot(iterations_random_init, iterations_plus_plus):
    plt.figure(figsize = (15, 10))
    plt.plot(iterations_random_init.values(), c = 'khaki', linewidth = 5)
    plt.plot(iterations_plus_plus.values(), c = 'cyan', linewidth = 5)
    plt.xlabel('Simulation iterations', fontsize = 20)
    plt.ylabel('Iteration', fontsize = 20)
    plt.title('K-means iterations', fontsize = 25, pad = 15)
    plt.legend(['Random', 'K-means++'], title = 'Initialization', title_fontsize = 'xx-large', 
               fancybox = True, shadow = True, fontsize = 'x-large')
    plt.ylim(bottom = 0, top = 8)
