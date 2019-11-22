# Custom functions to perform clustering tasks


import math
import pandas as pd
import seaborn as sb
import numpy as np
from math import sqrt
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def dataframe():
    cols = ['Type', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 
        'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
        'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names = cols)
    data_unlabelled = data.copy()
    data_unlabelled.drop(columns = 'Type', inplace = True)
    scaled_data_unlabelled = scale(data_unlabelled)
    pca = PCA()
    pca.fit(scaled_data_unlabelled)
    pca_data = pca.transform(scaled_data_unlabelled)
    pca_df = pd.DataFrame(pca_data[:,0:2], columns = ['Principal Component 1', 'Principal Component 2'])
    merged_dataframes = new = pd.concat([data, pca_df], axis = 1)
    #data.index = data.index + 1
    return data, merged_dataframes, data_unlabelled

def scatterplot(data, variable_1, variable_2, color, centroids = None):
    sb.set(style = "whitegrid")
    plt.figure(figsize = (16, 11))
    scatterplot = plt.scatter(data[variable_1], data[variable_2], alpha = 1,
                         c = data[color], s= 150, cmap = 'Spectral', edgecolors = 'grey')

    cbar = plt.colorbar(scatterplot)
    cbar.set_label(color, size = 21, labelpad = 35, rotation = 0)
    rcParams['axes.titlepad'] = 25
    plt.title(variable_1 + ' vs. ' + variable_2 + ' scatterplot', fontsize = 25, fontweight = 'bold')
    plt.xlabel('$\it{' + variable_1.replace(' ', '\ ') + '}$', fontsize = 20)
    plt.ylabel('$\it{' + variable_2.replace(' ', '\ ') + '}$', fontsize = 20)
    if centroids != None:
        for i in range(len(centroids)):
            plt.scatter(data.iloc[centroids[i]][variable_1], data.iloc[centroids[i]][variable_2],
                        marker='*', s=800, c='gold', edgecolors = 'black')
        plt.title(variable_1 + ' vs. ' + variable_2 + ' scatterplot \n (with centroids)', fontsize = 25, fontweight = 'bold')
    
def random_init(k, seed):
    rand_int = tuple()
    for i in range(int(k)):
        seed = (seed * 5) % 178
        rand_int += (seed, )
    return rand_int

def centroids_generator(dataframe, k, seed):
    rand_int = tuple()
    for i in range(int(k)):
        seed = (seed * 5) % len(dataframe)
        rand_int += (seed, )
    rand_int = np.array(rand_int)
    return dataframe.loc[rand_int]

def euclidean_distance(x, y):
    squared_distance = 0
    for i in range(len(x)):
            squared_distance += (x[i] - y[i])**2
    ed = sqrt(squared_distance)
    return ed

def distance_and_minimum(row, centroids):
    dist_and_min = [euclidean_distance(row, centroids.loc[i]) for i in centroids.index]
    dist_and_min.append(np.array(dist_and_min).argmin())
    return dist_and_min

def clustering_plot(data, variable_1, variable_2, color, centroids):
    sb.set(style = "whitegrid")
    plt.figure(figsize = (16, 11))
    scatterplot = plt.scatter(data[variable_1], data[variable_2], alpha = 1,
                     c = data[color], s= 150, cmap = 'Spectral', edgecolors = 'grey')

    cbar = plt.colorbar(scatterplot)
    cbar.set_label(color, size = 21, labelpad = 35, rotation = 0)
    rcParams['axes.titlepad'] = 25 
    plt.title('Clustering', fontsize = 25, fontweight = 'bold')
    plt.xlabel('$\it{' + variable_1.replace(' ', '\ ') + '}$', fontsize = 20)
    plt.ylabel('$\it{' + variable_2.replace(' ', '\ ') + '}$', fontsize = 20)
    for i in centroids.index:
        plt.scatter(centroids.loc[i][variable_1], centroids.loc[i][variable_2],
                    marker='*', s=800, c='gold', edgecolors = 'black')


    
