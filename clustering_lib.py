# Custom functions to perform clustering tasks

import math
import random
import numpy as np
import pandas as pd
import seaborn as sb
from math import sqrt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 85
from matplotlib import rcParams
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib import pyplot as plt



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
    scaled_data_unlabelled_df = pd.DataFrame(scaled_data_unlabelled, columns = cols[1:])
    pca_df = pd.DataFrame(pca_data[:,0:3], columns = ['Principal Component 1', 'Principal Component 2', 'Principal Component 3'])
    merged_dataframes = pd.concat([data['Type'], scaled_data_unlabelled_df, pca_df], axis = 1)
    return merged_dataframes

def scatterplot(data, variable_1, variable_2, color):
    sb.set(style = "whitegrid")
    plt.figure(figsize = (16, 11))
    scatterplot = plt.scatter(data[variable_1], data[variable_2], alpha = 1,
                         c = data[color], s = 150, cmap = 'Spectral', edgecolors = 'grey')

    cbar = plt.colorbar(scatterplot)
    cbar.set_label(color, size = 21, labelpad = 30, rotation = 270)
    rcParams['axes.titlepad'] = 25
    plt.title(variable_1 + ' vs. ' + variable_2 + ' scatterplot', fontsize = 25, fontweight = 'bold')
    plt.xlabel('$\it{' + variable_1.replace(' ', '\ ') + '}$', fontsize = 20)
    plt.ylabel('$\it{' + variable_2.replace(' ', '\ ') + '}$', fontsize = 20)
    
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

def k_means(dataframe, max_iterations = 500, k = 1):

    centroids = centroids_generator(dataframe.drop(columns = ['Type']), k, 13579)
    clusters_changes = []
    iterations = 0
    for i in range(1, max_iterations):
        convergence = True
        if i == 1:
            dist_and_min_df = dataframe.drop(columns = ['Type', 
                                                        'Principal Component 1', 
                                                        'Principal Component 2', 
                                                        'Principal Component 3']).apply(distance_and_minimum, axis = 1, args=[centroids.drop(columns = ['Principal Component 1', 
                                                                                                                                                        'Principal Component 2',
                                                                                                                                                        'Principal Component 3'])], result_type = 'expand').rename(columns = {i: 'Distance from Centroid ' + str(i+1) if i < len(centroids) else 'Centroids' for i in range(len(centroids)+1)})
            data_and_distances = pd.concat([dataframe.drop(columns = ['Type']), dist_and_min_df], axis = 1)
            clusters_changes.append(data_and_distances['Centroids'])
            iterations +=1
        else:    
            dist_and_min_df = dataframe.drop(columns = ['Type', 
                                                        'Principal Component 1', 
                                                        'Principal Component 2', 
                                                        'Principal Component 3']).apply(distance_and_minimum, axis = 1, args=[centroids], result_type = 'expand').rename(columns = {i: 'Distance from Centroid ' + str(i+1) if i < len(centroids) else 'Centroids' for i in range(len(centroids)+1)})
            for i in range(1, len(centroids) + 2):
                if i < len(centroids) + 1:
                    data_and_distances[['Distance from Centroid ' + str(i)]] = dist_and_min_df[['Distance from Centroid ' + str(i)]]
                else:
                    data_and_distances[['Centroids']] = dist_and_min_df[['Centroids']]
            previous_centroids = centroids
            centroids = data_and_distances.groupby('Centroids').mean()
            for j in range(len(centroids)):
                if abs(np.sum((centroids.iloc[j] - previous_centroids.iloc[j])/previous_centroids.iloc[j]*100)) > 0.0001:
                    convergence = False
            iterations += 1
            clusters_changes.append(dist_and_min_df['Centroids'])
            if convergence:
                print('The K-means algorithm converged after ' + str(iterations) + ' iterations.')
                break
    return centroids, clusters_changes, iterations, data_and_distances

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, data):
        self.data = data
        self.colors = mpl.cm.get_cmap('Spectral')
        
        fig = plt.figure(figsize = (11, 9))
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = fig.add_subplot(1, 2, 1)
        ax2.set_visible(False)
        
        im1 = ax2.imshow(self.data['clusters_changes'], cmap = 'Spectral', interpolation='None')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.5)
        cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
        cbar.set_label('Centroids', size = 20, labelpad = 20, rotation = 270)

        ax1.set_xlabel('$\it{Principal\  Component\  1}$', fontsize = 15)
        ax1.set_ylabel('$\it{Principal\  Component\  2}$', fontsize = 15)
        ax1.set_title('K-means clustering', fontsize = 20, fontweight = 'bold')
        
        self.lines = []
        for k in range(data['K']):
            self.lines.append(Line2D(
                [], [], color = self.colors(random.random()), marker = 'o', markersize = 11, markeredgecolor = 'grey', lw = 0, alpha = 1))
            self.lines.append(Line2D(
                [], [], color = 'gold', marker = '*', mew = 1, ms = 22, markeredgecolor = 'black', lw = 0))
            ax1.add_line(self.lines[-2])
            ax1.add_line(self.lines[-1])
        ax1.set_xlim(self.data['data']['Principal Component 1'].min() - 1, self.data['data']['Principal Component 1'].max() + 1)
        ax1.set_ylim(self.data['data']['Principal Component 2'].min() - 1, self.data['data']['Principal Component 2'].max() + 1)

        animation.TimedAnimation.__init__(self, fig, interval = 200, blit = True)

    def _draw_frame(self, framedata):
        i = framedata
        for k in range(self.data['K']):
            self.lines[2*k].set_data(self.data['data']['Principal Component 1'][self.data['clusters_changes'][i] == k],
                                   self.data['data']['Principal Component 2'][self.data['clusters_changes'][i] == k])
            self.lines[2*k+1].set_data(self.data['data']['Principal Component 1'][self.data['clusters_changes'][i] == k].mean(),
                                       self.data['data']['Principal Component 2'][self.data['clusters_changes'][i] == k].mean())

        self._drawn_artists = self.lines

    def new_frame_seq(self):
        return iter(range(self.data['iterations']))

    def _init_draw(self):
        for l in self.lines:
            l.set_data([], [])


def scatterplot_3D(dataframe, centroids):
    fig = plt.figure(figsize = (10.5, 8))
    ax = Axes3D(fig, elev = 24, azim = 18)

    ax.scatter(dataframe['Principal Component 1'], 
               dataframe['Principal Component 2'], 
               dataframe['Principal Component 3'], 
               c = dataframe['Centroids'], cmap = 'Spectral',
               edgecolor = 'grey', s = 50, alpha = 1)

    cmap = mpl.cm.get_cmap('Spectral')

    for i in range(len(centroids)):
        ax.scatter(centroids.iloc[i]["Principal Component 1"], centroids.iloc[i]["Principal Component 2"], 
                   centroids.iloc[i]["Principal Component 3"], 
                   s = 500, c = cmap(random.random()),  marker = 'o', label = 'Centroid', edgecolor = 'black')

    ax.set_title("K-means clusters", fontsize = 15, fontweight = 'bold', pad = 15)
    ax.set_xlabel("Principal Component 1", fontsize = 10, labelpad = 10)
    ax.set_ylabel("Principal Component 2", fontsize = 10, labelpad = 10)
    ax.set_zlabel("Principal Component 3", fontsize = 10, labelpad = 10)
    ax.dist = 10
