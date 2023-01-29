#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 19:16:21 2022

@author: joanguich

WE TRY DIFFERENT FILTER FUNCTIONS FOR OUR DATASET
"""

import pandas as pd
import numpy as np
import kmapper as km
import sklearn
from sklearn import ensemble
import mapper
 

import glob

import matplotlib.pyplot as plt

import plotly.express as px






df = pd.read_csv("/Users/joanguich/Documents/mates/TFG/noves_dades/DADES_FIS19.csv", usecols = [0], nrows=0)



path = r'/Users/joanguich/Documents/mates/TFG/noves_dades' # use your path
all_files = glob.glob(path + "/*.csv")


#print(df)

li = []



for filename in all_files:
    df1 = pd.read_csv(filename, usecols = [0, 2, 3, 4, 5])
    df = pd.concat([df, df1])
    
    
df.rename(columns={"Unnamed: 0": "Zona"}, inplace=True)



EndoEpi = [0] * ((len(df)))
Porc = [0] * ((len(df)))

for i in range(int(len(df)/12)): #Dividim per 12 ja que cada posició té 6 paramètres, i després de cada EPI ve un ENDO. (És a dir, als 6 EPIs els hi assignem 1 i els 6 següents ENDOs 0)
    for j in range(6):
        EndoEpi[j + i*12] = 1
      
  
for i in range(int(len(df)/96)): #recorda que cada porc té 96 columnes de dades, i que 576/96 = 6, que és el nombre de porcs que tenim
    for j in range(96):
        Porc[j + i*96] = i
        
    
df['EndoEpi'] = EndoEpi
df['Porc'] = Porc


feature_names = [c for c in df.columns if c not in ["EndoEpi", "Zona", "Porc"]]



X = np.array(df[feature_names])
Y = df['EndoEpi']





#lens = km.KeplerMapper().fit_transform(X, projection="mean")
#lens = km.KeplerMapper().fit_transform(X, projection="max")
#lens = km.KeplerMapper().fit_transform(X, projection="knn_distance_10")
#lens = sklearn.manifold.TSNE(n_components=2, init='pca', perplexity = 75, n_iter = 5000, metric = 'euclidean').fit_transform(X)
#lens = sklearn.manifold.MDS(n_components=2, metric = True).fit_transform(X)
#lens = sklearn.manifold.SpectralEmbedding(n_components=2).fit_transform(X)
#lens = sklearn.manifold.LocallyLinearEmbedding(n_components=2).fit_transform(X)
#lens = sklearn.manifold.LocallyLinearEmbedding(n_components=2, method = 'hessian').fit_transform(X)
#lens = sklearn.manifold.LocallyLinearEmbedding(n_components=2, method = 'modified').fit_transform(X)
#lens = sklearn.manifold.Isomap(n_components=2).fit_transform(X)
#lens = sklearn.decomposition.KernelPCA(n_components=2).fit_transform(X)
lens = sklearn.decomposition.PCA(n_components=2).fit_transform(X)
#lens = sklearn.decomposition.TruncatedSVD(n_components=2).fit_transform(X)
#lens = sklearn.decomposition.FastICA(n_components=2).fit_transform(X)
#lens = sklearn.decomposition.IncrementalPCA(n_components=2).fit_transform(X)
#lens = sklearn.decomposition.SparsePCA(n_components=2).fit_transform(X)
#lens = sklearn.decomposition.PCA(n_components=2, svd_solver = 'arpack').fit_transform(X)      #svd_solver{‘auto’, ‘full’, ‘arpack’, ‘randomized’}, default=’auto’
#lens = mapper.filters.eccentricity(X, exponent=1.0, metricpar={}, callback=None)
#lens = mapper.filters.Gauss_density(X, sigma = 10, metricpar={}, callback=None)
#lens = mapper.filters.distance_to_measure(X, 10, metricpar={}, callback=None)
#lens = mapper.filters.graph_Laplacian(X, 0.5, n=1, k=1, weighted_edges=False, sigma_eps=1., normalized=True, metricpar={}, verbose=True, callback=None)
#lens = mapper.filters.dm_eigenvector(X, k=3, metricpar={}, verbose=True, callback=None)




df2 = pd.DataFrame(lens, columns = ['Column_A','Column_B'])


"""
df2 = pd.DataFrame(lens, columns = ['Column_A'])

df2['Column_B'] =  lens2


"""

"""

df2 = pd.DataFrame(lens, columns = ['Column_A'])

ar = 0 * len(lens)

df2['Column_B'] = ar

"""




df2['Zona'] = df['Zona'].to_numpy()





colors = {'1-EPI_base_anterior' : 'tab:red','3-EPI_base_lateral' : 'tab:red', '5-EPI_base_posterior' : 'tab:red', 
          '2-ENDO_base_anterior' : 'tab:red', '4-ENDO_base_lateral' : 'tab:red', '6-ENDO_base_posterior' : 'tab:red', 
          '7-EPI_media_anterior' : 'blue', '9-EPI_media_lateral' : 'blue', '11-EPI_media_posterior' : 'blue', 
         '8-ENDO_media_anterior' : 'blue', '10-ENDO_media_lateral' : 'blue', '12-ENDO_media_posterior' : 'blue', 
         '13-EPI_apical1' : 'tab:green', '14-ENDO_apical1' : 'tab:green', '15-EPI_apical2' : 'tab:green', '16-ENDO_apical2' : 'tab:green'}
"""

colors = {0 : 'tab:red', 1 : 'tab:blue'} 
"""



"""
fig = px.scatter(lens, x=0, y=1, color=df['Zona'])
fig.show()


"""
"""""
threedee = plt.subplot()
threedee.scatter(df2['Column_A'], df2['Column_B'], c = df2['Zona'].map(colors))

threedee.set_xlabel('Column_A')
threedee.set_ylabel('Column_B')


#LA LLEGENDA TAMPOC FUNCIONA DEL TOT
#plt.legend(title = 'Porc', loc="upper right")

plt.show()

"""


plt.scatter(df2['Column_A'], df2['Column_B'], c = df2['Zona'].map(colors))

plt.legend(handles = ['Base', 'Media', 'Apical'], label = 'colors')




plt.show()














