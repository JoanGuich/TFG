#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:36:05 2022

@author: joanguich

WE APPLY THE MAPPER ALGORITHM TO OUR DATASET WITH THE SELECTED PARAMETERS
"""


import pandas as pd
import numpy as np
import kmapper as km
import sklearn
from sklearn import ensemble

import glob

import matplotlib.pyplot as plt








df = pd.read_csv("/Users/joanguich/Documents/mates/TFG/noves_dades/valors_bons/Bo_FIS19.csv", usecols = [0], nrows=0)



path = r'/Users/joanguich/Documents/mates/TFG/noves_dades/valors_bons' # use your path
all_files = glob.glob(path + "/*.csv")


#print(df)

li = []



for filename in all_files:
    df1 = pd.read_csv(filename, usecols = [0, 2, 3, 4])
    df = pd.concat([df, df1])
    
    
df.rename(columns={"Unnamed: 0": "Zona"}, inplace=True)

df.loc[df['Zona'].str.contains('base'), 'Zona'] = 'base'
df.loc[df['Zona'].str.contains('media'), 'Zona'] = 'media'
df.loc[df['Zona'].str.contains('apical'), 'Zona'] = 'apical'




EndoEpi = [0] * ((len(df)))
Porc = [0] * ((len(df)))

for i in range(int(len(df)/2)): #Dividim per 2 ja que EPI i ENDO es van intercalant. EPI = 1, ENDO = 0
    EndoEpi[2*i] = 1
      
  
for i in range(6): #recorda que cada porc té 96 columnes de dades, i que 576/96 = 6, que és el nombre de porcs que tenim
    for j in range(16):
        Porc[i*16 + j] = i
   
        

df['EndoEpi'] = EndoEpi
df['Porc'] = Porc



feature_names = [c for c in df.columns if c not in ["Diagnosis", "Zona", "Porc"]]


X = np.array(df[feature_names])
y = np.array(df["Zona"])


#lens = sklearn.manifold.TSNE(n_components=2, init='pca', perplexity = 75, n_iter = 5000).fit_transform(X)
lens = sklearn.decomposition.PCA(n_components=2, svd_solver = 'arpack').fit_transform(X)


label = sklearn.cluster.KMeans(n_clusters = 3).fit_predict(lens)
#label = sklearn.cluster.AgglomerativeClustering(n_clusters=3).fit_predict(lens)
#label = sklearn.cluster.AffinityPropagation(damping = 0.8665).fit_predict(lens)
#label = sklearn.cluster.Birch(threshold=0.000001, n_clusters=3).fit_predict(lens)
#label = sklearn.cluster.DBSCAN(eps=1.4, min_samples=3).fit_predict(lens)
#label = sklearn.cluster.MiniBatchKMeans(n_clusters = 3).fit_predict(lens)
#label = sklearn.cluster.MeanShift().fit_predict(lens)
#label = sklearn.cluster.OPTICS(eps=4.5, min_samples=4).fit_predict(lens)
#label = sklearn.cluster.SpectralClustering(n_clusters=3).fit_predict(lens)
#label = GaussianMixture(n_components=3).fit_predict(lens)




#Getting unique labels
 
u_labels = np.unique(label)



#plotting the results:

import matplotlib.pyplot as plt




for i in u_labels:
    plt.scatter(lens[label == i , 0] , lens[label == i , 1] , label = i)
    

plt.legend()
plt.show()


df.loc[df['Zona'].str.contains('base'), 'Color'] = 0.0
df.loc[df['Zona'].str.contains('media'), 'Color'] = 1.0
df.loc[df['Zona'].str.contains('apical'), 'Color'] = 2.0




mapper = km.KeplerMapper()



graph = mapper.map(
    lens,
    X,
    cover=km.Cover(n_cubes=5, perc_overlap=0.425),
    clusterer=sklearn.cluster.KMeans(n_clusters=3),
)

# Visualization
mapper.visualize(
    graph,
    path_html="prova1.html",
    title="Arritmies - UB",
    #color_values = lens,
    color_function = np.array(df['Color']),
    color_function_name = "Zona",
    custom_tooltips= np.array(df["Zona"]), 
    node_color_function=["mean", "std", "median", "max"],
)
















