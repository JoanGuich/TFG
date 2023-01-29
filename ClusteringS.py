#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 17:12:08 2022

@author: joanguich

WE PLOT DIFFERENT CLUSTERING ALGORITHMS FOR THE POINT CLOUD AFTER USING THE FIRST 2 PRINCIPAL COMPONENTS TO FILTER
"""


# k-means clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
#from sklearn.cluster import KMeans
from matplotlib import pyplot

import pandas as pd

import glob

import numpy as np
import sklearn
from sklearn import ensemble

from sklearn.mixture import GaussianMixture

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
Y = df['Zona']


#lens = mapper.fit_transform(X, projection="l2norm")
lens = sklearn.decomposition.PCA(n_components=2).fit_transform(X)


#label = sklearn.cluster.KMeans(n_clusters = 3).fit_predict(lens)
#label = sklearn.cluster.AgglomerativeClustering(n_clusters=3).fit_predict(lens)
#label = sklearn.cluster.AffinityPropagation(damping = 0.8665).fit_predict(lens)
#label = sklearn.cluster.Birch(threshold=0.000001, n_clusters=3).fit_predict(lens)
label = sklearn.cluster.DBSCAN(eps=2.2, min_samples=3).fit_predict(lens)
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










