#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:36:20 2022

@author: joanguich
"""




import pandas as pd
import numpy as np
import kmapper as km
import sklearn
from sklearn import ensemble

import glob

import matplotlib.pyplot as plt





df = pd.read_csv("/Users/joanguich/Documents/mates/TFG/noves_dades/DADES_FIS19.csv", usecols = [0], nrows=0)



path = r'/Users/joanguich/Documents/mates/TFG/noves_dades' # use your path
all_files = glob.glob(path + "/*.csv")


#print(df)

li = []



for filename in all_files:
    df1 = pd.read_csv(filename, usecols = [0, 2, 3, 4, 5])
    df = pd.concat([df, df1])
    
    
df.rename(columns={"Unnamed: 0": "Zona"}, inplace=True)


Zona2 = [0] * (len(df))



df.loc[df['Zona'].str.contains('base'), 'Zona'] = 'base'
df.loc[df['Zona'].str.contains('media'), 'Zona'] = 'media'
df.loc[df['Zona'].str.contains('apical'), 'Zona'] = 'apical'

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


dfEpi = df.loc[df['EndoEpi'] == 1]
dfEndo = df.loc[df['EndoEpi'] == 0]


feature_names = [c for c in df.columns if c not in ["EndoEpi", "Zona", "Porc"]]



X = np.array(dfEndo[feature_names])
Y = df['Zona']


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



df2 = pd.DataFrame(lens, columns = ['Column_A','Column_B'])


"""

df2 = pd.DataFrame(lens, columns = ['Column_A'])

ar = 0 * len(lens)

df2['Column_B'] = ar
"""

df2['Zona'] = dfEndo['Zona'].to_numpy()




colors = {'base' : 'tab:red','media' : 'blue', 'apical' : 'tab:green'}

"""

colors = {0 : 'tab:red', 1 : 'tab:blue'} 

"""

# PLOT DELS VALORS FILTRATS

threedee = plt.subplot()
threedee.scatter(df2['Column_A'], df2['Column_B'], c = df2['Zona'].map(colors))

threedee.set_xlabel('Column_A')
threedee.set_ylabel('Column_B')


#LA LLEGENDA TAMPOC FUNCIONA DEL TOT
#plt.legend(title = 'Porc', loc="upper right")

plt.show()




mapper = km.KeplerMapper()



graph = mapper.map(
    lens,
    X,
    cover=km.Cover(n_cubes=4, perc_overlap=0.375),
    clusterer=sklearn.cluster.KMeans(n_clusters=3),
)

# Visualization
mapper.visualize(
    graph,
    path_html="prova1.html",
    title="Arritmies - UB",
    color_values = lens,
    color_function_name = ["PC1", "PC2"],
    custom_tooltips= np.array(df["Zona"]), 
    node_color_function=["mean", "std", "median", "max"],
)




