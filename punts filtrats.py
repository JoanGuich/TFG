#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:51:00 2022

@author: joanguich

WE PLOT THE POINT CLOUD FILTERED BY THE FIRST TWO PRINCIPAL COMPOMNENTS 
"""

import pandas as pd
import numpy as np
import kmapper as km
import sklearn
from sklearn import ensemble

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

#from sklearn.preprocessing import StandardScaler

X = np.array(df[feature_names])
Y = df['Zona']

#X = StandardScaler().fit_transform(X)



#lens = mapper.fit_transform(X, projection="l2norm")
lens = sklearn.decomposition.PCA(n_components=2).fit_transform(X)



scores_df = pd.DataFrame(lens, columns=['PC1', 'PC2'])
print(scores_df)

Y_label = []

for i in Y:
    if i == '1-EPI_base_anterior' or i == '2-ENDO_base_anterior' or i == '3-EPI_base_lateral' or i == '4-ENDO_base_latera' or i == '5-EPI_base_posterior' or i == '6-ENDO_base_posterior':
        Y_label.append('Base')
    elif i == '7-EPI_media_anterior' or i == '8-ENDO_media_anterior' or i == '9-EPI_media_lateral' or i == '10-ENDO_media_lateral' or i == '11-EPI_media_posterior' or i == '12-ENDO_media_posterior':
        Y_label.append('Media')
    else:
        Y_label.append('Apical')

'''
for i in Y:
    if i == 1:
       Y_label.append('Epi')
    else:
        Y_label.append('Endo')

'''
'''
for i in Y:
    if i == 0:
        Y_label.append('FIS13')
    elif i == 1:
        Y_label.append('FIS14')
    elif i == 2:
        Y_label.append('FIS15')
    elif i == 3:
        Y_label.append('FIS18')
    elif i == 4:
        Y_label.append('FIS19')
    else:
        Y_label.append('FIS6')
'''


labels = pd.DataFrame(Y_label, columns=['Zona'])

df_scores = pd.concat([scores_df, labels], axis=1)

features = {'DPDT+', 'DPDT-', 'LV', 'FA'}

pca = sklearn.decomposition.PCA(n_components=2)
pca.fit(X)

loadings = pca.components_.T
df_loadings = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=features)
print(df_loadings)


explained_variance = pca.explained_variance_ratio_


print(explained_variance)


explained_variance = np.insert(explained_variance, 0, 0)

cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))

pc_df = pd.DataFrame(['','PC1', 'PC2'], columns=['PC'])
explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'])
cumulative_variance_df = pd.DataFrame(cumulative_variance, columns=['Cumulative Variance'])

df_explained_variance = pd.concat([pc_df, explained_variance_df, cumulative_variance_df], axis=1)
print(df_explained_variance)


fig = px.bar(df_explained_variance, 
             x='PC', y='Explained Variance',
             text='Explained Variance',
             width=800)

fig = px.scatter(df_scores, x='PC1', y='PC2',
              color='Zona')


import plotly.io as pio

pio.renderers.default='browser'
fig.show()

colors = {'1-EPI_base_anterior' : 'tab:red','3-EPI_base_lateral' : 'tab:red', '5-EPI_base_posterior' : 'tab:red', 
          '2-ENDO_base_anterior' : 'tab:red', '4-ENDO_base_lateral' : 'tab:red', '6-ENDO_base_posterior' : 'tab:red', 
          '7-EPI_media_anterior' : 'blue', '9-EPI_media_lateral' : 'blue', '11-EPI_media_posterior' : 'blue', 
         '8-ENDO_media_anterior' : 'blue', '10-ENDO_media_lateral' : 'blue', '12-ENDO_media_posterior' : 'blue', 
         '13-EPI_apical1' : 'tab:green', '14-ENDO_apical1' : 'tab:green', '15-EPI_apical2' : 'tab:green', '16-ENDO_apical2' : 'tab:green'}


plt.scatter(df_scores['PC1'], df_scores['PC2'], c = df['Zona'].map(colors))
plt.legend(handles = ['Base', 'Media', 'Apical'], label = 'colors')

plt.show()


