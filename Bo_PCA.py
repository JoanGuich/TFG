#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:09:11 2022

@author: joanguich
"""

import numpy as np

import glob

import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import scale # Data scaling
from sklearn import decomposition #PCA
import plotly.express as px


path = r'/Users/joanguich/Documents/mates/TFG/noves_dades/valors_bons' # use your path
all_files = glob.glob(path + "/*.csv")


df = pd.read_csv("/Users/joanguich/Documents/mates/TFG/noves_dades/valors_bons/Bo_FIS19.csv", usecols = [0], nrows=0)

li = []


print(all_files)
for filename in all_files:
    if 'FIS13' in filename:
        df1 = pd.read_csv(filename, usecols = [0, 1, 2, 3, 4], skiprows = [4])
        df = pd.concat([df, df1])
    else:
        df1 = pd.read_csv(filename, usecols = [0, 1, 2, 3, 4])
        df = pd.concat([df, df1])
    
    
df.rename(columns={"Unnamed: 0": "Zona"}, inplace=True)

print(df['Zona'])

EndoEpi = [0] * ((len(df)))
Porc = [0] * ((len(df)))

EndoEpi[0] = 1
EndoEpi[2] = 1

for i in range(int(len(df)) - 3): #Dividim per 12 ja que cada posició té 6 paramètres, i després de cada EPI ve un ENDO. (És a dir, als 6 EPIs els hi assignem 1 i els 6 següents ENDOs 0)
     if (i%2) == 0:
        EndoEpi[i+3] = 1
      
for i in range(15):
    Porc[i] = 0

for i in range(5): #recorda que cada porc té 96 columnes de dades, i que 576/96 = 6, que és el nombre de porcs que tenim
    for j in range(16):
        Porc[j + (i+1)*16 - 1] = i + 1
"""
EndoEpi = [0] * ((len(df)))
Porc = [0] * ((len(df)))

for i in range(int(len(df))): #Dividim per 12 ja que cada posició té 6 paramètres, i després de cada EPI ve un ENDO. (És a dir, als 6 EPIs els hi assignem 1 i els 6 següents ENDOs 0)
    if (i%2) == 0:
        EndoEpi[i] = 1
      
  
for i in range(6): #recorda que cada porc té 96 columnes de dades, i que 576/96 = 6, que és el nombre de porcs que tenim
    for j in range(16):
        Porc[j + i*16] = i
        
"""        

df['EndoEpi'] = EndoEpi
df['Porc'] = Porc


X = np.column_stack((df['DPDT+'], df['DPDT-'], df['LV'], df['FA']))
Y = df['Zona']


features = {'DPDT+', 'DPDT-', 'LV', 'FA'}


X = scale(X)

pca = decomposition.PCA(n_components=3)
pca.fit(X)

scores = pca.transform(X)

scores_df = pd.DataFrame(scores, columns=['PC1', 'PC2', 'PC3'])
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

loadings = pca.components_.T
df_loadings = pd.DataFrame(loadings, columns=['PC1', 'PC2','PC3'], index=features)
print(df_loadings)


explained_variance = pca.explained_variance_ratio_


print(explained_variance)


explained_variance = np.insert(explained_variance, 0, 0)

cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))

pc_df = pd.DataFrame(['','PC1', 'PC2', 'PC3'], columns=['PC'])
explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'])
cumulative_variance_df = pd.DataFrame(cumulative_variance, columns=['Cumulative Variance'])

df_explained_variance = pd.concat([pc_df, explained_variance_df, cumulative_variance_df], axis=1)
print(df_explained_variance)


fig = px.bar(df_explained_variance, 
             x='PC', y='Explained Variance',
             text='Explained Variance',
             width=800)

fig = px.scatter_3d(df_scores, x='PC1', y='PC2', z='PC3',
              color='Zona')


import plotly.io as pio

pio.renderers.default='browser'
fig.show()

