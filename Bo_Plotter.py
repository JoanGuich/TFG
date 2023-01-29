#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 09:45:14 2022

@author: joanguich
"""

import glob

import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


path = r'/Users/joanguich/Documents/mates/TFG/noves_dades/valors_bons' # use your path
all_files = glob.glob(path + "/*.csv")


df = pd.read_csv("/Users/joanguich/Documents/mates/TFG/noves_dades/valors_bons/Bo_FIS19.csv", usecols = [0], nrows=0)

li = []


print(all_files)
for filename in all_files:
    df1 = pd.read_csv(filename, usecols = [0, 1, 3, 4])
    df = pd.concat([df, df1])
    
    
df.rename(columns={"Unnamed: 0": "Zona"}, inplace=True)


EndoEpi = [0] * ((len(df)))
Porc = [0] * ((len(df)))

for i in range(int(len(df))): #Dividim per 12 ja que cada posició té 6 paramètres, i després de cada EPI ve un ENDO. (És a dir, als 6 EPIs els hi assignem 1 i els 6 següents ENDOs 0)
    if (i%2) == 0:
        EndoEpi[i] = 1
      
  
for i in range(6): #recorda que cada porc té 96 columnes de dades, i que 576/96 = 6, que és el nombre de porcs que tenim
    for j in range(16):
        Porc[j + i*16] = i
        


df['EndoEpi'] = EndoEpi
df['Porc'] = Porc

sdf0 = df[df['EndoEpi']==0]
sdf1 = df[df['EndoEpi']==1]

colors = {'1-EPI_base_anterior' : 'tab:red','3-EPI_base_lateral' : 'tab:red', '5-EPI_base_posterior' : 'tab:red', 
          '2-ENDO_base_anterior' : 'tab:red', '4-ENDO_base_lateral' : 'tab:red', '6-ENDO_base_posterior' : 'tab:red', 
          '7-EPI_media_anterior' : 'tab:blue', '9-EPI_media_lateral' : 'tab:blue', '11-EPI_media_posterior' : 'tab:blue', 
         '8-ENDO_media_anterior' : 'tab:blue', '10-ENDO_media_lateral' : 'tab:blue', '12-ENDO_media_posterior' : 'tab:blue', 
         '13-EPI_apical1' : 'tab:green', '14-ENDO_apical1' : 'tab:green', '15-EPI_apical2' : 'tab:green', '16-ENDO_apical2' : 'tab:green'}

colors2 = {'1-EPI_base_anterior' : 'pink','3-EPI_base_lateral' : 'pink', '5-EPI_base_posterior' : 'pink', 
          '2-ENDO_base_anterior' : 'pink', '4-ENDO_base_lateral' : 'pink', '6-ENDO_base_posterior' : 'pink', 
          '7-EPI_media_anterior' : 'aqua', '9-EPI_media_lateral' : 'aqua', '11-EPI_media_posterior' : 'aqua', 
         '8-ENDO_media_anterior' : 'aqua', '10-ENDO_media_lateral' : 'aqua', '12-ENDO_media_posterior' : 'aqua', 
         '13-EPI_apical1' : 'tab:yellow', '14-ENDO_apical1' : 'tab:yellow', '15-EPI_apical2' : 'tab:yellow', '16-ENDO_apical2' : 'tab:yellow'}


#colors = {0 : 'tab:red', 1 : 'tab:blue', 2 : 'tab:green', 3 : 'black', 4 : 'tab:purple', 5 : 'yellow'} 





#colors = {'^' : 'tab:red', 'o' : 'tab:blue'}



threedee = plt.subplot(projection='3d')
threedee.scatter(sdf0['DPDT+'], sdf0['LV'], sdf0['FA'], c = sdf0['Zona'].map(colors), marker = "^")

threedee.scatter(sdf1['DPDT+'], sdf1['LV'], sdf1['FA'], c = sdf1['Zona'].map(colors), marker = "s")
threedee.set_xlabel('DPDT+')
threedee.set_ylabel('LV')
threedee.set_zlabel('FA')

#LA LLEGENDA TAMPOC FUNCIONA DEL TOT
#plt.legend(title = 'Porc', loc="upper right")

plt.show()


