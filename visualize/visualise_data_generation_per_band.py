# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:28:48 2021
@author: AkibZaman
"""
import pandas as pd
# import numpy as np


data = pd.read_csv("clean_data-visualise.csv")


unique_val_band2=data.B2Reft2.unique()
unique_val_band3= data.B3Reft3.unique()
unique_val_band4=data.B4Reft4.unique()
unique_val_band5= data.B5Reft5.unique()
unique_val_band6=data.B6Reft6.unique()
unique_val_band7= data.B7Reft7.unique()

unique_val_band2.sort()
unique_val_band3.sort()
unique_val_band4.sort()
unique_val_band5.sort()
unique_val_band6.sort()
unique_val_band7.sort()

# df = pd.DataFrame(ans, names='0,01')
dataset_band2=pd.DataFrame()
dataset_band3=pd.DataFrame()
dataset_band4=pd.DataFrame()
dataset_band5=pd.DataFrame()
dataset_band6=pd.DataFrame()
dataset_band7=pd.DataFrame()
# df['Address'] = address

# ###Asigning the Variables
# list=[]
# for i,item in zip(range(18),datadict):
#   vars()["p"+list[i]]=datadict[item]

#test=data[['Phosphorus','Potassium','Boron','Calcium','Magnesium','Manganese']].loc[data['B2Reft2'] == 0.01].mean(axis=0)

for x in unique_val_band2:
    temp=data[['Phosphorus','Potassium','Boron','Calcium','Magnesium','Manganese']].loc[data['B2Reft2'] == x].mean(axis=0)
    var=str(x)
    dataset_band2[x]=temp
    
    
for x in unique_val_band3:
    temp=data[['Phosphorus','Potassium','Boron','Calcium','Magnesium','Manganese']].loc[data['B3Reft3'] == x].mean(axis=0)
    var=str(x)
    dataset_band3[x]=temp
    
for x in unique_val_band4:
    temp=data[['Phosphorus','Potassium','Boron','Calcium','Magnesium','Manganese']].loc[data['B4Reft4'] == x].mean(axis=0)
    var=str(x)
    dataset_band4[x]=temp
    
for x in unique_val_band5:
    temp=data[['Phosphorus','Potassium','Boron','Calcium','Magnesium','Manganese']].loc[data['B5Reft5'] == x].mean(axis=0)
    var=str(x)
    dataset_band5[x]=temp
    
for x in unique_val_band6:
    temp=data[['Phosphorus','Potassium','Boron','Calcium','Magnesium','Manganese']].loc[data['B6Reft6'] == x].mean(axis=0)
    var=str(x)
    dataset_band6[x]=temp
    
for x in unique_val_band7:
    temp=data[['Phosphorus','Potassium','Boron','Calcium','Magnesium','Manganese']].loc[data['B7Reft7'] == x].mean(axis=0)
    var=str(x)
    dataset_band7[x]=temp



dataset_band2.to_csv("visualize/band2.csv")
dataset_band3.to_csv("visualize/band3.csv")
dataset_band4.to_csv("visualize/band4.csv")
dataset_band5.to_csv("visualize/band5.csv")
dataset_band6.to_csv("visualize/band6.csv")
dataset_band7.to_csv("visualize/band7.csv")
