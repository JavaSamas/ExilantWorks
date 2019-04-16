g#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:25:50 2018

@author: samas
"""

import glob
import pandas as pd


#Reading Multiple CSV into one CSV
path=r'/Users/samas/Downloads/Predictive-Maintenance-master/dataset/csv/original'

path1='/Users/samas/Downloads/Predictive-Maintenance-master/dataset/csv/Sorting'

df1=pd.read_csv(path+'/1705_FT-202B.csv')
df1['time']=pd.to_datetime(df1['time'])
df1=df1.resample('3H', on='time').mean()

df1.to_csv(path1+'/1705_FT-202B.csv')

'''df1=glob.glob(path+'/1705_FT-202B.csv')
print(df1)
for df1 in df1:
    print(df1)
    df1=pd.read_csv(df1)
 
ts.resample(rule='3H', closed='left', label='left', base=17).sum()
df1.to_csv(path1+'/df1.csv',index=False)
   

df1=pd.read_csv(r'/Users/samas/Downloads/Predictive-Maintenance-master/dataset/csv/original/1705_FT-202B.csv')
df2=pd.read_csv(r'/Users/samas/Downloads/Predictive-Maintenance-master/dataset/csv/original/1705_FT-204B.csv')

mergedStuff = pd.merge(df1, df2, on=['time'], how='outer')
mergedStuff.head()'''


sensor_names = ['MAIN_FILTER_IN_PRESSURE','MAIN_FILTER_OIL_TEMP','MAIN_FILTER_OUT_PRESSURE','OIL_RETURN_TEMPERATURE',
    'TANK_FILTER_IN_PRESSURE','TANK_FILTER_OUT_PRESSURE','TANK_LEVEL','TANK_TEMPERATURE','FT-202B',
    'FT-204B','PT-203','PT-204']



for names in sensor_names:
    print(names) 
    filelist=glob.glob(path+'/*names.csv')
    print(filelist)
    for file in filelist:
        name=pd.read_csv(file)
        
print("Hello")
