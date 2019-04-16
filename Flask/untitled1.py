#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:41:12 2019

@author: samas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r"/Users/anaconda/Desktop/Wholesale customers data.csv")
df2=df[['Fresh','Milk']]
df2.shape


from sklearn.cluster import KMeans
model=KMeans(n_clusters=3,random_state=5)
model.fit(df2)

model.cluster_centers_

plt.hist(model.labels_)

df2['cluster']=model.labels_

plt.figure(figsize=(12,5))

plt.scatter(x='Fresh',y='Milk',c='r',
            data=df2[df2['cluster']==0],
                     label="Low Fresh MEdium Milk")

plt.scatter(x='Fresh',y='Milk',c='g',
            data=df2[df2['cluster']==1],
                     label="High Fresh High Milk")

plt.scatter(x='Fresh',y='Milk',c='b',
            data=df2[df2['cluster']==2],
                     label="Medium Fresh Low Milk")

plt.legend()
plt.show()

df3=df[['Fresh','Milk','Grocery','Frozen']]
model=KMeans(n_clusters=4,verbose=True)
model.fit(df3)

model.cluster_centers_

plt.hist(model.labels_)

