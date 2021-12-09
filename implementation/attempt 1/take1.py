# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:18:32 2019

@author: Midhun Krishnan K S
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('movielens\\test.csv', header = None)

df1 = df.fillna(0)

m = df.mean(axis=1)

df1 = df - m

df1 = df1.fillna(0);

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(df1, k = 2) 

Vt = Vt.transpose()

n = np.shape(df1)[0]
m = np.shape(df1)[1]
x =[]
y = []
k = 0
for i in range(n):
    for j in range(m):
        if(np.isnan(df[j][i]) == False):
            z = []
            z.append(U[i][0])
            z.append(Vt[j][0])
            x.append(z)
            y.append(df[j][i])