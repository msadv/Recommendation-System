# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

ratings = pd.read_csv('ratings.dat', 
                    sep='::',
                    engine = 'python',
                    usecols = [0, 1, 2],
                    names=['user_id', 'movie_id', 'rating'])

ratings.head()

ratings['user_id'] = ratings['user_id'] - 1
ratings['movie_id'] = ratings['movie_id'] - 1
ratings.head()

Ratings1 = ratings.pivot(index = 'user_id', columns = 'movie_id', values = 'rating')
Ratings1.head()

a = []
for i in range(3952):
    a.append(i)

Ratings1 = Ratings1.reindex(columns=a)

Ratings1

Ratings = ratings.pivot(index = 'user_id', columns = 'movie_id', values = 'rating').fillna(0)
Ratings = Ratings.reindex(columns=a)
Ratings.head()

Ratings.mean(axis = 0)

user_ratings_mean = Ratings.mean(axis = 0)
Ratings_demeaned = Ratings - user_ratings_mean

Ratings_demeaned

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(Ratings_demeaned, k = 5)

sigma = np.diag(sigma)

sigma

Vt

U

np.shape(U)

Vt = Vt.transpose()

np.shape(Vt)

import sys
np.set_printoptions(threshold = sys.maxsize)

n = np.shape(Ratings)[0]
m = np.shape(Ratings)[1]
print(n)
print(m)
x =[]
y = []
k = 0
for i in range(n):
    for j in range(m):
#         print("hi")
        if(np.isnan(Ratings1[j][i]) == False):
#             print("not printed")
            print("i j k ", i, j, k)
            z = []
            for k in range(5):
                z.append(U[i][k])
            for k in range(5):
                z.append(Vt[j][k])
            print("printed")
            x.append(z)
            if j in Ratings1.columns: 
                y.append(Ratings1[j][i])
            else:
                print("not found")
            print("not printed")
#             print(Ratings[j][i], U[i], Vt[j])

print(np.shape(x))
print(np.shape(y))

y = y.reshape(-1, 1)

x = np.array(x)

y = np.array(y)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim = 10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x, y, epochs = 20)
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))