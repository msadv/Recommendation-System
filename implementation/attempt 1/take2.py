# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:25:11 2019

@author: Midhun Krishnan K S
"""

import pandas as pd
import numpy as np

ratings = pd.read_csv('C:/Users/Midhun Krishnan K S/Downloads/7th sem/recommendation system/implementation/attempt 1/movielens/dat/ratings.dat', 
                    sep='::',
                    engine = 'python',
                    usecols = [0, 1, 2],
                    names=['user_id', 'movie_id', 'rating'])

Ratings = ratings.pivot(index = 'user_id', columns = 'movie_id', values = 'rating').fillna(0)

Ratings.head()

R = Ratings.values
user_ratings_mean = np.mean(R, axis = 1)
Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(Ratings_demeaned, k = 50)