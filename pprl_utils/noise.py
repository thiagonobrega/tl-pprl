#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   noise.py
@Time    :   2022/05/23 09:54:02
@Author  :   Thiago Nóbrega 
@Contact :   thiagonobrega@gmail.com
'''
import numpy as np
import pandas as pd

def laplace(x, μ, b):
    return 1 / (2 * b) * np.exp(-np.abs(x - μ) / b)


def add_noise(df,noise,epislon,columns=['dice', 'jaccard', 'overlap', 'hamming', 'entropy']):
    
    for col in columns:
        media = df[col].mean()
        x = df[col].to_numpy()

        dist1 = laplace(x, μ, 1 / ε)
