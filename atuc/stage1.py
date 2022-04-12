#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   stage1.py
@Time    :   2021/03/03 09:14:32
@Author  :   Thiago Nóbrega 
@Contact :   thiagonobrega@gmail.com
'''

import pandas as pd
from sklearn.model_selection import train_test_split


def s1_prepareTrainingData_v1(s, t, ts=.3, atributos=[2, 3, 4, 5, 6]):
    '''
    Remove os IDs e elementos que nao serao utilizados para treinar os modelos sobre os dados da matriz de similaridade
    retorna X e Y de teste e treino 7/3
    @param s source
    @param t target

    @return X treino
    @return X test
    @return Y treino
    @return Y test
    '''
    ds = s.iloc[:, atributos]
    dt = t.iloc[:, atributos]
    ds['ds'] = 0
    dt['ds'] = 1
    df = pd.concat([ds, dt])
    y = df.pop('ds')

    return train_test_split(df, y, test_size=ts)


def s1_prepareTrainingData(s , t , ts=.3 , atributos=[2 , 3 , 4 , 5 , 6]):
    '''
    Remove os IDs e elementos que nao serao utilizados para treinar os modelos sobre os dados da matriz de similaridade
    retorna X e Y de teste e treino 7/3
    @param s source
    @param t target

    @return X treino
    @return X test
    @return Y treino
    @return Y test
    '''
    #     ds = s.iloc[:, atributos]
    #     dt = t.iloc[:, atributos]

    ds = s.iloc[: , atributos].round(2)
    dt = t.iloc[: , atributos].round(2)

    ds['i'] = ds.iloc[: , 0:].astype(str).apply(lambda x: ' '.join(x) , axis=1)
    dt['i'] = dt.iloc[: , 0:].astype(str).apply(lambda x: ' '.join(x) , axis=1)
    ds = ds.set_index(['i'])
    dt = dt.set_index(['i'])

    a = list(ds.index.unique())
    b = list(dt.index.unique())

    u = set(a).intersection(set(b))
    dt.drop(list(u) , inplace=True)
    ds.reset_index(drop=True , inplace=True)
    dt.reset_index(drop=True , inplace=True)

    ds['ds'] = 0
    dt['ds'] = 1
    df = pd.concat([ds , dt])
    y = df.pop('ds')

    return train_test_split(df , y , test_size=ts)


def s1_trainDSClassifier(X_train, Y_train, model):
    '''
    Treina o modelo em uma regressão logisticax
    '''
    model.fit(X_train, Y_train)
    return model


def s1_selectData2Train(modelo, source_ds, lambada_corte=.51, atributos=[2, 3, 4, 5], col_gab=[7]):
    '''
        Seleciona os dados para treinarem o classificador target

        @return dados que serao utilizados pas1_trainDSClassifierra o modelo com o gabarito [dice,jaccard,overlap,hamming,entropy,is_match,pp]
        @return pesos que devam ser dados aos pares de entidades

    '''

    pp = modelo.predict_proba(source_ds.iloc[:, 2:(2 + len(atributos))])
    fds = source_ds.copy()
    fds['pp'] = pp[:, 1].tolist()

    return fds[(fds['pp'] >= lambada_corte)].iloc[:, atributos + col_gab + [len(fds.columns) - 1]]  # ,fds['pp']
