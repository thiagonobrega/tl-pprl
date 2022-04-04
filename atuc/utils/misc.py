#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   tl_base.py
@Time    :   2022/04/03 08:59:05
@Author  :   Thiago Nóbrega 
@Contact :   thiagonobrega@gmail.com
'''

from decimal import *

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score , recall_score , f1_score , accuracy_score


def showPredictionsSummary(Y):
    '''
     Show the number of matches and unmatches in Y [list]
    :param Y:
    :return:
    '''
    unique, counts = np.unique(Y, return_counts=True)
    return dict(zip(unique, counts))


# data.sort_values('pp',ascending=False)
def calcularTamanho(nm, pm):
    '''
     Metodo para calcular o tamanho do dataset para o metodo abaixo
    :param nm:
    :param pm:
    :return:
    '''
    assert pm < 1
    assert pm > 0
    pu = 1 - pm

    return int((nm * pu) / pm)


def ajustar_treino(dfg, length_matches, p=0.01):
    '''
    Adjust the number of matche and unmatcher (acording to p vairable) to stage1 of the approach.

    :param dfg:
    :param length_matches:
    :param p:
    :return: a dataframe with the features [dice,jaccard,overlap,hamming,entropy,is_match], and their weights
    '''
    s_um = calcularTamanho(length_matches, p)
    # sort
    um = dfg[dfg['is_match'] == 0].sort_values('pp', ascending=False).head(s_um)
    t_ = pd.concat([um, dfg[dfg['is_match'] == 1]])
    w = t_.pop('pp')
    df_ = t_
    del t_
    return df_, w


def round_of_rating(number):
    """Round a number to the closest half integer.
    >>> round_of_rating(1.3)
    1.5
    >>> round_of_rating(2.6)
    2.5
    >>> round_of_rating(3.0)
    3.0
    >>> round_of_rating(4.1)
    4.0"""
    # round_of_rating(0.16)
    x = Decimal(number)
    return (x * 2).quantize(Decimal('.1'), rounding=ROUND_UP) / 2


def generate_logs_s1(_y, _ty, s1_name, s1_sample, model_dt, model_s1, peso='none'):
    nY_g = showPredictionsSummary(_y)
    nY_b = showPredictionsSummary(_ty)

    # caso nao tenha nenhum match
    try:
        actual_1 = nY_g[1]
    except KeyError:
        actual_1 = 0

    base = {'data_train_model': model_dt ,
            'stage1_model': model_s1 ,
            'stage1_strategy': s1_name ,
            'stage1_weight': peso ,
            'stage1_sample_strategy': s1_sample ,
            'precision': precision_score(_y , _ty) ,
            'recall': recall_score(_y , _ty) ,
            'f1': f1_score(_y , _ty) ,
            'accuracy': accuracy_score(_y , _ty) ,
            'predicted_0': nY_b.get(0, 0),
            'predicted_1': nY_b.get(1, 0),
            'actual_0': nY_g[0],
            'actual_1': actual_1
            }
    return base


# metodos que buscam o melhor resultado utilizando o limiar
def executar_limiar(__t, _limiar, metricas=[2, 3, 4, 5, 6]):
    _t = __t.copy()
    _t['mean_y'] = _t.iloc[:, metricas].mean(axis=1)
    _t['y_th'] = 0

    _t.loc[_t['mean_y'] > _limiar, 'y_th'] = 1
    __Y = _t.pop('is_match')
    _Yt = _t.pop('y_th')

    return generate_logs_s1(__Y, _Yt, 'threshold', str(_limiar), 'none', 'none')


def buscar_melhor_limiar(__t, limiares=[.5, .6, .7, .8, .9], _metricas=[2, 3, 4, 5, 6]):
    indice = []
    dados = []

    for i in limiares:
        r = executar_limiar(__t, i)
        indice.append(r['f1'])
        dados.append(r)

    k = indice.index(max(indice))
    return dados[k]

def testar_limiar(__t, limiares=[.7, .8, .9, 1], _metricas=[2, 3, 4, 5, 6]):
    indice = []
    resultados = []

    for i in limiares:
        r = executar_limiar(__t, i)
        resultados.append(r)
    return resultados


#####
##### stage 02

def generate_logs_s2(_y, _ty, model_s1='Logistic'):
    nY_b = showPredictionsSummary(_ty)

    base = {'stage2_model': model_s1,
            's2_precision': precision_score(_y, _ty),
            's2_recall': recall_score(_y, _ty),
            's2_f1': f1_score(_y, _ty),
            's2_accuracy': accuracy_score(_y, _ty),
            's2_predicted_0': nY_b.get(0, 0),
            's2_predicted_1': nY_b.get(1, 0)
            }
    return base
