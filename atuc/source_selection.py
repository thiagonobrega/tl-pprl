#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   source_selection.py
@Time    :   2022/04/03 10:31:46
@Author  :   Thiago Nóbrega 
@Contact :   thiagonobrega@gmail.com
@Desc    :   "Methods related to the feature and source selection stage."
'''
import pandas as pd
import numpy as np
import logging

from tqdm.notebook import trange, tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, matthews_corrcoef

from atuc.utils.file import open_ds, load_data
from atuc.utils import ajustar_treino, showPredictionsSummary
from atuc.stage1 import s1_prepareTrainingData, s1_trainDSClassifier, s1_selectData2Train


def calcular_metricas_dados_relacionado(source_,target_,atts = [2,3,4,5,6],
                                        lr_model_name='Logistic',
                                        _s1_percentualmatch=0.1,
                                        percentual_minimo_de_match=2):
    """
        Calculate the source and target distance according to the BenDavid work [1]

        [1] - Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. 
                A theory of learning from different domains.
                Machine Learning, 79(1-2):151–175, 5 2010. ISSN 0885- 6125. doi: 10.1007/s10994-009-5152-4.
                https://link.springer.com/article/10.1007/s10994-009-5152-4
    """

    X_train, X_test, Y_train, Y_test  = s1_prepareTrainingData(source_,target_,ts=.3,atributos=atts)
    # utilizando apenas regressao logistica
    lr_model = LogisticRegression(random_state=10100,n_jobs=-1)
    
    if lr_model_name == 'Logistic':
        lr_model = LogisticRegression(random_state=10100,n_jobs=-1)
    if lr_model_name == 'SVM':
        svc = LinearSVC(random_state=10100)
        lr_model = CalibratedClassifierCV(base_estimator=svc, cv=2)

    modelo = s1_trainDSClassifier(X_train,Y_train,lr_model)
    
    # metodo que sera alterado
    data_ = s1_selectData2Train(modelo,source_,
                                    lambada_corte=.51,
                                    atributos=atts)

    numero_matches = len(data_[data_['is_match'] == 1])  # salvar
    numero_unmatches = len(data_[data_['is_match'] == 0]) # salvar

    

    if numero_matches < 1:
        raise ValueError("Few Matches " + str(numero_matches))

    if numero_unmatches < numero_matches * percentual_minimo_de_match:
      if numero_unmatches > 0:
        numero_matches = int(numero_unmatches/percentual_minimo_de_match)
        data_m = data_[data_['is_match'] == 1].head(numero_matches)
        data_um = data_[data_['is_match'] == 0]
        data_ = pd.concat([data_m,data_um])
      else:
        # print(numero_matches,'/',numero_unmatches)
        raise ValueError("Few Un Matches : " + str(numero_matches) +"matches / " + str(numero_unmatches) + "unmatches" )

    data , w = ajustar_treino(data_,numero_matches,p=_s1_percentualmatch)

    # escolhe um sample para dh_uu
    menor = min(len(source_),len(target_))
    sample_size_u = int(menor/2)

    #calcula dh_uu
    s = source_.iloc[:, 2:(2 + len(atts))].sample(sample_size_u).copy()
    t = target_.iloc[:, 2:(2 + len(atts))].sample(sample_size_u).copy()
    s['Y'] = 0
    t['Y'] = 1
    U = pd.concat([s,t])
    Ydu = U.pop('Y')
    s.pop('Y')
    t.pop('Y')
    predicao = modelo.predict(U)

    i_s = showPredictionsSummary(modelo.predict(s)).get(0, 0)/len(U)
    i_t = showPredictionsSummary(modelo.predict(t)).get(1, 0)/len(U)

    dh_UU = 2*(1 - ( i_s + i_t ))
    
    # calcula o complemento terceiro termo da equacao (raiz quadrada)
    d = len(atts)
    mm = len(U)
    m = int(mm/2)
    #ou sera log2
    div = 2*d* np.log(mm) + np.log(2/.5)
    complemento = 4*np.sqrt((div/m))
    
    # calcula o priumeiro termo da equacao error s
    y_s1b = data.pop('is_match')
    model_s1b   = LogisticRegression(random_state=10100,n_jobs=-1)
    model_s1b.fit(data,y_s1b.astype('int')) # treinando modelo para o source
    
    t_ = source_.iloc[:,atts + [-1] ].copy()
    Y = t_.pop('is_match').values.astype('int')
    Y_p = model_s1b.predict(t_)
    
    e_s = mean_absolute_error(Y,Y_p)
    error_t = e_s + dh_UU/2 + complemento    
    
    mcc = matthews_corrcoef(Ydu,predicao)

    
    return error_t, e_s,dh_UU/2,complemento, mcc

def search_best_dr(sources,targets,atts = [2,3,4,5,6],lr_model_name='Logistic'):
    """
        search best source and target distance considering the BenDavid work [1]
        1. Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. A theory of learning from different domains. Machine Learning, 79(1-2):151–175, 5 2010. ISSN 0885- 6125. doi: 10.1007/s10994-009-5152-4.
    """
    resultado = []


    for es_ in trange(len(sources),desc='Datasets 1', leave=False):
        for et_ in trange(len(targets),desc='Datasets 2', leave=False):

            s_  = sources[es_][0]
            t_  = targets[et_][0]
            if s_ != t_:
                #print(s_,t_)
                _fs = sources[es_][1]
                _ft = targets[et_][1]
                atts_s = sources[es_][2]
                atts_t = targets[et_][2]
                blk_s = sources[es_][3]
                blk_t = targets[et_][3]

                for fs in _fs: # itera nos arquivos
                    for ft in _ft:
                        source_,target_,log_ds = load_data(s_,t_,fs,ft,atts_s,atts_t,blk_s,blk_t)
                        
                        try:
                            et, e_s, dhh,complemento, mcc = calcular_metricas_dados_relacionado(source_,target_,atts=atts,lr_model_name=lr_model_name)    
                            logs_dr = {'e_s':e_s,'dhh':dhh,'complemento':complemento,'et':et, 'mcc':mcc}
                        # except Exception as e:
                        except ValueError as e:
                            nome = e.__class__.__name__
                            logging.debug("Error",nome,":",s_,t_,fs,ft)
                            logging.debug(e)
                            logs_dr = {'e_s':-1,'dhh':-1,'complemento':-1,'et':-1, 'mcc':-1}
                        
                        resultado.append({**log_ds , **logs_dr})
    return resultado