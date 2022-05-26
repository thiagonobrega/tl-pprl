#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   stage1.py
@Time    :   2021/03/03 09:14:32
@Author  :   Thiago Nóbrega 
@Contact :   thiagonobrega@gmail.com
'''

import time

import numpy as np
from scipy import linalg
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from atuc.utils import ajustar_treino, generate_logs_s1

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

###
### Instance picking
###

def coral(Xs,Xt,lambda_=0.00001):
  cov_Xs = np.cov(Xs, rowvar=False)
  cov_Xt = np.cov(Xt, rowvar=False)

  Cs = cov_Xs + lambda_ * np.eye(Xs.shape[1])
  Ct = cov_Xt + lambda_ * np.eye(Xt.shape[1])

  Cs_sqrt_inv = linalg.inv(linalg.sqrtm(Cs))
  Ct_sqrt = linalg.sqrtm(Ct)
  
  if np.iscomplexobj(Cs_sqrt_inv):
    Cs_sqrt_inv = Cs_sqrt_inv.real
  if np.iscomplexobj(Ct_sqrt):
      Ct_sqrt = Ct_sqrt.real
        
  Xs_emb = np.matmul(Xs, Cs_sqrt_inv)
  Xs_emb = np.matmul(Xs_emb, Ct_sqrt)
  return Xs_emb

def apply_source_adptation(src,target):
  Xt = target.iloc[:,2:-1]
  Xs = src.iloc[:,:-1]
  Xs = coral(Xs,Xt,lambda_=1)
  adjusted_src = src.copy()
  adjusted_src.iloc[:,:-1]=Xs

  # falta corrigir o maiores que 1 e menores 0
  return adjusted_src

###
###
###

def execute_classifier_manufacturing(source_,target_,
                                        ts=.3,
                                        atributos=[2,3,4,5,6],columns_y=[7],
                                        _s1_percentualmatch=0.1,
                                        lr_model_name='Logistic',_lambda=.51,
                                        stage1_model='Logistic',
                                        funcao_w=None,
                                        min_match_number=3,
                                        proporcao_match_nonmatch=2):
    """
        execute the classifier manufactorin stage of AT-UC
        
        - *source_*, source dataset
        - *target_*, target datase
        - *ts*, percentual of the datatset that will be employed as test (second filtering)
        - *attributos*, a list with all attributes (similarity metrics) taht will be employed
        - *lr_model_name*, model that will be employed to train the dataset classifier.
            - Logistic: LogisticRegresion
            - SVM: Suport Vector Machine (LinearSVC)
            - DT: Decision Tree (RandomForestClassifier)
        - *_lambda*, the minimal confidence needed to use the pair to train the target (default=.51)
        - *stage1_model*, model that will be employed to train the dataset classifier
            - Logistic: LogisticRegresion
            - SVM: Suport Vector Machine (LinearSVC)
            - DT: Decision Tree (RandomForestClassifier)
        - min_match_number : minim number of match
        - proporcao_match_nonmatch : proportion of match and unmatch instance
    """

    assert lr_model_name in ['Logistic','SVM','DT','GBC']
    assert stage1_model in ['Logistic','SVM','DT','GBC']

    X_train, X_test, Y_train, Y_test  = s1_prepareTrainingData(source_,target_,ts=.3,atributos=atributos)

    if lr_model_name == 'Logistic':
        lr_model = LogisticRegression(random_state=10100,n_jobs=-1)
    if lr_model_name == 'SVM':
        svc = LinearSVC(random_state=10100)
        lr_model = CalibratedClassifierCV(base_estimator=svc, cv=2)
#         lr_model = SVC(random_state=10100,probability=True)
    if lr_model_name == 'DT':
        lr_model = RandomForestClassifier(n_estimators=10,random_state=101010,n_jobs=-1)
    if lr_model_name == 'GBC':
        lr_model = GradientBoostingClassifier(random_state=101010)

    # treinando modelo para dar peso aos pares
    start_time = time.time()
    modelo = s1_trainDSClassifier(X_train,Y_train,lr_model)
    sep_model_time = time.time() - start_time

    #################################################################
    ### selecionando os dados para serem utilizados no treinamento
    ### target
    #################################################################

    start_time = time.time()
    data_ = s1_selectData2Train(modelo,source_,
                                lambada_corte=_lambda,
                                atributos=atributos)
    select_train_data_time = time.time() - start_time
    
    numero_matches = len(data_[data_['is_match'] == 1])  # salvar
    numero_unmatches = len(data_[data_['is_match'] == 0]) # salvar

    if numero_matches < min_match_number:
        raise ValueError("Few Matches " + str(numero_matches))

    if numero_unmatches < numero_matches * proporcao_match_nonmatch:
        raise ValueError("Few Un Matches : " + str(numero_matches) +"matches / " + str(numero_unmatches) + "unmatches" )

    #### selecionando os dados
    data_naive = data_.iloc[:,:-1]
    start_time = time.time()
    data , w = ajustar_treino(data_,numero_matches,p=_s1_percentualmatch)
    adjust_train_data_time = time.time() - start_time

    ###
    ### regularizacao
    ###
    data_adapted = apply_source_adptation(data,target_)
    
    
    #caso utilizar antes da selecao
    # data_.iloc[:,:-2]
    
    # selecao naive
    Y_naive = data_naive.pop('is_match').values
    y_s1b = data.pop('is_match')
    y_s1r = data_adapted.pop('is_match')

    #treinando classificador no target
    model_base  = LogisticRegression(random_state=10100,n_jobs=-1)
    model_s1b   = LogisticRegression(random_state=10100,n_jobs=-1)
    model_w_s1b = LogisticRegression(random_state=10100,n_jobs=-1)
    model_s1r   = LogisticRegression(random_state=10100,n_jobs=-1)

    if stage1_model == 'Logistic':
        model_base  = LogisticRegression(random_state=10100,n_jobs=-1)
        model_s1b   = LogisticRegression(random_state=10100,n_jobs=-1)
        model_w_s1b = LogisticRegression(random_state=10100,n_jobs=-1)
        model_s1r   = LogisticRegression(random_state=10100,n_jobs=-1)
    if stage1_model == 'SVM':
        svc = LinearSVC(random_state=10100)
        model_base  = CalibratedClassifierCV(base_estimator=svc, cv=2)
        model_s1b   = CalibratedClassifierCV(base_estimator=svc, cv=2)
        model_w_s1b = CalibratedClassifierCV(base_estimator=svc, cv=2)
        model_s1r   = CalibratedClassifierCV(base_estimator=svc, cv=2)
    if stage1_model == 'DT':
        model_base  = RandomForestClassifier(n_estimators=10,random_state=101010,n_jobs=-1)
        model_s1b   = RandomForestClassifier(n_estimators=10,random_state=101010,n_jobs=-1)
        model_w_s1b = RandomForestClassifier(n_estimators=10,random_state=101010,n_jobs=-1)
        model_s1r   = RandomForestClassifier(n_estimators=10,random_state=101010,n_jobs=-1)
    if lr_model_name == 'GBC':
        model_base  = GradientBoostingClassifier(random_state=101010,loss="exponential")
        model_s1b   = GradientBoostingClassifier(random_state=101010,loss="exponential")
        model_w_s1b = GradientBoostingClassifier(random_state=101010,loss="exponential")
        model_s1r   = GradientBoostingClassifier(random_state=101010,loss="exponential")
    

    model_base.fit(data_naive,Y_naive.astype('int'))
    
    start_time = time.time()
    model_s1b.fit(data,y_s1b.astype('int'))
    train_classifier_time = time.time() - start_time

    model_s1r.fit(data_adapted,y_s1r.astype('int')) # ajustado
    

    ######################################
    ### Classificando
    ######################################
    alvo = atributos + columns_y

    t_ = target_.iloc[:,alvo]
    Y_t = t_.pop('is_match').values

    Y_naive = modelo.predict(t_) # modelo totalmente treinado no source
    Y_base = model_base.predict(t_)
    
    start_time = time.time()
    Y_topk = model_s1b.predict(t_)
    execute_classifier_time = time.time() - start_time
    
    Y_topk_adpt = model_s1r.predict(t_)

    naive_log = generate_logs_s1(Y_t, Y_naive,'naive','all',lr_model_name,stage1_model)
    base_log = generate_logs_s1(Y_t, Y_base,'tl-base','all',lr_model_name,stage1_model)
    topk_log = generate_logs_s1(Y_t, Y_topk,'tl_pprl','top_k',lr_model_name,stage1_model)
    topk_adpt_log = generate_logs_s1(Y_t, Y_topk_adpt,'tl_pprl++','top_k_adpt',lr_model_name,stage1_model)

    logs = [naive_log,base_log,topk_log,topk_adpt_log]

    # passando funcao com peso para o modelo
    # TODO: Colocar no log a funcao
    if funcao_w == None:
        model_w_s1b.fit(data,y_s1b.astype('int'),sample_weight=w)
        Y_topkw = model_w_s1b.predict(t_)
        topkw_log = generate_logs_s1(Y_t, Y_topkw,'tl_pprl-w','top_k',lr_model_name,stage1_model)
    else:
        for fx in funcao_w:
            w_linha = fx(w)
            model_w_s1b.fit(data,y_s1b.astype('int'),sample_weight=w_linha)

            try:
                fw = fx.__name__
            except AttributeError:
                fw = 'naosei'

            Y_topkw = model_w_s1b.predict(t_)
#             print("\t" + lr_model_name)
            topkw_log = generate_logs_s1(Y_t, Y_topkw,'tl_pprl-w','top_k',lr_model_name,stage1_model,peso=fw)
            logs.append(topkw_log)

    # nesses dados estao faltando o adptados
    _dnaive = target_.copy()
    _dbase = target_.copy()
    _dtopk = target_.copy()
    _dtopkw = target_.copy() # pega so o ultimo kw

    _dnaive['y'] = Y_naive
    _dbase['y']  = Y_base
    _dtopk['y']  = Y_topk
    _dtopkw['y'] = Y_topkw

    data = [_dnaive,_dbase,_dtopk,_dtopkw]
    
    time_log = {'sep_model_training':sep_model_time,
#                 'exec_sep_model' : execute_classifier_time,
                'sep_model_execution':select_train_data_time,
                'instance_adptation':adjust_train_data_time,
                'train_classifier':train_classifier_time,
                'execute_classifier':execute_classifier_time}
    
#     sep_model_time, select_train_data_time, adjust_train_data_time, train_classifier_time, execute_classifier_time

    return data, logs, time_log#, data_ # colocado o data