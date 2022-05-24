#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   file.py
@Time    :   2022/04/03 08:59:40
@Author  :   Thiago Nóbrega 
@Contact :   thiagonobrega@gmail.com
'''

import os
import pandas as pd
import zipfile

def open_ds(infile , n_atts='atts-1'):
    '''

    Open the comparisons files

    :param infile:
    :param n_atts:
    :param deduplica:
    :return:
    '''
    zf = zipfile.ZipFile(infile)
    nl = zipfile.ZipFile.namelist(zf)

    for i in range(0, len(nl)):

        fn = nl[i]

        if (n_atts in fn):
            a = pd.read_csv(zf.open(fn), header=0, sep=";",
                            index_col=False)

            # remove comparacoes com mesma id
            # colocar flag
            # if (deduplica):
            #     a = a[a['id1'] != a['id2']]

    return a

def load_data(context_s,context_t,
              s_compfile,t_compfile,
              att_s,att_t,
             ds_dir="./datasets/"):



    s_file = ds_dir + os.sep + context_s +os.sep+ s_compfile
    t_file = ds_dir + os.sep + context_t +os.sep+ t_compfile

    source_ = open_ds(s_file,n_atts='atts-'+str(att_s))
    target_ = open_ds(t_file,n_atts='atts-'+str(att_t))

    log_ = {'source':context_s[:-1],
            'src_len':s_file.split('_')[2],
            'target':context_t[:-1],
            'tgt_len':t_file.split('_')[2],
            'atts_source':att_s,'atts_target':att_t}

    return source_,target_,log_

def write_results_2_google(df, google_auth,
                  spreadsheetId='invalid_id',
                  sheetName = 'Sheet1'):
  
  """
  Colab only
  usage:
    auth.authenticate_user()
    creds, _ = default()
    gc = gspread.authorize(creds) #google_auth

    write_results_2_google(df,gc)
  """
  from google.colab import auth
  import gspread
  from google.auth import default

  sh = google_auth.open_by_key(spreadsheetId)
  sheet = sh.worksheet(sheetName)

  values = df.values.tolist()
  #append_row
  sh.values_append(sheetName, {'valueInputOption': 'USER_ENTERED'}, {'values': values})



def merge_experiments_results(df_quality_results , df_source_selection_results , s1_model='Logistic' , pesos=[], start=-5):
    # agrupar_por_dft
    '''
    MERGE tl_pprl with dr_pprl to

    :param df_quality_results: tl_pprl quality results (pandas.DataFrame)
    :param df_source_selection_results: source selection results (pandas.DataFrame)
    :param s1_model: Select weigthing schema employed in stage 01 default value: Logistic
    :param pesos: chose weith
    :return:
    '''
    # filtrandondo
    if len(pesos) != 0:
        df_quality_results = df_quality_results[df_quality_results['stage1_weight'].isin(pesos)]
        df_quality_results = df_quality_results[(df_quality_results.stage1_model == s1_model)]

    df = pd.merge(df_quality_results , df_source_selection_results , how='left' , on='scenario')
    df = df.iloc[: , [i for i in range(len(df_quality_results.columns))] + [i for i in range(start , 0)]]
    # cols = dft.columns + dfr.columns[-6:-1]

    end = -1
    scols = start - 1

    df.columns = list(df_quality_results.columns) + list(df_source_selection_results.columns[scols:end])
    del df['scenario']
    # dfr.iloc[:,-6:-1]
    return df