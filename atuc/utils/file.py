#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   file.py
@Time    :   2022/04/03 08:59:40
@Author  :   Thiago Nóbrega 
@Contact :   thiagonobrega@gmail.com
'''

import pandas as pd
import zipfile

def open_ds(infile , n_atts='atts-1' , deduplica=True):
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
            if (deduplica):
                a = a[a['id1'] != a['id2']]

    return a


def load_data(context_s,context_t,
              s_compfile,t_compfile,
              att_s,att_t,
              dedup_s=False,dedup_t=False,
             ds_dir="../datasets/"):

    s_file = s_compfile
    t_file = t_compfile

    source_ = open_ds(s_file,n_atts='atts-'+str(att_s),deduplica=dedup_s)
    target_ = open_ds(t_file,n_atts='atts-'+str(att_t),deduplica=dedup_t)

    log_ = {'source':context_s[:-1],
            'src_len':s_file.split('_')[2],
            'target':context_t[:-1],
            'tgt_len':t_file.split('_')[2],
            'atts_source':att_s,'atts_target':att_t}

    return source_,target_,log_