import pandas as pd
import zipfile
from collections import defaultdict


def save_zip(data1,data2,gabarito,estatisca,outfile):
    '''
    salva o arquivo pronto para ser utilizado

    :param data1:
    :param data2:
    :param gabarito:
    :param estatisca:
    :param outfile:
    :return:
    '''
    with zipfile.ZipFile(outfile, 'w' , zipfile.ZIP_DEFLATED) as csv_zip:
        csv_zip.writestr("a.csv", pd.DataFrame(data1).to_csv(sep=';',index=False))
        csv_zip.writestr("b.csv", pd.DataFrame(data2).to_csv(sep=';',index=False))
        csv_zip.writestr("gold.csv", pd.DataFrame(gabarito).to_csv(sep=';',index=False))
        csv_zip.writestr("stats.csv", pd.DataFrame(estatisca).to_csv(sep=';',index=True))

def save_zip2(data1,data2,gabarito,esta,estb,outfile):
    '''
        salva o arquivo pronto para ser utilizado


    :param data1:
    :param data2:
    :param gabarito:
    :param esta:
    :param estb:
    :param outfile:
    :return:
    '''

    with zipfile.ZipFile(outfile, 'w' , zipfile.ZIP_DEFLATED) as csv_zip:
        csv_zip.writestr("a.csv", pd.DataFrame(data1).to_csv(sep=';',index=False))
        csv_zip.writestr("b.csv", pd.DataFrame(data2).to_csv(sep=';',index=False))
        csv_zip.writestr("gold.csv", pd.DataFrame(gabarito).to_csv(sep=';',index=False))
        csv_zip.writestr("stats_a.csv", pd.DataFrame(esta).to_csv(sep=';',index=True))
        csv_zip.writestr("stats_b.csv", pd.DataFrame(estb).to_csv(sep=';', index=True))

def gerar_estatiscas_df(ln, est):
    '''
    Gera as estatiscas dos datasets

    numeros de valores
    ...
    numero de nulos
    '''

    lin = ln.values[0:].tolist()

    zi = est.index.values.tolist()
    zi.append('nan')

    dict_ = {}
    for i in range(0, len(list(est.columns))):
        att = list(est.columns)[i]
        dict_[att] = lin[i]

    est = est.append(dict_, ignore_index=True)
    est['metrica'] = zi
    est.set_index('metrica', inplace=True)

    return est

def substituir_valores_nulos(df,strrpl=''):
    '''
     Substitui os valores nulo de um dataset por um string

    :param df:
    :param strrpl: a string que sera substituito
    :return:
    '''
    for c in df.columns:
        df[c] = df[c].fillna(strrpl)
    return df


def verify_gg4cc(df_gab, a='id1', b='id2'):
    '''
     verifica se o gabarito e de um dataset clean-clean

    :param df_gab: Dataframe (a,b) com os true_match
    :param a: nome de a
    :param b: nome de b
    :return dupla: retorna as tuplas que i a aparece mais que 1 x
    :return duplb: retorna as tuplas que i b aparece mais que 1 x
    '''
    counts = df_gab[a].value_counts()
    dupl = df_gab[df_gab[a].isin(counts.index[counts > 1])]
    counts = df_gab[b].value_counts()
    dupl2 = df_gab[df_gab[b].isin(counts.index[counts > 1])]

    return dupl, dupl2

def open_processed_ds(infile):
    '''

    Ler os dados pre-processados dos dataset

    :param infile:
    :return: a - dsa
    :return: b - dsb
    :return: gs - goldstandard
    '''
    zf = zipfile.ZipFile(infile)
    nl = zipfile.ZipFile.namelist(zf)

    for i in range(0, len(nl)):

        fn = nl[i]

        if ('a.csv' == fn):
            a = pd.read_csv(zf.open(fn), header=0, sep=";",
                            index_col=False)
            a = a.fillna('')

        if ('b.csv' == fn):
            b = pd.read_csv(zf.open(fn), header=0, sep=";",
                            index_col=False)
            b = b.fillna('')

        if ('gold.csv' in fn):
            gs = pd.read_csv(zf.open(fn), header=0, sep=";",
                             index_col=False)
    return a, b, gs

def gerar_gabarito(gs):
    '''
     Gera um dicionario com as duas chaves para um gabarito

    :param gs:
    :return:
    '''
    d = defaultdict(dict)
    for i, row in gs.iterrows():
#         return row
        d[row[0]][row[1]] = 1
    return d