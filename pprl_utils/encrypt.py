import sys
import logging
import math

import ngram

import pandas as pd
# from tqdm.notebook import trange, tqdm
import tqdm

from pprl_utils.encoding.bloomfilter import BloomFilter
from pprl_utils.encoding.bloomfilter import dice_coefficient, jaccard_coefficient, entropy_coefficient, overlap_coefficient , hamming_coefficient



def encrypt_ds(df, atts, bf_len, num_ngrams=2):
    '''
    Criptografa todo o dataset

    :param df: dataset
    :param atts:  atributos e;g; [3,6,8]
    :param bf_len: tamanho do bf (utilizar o get_max_length_of_record())
    :param bigrams: numero de bigramas
    :return: o dataset anonimizado como uma lista de lista [ [id, bf] ]
    '''

    index = ngram.NGram(N=num_ngrams)
    total_num_q_gram = 0
    total_num_val =    0

    for i, row in df.iterrows():
        id_ = row[0]
        data_ = ''
        for att in atts:
            data_ += str(row[att])
        
        
    ngramas = list(index.ngrams(index.pad(str(data_))))
    total_num_q_gram += len(ngramas)
    total_num_val +=    1
        


    avrg_num_q_gram = float(total_num_q_gram) / total_num_val

    # Set number of hash functions to have in average 50% of bits set to 1
    #
    bf_num_hash_funct = int(round(math.log(2.0)*float(bf_len) / \
                            avrg_num_q_gram))

    logging.info(' Number of BF hash functions to: %d' % (bf_num_hash_funct))

    ds = []
    for index, row in df.iterrows():
        id_ = row[0]
        data_ = ''
        for att in atts:
            data_ += str(row[att])

        e_data = encryptData(data_, bf_len,bf_num_hash_funct,num_ngrams=num_ngrams)

        ds.append((id_, e_data))
    return ds

def encryptData(data,bf_len,num_of_hash,num_ngrams=2):
    """
        Encode data

        bigrams : 2 = Bigrams
        bf_len : Size of BF
        num_of_hash : False positive rate
    """

    bloomfilter = BloomFilter(bf_len,num_of_hash)

    index = ngram.NGram(N=num_ngrams)
    ngramas = list(index.ngrams(index.pad(str(data))))

    for bigram in ngramas:
        bloomfilter.add(str(bigram))

    return bloomfilter

def get_max_length_of_record(df, atts):
    '''
    Informa o tamanho do maior registro de um dataset

    :param df: dataset
    :param atts: a posição dos atributis eg. [1,2,3]
    :return:
    '''
    soma = 0
    for att in atts:

        soma += df.iloc[:, att].map(len).max()
    return soma

def number_of_bigrams(n, r=2):
    '''
    retorna o numero de ngrams para o tamanho n
    :param n: tamanho do string
    :param r: numero de ngrams
    :return:
    '''
    index = ngram.NGram(N=r)
    return len(list(index.ngrams(index.pad(str('a' * n)))))
    #return int(math.factorial(n)/ ( math.factorial(r) * math.factorial(n-r)))

def compare_ds(dfa, dfb, atts, golds, bf_len,  bigrams=2):
    '''
    Compara dois dadataset
    calcula a similaridade de todos os itens de dois dadataset
    metricas [ dice, jaccard, overlap, hamming , entropy ]
    :param dfa:
    :param dfb:
    :param atts: a lista com a posiçãod os atributis

    :param golds: gabarito ja passado pelo metodo datasetutil.gerar_gabarito()
    :param bigrams:

    :return:
    '''

    _dfa = encrypt_ds(dfa, atts, bf_len,num_ngrams=bigrams)
    _dfb = encrypt_ds(dfb, atts, bf_len,num_ngrams=bigrams)

    bft = _dfa[0][1]

    # estatisticas
    bf_hash_functions = bft.hash_functions
    bf_bit_size = bft.bit_size

    # comparando tudo
    otodo = []
    # for e1 in tqdm(_dfa):
    for e1 in _dfa:
        id1 = e1[0]
        bf1 = e1[1]
        for e2 in _dfb:
            # for e2 in tqdm(eb, leave=False):
            id2 = e2[0]
            bf2 = e2[1]

            dice = dice_coefficient(bf1, bf2)
            jac = jaccard_coefficient(bf1, bf2)
            ol = overlap_coefficient(bf1, bf2)
            ha = hamming_coefficient(bf1, bf2)
            hx = entropy_coefficient(bf1, bf2)

            try:
                classificacao = golds[id1][id2]  # mudar variavel
            except KeyError:
                classificacao = 0  # nao e match

            # linha = {'id1': id1, 'id2': id2,
            #          'dice': dice, 'jaccard': jac, 'overlap': ol,
            #          'hamming': ha,
            #          'entropy': hx,
            #          'is_match': classificacao}
            linha = [id1 , id2 , dice , jac , ol , ha , hx , classificacao]
            otodo.append(linha)

            # pd.DataFrame(otodo)

    cnames = ['id1' , 'id2' , 'dice' , 'jaccard' , 'overlap' ,
              'hamming' , 'entropy' , 'is_match']

    return pd.DataFrame(otodo , columns=cnames) , {'n_hash': bf_hash_functions , 'bits': bf_bit_size ,
                                                   'cap': 0 , 'atts': len(atts)}
    # return pd.DataFrame(otodo), {'n_hash': bf_hash_functions,'bits': bf_bit_size,'cap': bf_capacity , 'atts' : len(atts)}


def compare_ds_based_on_blk(dfa, dfb, atts, gold, bf_len, comps, bigrams=2, use_comps_in_gold=False):
    '''
    Compara dois dadataset, diferente do anterior ele so compara os pares indicados no goldstandart

    calcula a similaridade de todos os itens de dois dadataset
    metricas [ dice, jaccard, overlap, hamming , entropy ]
    :param dfa:
    :param dfb:
    :param atts: a lista com a posiçãod os atributis

    :param golds: gabarito ja passado pelo metodo datasetutil.gerar_gabarito()
    :param bigrams:
    :use_comps_in_gold: so utiliza as comparacoes do gold (defaul: False)

    :return:
    '''

    _dfa = encrypt_ds(dfa , atts , bf_len , num_ngrams=bigrams)
    _dfb = encrypt_ds(dfb , atts , bf_len , num_ngrams=bigrams)
    del dfa, dfb

    bft = _dfa[0][1]

    # _dfa = pd.DataFrame(_dfa)
    # _dfa.columns = ['id' , 'bf']
    # _dfb = pd.DataFrame(_dfb)
    # _dfb.columns = ['id' , 'bf']

    dict_dfa = {}
    for e in _dfa:
        dict_dfa[e[0]] = e[1]
    
    dict_dfb = {}
    for e in _dfb:
        dict_dfb[e[0]] = e[1]

    # estatisticas
    bf_hash_functions = bft.hash_functions
    bf_bit_size = bft.bit_size

    otodo = []
    contador = 0
    # comparando apenas os do gold
    for i , row in comps.iterrows():
        # id1 = _dfa[_dfa.id == row.id1].id.values[0]
        # bf1 = _dfa[_dfa.id == row.id1].bf.values[0]
        # id2 = _dfb[_dfb.id == row.id2].id.values[0]
        # bf2 = _dfb[_dfb.id == row.id2].bf.values[0]
        id1 = row.id1
        id2 = row.id2
        
        try:
            bf1 = dict_dfa[id1]
            bf2 = dict_dfb[id2]

            dice = dice_coefficient(bf1 , bf2)
            jac = jaccard_coefficient(bf1 , bf2)
            ol = overlap_coefficient(bf1 , bf2)
            ha = hamming_coefficient(bf1 , bf2)
            hx = entropy_coefficient(bf1 , bf2)

            
            try:
                classificacao = gold[id1][id2]  # mudar variavel
                contador+=1
                if use_comps_in_gold:
                    linha = [id1 , id2 , dice , jac , ol , ha , hx , classificacao]
                    otodo.append(linha)
            except KeyError:
                classificacao = 0  # nao e match

            if not use_comps_in_gold:
                linha = [id1 , id2 , dice , jac , ol , ha , hx , classificacao]
                otodo.append(linha)

        except KeyError:
            #caso nao tenha os valores nos dicionarios
            pass
        

    cnames = ['id1' , 'id2' , 'dice' , 'jaccard' , 'overlap' ,
              'hamming' , 'entropy' , 'is_match']
    
    # print("::::: " + str(contador))

    return pd.DataFrame(otodo , columns=cnames) , {'n_hash': bf_hash_functions , 'bits': bf_bit_size ,
                                                   'cap': 0 , 'atts': len(atts)}


