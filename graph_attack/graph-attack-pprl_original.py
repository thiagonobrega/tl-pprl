# graph-attack-pprl.py - Implementation of a graph alignment based attack
# on a data set encoded for PPRL. The attack aims to identify similar nodes
# across two similarity graphs, one based on the comparison of plain-text
# values while the other is based on the comparison of encoded values. For
# both graphs nodes represent records while edges represent the similarities
# between these records.
#
# Graphs are saved into binary pickle files, and if available these files will
# be used instead of re-calculated
#
# Graph features are also saved into csv files, and if available these files will
# be used instead of re-calculated
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Description of the main steps of the approach and corresponding functions
# used (also see the code in the main program towards the end of the program):
#
# 1) Load the files to be used as plain-text and encoded data sets - if both
#    file names are the same we use this file both as plain-text and encoded
#    data set (i.e. known plain-text attack).
#    Function: load_data_set()
#
# 2) Convert the data set(s) into q-gram sets (one per record).
#    Function: gen_q_gram_sets()
#
# 3) Encode the second data set into a encoding (currently either
#    Bloom filters, BFs, Duncan Smith's tabulation min-hash, TMH,
#    approach, or Ranbaduge's Two-step hashing approach). 
#    Any other encoding can be used where approximate similarities 
#    between encodings can be calculated.
#    Function: uses modules hashing.py (for BF), tabminhash.py (for TMH)
#    or colminhash.py (for 2SH)
#
# 4) Generate two similarity graphs, one by calculating similarities between
#    q-gram sets (plain-text similarity graph), the other by calculating
#    similarities between encodings (encoded similarity graph). To prevent
#    full pair-wise comparison between all records in a data set we use
#    min-hashing LSH on the q-gram sets.
#
# 5) To identify the differences between plain-text and encoded similarities,
#    assuming we do know the ground truth (which pairs of encodings correspond
#    to which q-gram set pairs), we can calculate similarity differences and
#    use these differences in a regression model to adjust the encoded similarities 
#    to what they should have been if they would correspond to q-gram pairs.
#    If this step is done then we can adjust either the encoded or plaintext 
#    similarities accordingly before they are inserted into the similarity graphs.
#
# 6) We get the summary characteristics of the two graphs for different
#    minimum similarities.
#    Function: show_graph_characteristics()
#
# 7) We save the two graphs into two pickle (binary) files.
#    Function: networkx.write_gpickle()

# Note that step 3) to 6) are only done if the corresponding pickle files are
# not available, if they are we skip steps 3) to 6) and instead load these
# pickle files. Function: networkx.read_gpickle()
#
# The attack phase starts here (in the code several nested loops iterate over
# the different parameter settings provided at the beginning of the program
# (i.e. many different attacks are conducted on the two similarity graphs):
#
# 8) To limit the attack to nodes (records) that have information about their
#    graph neighbours, remove all connected components in the graph that
#    contain less that a certain minimum number of nodes.
#    Function: gen_min_conn_comp_graph()
#
# 9) Generate the features where for each node in a similarity graph a feature
#    vector is calculated.
#    Function: calc_features()
#
# 10) Normalise all features into the [0,1] interval, because the sim-hash
#     approach uses random vectors where each element is in the range [0,1].
#     Function: norm_features_01()
#
# 11) Apply a possible feature selection: all, only those with non-zero
#     values, or those with the highest variance.
#     Functions: get_std_features() and select_features()
#
# 12) Because we aim to have 1-to-1 matching nodes across the two graphs,
#     remove all feature vectors that occur more than once.
#     Function: remove_freq_feat_vectors()
#
# 13) Generate the sim-hashes for all feature vectors, i.e. one binary vector
#     representing each numerical feature vector.
#     Function: see class CosineLSH
#
# 14) As blocking on these bit arrays either compare all bit array pairs or
#     use Hamming LSH.
#     Functions: all_in_one_blocking() and hlsh_blocking()
#
# 15) Compare the encodings across the two graphs and only keep those pairs
#     with a high similarity. Either compare all pairs (coming out of the
#     previous blocking step) or only those pairs where their edge with the
#     highest similarity is comparable, and also their total numbers of edges
#     are comparable.
#     Functions: calc_cos_sim_all_pairs() and calc_cos_sim_edge_comp()
#
# 16) For the compared bit-array pairs with high similarities finally
#     calculate their actual Cosine similarity using their actual feature
#     vectors. Then sort the resulting pairs according to their similarities
#     with highest first, and report how many in the top 10, 20, 50, 100 and
#     all, are correct matches (i.e. a feature vector from the plain-text data
#     set was correctly matched with a feature vector from the encoded data
#     set).
#
#

# Anushka Vidanage, Peter Christen, Thilina Ranbaduge, Rainer Schnell
# July 2020
#
# Usage: python graph-attack-pprl.py [encode_data_set_name] [encode_ent_id_col]
#                                    [encode_col_sep_char] [encode_header_line_flag]
#                                    [encode_attr_list] [encode_num_rec]
#                                    [plain_data_set_name] [plain_ent_id_col]
#                                    [plain_col_sep_char] [plain_header_line_flag]
#                                    [plain_attr_list] [plain_num_rec]
#                                    [q] [padded_flag]
#                                    [plain_sim_funct_name]
#                                    [sim_diff_adjust_flag]
#                                    [encode_sim_funct_name]
#                                    [encode_method] {encode_method_param}
#                                    
# where:
#
# encode_data_set_name     is the name of the CSV file to be encoded into BFs.
# encode_ent_id_col        is the column in the CSV file containing entity
#                          identifiers.
# encode_col_sep_char      is the character to be used to separate fields in
#                          the encode input file.
# encode_header_line_flag  is a flag, set to True if the file has a header
#                          line with attribute (field) names.
# encode_attr_list         is the list of attributes to encode and use for
#                          the linkage.
# encode_num_rec           is the number of records to be loaded from the data
#                          set, if -1 then all records will be loaded,
#                          otherwise the specified number of records (>= 1).
#
# plain_data_set_name      is the name of the CSV file to use plain text
#                          values from.
# plain_ent_id_col         is the column in the CSV file containing entity
#                          identifiers.
# plain_col_sep_char       is the character to be used to separate fields in
#                          the plain text input file.
# plain_header_line_flag   is a flag, set to True if the file has a header
#                          line with attribute (field) names.
# plain_attr_list          is the list of attributes to get values from to
#                          guess if they can be re-identified.
# plain_num_rec            is the number of records to be loaded from the data
#                          set, if -1 then all records will be loaded,
#                          otherwise the specified number of records (>= 1).
#
# q                        is the length of q-grams to use when converting
#                          values into q-gram set.
# padded_flag              if set to True then attribute values will be padded
#                          at the beginning and end, otherwise not.
#
# plain_sim_funct_name     is the function to be used to calculate similarities
#                          between plain-text q-gram sets. Possible values are
#                          'dice' and 'jacc'.
# sim_diff_adjust_flag     is a flag, if set to True then the encoded
#                          similarities will be adjusted based on calculated
#                          similarity differences of true matching edges
#                          between plain-text and encoded similarities.
# encode_sim_funct_name    is the function to be used to calculate similarities
#                          between encoded values (such as bit-arrays).
#                          Possible values are 'dice', 'hamm', and 'jacc'.
#
# encode_method            is the method to be used to encode values from the
#                          encoded data set (assuming these have been converted
#                          into q-gram sets). Possible are: 'bf' (Bloom filter
#                          encoding) or 'tmh' (tabulation min hash encoding) or
#                          '2sh' (two-step hash encoding).
# encode_method_param      A set of parameters that depend upon the encoding
#                          method.
#                          For Bloom filters, these are:
#                          - bf_hash_type       is either DH (double-hashing)
#                                               or RH (random hashing).
#                          - bf_num_hash_funct  is a positive number or 'opt'
#                                               (to fill BF 50%).
#                          - bf_len             is the length of Bloom filters.
#                          - bf_encode          is the Bloom filter encoding method.
#                                               can be 'clk', 'abf', 'rbf-s', or
#                                               'rbf-d'
#                          - bf_harden          is either None, 'balance' or
#                                               or 'fold' for different BF
#                                               hardening techniques.
#                          - bf_enc_param       parameters for Bloom filter encoding
#                                               method 
#                          For tabulation min-hashing, these are:
#                          - tmh_num_hash_bits  The number of hash bits to
#                                               generate per encoding.
#                          - tmh_hash_funct     is the actual hash function to
#                                               use. Possible values are:
#                                               'md5', 'sha1', and 'sha2'.
#                          - tmh_num_tables     is the number of tabulation
#                                               hash tables to be generated.
#                          - tmh_key_len        is the length of the keys into
#                                               these tables.
#                          - tmh_val_len        is the length of the random bit
#                                               strings to be generated for
#                                               each table entry.
#                          For two-step hash encoding, these are:
#                          - cmh_num_hash_funct The number of hash functions to
#                                               be considered.
#                          - cmh_num_hash_col   is the length of the generated
#                                               encodings (number of random integers
#                                               per encoding)
#                          - cmh_max_rand_int   maximum integer value to select the
#                                               random integers
#
#
#
# Some example calls:
#
# python graph-attack-pprl.py euro-census.csv 0 , True [1,2,8] -1 euro-census.csv 0 , True [1,2,8] -1 2 False dice True bf rh 15 1000 clk none None dice
#
# python graph-attack-pprl4.py euro-census.csv 0 , True [1,2,8] -1 euro-census.csv 0 , True [1,2,8] -1 2 False dice True bf rh 15 1000 rbf-s none [0.2,0.3,0.5] dice
#
# python graph-attack-pprl4.py euro-census.csv 0 , True [1,2,8] -1 euro-census.csv 0 , True [1,2,8] -1 2 False jacc True tmh 1000 sha2 8 8 64 jacc
#
# python graph-attack-pprl4.py euro-census.csv 0 , True [1,2,8] -1 euro-census.csv 0 , True [1,2,8] -1 2 False jacc True 2sh 15 1000 100000 jacc
#

# -----------------------------------------------------------------------------
# The actual parameters for the graph attack are set below - once the
# plain-text and encoded similarity graphs have been generated (or loaded from
# pickled files) they can be matched using various settings for the following
# parameters:
#
# graph_node_min_degr  is the minimum degree (number of neighbours) a node in
#                      the graph needs to be considered in the graph matching.
# graph_sim_list       is the list of similarity values to be used when
#                      generating the features for the graph. The minimum
#                      similarity value is also used to decide what edges not
#                      to include in the graph (those below that minimum
#                      similarity).
# graph_feat_list      is the features to be used to characterise the nodes in
#                      a graph.
# graph_feat_select    is the method to select features based on their
#                      variance. Possible values are 'all', 'nonzero', k (an
#                      integer, in which case the top k features will be
#                      selected), or f (a floating-point number, in which case
#                      features with at least this standard deviation will be
#                      selected).
#
# sim_hash_num_bit      is the number of bits to be used when generating
#                       Cosine LSH bit arrays from the feature arrays
# sim_hash_match_sim    is the minimum similarity for two records to be
#                       considered a match across the two graphs.
# sim_hash_block_funct  is the method used for blocking the similarity hash
#                       dictionaries, can be one of 'all_in_one' (one block
#                       with all records from a data set) or 'hlsh' (apply
#                       Hamming LSH).
# sim_comp_funct        is the way similarity hashes are compared, this can
#                       'allpairs' (all pairs - based on blocking) or
#                       'simwindow' (only pairs with similar highest edges
#                       and similar numbers of edges).
#
# Several attacks can be conducted on the two graphs with different parameter
# settings - the full combination of the following settings will be run (so be
# careful of the number of settings per parameter)
#
# For each setting provide a name as key and then its parameter values as a
# list or dictionary
#

# Set the graph generation parameters depending upon data sets used
#
import sys
import sklearn

encode_data_set_name = sys.argv[1]

if ('passenger' in encode_data_set_name):  # Titanic names
  soundex_attr_list = [2] # 2, 3
  
  graph_sim_list = [('[0.2]',[0.2])] #('[0.9-0.5]',[0.9,0.7,0.5]),
                                     #('[0.5]',[0.5])

  # Important: this needs to be smallest first because connected components
  # smaller than a (minimum degree+1) are removed from the input graph
  #
  graph_node_min_degr_list = [('5',5)]

  sim_hash_match_sim = 0.9
  sim_hash_num_bit_list = [('500',500)] #('1000',1000), ('500',500), ('100',100)
  sim_hash_block_funct_list = ['all_in_one'] # 'all_in_one','hlsh'
  sim_comp_funct_list = ['allpairs'] # 'allpairs', 'simtol'

  graph_feat_list_list = [('all',['node_freq','max_sim','min_sim','degree',
                                 'degree_histo1','degree_histo2','sim_avr',
                                 'sim_std','egonet_degree','egonet_density',
                                  #'pagerank', # very time consuming
                                  'between_central',
                                  #'closeness_central', # very time consuming
                                  'degree_central',
                                  #'eigenvec_central', # very time consuming
                                 ]),
                                  
                          ('no-histo',['node_freq','max_sim','min_sim','degree',
                                       'sim_avr', 'sim_std','egonet_degree',
                                       'egonet_density','between_central',
                                       'degree_central', 'eigenvec_central',
                                 ]),
                          
                          ('one-central',['node_freq','max_sim','min_sim','degree',
                                 'degree_histo1', 'degree_histo2', 'sim_avr',
                                 'sim_std','egonet_degree', 'egonet_density',
                                 'degree_central',
                                 ]),
                          
                          ('no-nf-central',['max_sim','min_sim','degree',
                                            'degree_histo1','degree_histo2',
                                            'sim_avr','sim_std',
                                            'egonet_degree','egonet_density']),
                          
                          ('no-central',['node_freq','max_sim','min_sim',
                                         'degree','degree_histo1',
                                         'degree_histo2','sim_avr','sim_std',
                                         'egonet_degree','egonet_density']),
                          
                          ('node2vec', ['node2vec']),
                       ]

elif ('titanic' in encode_data_set_name):  # Titanic names
  soundex_attr_list = [2] # 2, 1
  
  graph_sim_list = [('[0.2]',[0.2])]

  # Important: this needs to be smallest first because connected components
  # smaller than a (minimum degree+1) are removed from the input graph
  #
  graph_node_min_degr_list = [('3',3)]

  sim_hash_match_sim = 0.9
  sim_hash_num_bit_list = [('500',500)]
  sim_hash_block_funct_list = ['hlsh']
  sim_comp_funct_list = ['simtol']

  graph_feat_list_list = [('all',['node_freq','max_sim','min_sim','degree',
                                 'degree_histo1','degree_histo2','sim_avr',
                                 'sim_std','egonet_degree','egonet_density',
                                  #'pagerank', # very time consuming
                                  'between_central',
                                  #'closeness_central', # very time consuming
                                  'degree_central',
                                  #'eigenvec_central', # very time consuming
                                 ]),
                                  
                          ('no-histo',['node_freq','max_sim','min_sim','degree',
                                       'sim_avr', 'sim_std','egonet_degree',
                                       'egonet_density','between_central',
                                       'degree_central', 'eigenvec_central',
                                 ]),
                          
                          ('one-central',['node_freq','max_sim','min_sim','degree',
                                 'degree_histo1', 'degree_histo2', 'sim_avr',
                                 'sim_std','egonet_degree', 'egonet_density',
                                 'degree_central',
                                 ]),
                          
                          ('no-nf-central',['max_sim','min_sim','degree',
                                            'degree_histo1','degree_histo2',
                                            'sim_avr','sim_std',
                                            'egonet_degree','egonet_density']),
                          
                          ('no-central',['node_freq','max_sim','min_sim',
                                         'degree','degree_histo1',
                                         'degree_histo2','sim_avr','sim_std',
                                         'egonet_degree','egonet_density']),
                          
                          ('node2vec', ['node2vec']),
                       ]
  
elif ('ncvoter' in encode_data_set_name or 'ncvr' in encode_data_set_name):  # NCVR
  soundex_attr_list = [3] # 3, 5
  
  graph_sim_list = [('[0.2]',[0.2])]
  
  graph_node_min_degr_list = [('5',5)]

  sim_hash_num_bit_list = [('1000',1000)]
  sim_hash_match_sim = 0.9
  sim_hash_block_funct_list = ['hlsh']
  sim_comp_funct_list = ['simtol']

  graph_feat_list_list = [('all',['node_freq','max_sim','min_sim','degree',
                                 'degree_histo1','degree_histo2','sim_avr',
                                 'sim_std','egonet_degree','egonet_density',
                                  #'pagerank', # very time consuming
                                  'between_central',
                                  #'closeness_central', # very time consuming
                                  'degree_central',
                                  #'eigenvec_central', # very time consuming
                                 ]),
                                  
                          ('no-histo',['node_freq','max_sim','min_sim','degree',
                                       'sim_avr', 'sim_std','egonet_degree',
                                       'egonet_density','between_central',
                                       'degree_central', 'eigenvec_central',
                                 ]),
                          
                          ('one-central',['node_freq','max_sim','min_sim','degree',
                                 'degree_histo1', 'degree_histo2', 'sim_avr',
                                 'sim_std','egonet_degree', 'egonet_density',
                                 'degree_central',
                                 ]),
                          
                          ('no-nf-central',['max_sim','min_sim','degree',
                                            'degree_histo1','degree_histo2',
                                            'sim_avr','sim_std',
                                            'egonet_degree','egonet_density']),
                          
                          ('no-central',['node_freq','max_sim','min_sim',
                                         'degree','degree_histo1',
                                         'degree_histo2','sim_avr','sim_std',
                                         'egonet_degree','egonet_density']),
                          
                          ('node2vec', ['node2vec']),
                       ]

  
elif('euro' in encode_data_set_name):
  soundex_attr_list = [1] # 1, 2
  
  graph_sim_list = [('[0.2]',[0.2])]

  # Important: this needs to be smallest first because connected components
  # smaller than a (minimum degree+1) are removed from the input graph
  #
  graph_node_min_degr_list = [('20',20)]

  sim_hash_num_bit_list = [('500',500)]
  sim_hash_match_sim = 0.9
  sim_hash_block_funct_list = ['hlsh']
  sim_comp_funct_list = ['simtol']

  graph_feat_list_list = [('all',['node_freq','max_sim','min_sim','degree',
                                 'degree_histo1','degree_histo2','sim_avr',
                                 'sim_std','egonet_degree','egonet_density',
                                  #'pagerank', # very time consuming
                                  'between_central',
                                  #'closeness_central', # very time consuming
                                  'degree_central',
                                  #'eigenvec_central', # very time consuming
                                 ]),
                                  
                          ('no-histo',['node_freq','max_sim','min_sim','degree',
                                       'sim_avr', 'sim_std','egonet_degree',
                                       'egonet_density','between_central',
                                       'degree_central', 'eigenvec_central',
                                 ]),
                          
                          ('one-central',['node_freq','max_sim','min_sim','degree',
                                 'degree_histo1', 'degree_histo2', 'sim_avr',
                                 'sim_std','egonet_degree', 'egonet_density',
                                 'degree_central',
                                 ]),
                          
                          ('no-nf-central',['max_sim','min_sim','degree',
                                            'degree_histo1','degree_histo2',
                                            'sim_avr','sim_std',
                                            'egonet_degree','egonet_density']),
                          
                          ('no-central',['node_freq','max_sim','min_sim',
                                         'degree','degree_histo1',
                                         'degree_histo2','sim_avr','sim_std',
                                         'egonet_degree','egonet_density']),
                          
                          ('node2vec', ['node2vec']),
                       ]

else:
  print '*** Error: Unknown input data set *****'
  sys.exit()

print 'Use similarity lists for graph generation:', graph_sim_list
  
graph_feat_select_list = [('all','all'),
                          #('non-zero','nonzero'),
                          #'top-10':10,
                          #('top-10',10)
                          #('min-std-0.4',0.4),
                         ]

# For the 'simtol' comparison approach set the similarity tolerance values:
# lower: how much lower the encoded similarity can be,
# upper: how much higher the encoded similarity can be
#
bf_hash_sim_low_tol =  0.01  # If the encoding are Bloom filters
bf_hash_sim_up_tol =   0.25

tmh_hash_sim_low_tol =  0.05  # If the encoding are tabulation min hashes
tmh_hash_sim_up_tol =   0.05

cmh_hash_sim_low_tol =  0.05  # If the encoding are tabulation min hashes
cmh_hash_sim_up_tol =   0.01

adj_hash_sim_low_tol =  0.05  # If the encoded similarities have been adjusted
adj_hash_sim_up_tol =   0.05

hash_sim_min_tol = 0.05  # Also for the 'simtol' comparison approach, how much
                         # tolerance to count the number of edges

# Numpy and Scipy Cosine give the same results and are always best - so only
# use those
#
#final_sim_funct_list = ['lsh_cosine','cosine_scipy','cosine_numpy',
#                        'euclidean']
#final_sim_funct_list = ['edge_sim_diff']

# The parameters for the Hamming LSH blocking of the similarity hashes in the
# graph matching step
#
if('ncvoter' in encode_data_set_name or 'ncvr' in encode_data_set_name):
  graph_block_hlsh_num_sample =      20
  graph_block_hlsh_rel_sample_size = 10  # Divide bit array length by this

else:
  graph_block_hlsh_num_sample =      50
  graph_block_hlsh_rel_sample_size = 50  # Divide bit array length by this

num_exp = len(graph_sim_list) * len(graph_node_min_degr_list) * \
          len(graph_feat_list_list) * len(graph_feat_select_list) * \
          len(sim_hash_num_bit_list) * \
          len(sim_hash_block_funct_list) * len(sim_comp_funct_list)
          # * len(final_sim_funct_list)
print
print '*** Number of graph attack experiments to be conducted: %d ***' % \
      (num_exp)
print

# Get the overall minimum similarity for the graphs
#
min_sim = 1.0

# Also get a sorted list of all similarities over all parameter settings
#
all_sim_set = set()

for (sim_list_name, sim_list) in graph_sim_list:
  min_sim = min(min_sim, min(sim_list))
  all_sim_set = all_sim_set | set(sim_list)

all_sim_list = sorted(all_sim_set)

# -----------------------------------------------------------------------------

import binascii
import csv
import gzip
import hashlib
import math
import os.path
import random
import sys
import time

import bitarray
import itertools
import numpy
import numpy.linalg
import scipy.stats
import scipy.spatial.distance
import sklearn.tree
import pickle

import networkx
import networkx.drawing

import matplotlib
matplotlib.use('Agg')  # For running on adamms4 (no display)
import matplotlib.pyplot as plt

import auxiliary
import hashing   # Bloom filter based PPRL functions
import encoding
import hardening
import tabminhash  # Tabulation min-hash PPRL functions
import colminhash # Two step column hashing for PPRL
import simcalc  # Similarity functions for q-grams and bit-arrays
import node2vec # Node2vec module for generating features using random walks

from gensim.models import Word2Vec

from matplotlib import pylab as pl

from sklearn import linear_model
from sklearn.model_selection import train_test_split

from sklearn.isotonic import IsotonicRegression

from sklearn.preprocessing import PolynomialFeatures

from scipy.optimize import linear_sum_assignment

from sklearn import metrics

# -----------------------------------------------------------------------------

# Set the maximum size of a graph for plots to be generated
#
MAX_DRAW_PLOT_NODE_NUM = 200

PAD_CHAR = chr(1)   # Used for q-gram padding

BF_HASH_FUNCT1 = hashlib.sha1
BF_HASH_FUNCT2 = hashlib.md5

#BF_HASH_FUNCT1 = hashlib.md5
#BF_HASH_FUNCT2 = hashlib.sha256

random_seed = 17
if (random_seed != None):
  random.seed(random_seed)

today_str = time.strftime("%Y%m%d", time.localtime())
now_str =   time.strftime("%H%M", time.localtime())

today_time_str = time.strftime("%Y%m%d %H:%M:%S", time.localtime())

numpy.set_printoptions(precision=4, linewidth=120)

PLT_PLOT_RATIO = 1.0
FILE_FORMAT  = '.eps' #'.png'

PLT_FONT_SIZE    = 20# 28 # used for axis lables and ticks
LEGEND_FONT_SIZE = 20 # 28 # used for legends
TITLE_FONT_SIZE  = 19 # 30 # used for plt title
TICK_FONT_SIZE   = 18

# =============================================================================

def load_data_set(data_set_name, attr_num_list, ent_id_col, soundex_attr_val_list,
                  num_rec=-1, col_sep_char=',', header_line_flag=False):
  """Load a data set and extract required attributes.

     It is assumed that the file to be loaded is a comma separated values file.

     Input arguments:
       - data_set_name          File name of the data set.
       - attr_num_list          The list of attributes to use (of the form
                                [1,3]).
       - ent_id_col             The column number of the entity identifiers.
       - num_rec                Number of records to be loaded from the file
                                If 'num_rec' is -1 all records will be read
                                from file, otherwise (assumed to be a positive
                                integer number) only the first 'num_rec'
                                records will be read.
       - col_sep_char           The column separate character.
       - header_line_flag       A flag, set this to to True if the data set 
                                contains a header line describing attributes, 
                                otherwise set it to False. 
                                The default value False.

     Output:
       - rec_attr_val_dict  A dictionary with entity identifiers as keys and
                            where values are lists of attribute values.
       - attr_name_list     The list of the attributes to be used.
       - num_rec_loaded     The actual number of records loaded.
  """

  rec_attr_val_dict = {}  # The dictionary of attribute value lists to be 
                          # loaded from file
                          
  rec_soundex_attr_val_dict = {}

  # Check if the file name is a gzip file or a csv file
  #
  if (data_set_name.endswith('gz')):
    in_f = gzip.open(data_set_name)
  else:
    in_f = open(data_set_name)

  # Initialise the csv reader
  #
  csv_reader = csv.reader(in_f, delimiter=col_sep_char)

  # The names (if available from the header line) of the attributes to be used
  #
  attr_name_list = []

  # Read the header line if available
  #
  if (header_line_flag == True):
    header_list = csv_reader.next()

    print 'File header line:', header_list
    print '  Attributes to be used:',
    for attr_num in attr_num_list:
      print header_list[attr_num],
      attr_name_list.append(header_list[attr_num])
    print

  max_attr_num = max(attr_num_list)+1

  # Read each line in the file and store the required attribute values in a
  # list
  #
  rec_num = 0 

  for rec_val_list in csv_reader:

    if (num_rec > 0) and (rec_num >= num_rec):
      break  #  Read enough records

    rec_num += 1

    use_rec_val_list = []
    soundex_val_list = []
    
    # Read the entity identifier 
    #
    ent_id = rec_val_list[ent_id_col].strip().lower()
  
    for attr_num in xrange(max_attr_num):

      if attr_num in attr_num_list:
        use_rec_val_list.append(rec_val_list[attr_num].lower().strip())
        
      if(attr_num in soundex_attr_val_list):
        soundex_val_list.append(rec_val_list[attr_num].lower().strip())
#      else:
#        use_rec_val_list.append('')  # Not needed

    # Don't use completely empty list of values
    #
    if (len(''.join(use_rec_val_list)) > 0):
      rec_attr_val_dict[ent_id] = use_rec_val_list
      rec_soundex_attr_val_dict[ent_id] = soundex_val_list

  in_f.close()

  print '  Loaded %d records from file' % (rec_num)
  print '    Stored %d values' % (len(rec_attr_val_dict))

  # Get the frequency distribution of values
  #
  rec_tuple_count_dict = {}
  for use_rec_val_list in rec_attr_val_dict.itervalues():
    rec_tuple = tuple(use_rec_val_list)
    rec_tuple_count_dict[rec_tuple] = \
                       rec_tuple_count_dict.get(rec_tuple, 0) + 1
  count_dist_dict = {}
  for rec_tuple_count in rec_tuple_count_dict.itervalues():
    count_dist_dict[rec_tuple_count] = \
                       count_dist_dict.get(rec_tuple_count, 0) + 1

  print '  Count distribution of value occurrences:', \
        sorted(count_dist_dict.items())

  rec_tuple_count_dict.clear()  # Not needed anymore
  count_dist_dict.clear()

  return rec_attr_val_dict, attr_name_list, rec_num, rec_soundex_attr_val_dict

# -----------------------------------------------------------------------------

def gen_q_gram_sets(rec_attr_val_dict, q, padded_flag):
  """Convert the attribute value(s) for each record into one q-gram set. If
     there are several attributes they are concatenated by a single space
     character.

     Only add a record to the q-gram dictionary if its q-gram set is not empty.

     Input arguments:
       - rec_attr_val_dict  A dictionary with entity identifiers as keys and
                            where values are lists of attribute values.
       - q                  The number of characters per q-gram.
       - padded_flag        if set to True then values will be padded at the
                            beginning and end, otherwise not.

     Output:
       - q_gram_dict  A dictionary with entity identifiers as keys and one
                      q-gram set per record.
  """

  q_gram_dict = {}

  qm1 = q-1  # Shorthand

  for (ent_id, attr_val_list) in rec_attr_val_dict.iteritems():
    
    rec_q_gram_set = set()
    
    for attr_val in attr_val_list:
      attr_val_str = attr_val.strip()
      
      if (padded_flag == True):  # Add padding start and end characters
        attr_val_str = PAD_CHAR*qm1+attr_val_str+PAD_CHAR*qm1
  
      attr_val_len = len(attr_val_str)
  
      # Convert into q-grams and process them
      #
      attr_q_gram_list = [attr_val_str[i:i+q] for i in range(attr_val_len - qm1)]
      attr_q_gram_set = set(attr_q_gram_list)
      
      rec_q_gram_set = rec_q_gram_set.union(attr_q_gram_set)

    if (len(rec_q_gram_set) > 0):
      q_gram_dict[ent_id] = rec_q_gram_set

  return q_gram_dict

# -----------------------------------------------------------------------------

def plot_save_graph(sim_graph, file_name, min_sim):
  """Generate a plot of the given similarity graph and save it into the given
     file name. It is assumed that the edge similarities are given in the
     range [min_sim, 1.0]. Different grey levels will be assigned to different
     edge similarities, with black being the lowest similarity and light grey
     the highest.

     Input arguments:
       - sim_graph  An undirected graph where edges have similarities (sim).
       - file_name  The name into which the plot is to be saved.
       - min_sim    The lowest similarity assumed to be assigned to edges.

     Output:
         - This method does not return anything.
  """

  # Generate a list with different grey scales for different similarities
  # (set to 4 levels)
  #
  color_str_list = ['#333333', '#666666', '#999999', '#CCCCCC']

  sim_interval = (1.0 - min_sim) / len(color_str_list)

  edge_color_list = [(min_sim,'#000000')]

  for (i, color_str) in enumerate(color_str_list):
    edge_color_list.append((min_sim + (1+i)*sim_interval, color_str))

  graph_edge_list =       sim_graph.edges()
  graph_edge_color_list = []  # Based on sim generate grey colors

  node_key_val_set = set()

  for (node_key_val1, node_key_val2) in graph_edge_list:
    node_key_val_set.add(node_key_val1)
    node_key_val_set.add(node_key_val2)
    edge_sim = sim_graph.edge[node_key_val1][node_key_val2]['sim']

    for (color_sim, grey_scale_color_str) in edge_color_list:
      if (edge_sim <= color_sim):
        graph_edge_color_list.append(grey_scale_color_str)
        break

  assert len(graph_edge_color_list) == len(graph_edge_list)

  networkx.draw(sim_graph, node_size=20, width=2, edgelist=graph_edge_list,
                edge_color=graph_edge_color_list, with_labels=False)

  print 'Save generated graph figure with %d nodes into file: %s' % \
        (len(node_key_val_set), file_name)

  plt.savefig(file_name)

# -----------------------------------------------------------------------------

def plot_sim_graph_histograms(sim_graph_list, plot_file_name,
                              round_num_digits=3, y_scale_log=True):
  """Generate a histogram plot with one histogram for each of the given
     similarity graphs.

     Input arguments:
       - sim_graph_list   A list of pairs (similarity graph, label_str) where
                          for each given similarity graph a histogram will
                          be generated.
       - plot_file_name   The name into which the plot is to be saved.
       - round_num_digit  The number of digits values are to be rounded to.
       - y_scale_log      A flag, if set to True (default) then the y-axis is
                          shown in log scale.

     Output:
         - This method does not return anything.
  """

  plot_color_list = ['r','b','g','m','k']  # Colors to be used

  plot_list = []  # One list of x and y values, plus label, for each graph

  min_count = 99999999999  # Needed for plot maximum y-axis value
  max_count = 0
  min_sim =   1.0
  max_sim =   0.0

  for (sim_graph, label_str) in sim_graph_list:

    # Generate a dictionary where keys will be similarities and values counts
    # of how many edges have that similarity
    #
    sim_histo_dict = {}

    for (node_key_val1, node_key_val2) in sim_graph.edges():
      edge_sim = sim_graph.edge[node_key_val1][node_key_val2]['sim']

      egde_sim = round(edge_sim, round_num_digits)

      sim_histo_dict[egde_sim] = sim_histo_dict.get(egde_sim, 0) + 1

    sim_graph_x_val_list = []
    sim_graph_y_val_list = []
    for (edge_sim, count) in sorted(sim_histo_dict.items()):
      sim_graph_x_val_list.append(edge_sim)
      sim_graph_y_val_list.append(count)

    plot_list.append((sim_graph_x_val_list, sim_graph_y_val_list, label_str))

    min_count = min(min_count, min(sim_graph_y_val_list))
    max_count = max(max_count, max(sim_graph_y_val_list))
    min_sim =   min(min_sim,   min(sim_graph_x_val_list))
    max_sim =   max(max_sim,   max(sim_graph_x_val_list))

  sim_histo_dict.clear()  # Not needed anymore
    
  # Plot the generated lists
  #
  w,h = plt.figaspect(PLT_PLOT_RATIO)
  plt.figure(figsize=(w,h))

  plt.title('Graph edge similarity histograms')

  plt.xlabel('Edge similarities')

  if (y_scale_log == True):
    plt.ylabel('Counts (log-scale)')
    plt.yscale('log')
    plt.ylim(min_count*0.667, max_count*1.5)
  else:
    plt.ylabel('Counts')
    plt.ylim(min_count-max_count*0.05, max_count*1.05)

  x_tol = (max_sim-min_sim)/20.0

  plt.xlim(min_sim-x_tol, max_sim+x_tol)

  i = 0
  for sim_graph_x_val_list, sim_graph_y_val_list, label_str in plot_list:
    c = plot_color_list[i]
    i += 1

    plt.plot(sim_graph_x_val_list, sim_graph_y_val_list, color = c, #lw=2,
             label=label_str)

  plt.legend(loc="best") #, prop={'size':14}, bbox_to_anchor=(1, 0.5))

  plt.savefig(plot_file_name, bbox_inches='tight')

# -----------------------------------------------------------------------------

def plot_save_sim_diff(plot_list_dict, plot_file_name, round_num_digit=3,
                       density_plot=True):
  """Generate a frequency plot of the given dictionary with similarity
     differences for different minimum similarities.

     Input arguments:
       - plot_list_dict   A dictionary with minimum similarities as keys and
                          lists of similarity differences between true-matching
                          encoded and plain-text value pairs.
       - plot_file_name   The name into which the plot is to be saved.
       - round_num_digit  The number of digits values are to be rounded to.
       - density_plot     A flag, if set to True then a density plot is
                          generated instead of a frequency plot.

     Output:
         - This method does not return anything.
  """

  # PC 20181025: TODO: Density plot should be normalised, buty-axis seems to
  #              be larger than 1 ??

  assert density_plot in [True, False], density_plot

  # Generate a list of grey levels to be used, with highest similarity having
  # black and lowest similarity a middle grey tone
  #
  plot_color_list = []

  min_level = 0.0
  max_level = 0.6
  level_diff = max_level - min_level

  min_sim_list = sorted(plot_list_dict.keys())
  num_sim =      len(min_sim_list)

  if (num_sim == 1):
    plot_color_list.append(0.0)  # Only one color needed - make it black
  else:
    for (i, sim) in enumerate(min_sim_list):
      c = max_level - i * level_diff/(num_sim-1.0)
      plot_color_list.append(c)

  # Get the overall minimum and maximum values for the density plot
  #
  min_x = 9999
  max_x = -999

  for edge_sim_diff_list in plot_list_dict.itervalues():
    min_x = min(min_x, min(edge_sim_diff_list))
    max_x = max(max_x, max(edge_sim_diff_list))

  # For density plot increase by 50% to set plotting min and max values
  #
  if (density_plot == True):
    plot_min_x = min_x*1.5
    plot_max_x = max_x*1.5

  w,h = plt.figaspect(PLT_PLOT_RATIO)
  plt.figure(figsize=(w,h))

#  font = {'family':'sans-serif', 'weight':'normal', 'size':24}
#  plt.rc('font', **font)
#  plt.rcParams['text.usetex'] = True

  plt.title('Similarity differences between plain-text versus encoded ' + \
            'value pairs')

  plt.xlabel('Plain-text edge similarity minus encoded edge similarity')

  if (density_plot == True):
    plt.ylabel('Density of similarity differences')
  else:
    plt.ylabel('Frequency of similarity differences')

  max_x = -1
  max_y = -1
  min_x = 9999
  min_y = 9999

  for (i, min_sim) in enumerate(min_sim_list):
    edge_sim_diff_list = plot_list_dict[min_sim]

    c = plot_color_list[i]

    if (density_plot == True):

      if (len(edge_sim_diff_list) > 1):
        density = scipy.stats.kde.gaussian_kde(edge_sim_diff_list)
        x_list = numpy.arange(plot_min_x, plot_max_x,
                              (plot_max_x-plot_min_x)/50)
        y_list = density(x_list)
      else:
        x_list = [edge_sim_diff_list[0]]
        y_list = [1.0]

      plt.plot(x_list, y_list, color=(c,c,c,), #lw=2,
               label='MinSim='+str(min_sim))

      min_y = min(min_y, min(y_list))
      max_y = max(max_y, max(y_list))

    else:  # Frequency plot

      # Get counts of how often each similarity difference occurs
      #
      min_sim_plot_dict = {}
      for sim_diff in edge_sim_diff_list:
        round_sim_diff = round(sim_diff, round_num_digit)
        min_sim_plot_dict[round_sim_diff] = \
                                 min_sim_plot_dict.get(round_sim_diff, 0) + 1
      plot_x_list = []
      plot_y_list = []

      for (sim_diff, count) in sorted(min_sim_plot_dict.items()):
        plot_x_list.append(sim_diff)
        plot_y_list.append(count)

      max_x = max(max_x, max(plot_x_list))
      max_y = max(max_y, max(plot_y_list))

      min_x = min(min_x, min(plot_x_list))
      min_y = min(min_y, min(plot_y_list))

      plt.plot(plot_x_list, plot_y_list, color=(c,c,c,), #lw=2,
               label='MinSim='+str(min_sim))

  if (density_plot == False):
    x_margin = max(abs(min_x), max_x)*0.05  # Margin for plotting range
    plot_min_x = min_x - x_margin
    plot_max_x = max_x + x_margin

  y_margin = max_y*0.05

  plt.xlim(plot_min_x, plot_max_x)
  plt.ylim(min_y-y_margin, max_y+y_margin)

  plt.legend(loc="best") #, prop={'size':14}, bbox_to_anchor=(1, 0.5))

  plt.savefig(plot_file_name, bbox_inches='tight')

# -----------------------------------------------------------------------------
  
def calc_plot_histogram_feat_vec_cosine_sim(sim_graph1, sim_graph2,
                                            feat_vec_dict1, feat_vec_dict2,
                                            plot_file_name, round_num_digits=3,
                                            y_scale_log=True, over_sample=1):
  """Calculate the Cosine similarity between all true matching nodes across the
     two graphs as well as the same number of randomly selected non-matching
     nodes, and plot a histogram of the resulting similarity distributions.

     Input arguments:
       - sim_graph1       The first similarity graph from where edges are to
                          be compared.
       - sim_graph2       The second similarity graph from where edges are to
                          be compared.
       - feat_vec_dict1   The generated node feature vectors for the first
                          similarity graph.
       - feat_vec_dict2   The generated node feature vectors for the second
                          similarity graph.
       - plot_file_name   The name into which the plot is to be saved.
       - round_num_digit  The number of digits values are to be rounded to.
       - y_scale_log      A flag, if set to True (default) then the y-axis is
                          shown in log scale.
       - over_sample      The number of times more non-matches should be
                          sampled and compared.

     Output:
         - This method does not return anything.
  """

  # Get record values from both similarity graphs
  #
  sim_graph_node_dict1 = {}  # Keys will be original record values, values
  sim_graph_node_dict2 = {}  # their node keys

  for node_key_val in sim_graph1.nodes():

    # If a node has several record value tuples take the first one only
    #
    org_val = ' '.join(sorted(sim_graph1.node[node_key_val]['org_val_set'])[0])

    # Only use if the node has a non-empty original value
    #
    if (len(org_val) > 0):
      assert org_val not in sim_graph_node_dict1, org_val
      sim_graph_node_dict1[org_val] = node_key_val

  for node_key_val in sim_graph2.nodes():
    org_val = ' '.join(sorted(sim_graph2.node[node_key_val]['org_val_set'])[0])

    # Only use if the node has a non-empty original value
    #
    if (len(org_val) > 0):
      assert org_val not in sim_graph_node_dict2, org_val
      sim_graph_node_dict2[org_val] = node_key_val

  # Now get true matching original values across the two graphs
  #
  common_node_key_set = set(sim_graph_node_dict1.keys()) & \
                        set(sim_graph_node_dict2.keys())
  num_common_nodes = len(common_node_key_set)

  # Generate two dictionaries where keys will be Cosine similarities and
  # values counts of how many node pairs have that similarity
  #
  match_sim_histo_dict =     {}
  non_match_sim_histo_dict = {}

  for org_val in common_node_key_set:
    node_key_val1 = sim_graph_node_dict1[org_val]
    node_key_val2 = sim_graph_node_dict2[org_val]

    if (node_key_val1 in feat_vec_dict1) and (node_key_val2 in feat_vec_dict2):
      feat_vec1 = feat_vec_dict1[node_key_val1]
      feat_vec2 = feat_vec_dict2[node_key_val2]

      # Calculate the Cosine similarity beween the featue vectors
      #
      vec_len1 = math.sqrt(numpy.dot(feat_vec1, feat_vec1))
      vec_len2 = math.sqrt(numpy.dot(feat_vec2, feat_vec2))
      cosine = numpy.dot(feat_vec1,feat_vec2) / vec_len1 / vec_len2
      cos_sim = 1.0 - math.acos(min(cosine,1.0)) / math.pi

      cos_sim = round(cos_sim, round_num_digits)
      match_sim_histo_dict[cos_sim] = match_sim_histo_dict.get(cos_sim, 0) + 1

  num_match_sim = sum(match_sim_histo_dict.values())
  print 'Calculated %d match Cosine similarities based on feature vectors' % \
        (num_match_sim)

  # The number of non-matches we want
  num_non_match_sim_needed = num_match_sim * over_sample

  # Get all node key values from both feature vectors
  #
  node_key_val_list1 = feat_vec_dict1.keys()
  node_key_val_list2 = feat_vec_dict2.keys()

  non_match_key_pair_set = set()

  while (len(non_match_key_pair_set) < num_non_match_sim_needed):
    node_key_val1 = random.choice(node_key_val_list1)
    node_key_val2 = random.choice(node_key_val_list2)
    node_key_val_pair = (node_key_val1,node_key_val2)
    if node_key_val_pair not in non_match_key_pair_set:
      non_match_key_pair_set.add(node_key_val_pair)

      feat_vec1 = feat_vec_dict1[node_key_val1]
      feat_vec2 = feat_vec_dict2[node_key_val2]

      # Calculate the Cosine similarity beween the featue vectors
      #
      vec_len1 = math.sqrt(numpy.dot(feat_vec1, feat_vec1))
      vec_len2 = math.sqrt(numpy.dot(feat_vec2, feat_vec2))
      cosine = numpy.dot(feat_vec1,feat_vec2) / vec_len1 / vec_len2
      cos_sim = 1.0 - math.acos(min(cosine,1.0)) / math.pi

      cos_sim = round(cos_sim, round_num_digits)
      non_match_sim_histo_dict[cos_sim] = \
                         non_match_sim_histo_dict.get(cos_sim, 0) + 1

  match_x_val_list =     []
  match_y_val_list =     []
  non_match_x_val_list = []
  non_match_y_val_list = []

  for (cos_sim, count) in sorted(match_sim_histo_dict.items()):
    match_x_val_list.append(cos_sim)
    match_y_val_list.append(count)
  for (cos_sim, count) in sorted(non_match_sim_histo_dict.items()):
    non_match_x_val_list.append(cos_sim)
    non_match_y_val_list.append(count)

#  min_count = min(min(match_y_val_list), min(non_match_y_val_list))
  max_count = max(max(match_y_val_list), max(non_match_y_val_list))
  min_sim =   min(min(match_x_val_list), min(non_match_x_val_list))
  max_sim =   max(max(match_x_val_list), max(non_match_x_val_list))

  non_match_sim_histo_dict.clear()  # Not needed anymore
  match_sim_histo_dict.clear()
    
  # Plot the generated lists
  #
  w,h = plt.figaspect(PLT_PLOT_RATIO)
  plt.figure(figsize=(w,h))

  plt.title('Node Cosine similarity histograms')

  plt.xlabel('Node similarities')

  if (y_scale_log == True):
    plt.ylabel('Counts (log-scale)')
    plt.yscale('log')
    plt.ylim(0.9, max_count*1.5)
#    plt.ylim(min_count*0.667, max_count*1.5)
  else:
    plt.ylabel('Counts')
    plt.ylim(0, max_count*1.05)
#    plt.ylim(min_count-max_count*0.05, max_count*1.05)

  x_tol = (max_sim-min_sim)/20.0

  plt.xlim(min_sim-x_tol, max_sim+x_tol)

  plt.plot(match_x_val_list, match_y_val_list, color = 'r', #lw=2,
           label='Matches')
  plt.plot(non_match_x_val_list, non_match_y_val_list, color = 'b', #lw=2,
           label='Non-matches')

  plt.legend(loc="best") #, prop={'size':14}, bbox_to_anchor=(1, 0.5))

  plt.savefig(plot_file_name, bbox_inches='tight')

  print '  Written plot into file:', plot_file_name,
  print
  
#------------------------------------------------------------------------------
  
def encode_hlsh_blocking(encode_data_dict, hlsh_sample_size, hlsh_num_sample, 
                         bit_array_len, random_seed=None):
  
  start_time = time.time()
  
  encode_block_dict = {}

  if (random_seed != None):
    numpy.random.seed(random_seed)

  # First generate the required list of bit position arrays to be used for
  # sampling ('hlsh_num_sample' arrays each of length 'hlsh_sample_size')
  #
  bit_sample_list = []

  for sample_num in xrange(hlsh_num_sample):
    bit_sample_list.append(random.sample(xrange(bit_array_len),
                                         hlsh_sample_size))
    if (len(bit_sample_list) > 1):
      assert bit_sample_list[-1] != bit_sample_list[-2]  # Check uniqueness

  # Now apply HLSH on both similarity hash dictionaries
  #
  for (node_key_val, q_gram_set_bit_array) in encode_data_dict.iteritems():
    q_gram_set, dict_bit_array = q_gram_set_bit_array
    
    # Loop over all HSLH bit position sample lists
    #
    for (sample_num, bit_pos_list) in enumerate(bit_sample_list):
      sample_bit_array = bitarray.bitarray(hlsh_sample_size)
      sample_num_str = str(sample_num)+'-'

      for (i, bit_pos) in enumerate(bit_pos_list):
        sample_bit_array[i] = dict_bit_array[bit_pos]

      # Generate the HLSH block (dictionary) key
      #
      hlsh_dict_key = sample_num_str+sample_bit_array.to01()

      # Add the current key value into the corresponding HLSH block
      #
      hlsh_block_key_set = encode_block_dict.get(hlsh_dict_key, set())
      hlsh_block_key_set.add(node_key_val)
      encode_block_dict[hlsh_dict_key] = hlsh_block_key_set

  # Print summary statistics about the generated LSH blocks
  #
  print 'Number of blocks for the Encoded HLSH index: %d' % \
        (len(encode_block_dict)) + \
        '  (with sample size: %d, and number of samples: %d' % \
        (hlsh_sample_size, hlsh_num_sample)
  hlsh_block_size_list = []
  for hlsh_block_key_set in encode_block_dict.itervalues():
    hlsh_block_size_list.append(len(hlsh_block_key_set))
  print '  Minimum, average, median and maximum block sizes: ' + \
        '%d / %.2f / %d / %d' % (min(hlsh_block_size_list),
                             numpy.mean(hlsh_block_size_list),
                             numpy.median(hlsh_block_size_list),
                             max(hlsh_block_size_list))
  
  print '  Time used: %.3f sec' % (time.time() - start_time)
  print

  return encode_block_dict

#------------------------------------------------------------------

def rec_soundex_blocking(soundex_val_dict, graph_node_key_id_dict):
  
  block_dict = {}
  
  for ent_id, soundex_val_list in soundex_val_dict.iteritems():
    
    rec_bkv = ''  # Initialise the blocking key value for this record

    # Process selected blocking attributes
    #
    for attr_val in soundex_val_list:

      if (attr_val == ''):
        rec_bkv += 'z000'  # Often used as Soundex code for empty values

      else:  # Convert the value into its Soundex code

        attr_val = attr_val.lower()
        sndx_val = attr_val[0]  # Keep first letter

        for c in attr_val[1:]:  # Loop over all other letters
          if (c in 'aehiouwy'):  # Not inlcuded into Soundex code
            pass
          elif (c in 'bfpv'):
            if (sndx_val[-1] != '1'):  # Don't add duplicates of digits
              sndx_val += '1'
          elif (c in 'cgjkqsxz'):
            if (sndx_val[-1] != '2'):  # Don't add duplicates of digits
              sndx_val += '2'
          elif (c in 'dt'):
            if (sndx_val[-1] != '3'):  # Don't add duplicates of digits
              sndx_val += '3'
          elif (c in 'l'):
            if (sndx_val[-1] != '4'):  # Don't add duplicates of digits
              sndx_val += '4'
          elif (c in 'mn'):
            if (sndx_val[-1] != '5'):  # Don't add duplicates of digits
              sndx_val += '5'
          elif (c in 'r'):
            if (sndx_val[-1] != '6'):  # Don't add duplicates of digits
              sndx_val += '6'

        if (len(sndx_val) < 4):
          sndx_val += '000'  # Ensure enough digits

        sndx_val = sndx_val[:4]  # Maximum length is 4
        rec_bkv += sndx_val
      
    # Insert the blocking key value and record into blocking dictionary
    #
    if (rec_bkv in block_dict): # Block key value in block index

      # Only need to add the record
      #
      rec_id_set = block_dict[rec_bkv]
      node_key_val = graph_node_key_id_dict[ent_id]
      rec_id_set.add(node_key_val)

    else: # Block key value not in block index

      # Create a new block and add the record identifier
      #
      node_key_val = graph_node_key_id_dict[ent_id]
      rec_id_set = set([node_key_val])
      

    block_dict[rec_bkv] = rec_id_set  # Store the new block
    
  return block_dict

# =============================================================================

class SimGraph():
  """A class to implement a similarity graph, where nodes correspond to records
     and edges to the similarities calculated between them.

     The node identifiers are record values, such as strings, q-gram sets, or
     bit arrays (encodings).

     The attributes to be stored in the graph are:
     - For edges, their similarity as 'sim' (assumed to be normalised into
       the range 0..1)
     - For nodes, a set off record identifiers (that have the node's value).
  """

  def __init__(self):
    """Initialise a similarity graph.

       Input arguments:
         - This method has no input.

       Output:
         - This method does not return anything.
    """

    # Initialise an empty similarity graph
    #
    self.sim_graph = networkx.Graph()

  # ---------------------------------------------------------------------------

  def add_rec(self, node_key_val, ent_id, org_val):
    """Add a new node to the graph, where the identifier (key) of this node
       is the object that will be compared when similarities are calculated
       between records (this can be a string or q-gram set for a plain-text
       data set, and a bit array for an encoded data set). We also store the
       entity identifier of this record with this node (as a set of entity
       identifiers - i.e. all entities that have that node_key_val), and a set
       of the original values (attribute strings as a tuple) of these records.

       Input arguments:
         - node_key_val  The identifier for this node.
         - ent_id        The entity identifier for this record from the data
                         set.
         - org_val       The original (attribute) value of this record.

       Output:
         - This method does not return anything.
    """

    sim_graph = self.sim_graph

    # If the node key value already exists then add the entity identifier and
    # the original value to the corresponding sets
    #
    if (node_key_val in sim_graph):
      ent_id_set = sim_graph.node[node_key_val]['ent_id_set']
      ent_id_set.add(ent_id)
      org_val_set = sim_graph.node[node_key_val]['org_val_set']
      org_val_set.add(tuple(org_val))

    else:  # Create a new node
      sim_graph.add_node(node_key_val)
      ent_id_set = set([ent_id])
      org_val_set = set([tuple(org_val)])

    sim_graph.node[node_key_val]['ent_id_set'] =  ent_id_set
    sim_graph.node[node_key_val]['org_val_set'] = org_val_set

  
  # ---------------------------------------------------------------------------
  
  def add_feat_sim_node(self, node_key_val, node_type):
    
    feat_sim_graph = self.sim_graph
    
    if (node_key_val not in feat_sim_graph):
      feat_sim_graph.add_node(node_key_val)
      feat_sim_graph.node[node_key_val]['type'] = node_type
    else:
      assign_node_type = feat_sim_graph.node[node_key_val]['type']
      assert assign_node_type == node_type, (assign_node_type, node_type)
      
  # ---------------------------------------------------------------------------
  
  def add_feat_sim_edge(self, encode_node_key, plain_node_key, sim):
    
    feat_sim_graph = self.sim_graph
    
    if feat_sim_graph.has_edge(encode_node_key, plain_node_key):  # Check existing similarity is same
      if (feat_sim_graph.edge[encode_node_key][plain_node_key]['feat_sim'] != sim):
        print '  *** Warning: Edge between %s and %s has a different ' % \
              (encode_node_key,plain_node_key) + \
              'similarity to new similarity: %.3f versus %.3f' \
              % (sim_graph.edge[encode_node_key][plain_node_key]['feat_sim'], sim) + \
              '(old value will be overwritten)'
    else:
      feat_sim_graph.add_edge(encode_node_key,plain_node_key)
    
    feat_sim_graph.edge[encode_node_key][plain_node_key]['feat_sim'] = sim  # Add or update edge
  
  # ---------------------------------------------------------------------------
  
  def add_feat_conf_edge(self):
    
    print 'Adding confidence values to each edge of the feature graph'
    
    start_time = time.time()
    
    feat_sim_graph = self.sim_graph
    
    conf_val_list = [] # For basic statistics
    
    for (encode_key_val, plain_key_val) in feat_sim_graph.edges():
      
      edge_sim_val = feat_sim_graph.edge[encode_key_val][plain_key_val]['feat_sim']
      
      #encode_node_neighbors = feat_sim_graph.neighbors(encode_key_val)
      #encode_node_neighbors = feat_sim_graph.neighbors(plain_key_val)
      
      enc_neighbour_sim_sum = 0.0
      plain_neighbour_sim_sum = 0.0
      
      conf_val = 0.0
      neighbour_sim_list = []
        
      for neigh_key1, neigh_key2 in feat_sim_graph.edges(encode_key_val):
        
        if(neigh_key2 != plain_key_val):
          
          #enc_neighbour_sim_sum += feat_sim_graph[neigh_key1][neigh_key2]['feat_sim']
          neighbour_sim_list.append(feat_sim_graph.edge[neigh_key1][neigh_key2]['feat_sim'])
            
      for neigh_key1, neigh_key2 in feat_sim_graph.edges(plain_key_val):
        
        if(neigh_key2 != encode_key_val):
          
          #plain_neighbour_sim_sum += feat_sim_graph[neigh_key1][neigh_key2]['feat_sim']
          neighbour_sim_list.append(feat_sim_graph.edge[neigh_key1][neigh_key2]['feat_sim'])
      
      if(len(neighbour_sim_list) > 0):
        conf_val = edge_sim_val / numpy.mean(neighbour_sim_list)
      else:
        conf_val = 10.0
      conf_val_list.append(conf_val)
      
      feat_sim_graph.edge[encode_key_val][plain_key_val]['conf'] = conf_val
    
    conf_cal_time = time.time() - start_time
    
    print '  Minimum, average, median and maximum confidence: ' + \
        '%.3f / %.3f / %.3f / %.3f' % (min(conf_val_list),
                                 numpy.mean(conf_val_list),
                                 numpy.median(conf_val_list),
                                 max(conf_val_list))
  
  
  def get_conf_val(self, encode_key, plain_key):
    
    feat_sim_graph = self.sim_graph
    
    conf_val = feat_sim_graph.edge[encode_key][plain_key]['conf']
    
    return conf_val
  
  
  # ---------------------------------------------------------------------------
  
  def get_sim_pair_list(self):
    
    feat_sim_graph = self.sim_graph
  
  # ---------------------------------------------------------------------------

  def add_edge_sim(self, node_key_val1, node_key_val2, sim):
    """Add an (undirected) edge to the similarity graph between the two given
       nodes with the given similarity. It is assumed the two node key values
       are available as nodes in the graph, if this is not the case the
       function with exit with an error.

       Input arguments:
         - node_key_val1  The first node key value - needs to be available in
                          the graph.
         - node_key_val2  The second node key value - needs to be available in
                          the graph.
         - sim            The similarity value of the edge (edge attribute).

       Output:
         - This method does not return anything.
    """

    sim_graph = self.sim_graph

    # Sort values so we only insert an edge once
    #
    val1s, val2s = sorted([node_key_val1, node_key_val2])

    if sim_graph.has_edge(val1s, val2s):  # Check existing similarity is same
      if (sim_graph.edge[val1s][val2s]['sim'] != sim):
        print '  *** Warning: Edge between %s and %s has a different ' % \
              (node_key_val1,node_key_val2) + \
              'similarity to new similarity: %.3f versus %.3f' \
              % (sim_graph.edge[val1s][val2s]['sim'], sim) + \
              '(old value will be overwritten)'

    else:
      sim_graph.add_edge(val1s,val2s)
    sim_graph.edge[val1s][val2s]['sim'] = sim  # Add or update edge

  # ---------------------------------------------------------------------------
  
  def show_graph_characteristics(self, min_sim_list=None, graph_name=''):
    """Print the number of nodes and edges in the similarity graph, their
       degree distributions, number of connected components and their sizes.

       Edges, number and sizes of connected components, and number of
       singletons are calculated for different edge minimum similarities if
       the list 'min_sim_list' is provided.

       Input arguments:
         - min_sim_list  A list with similarity values, or None (default).
         - graph_name    A string to be printed with the name of the graph

       Output:
         - This method does not return anything.
    """

    sim_graph = self.sim_graph

    if (graph_name != ''):
      print 'Similarity graph characteristics for "%s":' % (graph_name)
    else:
      print 'Similarity graph characteristics:'
    print '  Number of nodes:    ', len(sim_graph.nodes())

    # Get a distribution of the node frequencies
    # (keys will be counts, values how many nodes have that count)
    #
    node_freq_dict = {}
    for node_key_val in sim_graph.nodes():
      node_freq = len(sim_graph.node[node_key_val]['ent_id_set'])
      node_freq_dict[node_freq] = node_freq_dict.get(node_freq, 0) + 1
    print '    Distribution of node frequencies:', \
          sorted(node_freq_dict.iteritems())

    node_freq_dict.clear()  # Not needed anymore

    print '  Number of all edges:', len(sim_graph.edges())

    # Count the number of singletons (identifiers with no edges)
    #
    sim_graph_degree_list = sim_graph.degree().values()
    num_singleton =         sim_graph_degree_list.count(0)
    sim_graph_degree_list = []  # Not needed anymore

    print '    Number of singleton nodes:', num_singleton

    # Calculate number of edges and singletons for the given similarity
    # thresholds
    #
    if (min_sim_list != None):
      for min_sim in min_sim_list:
        print '  For minimum similarity threshold %.2f:' % (min_sim)

        num_min_sim_edges = 0

        for (node_key_val1,node_key_val2) in sim_graph.edges():
          if (sim_graph.edge[node_key_val1][node_key_val2]['sim'] >= min_sim):
            num_min_sim_edges += 1
        print '    Number of edges:          ', num_min_sim_edges

        # For this minimum similarity, get the distribution of the degrees of
        # all nodes
        #
        min_sim_degree_dist_dict = {}

        for node_key_val in sim_graph.nodes():
          node_degree = 0

          # Loop over all the node's neighbours
          #
          for neigh_key_val in sim_graph.neighbors(node_key_val):

            if (sim_graph.edge[node_key_val][neigh_key_val]['sim'] >= min_sim):
              node_degree += 1
          min_sim_degree_dist_dict[node_degree] = \
                     min_sim_degree_dist_dict.get(node_degree, 0) + 1
        print '    Degree distribution:      ', \
              sorted(min_sim_degree_dist_dict.items())[:20], '.....', \
              sorted(min_sim_degree_dist_dict.items())[-20:]

        min_sim_degree_dist_dict.clear()  # Not needed anymore
      
        # Generate the graph with only edges of the current minimum similarity
        # in order to find connected components
        #
        min_sim_graph = networkx.Graph()
        for node_key_val in sim_graph.nodes():
          min_sim_graph.add_node(node_key_val)
        for (node_key_val1,node_key_val2) in sim_graph.edges():
           if (sim_graph.edge[node_key_val1][node_key_val2]['sim'] >= min_sim):
              min_sim_graph.add_edge(node_key_val1,node_key_val2)

        conn_comp_list = list(networkx.connected_components(min_sim_graph))

        conn_comp_size_dist_dict = {}

        for conn_comp in conn_comp_list:
          conn_comp_size = len(conn_comp)
          conn_comp_size_dist_dict[conn_comp_size] = \
                   conn_comp_size_dist_dict.get(conn_comp_size, 0) + 1        
        print '    Number of connected components:', len(conn_comp_list)
        print '      Size of connected components distribution:', \
              sorted(conn_comp_size_dist_dict.items())

        print '    Number of singletons: %d' % \
              (conn_comp_size_dist_dict.get(1, 0))

        min_sim_graph.clear()  # Not needed anymore
        conn_comp_list = []
        conn_comp_size_dist_dict.clear()
 
        print

  # ---------------------------------------------------------------------------

  def remove_singletons(self):
    """Remove all singleton nodes (nodes that have no edge to any other node).

       Input arguments:
         - This method has no input.

       Output:
         - num_singleton  The number of singletones removed
    """

    sim_graph = self.sim_graph

    num_singleton = 0

    for (node_key_val, node_degree) in sim_graph.degree().iteritems():
      if (node_degree == 0):
        sim_graph.remove_node(node_key_val)
        num_singleton += 1

    print 'Removed %d singleton nodes' % (num_singleton)
    print

    return num_singleton

  # ---------------------------------------------------------------------------

# Improved version of the function below: do piece-wise linear regression
#  See for example: https://datascience.stackexchange.com/questions/8457/python-library-for-segmented-regression-a-k-a-piecewise-regression

  # ---------------------------------------------------------------------------
  
  def get_sim_differences(self, other_node_val_dict, other_sim_funct,
                          sim_diff_inter_size, num_samples,
                          plot_file_name=None):
    """Sample edges of the similarity graph and calculate similarities between
       the corresponding same pair of nodes in the given other node dictionary
       (assumed to contain q-gram sets and bit arrays), and then calculate the
       similarity differences between the two edges. For the given different
       similarity intervals calculate and return an average similarity
       difference.

       Input arguments:
         - other_node_val_dict  A dictionary with plain-text node key values
                                as keys and bit arrays (encoded values) as
                                values.
         - other_sim_funct      The similarity function to be used to compare
                                the encoded bit arrays.
         - sim_diff_inter_size  The size of the similarity difference interval
                                to consider
         - num_samples          The total number of edges to be sampled.
         - plot_file_name       If provided then a plot will be generated with
                                the similarity differences as well as their
                                averages over the different similarity
                                intervals.

       Output:
         - sim_diff_dict  For each minimum similarity (key) given in the input
                          'sim_interval_list' list return a pair with the
                          average similarity difference in this interval and
                          the number of samples in the interval. Note that if
                          there are no samples in an interval then the
                          similarity difference will be set to 0.0 (no
                          difference).
    """

    sim_graph = self.sim_graph

    # Generate a list of all similarity intervals downwards then reverse
    #
    sim_interval_list = [1.0]
    while ((sim_interval_list[-1]-sim_diff_inter_size) > 0.0):
      sim_interval_list.append(sim_interval_list[-1] - sim_diff_inter_size)
    sim_interval_list[-1] = 0.0  # Ensure lowest similarity is not negative

    sim_interval_list.reverse()  # Make 1.0 last

    # Initialise an empty similarity difference dictionary
    #
    sim_diff_dict = {}
    for sim_interval in sim_interval_list:
      sim_diff_dict[sim_interval] = []

    all_edge_list = list(sim_graph.edges())

    # If a plot is to be generated create lists of x- and y values for plotting
    #
    if (plot_file_name != None):
      x_val_list = []  # The similarity differences
      y_val_list = []  # The plain-text similarities

    # Reduce number of samples is there are not enough edges
    #
    if (len(all_edge_list) < num_samples):
      num_samples = len(all_edge_list)-1

    print 'Sample %d from %d edges in graph to calculate similarity ' % \
          (num_samples, len(all_edge_list)) + 'differences'
    print '  Using similarity intervals:', sim_interval_list

    for s in xrange(num_samples):

      sample_edge = random.choice(all_edge_list)

      # Get the two key values of this edge and check they both also occur in
      # in the encoded data set
      #
      node_key_val1, node_key_val2 = sample_edge

      if ((node_key_val1 in other_node_val_dict) and \
          (node_key_val2 in other_node_val_dict)):

        bit_array1 = other_node_val_dict[node_key_val1]
        bit_array2 = other_node_val_dict[node_key_val2]

        ba_sim = other_sim_funct(bit_array1, bit_array2)

        edge_sim = sim_graph.edge[node_key_val1][node_key_val2]['sim']

        # If sim_diff is positive: encoded sim is larger than plain-text sim
        # If sim_diff is negative: encoded sim is smaller than plain-text sim
        #
        sim_diff = ba_sim - edge_sim

        if (plot_file_name != None):
          x_val_list.append(sim_diff)
          y_val_list.append(ba_sim)

        # Add the similarity difference to the dictionary to be returned in
        # the appropriate similarity interval
        i = 0
        while (ba_sim > sim_interval_list[i]):
          i += 1
        sim_interval = sim_interval_list[i]

        assert ba_sim <= sim_interval

        # The list of similarity differences for this interval
        #
        sim_interval_sim_diff_list = sim_diff_dict[sim_interval]
        sim_interval_sim_diff_list.append(sim_diff)
        sim_diff_dict[sim_interval] = sim_interval_sim_diff_list

        all_edge_list.remove(sample_edge)  # Ensure each edge is only used once
  
    # Convert similarity difference lists into averages and number of samples
    # (first remove all low ones that do not contain any samples)
    #
    for sim_interval in sim_interval_list:
      if (len(sim_diff_dict[sim_interval]) == 0):
        del sim_diff_dict[sim_interval]
      else:
        break  # First non-empty interval, exit the loop

    for sim_interval in sorted(sim_diff_dict.keys()):
      sim_interval_sim_diff_list = sim_diff_dict[sim_interval]

      if (len(sim_interval_sim_diff_list) == 0):
        print '  *** Warning no samples in similarity interval >= %.2f ***' % \
              (sim_interval)
        sim_diff_dict[sim_interval] = (0.0, 0)

      else:
        avr_sim_diff_inter = numpy.mean(sim_interval_sim_diff_list)
        num_samples_inter =  len(sim_interval_sim_diff_list)
        sim_diff_dict[sim_interval] = (numpy.mean(sim_interval_sim_diff_list),
                                       len(sim_interval_sim_diff_list))
        print '  Similarity interval >= %.2f has %d samples and average ' % \
              (sim_interval, num_samples_inter) + \
              'similarity difference is %.3f' % (avr_sim_diff_inter)
    print

    all_edge_list = []  # Not needed anymore

    # Generate the plot if needed
    #
    if (plot_file_name != None) and (len(x_val_list) > 0):

      x_min = min(x_val_list)
      x_max = max(x_val_list)
      x_diff = x_max - x_min 
      plot_min_x = x_min - x_diff*0.05
      plot_max_x = x_max + x_diff*0.05

      plot_min_y = min(y_val_list) - min(y_val_list)*0.05
      plot_max_y = 1.0 + min(y_val_list)*0.05

      # Plot the generated lists as scatter plot
      #
      w,h = plt.figaspect(PLT_PLOT_RATIO)
      plt.figure(figsize=(w,h))

      plt.title('Similarity differences of plain-text versus encoded edges')

      plt.xlabel('Encoded similarity minus plain-text similarity')
      plt.ylabel('Encoded similarity')

      plt.xlim(plot_min_x, plot_max_x)
      plt.ylim(plot_min_y, plot_max_y)

      plt.scatter(x_val_list, y_val_list, color='k')

      # Add the calculated averages as a line plot
      #
      avr_x_list = []
      avr_y_list = []

      for sim_interval in sorted(sim_diff_dict.keys()):
        sim_diff = sim_diff_dict[sim_interval][0]

        avr_x_list.append(sim_diff)
        avr_y_list.append(sim_interval)

      plt.plot(avr_x_list, avr_y_list, color='r', lw=2, label='Avr sim diff')

      plt.legend(loc="best",) #, prop={'size':14}, bbox_to_anchor=(1, 0.5))

      plt.savefig(plot_file_name, bbox_inches='tight')

    return sim_diff_dict

  # ---------------------------------------------------------------------------
  
  def comp_sim_differences(self, other_node_val_dict, other_sim_graph,
                           sim_diff_inter_size, num_samples,
                           plot_file_name=None):
    """Sample edges of the similarity graph that also occur in the other
       similarity graph, and calculate the similarity differences between
       these edges, report the average similarity difference for each interval
       based on the 'sim_diff_inter_size' and plot if a file name is given.

       Input arguments:
         - other_node_val_dict  A dictionary with plain-text node key values
                                as keys and bit arrays (encoded values) as
                                values.
         - other_sim_graph      The other similarity graph from where edges
                                are to be compared.
         - sim_diff_inter_size  The size of the similarity difference interval
                                to consider
         - num_samples          The total number of edges to be sampled.
         - plot_file_name       If provided then a plot will be generated with
                                the similarity differences as well as their
                                averages over the different similarity
                                intervals.

       Output:
         - This method does not return anything.
    """

    sim_graph = self.sim_graph

    # Generate a list of all similarity intervals downwards then reverse
    #
    sim_interval_list = [1.0]
    while ((sim_interval_list[-1]-sim_diff_inter_size) > 0.0):
      sim_interval_list.append(sim_interval_list[-1] - sim_diff_inter_size)
    sim_interval_list[-1] = 0.0  # Ensure lowest similarity is not negative

    sim_interval_list.reverse()  # Make 1.0 last

    # Initialise an empty similarity difference dictionary
    #
    sim_diff_dict = {}
    for sim_interval in sim_interval_list:
      sim_diff_dict[sim_interval] = []

    # Get pairs of actual record values from both similarity graphs an their
    # similarities
    #
    sim_graph_pair_dict = {}
    common_sim_graph_pair_dict = {} # With similarties from both graphs

    for (node_key_val1, node_key_val2) in sim_graph.edges():

      # If a node has several record value tuples take the first one only
      #
      str1 = ' '.join(sorted(sim_graph.node[node_key_val1]['org_val_set'])[0])
      str2 = ' '.join(sorted(sim_graph.node[node_key_val2]['org_val_set'])[0])

      # Only use if both the nodes have non-empty original values
      #
      if (len(str1) > 0) and (len(str2) > 0):
        rec_pair_dict_key = tuple(sorted([str1, str2]))
        assert rec_pair_dict_key not in sim_graph_pair_dict
        sim_graph_pair_dict[rec_pair_dict_key] = \
              sim_graph.edge[node_key_val1][node_key_val2]['sim']

    for (node_key_val1, node_key_val2) in other_sim_graph.edges():
      str1 = \
       ' '.join(sorted(other_sim_graph.node[node_key_val1]['org_val_set'])[0])
      str2 = \
       ' '.join(sorted(other_sim_graph.node[node_key_val2]['org_val_set'])[0])

      if (len(str1) > 0) and (len(str2) > 0):
        rec_pair_dict_key = tuple(sorted([str1, str2]))

        if (rec_pair_dict_key in sim_graph_pair_dict):
          sim_graph_sim = sim_graph_pair_dict[rec_pair_dict_key]
          common_sim_graph_pair_dict[rec_pair_dict_key] = \
            (sim_graph_sim,
             other_sim_graph.edge[node_key_val1][node_key_val2]['sim'])

          # Once we have enough pairs exit the loop
          #
          if (len(common_sim_graph_pair_dict) == num_samples):
            break  # Exit the for loop

    sim_graph_pair_dict.clear()  # Not needed anymore

    print 'Using %d common edges to calculate similarity differences' % \
          (len(common_sim_graph_pair_dict)), num_samples
    print '  Using similarity intervals:', sim_interval_list

    # If a plot is to be generated create lists of x- and y values for plotting
    #
    if (plot_file_name != None):
      x_val_list = []  # The similarity differences
      y_val_list = []  # The plain-text edge similarities

    for (edge_sim, other_edge_sim) in common_sim_graph_pair_dict.itervalues():

        # If sim_diff is positive: encoded sim is larger than plain-text sim
        # If sim_diff is negative: encoded sim is smaller than plain-text sim
        #
        sim_diff = other_edge_sim - edge_sim

        if (plot_file_name != None):
          x_val_list.append(sim_diff)
          y_val_list.append(other_edge_sim)

        # Add the similarity difference to the dictionary to be returned in
        # the appropriate similarity interval
        i = 0
        while (other_edge_sim > sim_interval_list[i]):
          i += 1
        sim_interval = sim_interval_list[i]

        assert other_edge_sim <= sim_interval

        # The list of similarity differences for this interval
        #
        sim_interval_sim_diff_list = sim_diff_dict[sim_interval]
        sim_interval_sim_diff_list.append(sim_diff)
        sim_diff_dict[sim_interval] = sim_interval_sim_diff_list

    # Convert similarity difference lists into averages and number of samples
    # (first remove all low ones that do not contain any samples)
    #
    for sim_interval in sim_interval_list:
      if (len(sim_diff_dict[sim_interval]) == 0):
        del sim_diff_dict[sim_interval]
      else:
        break  # First non-empty interval, exit the loop

    for sim_interval in sorted(sim_diff_dict.keys()):
      sim_interval_sim_diff_list = sim_diff_dict[sim_interval]

      if (len(sim_interval_sim_diff_list) == 0):
        print '  *** Warning no samples in similarity interval <= %.2f ***' % \
              (sim_interval)
        sim_diff_dict[sim_interval] = (0.0, 0)

      else:
        avr_sim_diff_inter = numpy.mean(sim_interval_sim_diff_list)
        num_samples_inter =  len(sim_interval_sim_diff_list)
        sim_diff_dict[sim_interval] = (numpy.mean(sim_interval_sim_diff_list),
                                       len(sim_interval_sim_diff_list))
        print '  Similarity interval >= %.2f has %d samples and average ' % \
              (sim_interval, num_samples_inter) + \
              'similarity difference is %.3f' % (avr_sim_diff_inter)
    print

    common_sim_graph_pair_dict.clear()  # Not needed anymore

    # Generate the plot if needed
    #
    if (plot_file_name != None):

      x_min = min(x_val_list)
      x_max = max(x_val_list)
      x_diff = x_max - x_min 
      plot_min_x = x_min - x_diff*0.05
      plot_max_x = x_max + x_diff*0.05

      plot_min_y = min(y_val_list) - min(y_val_list)*0.05
      plot_max_y = 1.0 + min(y_val_list)*0.05

      # Plot the generated lists as scatter plot
      #
      w,h = plt.figaspect(PLT_PLOT_RATIO)
      plt.figure(figsize=(w,h))

      plt.title('Adjusted similarity differences of plain-text versus ' + \
                'encoded edges')

      plt.xlabel('Encoded similarity minus plain-text similarity')
      plt.ylabel('Encoded similarity')

      plt.xlim(plot_min_x, plot_max_x)
      plt.ylim(plot_min_y, plot_max_y)

      plt.scatter(x_val_list, y_val_list, color='k')

      # Add the calculated averages as a line plot
      #
      avr_x_list = []
      avr_y_list = []

      for sim_interval in sorted(sim_diff_dict.keys()):
        sim_diff = sim_diff_dict[sim_interval][0]

        avr_x_list.append(sim_diff)
        avr_y_list.append(sim_interval)

      plt.plot(avr_x_list, avr_y_list, color='r', lw=2, label='Avr sim diff')

      plt.legend(loc="best",) #, prop={'size':14}, bbox_to_anchor=(1, 0.5))

      plt.savefig(plot_file_name, bbox_inches='tight')

  
  # ---------------------------------------------------------------------------
  
  def apply_regre(self, other_node_val_dict, other_sim_funct, 
                  num_samples, regre_model_str, plot_data_name, 
                  plot_file_name=None):
    
    """Sample edges of the similarity graph and calculate similarities between
       the corresponding same pair of nodes in the given other node dictionary
       (assumed to contain q-gram sets and bit arrays), and then calculate the
       similarity differences between the two edges. For the given different
       similarity intervals calculate and return an average similarity
       difference.

       Input arguments:
         - other_node_val_dict  A dictionary with plain-text node key values
                                as keys and bit arrays (encoded values) as
                                values.
         - other_sim_funct      The similarity function to be used to compare
                                the encoded bit arrays.
         - sim_diff_inter_size  The size of the similarity difference interval
                                to consider
         - num_samples          The total number of edges to be sampled.
         - plot_file_name       If provided then a plot will be generated with
                                the similarity differences as well as their
                                averages over the different similarity
                                intervals.

       Output:
         - bayes_regre_model  The trained Bayesian regression model
    """

    sim_graph = self.sim_graph # Encoded graph

    all_edge_list = list(sim_graph.edges())

    x_val_list = []  # The plain-text similarities (independent/predictor variable)
    q_val_list = []  # Number of unique q-grams in both records (independent/predictor variable)
    y_val_list = []  # The encoded similarities (dependent/response variable)
    

    # Reduce number of samples if there are not enough edges
    #
    if (len(all_edge_list) < num_samples):
      num_samples = len(all_edge_list)-1

    print 'Sample %d from %d edges in graph to calculate similarity ' % \
          (num_samples, len(all_edge_list)) + 'differences'
    print
    
    select_num_samples = 0
    
    #encode_node_val_dict[node_key_val] = (q_gram_set, bit_array)

    while(select_num_samples < num_samples and len(all_edge_list) > 0):
    #for s in xrange(num_samples):

      sample_edge = random.choice(all_edge_list)

      # Get the two key values of this edge and check they both also occur in
      # in the encoded data set
      #
      node_key_val1, node_key_val2 = sample_edge
      
      
      q_gram_set1, bit_array1 = other_node_val_dict[node_key_val1]
      q_gram_set2, bit_array2 = other_node_val_dict[node_key_val2]
      
      
      full_q_gram_set = q_gram_set1.union(q_gram_set2)

      qg_sim = other_sim_funct(q_gram_set1, q_gram_set2)

      edge_sim = sim_graph.edge[node_key_val1][node_key_val2]['sim'] # encoded similarity

      x_val_list.append(qg_sim)
      y_val_list.append(edge_sim)
      q_val_list.append(len(full_q_gram_set))

      all_edge_list.remove(sample_edge)  # Ensure each edge is only used once
      
      select_num_samples += 1
      
      if(select_num_samples%5000 == 0):
        print '  Selected %d number of samples' %select_num_samples
      
      
      
#===============================================================================
#       if ((q_gram_key_val1 in other_node_val_dict) and \
#           (q_gram_key_val2 in other_node_val_dict)):
# 
#         q_gram_set1 = other_node_val_dict[q_gram_key_val1]
#         q_gram_set2 = other_node_val_dict[q_gram_key_val2]
#         
#         full_q_gram_set = q_gram_set1.union(q_gram_set2)
# 
#         qg_sim = other_sim_funct(q_gram_set1, q_gram_set2)
# 
#         edge_sim = sim_graph.edge[node_key_val1][node_key_val2]['sim'] # encoded similarity
# 
#         x_val_list.append(qg_sim)
#         y_val_list.append(edge_sim)
#         q_val_list.append(len(full_q_gram_set))
# 
#         all_edge_list.remove(sample_edge)  # Ensure each edge is only used once
#         
#         select_num_samples += 1
#===============================================================================

    #Reshape the X value list if there is only one feature
    #x_val_array = numpy.array(x_val_list).reshape(-1, 1)
    
    
    
    if(regre_model_str == 'linear'):
      
      print 'Traning the linear regression model'
      
      x_val_array = numpy.array(zip(x_val_list,q_val_list))
    
      # Split the data for the training and testing samples
      x_train, x_test, y_train, y_test = train_test_split(x_val_array, y_val_list, test_size=0.25, random_state=42)
      
      # Train the model
      reg_model = linear_model.LinearRegression()
      
      reg_model.fit(x_train, y_train)
      
      # Testing the model
      y_predict = reg_model.predict(x_test)
      
    elif(regre_model_str == 'isotonic'):
      
      print 'Traning the isotonic regression model'
      
      x_train, x_test, y_train, y_test = train_test_split(x_val_list, y_val_list, test_size=0.25, random_state=42)
    
      reg_model = IsotonicRegression()
  
      reg_model.fit_transform(x_train, y_train)
      
      y_predict = reg_model.predict(x_test)
      
    elif(regre_model_str == 'poly'):
      
      print 'Traning the polynomial regression model'
      
      x_val_array = numpy.array(zip(x_val_list,q_val_list))
    
      # Split the data for the training and testing samples
      x_train, x_test, y_train, y_test = train_test_split(x_val_array, y_val_list, test_size=0.25, random_state=42)
      
      poly = PolynomialFeatures(degree=2)
      x_train_poly = poly.fit_transform(x_train)
      x_test_poly = poly.fit_transform(x_test)
      
      # Train the model
      reg_model = linear_model.LinearRegression()
      
      reg_model.fit(x_train_poly, y_train)
      
      # Testing the model
      y_predict = reg_model.predict(x_test_poly)
    
    else:
      raise Exception, '***WARNING!, wrong regression model'   
    
    y_p_new = []
    y_t_new = []
    for i, y_p in enumerate(y_predict):
      y_t = y_test[i]
      
      if(y_p > 0):
        y_p_new.append(y_p)
        y_t_new.append(y_t)
    
    y_predict = y_p_new
    y_test    = y_t_new
    
    err_val_list = [abs(y_p - y_t) for y_p, y_t in zip(y_predict, y_test)]
    
    mean_sqrd_lg_err = numpy.mean([(math.log(1+y_p) - math.log(1+y_t))**2 for y_p, y_t in zip(y_predict, y_test)])
    
    # Evaluating the model
    #
    model_var_score        = metrics.explained_variance_score(y_test, y_predict)
    model_min_abs_err      = min(err_val_list)
    model_max_abs_err      = max(err_val_list)
    model_avrg_abs_err     = numpy.mean(err_val_list)
    model_std_abs_err      = numpy.std(err_val_list)
    model_mean_sqrd_err    = metrics.mean_squared_error(y_test, y_predict)
    model_mean_sqrd_lg_err = mean_sqrd_lg_err
    model_r_sqrd           = metrics.r2_score(y_test, y_predict)
    
    print
    print 'Evaluation of the %s regression model' %regre_model_str
    print '  Explained variance score:       %.5f' %model_var_score
    print '  Min/Max/Average absolute error: %.5f / %.5f / %.5f' %(min(err_val_list), \
                                                                   max(err_val_list), \
                                                                   numpy.mean(err_val_list))
    print '  Standard deviation error:       %.5f' %model_std_abs_err
    print '  Mean-squared error:             %.5f' %model_mean_sqrd_err
    print '  Mean-squared log error:         %.5f' %model_mean_sqrd_lg_err
    print '  R-squared value:                %.5f' %model_r_sqrd
    print
    
    
    eval_res_tuple = (model_var_score, model_min_abs_err, model_max_abs_err, model_avrg_abs_err,
                      model_std_abs_err, model_mean_sqrd_err, model_mean_sqrd_lg_err, 
                      model_r_sqrd)
    
    # Generate the plot if needed
    #
    
    #plot_file_name = None
    
    if (plot_file_name != None) and (len(x_val_list) > 0):
      
      if(regre_model_str in ['linear', 'poly']):
        x_plot_train, q_val_train = zip(*x_train)
        x_plot_test, q_val_test = zip(*x_test)
      else:
        x_plot_train = x_train
        x_plot_test = x_test
      
      # Sampling for the plot
      x_plot_train = x_plot_train[:2000]
      y_train      = y_train[:2000]
      
      x_plot_test  = x_plot_test[:500]
      y_test       = y_test[:500]
      y_predict    = y_predict[:500]
      

      #x_min = min(x_val_list)
      #x_max = max(x_val_list)
      #x_diff = x_max - x_min 
      #plot_min_x = x_min - x_diff*0.05
      #plot_max_x = x_max + x_diff*0.05

      #plot_min_y = min(y_val_list) - min(y_val_list)*0.05
      #plot_max_y = 1.0 + min(y_val_list)*0.05

      
      fig, ax = plt.subplots(1, 1)
      ax.plot(x_plot_train, y_train, ".", markersize=1, color='#144ba4', label='Training set')
      ax.plot(x_plot_test, y_test, ".", markersize=1, color='#0bb015', label="Testing set")
      ax.plot(x_plot_test, y_predict, ".", markersize=2, color='red', label="Predictions")
      
      if(regre_model_str == 'linear'):
        plot_title = 'Linear Regression ' + plot_data_name
      elif(regre_model_str == 'isotonic'):
        plot_title = 'Isotonic Regression ' + plot_data_name
      elif(regre_model_str == 'poly'):
        plot_title = 'Polynomial Regression ' + plot_data_name
      
      ax.set_title(plot_title)
      ax.legend(loc="best",)
      
      # Plot the generated lists as scatter plot
      #
      #w,h = plt.figaspect(PLT_PLOT_RATIO)
      #plt.figure(figsize=(w,h))

      #plt.title('Similarity variance of plain-text versus encoded edges')

      plt.xlabel('Q-gram similarity')
      plt.ylabel('Encoded similarity')

      #plt.xlim(plot_min_x, plot_max_x)
      #plt.ylim(plot_min_y, plot_max_y)
      
      #sizes = [0.1*x for x in range(len(x_val_list))]

      #plt.scatter(x_val_list, y_val_list, s=1)

#===============================================================================
#       # Add the calculated averages as a line plot
#       #
#       avr_x_list = []
#       avr_y_list = []
# 
#       for sim_interval in sorted(sim_diff_dict.keys()):
#         sim_diff = sim_diff_dict[sim_interval][0]
# 
#         avr_x_list.append(sim_diff)
#         avr_y_list.append(sim_interval)
# 
#       plt.plot(avr_x_list, avr_y_list, color='r', lw=2, label='Avr sim')
#===============================================================================

      #plt.legend(loc="best",) #, prop={'size':14}, bbox_to_anchor=(1, 0.5))

      plt.savefig(plot_file_name, bbox_inches='tight')

    return reg_model, eval_res_tuple
  
  
  # ---------------------------------------------------------------------------

  def gen_min_conn_comp_graph(self, min_conn_comp_size):
    """Remove fro mthe graph all connected components (nodes and edges) that
       aresmaller than the given minimum component size, because smaller
       components do not contain enough information to distinguishing nodes
       from each other (i.e. their neighbourhoods are too small).

       Input arguments:
         - min_conn_comp_size  The minimum size of the connected components to
                               be kept.
       Output:
         - This method does not return anything.
    """

    sim_graph = self.sim_graph

    # Loop over all connected components, and remove nodes and edges from
    # those that are too small
    #
    conn_comp_list = list(networkx.connected_components(sim_graph))
    for conn_comp in conn_comp_list:
      if (len(conn_comp) < min_conn_comp_size):
        for node_key_val in conn_comp:
          node_neighbour_list = sim_graph.neighbors(node_key_val)
          sim_graph.remove_node(node_key_val)

          for other_node_key_val in node_neighbour_list:
            if sim_graph.has_edge(node_key_val, other_node_key_val):
              sim_graph.remove_edge(node_key_val, other_node_key_val)

    conn_comp_list = []  # Not needed anymore

    print 'Graph without connected components smaller than %d nodes ' % \
          (min_conn_comp_size) + 'contains %d nodes and %d edges' % \
          (len(sim_graph.nodes()), len(sim_graph.edges()))
    print

  # ---------------------------------------------------------------------------

  def calc_features(self, calc_feature_list, min_sim_list, min_node_degree,
                    max_node_degree):
    """Calculate various features for each node based on its edges, as well as
       node and edge attributes, where some of these features are calculated
       for each of the similarity values in the provided list. Only nodes that
       have a 'min_node_degree' in the graph for the lowest similarity will be
       considered. For each such node a list of numerical features is
       calculated, returned in a dictionary with node values as keys.

       Input arguments:
         - calc_feature_list  A list with the names of the features to be
                              calculated. Possible features are are listed
                              and described below.
         - min_sim_list       A list with similarity threshold values.
         - min_node_degree    The minimum degree a node requires in order to
                              be considered.
         - max_node_degree    The maximum degree of any node in the graph.

       Output:
         - node_feat_dict  A dictionary with one entry per node of the graph,
                           with its key being the node value and its value
                           being an array of the calculated numerical feature
                           values.
         - min_feat_array  An array which for each feature contains its
                           minimum calculated value.
         - max_feat_array  An array which for each feature contains its
                           maximum calculated value.
         - feat_name_list  A list with the names of all generated features.

       Possible features that can be included in the 'calc_feature_list' are:
         - node_freq          The count of a node.
         - max_sim            The overall maximum edge similarity for a node
                              (independent of the list of similarity values
                              given).
         - min_sim            The minimum edge similarity (per minimum
                              similarity value).
         - sim_avr            The average similarity of all edges connected
                              to a node (per minimum similarity value).
         - sim_std            The similarity standard deviation of all edges
                              connected to a node (per minimum similarity
                              value).
         - degree             The number of edges connecting the node to other
                              nodes (per minimum similarity value).
         - degree_histo1      Histogram of degree distribution of neighbours
                              directly connected to a node (via one edge)
                              (per minimum similarity value).
         - degree_histo2      Histogram of degree distribution of neighbours
                              connected via two edges to a node (per minimum
                              similarity value).
         - egonet_degree      Number of edges connecting the egonet of a node
                              (i.e. all its neighbours) to other nodes outside
                              the egonet.
         - egonet_density     The density of the egonet as the number of edges
                              between nodes (including the node itself) of the
                              egonet over all possible edges.
         - pagerank           The page rank of the node (per minimum
                              similarity value).
                              *** Note very slow for large graphs. ***
         - between_central    The betweenness centrality for each node (per
                              minimum similarity value).
         - closeness_central  The closeness centrality for each node (per
                              minimum similarity value).
                              *** Note very slow for large graphs. ***
         - degree_central     The degree centrality for each node (per
                              minimum similarity value).
         - eigenvec_central   The eigenvector centrality for each node (per
                              minimum similarity value).
                              *** Does not always converge ***
    """

    t1 = time.time()

    sim_graph = self.sim_graph

    # If required calculate the required length of the degree histograms
    #
    if ('degree_histo1' in calc_feature_list) or \
       ('degree_histo2' in calc_feature_list):

       # The neighbour node degree histograms are based on logarithmic scale
       # (following "REGAL: Representation Learning-based Graph Alignment",
       # M. Heimann et al., CIKM 2018):
       #
       degree_hist_len = int(math.log(max_node_degree,2))+1
    else:
       degree_hist_len = 0

    # Generate the list of the names of all features to be generated
    #
    feat_name_list = []

    if ('node_freq' in calc_feature_list):
      feat_name_list.append('node_freq')
    if ('max_sim' in calc_feature_list):
      feat_name_list.append('max_sim')

    # The following list is ordered in the same way how features are added in
    # the code below. If the below code is changed then this list needs to be
    # updated as well.
    #
    for min_sim in min_sim_list:
      for feat_name in ['min_sim','sim_avr','sim_std','degree',
                        'degree_histo1','degree_histo2','egonet_degree',
                        'egonet_density']:
        if feat_name in calc_feature_list:
          if ('degree_histo' not in feat_name):
            feat_name_list.append(feat_name+'-%.2f' % (min_sim))
          else:  # For degree histograms one feature per histogram bucket
            for histo_bucket in xrange(degree_hist_len):
              feat_name_list.append(feat_name+'-b%d-%.2f' % \
                                    (histo_bucket, min_sim))
    for min_sim in min_sim_list:
      for feat_name in ['pagerank','between_central','closeness_central',
                        'degree_central', 'eigenvec_central']:
        if feat_name in calc_feature_list:
          feat_name_list.append(feat_name+'-%.2f' % (min_sim))

    num_feat = len(feat_name_list)

    print
    print 'Generating %d features per graph node' % (num_feat) + \
          ' for %d nodes' % (sim_graph.number_of_nodes())
    print '  Minimum similarities to consider:', str(min_sim_list)
    print '  Minimum node degree required:    ', min_node_degree
    print '  Features generated:', feat_name_list

    node_feat_dict = {}  # One list of features per node

    num_nodes_too_low_degree = 0

    # Process each node and get is neighbours, then calculate the node's
    # features for different similarity thresholds
    #
    for node_key_val in sim_graph.nodes():

      # Get the node's neighbours
      #
      node_neigh_val_list = sim_graph.neighbors(node_key_val)

      if (len(node_neigh_val_list) < min_node_degree):
        num_nodes_too_low_degree += 1
        continue

      node_feat_array = numpy.zeros(num_feat)
      next_feat =       0  # The index into the feature array for this node

      if ('node_freq' in calc_feature_list):
        node_freq = len(sim_graph.node[node_key_val]['ent_id_set'])
        assert node_freq >= 1, (node_key_val, node_freq)
        node_feat_array[next_feat] = node_freq
        next_feat += 1

      # Get the node values and similarities of all edges of this node
      #
      all_node_neigh_sim_list = []

      max_edge_sim = 0.0
      for neigh_key_val in node_neigh_val_list:
        edge_sim = sim_graph.edge[node_key_val][neigh_key_val]['sim']
        max_edge_sim = max(max_edge_sim, edge_sim)
        all_node_neigh_sim_list.append((neigh_key_val, edge_sim))

      if ('max_sim' in calc_feature_list):
        node_feat_array[next_feat] = max_edge_sim
        next_feat += 1

      for min_sim in min_sim_list:

        # First get the edges and their similarities that are at least the
        # current minimum similarity
        #
        node_neigh_val_set = set()  # Nodes connected with similar enough edges
        node_neigh_sim_list = []    # Similarities of these edges

        for (neigh_key_val, edge_sim) in all_node_neigh_sim_list:
          if (edge_sim >= min_sim):
            node_neigh_val_set.add(neigh_key_val)
            node_neigh_sim_list.append(edge_sim)

        num_sim_neigh = len(node_neigh_val_set)

        # Calculate selected features for the extracted edge similarity list
        #
        if ('min_sim' in calc_feature_list):
          if (num_sim_neigh > 0):  # Nothing to do if 0
            node_feat_array[next_feat] = min(node_neigh_sim_list)
          next_feat += 1

        if ('sim_avr' in calc_feature_list):
          if (num_sim_neigh > 0):  # Nothing to do if 0
            node_feat_array[next_feat] = numpy.mean(node_neigh_sim_list)
          next_feat += 1

        if ('sim_std' in calc_feature_list):
          if (num_sim_neigh > 0):  # Nothing to do if 0
            node_feat_array[next_feat] = numpy.std(node_neigh_sim_list)
          next_feat += 1

        if ('degree' in calc_feature_list):
          node_feat_array[next_feat] = num_sim_neigh
          next_feat += 1

        if ('degree_histo1' in calc_feature_list):
          if (num_sim_neigh > 0):  # Nothing to do if no neighbours
            for neigh_key_val in node_neigh_val_set:
              node_neigh_degree = sim_graph.degree(neigh_key_val)
              node_neigh_log_degree = int(math.log(node_neigh_degree, 2))
              assert (node_neigh_log_degree >= 0) and \
                      (node_neigh_log_degree < degree_hist_len)
              node_feat_array[next_feat + node_neigh_log_degree] += 1

          next_feat += degree_hist_len  # Advance over the degree histrogram

        if ('degree_histo2' in calc_feature_list):
          if (num_sim_neigh > 0):  # Nothing to do if no neighbours
            hop2_neigh_val_set = set()  # Get all neighbours of neighbours

            for neigh_key_val in node_neigh_val_set:
              for hop2_neigh_key_val in sim_graph.neighbors(neigh_key_val):

                # Only consider all nodes two hops away from current node
                #
                if (hop2_neigh_key_val != node_key_val) and \
                   (hop2_neigh_key_val not in node_neigh_val_set):

                  edge_sim = \
                      sim_graph.edge[neigh_key_val][hop2_neigh_key_val]['sim']

                  if (edge_sim >= min_sim):
                    hop2_neigh_val_set.add(hop2_neigh_key_val)

            for hop2_neigh_key_val in hop2_neigh_val_set:
              node_neigh_degree = sim_graph.degree(hop2_neigh_key_val)
              node_neigh_log_degree = int(math.log(node_neigh_degree, 2))
              assert (node_neigh_log_degree >= 0) and \
                      (node_neigh_log_degree < degree_hist_len)
              node_feat_array[next_feat + node_neigh_log_degree] += 1

          next_feat += degree_hist_len  # Advance over the degree histrogram

        if ('egonet_degree' in calc_feature_list) or \
           ('egonet_density' in calc_feature_list):
          if (num_sim_neigh == 0):
            if ('egonet_degree' in calc_feature_list):
              next_feat += 1
            if ('egonet_density' in calc_feature_list):
              next_feat += 1

          else:  # There are neighbours
            egonet_degree = 0.0  # Number of edges with nodes outside egonet
            egonet_edge_set = set()  # Set of edges within egonet

            for neigh_key_val in node_neigh_val_set: # Loop over all neighbours

              # Loop over all neighbours of the node's neighbours
              #
              for other_key_val in sim_graph.neighbors(neigh_key_val):

                # Check if this other node is in the egonet or not
                #
                if (other_key_val != node_key_val) and \
                   (other_key_val not in node_neigh_val_set):
                  egonet_degree += 1
                else:  # Edge is within egonet, so add to egonet edge set
                  egonet_edge = tuple(sorted([neigh_key_val,other_key_val]))
                  egonet_edge_set.add(egonet_edge)

            if ('egonet_degree' in calc_feature_list):
              node_feat_array[next_feat] = egonet_degree
              next_feat += 1
            if ('egonet_density' in calc_feature_list):
              all_egonet_num_edges = num_sim_neigh*(num_sim_neigh+1)/2.0
              egonet_density_val = len(egonet_edge_set) / all_egonet_num_edges
              assert egonet_density_val <= 1.0, egonet_density_val
              node_feat_array[next_feat] = egonet_density_val
              next_feat += 1

      node_feat_dict[node_key_val] = node_feat_array

    # Calculate pagerank and/or betweenness centrality on the similarity graph
    # according to the current minimum similarity
    #
    if ('pagerank' in calc_feature_list) or \
       ('between_central' in calc_feature_list) or \
       ('closeness_central' in calc_feature_list) or \
       ('degree_central' in calc_feature_list) or \
       ('eigenvec_central' in calc_feature_list):

      for min_sim in min_sim_list:

        # Generate the sub-graph with the current minimum similarity (pagerank
        # requires a directed graph)
        #
        di_graph = networkx.DiGraph()
        for (node_key_val1, node_key_val2) in sim_graph.edges():
          if (sim_graph.edge[node_key_val1][node_key_val2]['sim'] >= min_sim):
            di_graph.add_edge(node_key_val1,node_key_val2)
            di_graph.add_edge(node_key_val2,node_key_val1)

        if ('pagerank' in calc_feature_list):
          page_rank_dict = networkx.pagerank_numpy(di_graph)

          for (node_key_val, node_feat_array) in node_feat_dict.iteritems():
            node_feat_array[next_feat] = page_rank_dict.get(node_key_val, 0.0)
            node_feat_dict[node_key_val] = node_feat_array
          next_feat += 1

        for (central_funct_name_str, central_funct_name) in \
            [('between_central',   networkx.betweenness_centrality),
             ('closeness_central', networkx.closeness_centrality),
             ('degree_central',    networkx.degree_centrality),
             ('eigenvec_central',  networkx.eigenvector_centrality)]:
          if (central_funct_name_str in calc_feature_list):
            central_dict = central_funct_name(di_graph)
            for (node_key_val, node_feat_array) in node_feat_dict.iteritems():
              node_feat_array[next_feat] = \
                                  central_dict.get(node_key_val, 0.0)
              node_feat_dict[node_key_val] = node_feat_array
            next_feat += 1
        
    assert next_feat == num_feat, (next_feat, num_feat)

    # Keep track of the minimum and maximum values per feature
    #
    min_feat_array = numpy.ones(num_feat)
    max_feat_array = numpy.zeros(num_feat)

    for node_feat_array in node_feat_dict.itervalues():
      min_feat_array = numpy.minimum(node_feat_array, min_feat_array)
      max_feat_array = numpy.maximum(node_feat_array, max_feat_array)

    print '  Feature generation for %d nodes took %.2f sec' % \
          (len(node_feat_dict), time.time() - t1)
    print '    Number of nodes with a degree below %d: %d' % \
          (min_node_degree, num_nodes_too_low_degree)
    print '    Minimum feature values:', min_feat_array
    print '    Maximum feature values:', max_feat_array
    print

    assert len(node_feat_dict) + num_nodes_too_low_degree == \
           len(sim_graph.nodes())

    return node_feat_dict, min_feat_array, max_feat_array, feat_name_list

  # ---------------------------------------------------------------------------

  def remove_freq_feat_vectors(self, node_feat_dict, max_count):
    """Remove from the given dictionary of feature vectors all those vectors
       that occur more than 'max_count' times, and return a new feature vector
       dictionary.

       Input arguments:
         - node_feat_dict  A dictionary with one entry per node of the graph,
                           with its key being the node value and its value
                           being an array of the calculated numerical feature
                           values.
         - max_count       An integer, where all unique feature vectors that
                           occur more than this number of times in the
                           dictionary will be removed.

      Output:
         - filter_node_feat_dict  A dictionary containing only those node
                                  identifiers as keys where their feature
                                  vectors does not occur frequently according
                                  'max_count'.
    """

    # Keys will be feature vectors and values the sets of node identifiers that
    # have this feature vector
    #
    feat_vec_count_dict = {}

    for (node_key_val, node_feat_array) in node_feat_dict.iteritems():
      node_feat_tuple = tuple(node_feat_array)  # So it can be a dictionary key

      feat_vec_node_set = feat_vec_count_dict.get(node_feat_tuple, set())
      feat_vec_node_set.add(node_key_val)
      feat_vec_count_dict[node_feat_tuple] = feat_vec_node_set

    filter_node_feat_dict = {}

    # Now copy those feature vectors that occur for maximum 'max_count' nodes
    #
    for feat_vec_node_set in feat_vec_count_dict.itervalues():

      if (len(feat_vec_node_set) <= max_count):
        for node_key_val in feat_vec_node_set:
          assert node_key_val not in filter_node_feat_dict

          filter_node_feat_dict[node_key_val] = node_feat_dict[node_key_val]

    print 'Reduced number of nodes in feature vector dictionary from %d ' % \
          (len(node_feat_dict)) + 'to %d by removing all feature vectors ' % \
          (len(filter_node_feat_dict)) + 'that occur more than %d times' % \
          (max_count)
    print

    return filter_node_feat_dict

  # ---------------------------------------------------------------------------

  def norm_features_01(self, node_feat_dict, min_feat_array, max_feat_array):
    """Normalise the feature vectors in the given dictionary with regard to
       the given minimum and maximum values provided such that all feature
       values will be in the 0..1 range.

       Input arguments:
         - node_feat_dict  A dictionary with one entry per node of the graph,
                           with its key being the node value and its value
                           being an array of the calculated numerical feature
                           values.
         - min_feat_array  An array which for each feature contains its
                           minimum calculated value.
         - max_feat_array  An array which for each feature contains its
                           maximum calculated value.

       Output:
         - norm_node_feat_dict  A dictionary with one entry per node of the
                                graph, with its key being the node value and
                                its value being an array of the 0 to 1
                                normalised numerical feature values.
    """

    norm_node_feat_dict = {}

    num_feat = len(min_feat_array)
    assert num_feat == len(max_feat_array)

    # Get the difference between the maximum and minimum values for each
    # feature
    #
    feat_range_array = max_feat_array - min_feat_array

    # Loop over all feature vectors and normalise them if their difference
    # between maximum and minimum values is larger than 0
    #
    for (node_key_val, node_feat_array) in node_feat_dict.iteritems():
      norm_feat_array = numpy.zeros(num_feat)

      for i in xrange(num_feat):
         if (feat_range_array[i] > 0):
           norm_feat_array[i] = (node_feat_array[i] - min_feat_array[i]) / \
                                feat_range_array[i]
         else:  # No value range, but check the feature is not ouside 0..1
           assert (node_feat_array[i] >= 0.0) and (node_feat_array[i] <= 1.0),\
                  (i, node_feat_array[i], node_feat_array)

      norm_node_feat_dict[node_key_val] = norm_feat_array

    # Check minimum and maximum values per feature are now normalised
    #
    min_feat_array = numpy.ones(num_feat)
    max_feat_array = numpy.zeros(num_feat)

    for node_feat_array in norm_node_feat_dict.itervalues():
      min_feat_array = numpy.minimum(node_feat_array, min_feat_array)
      max_feat_array = numpy.maximum(node_feat_array, max_feat_array)

    print '  Normalised minimum feature values:', min_feat_array
    print '  Normalised maximum feature values:', max_feat_array
    print

    assert min(min_feat_array) >= 0.0, min_feat_array
    assert max(max_feat_array) <= 1.0, max_feat_array

    return norm_node_feat_dict

  # ---------------------------------------------------------------------------

  def get_std_features(self, node_feat_dict, num_feat):
    """For each feature calculate its standard deviation, and return an array
       with these standard deviation values.

       Input argument:
         - node_feat_dict  A dictionary with one entry per node of the graph,
                           with its key being the node identifier and its
                           value being an array of the calculated numerical
                           feature values.
         - num_feat        The number of features for each node.

       Output:
         - feat_std_list  A list with pairs of (feature number, feature
                          standard deviation) for each feature.
    """

    feat_val_list = []  # One list of all values for each feature

    for i in xrange(num_feat):
      feat_val_list.append([])  # One list per feature

    for node_feat_array in node_feat_dict.itervalues():
      for i in xrange(num_feat):  # All feature values to their lists
        feat_val_list[i].append(node_feat_array[i])

    feat_std_list = []

    for i in xrange(num_feat):
      feat_std_list.append((i, numpy.std(feat_val_list[i])))

    del feat_val_list

    return feat_std_list

  # ---------------------------------------------------------------------------

  def select_features(self, node_feat_dict, use_feat_set, org_num_feat):
    """Select certain features as provided in the given set of features to
       use.

       Input argument:
         - node_feat_dict  A dictionary with one entry per node of the graph,
                           with its key being the node identifier and its
                           value being an array of the calculated numerical
                           feature values.
         - use_feat_set    A set with the numbers of features to use (while
                           all other features  are removed from the feature
                           array of each node).
         - org_num_feat    The original number of features for each node.

       Output:
         - sel_node_feat_dict  A new dictionary with one entry per node of
                               the graph, with its key being the node
                               identifier and its value being an array of the
                               selected numerical feature values.
    """

    num_use_feat = len(use_feat_set)

    sel_node_feat_dict = {}

    for (id, node_feat_array) in node_feat_dict.iteritems():
       use_feat_array = numpy.ones(num_use_feat)
       use_i = 0
       for i in xrange(org_num_feat):
         if i in use_feat_set:
           use_feat_array[use_i] = node_feat_array[i]
           use_i += 1

       sel_node_feat_dict[id] = use_feat_array

    return sel_node_feat_dict

# =============================================================================

def write_feat_matrix_file(graph_feat_dict, feat_name_list, feat_matrix_file_name):
  """A funtion to write feature matrix generated for each graph to
     a csv file.
  """
  
  writing_start_time = time.time()
  
  header_list = ['node_key'] + feat_name_list
  
  for node_key_val, feat_val_array in graph_feat_dict.iteritems():
    
    if('encode' in feat_matrix_file_name and '2sh' not in feat_matrix_file_name):
      res_list = [node_key_val] + list(feat_val_array)
    else:
      res_list = [','.join(node_key_val)] + list(feat_val_array)
  
    # Check if the result file exists, if it does append, otherwise create
    if (not os.path.isfile(feat_matrix_file_name)):
      write_file = open(feat_matrix_file_name, 'w')
      csv_writer = csv.writer(write_file)
      csv_writer.writerow(header_list)
    
      print 'Created new result file:', feat_matrix_file_name
    
    else:  # Append results to an existing file
      write_file = open(feat_matrix_file_name, 'a')
      csv_writer = csv.writer(write_file)
    
    csv_writer.writerow(res_list)
    write_file.close()
  
  writing_time = time.time() - writing_start_time
  
  print ' Wrote graph feature results into the csv file. Took %.3f seconds' %writing_time
  print ' ', auxiliary.get_memory_usage()
  print

#-----------------------------------------------------------------------  

def load_feat_matrix_file(feat_matrix_file_name):
  """A funtion to load feature matrix for each graph to
     a dictionary.
  """
  
  load_start_time = time.time()
  
  graph_feat_dict = {}
  col_sep_char = ','
  
  
  # Check if the file name is a gzip file or a csv file
  #
  if (feat_matrix_file_name.endswith('gz')):
    in_f = gzip.open(feat_matrix_file_name)
  else:
    in_f = open(feat_matrix_file_name)

  # Initialise the csv reader
  #
  csv_reader = csv.reader(in_f, delimiter=col_sep_char)
  
  header_list = csv_reader.next()
  feat_name_list = header_list[1:]

  # Read each line in the file and store the required attribute values in a
  # list
  #
  vec_num = 0 

  for vec_val_list in csv_reader:
    
    if('encode' in feat_matrix_file_name and '2sh' not in feat_matrix_file_name):
      node_key_val = vec_val_list[0]
    else:
      node_key_val_str = vec_val_list[0]
      
      node_key_val_list = [i for i in node_key_val_str.split(',')]
      node_key_val = tuple(node_key_val_list)
      
    feat_val_array = numpy.array([float(val) for val in vec_val_list[1:]])
    
    graph_feat_dict[node_key_val] = feat_val_array
  
  # Keep track of the minimum and maximum values per feature
  num_feat = len(feat_name_list)
  min_feat_array = numpy.ones(num_feat)
  max_feat_array = numpy.zeros(num_feat)

  for node_feat_array in graph_feat_dict.itervalues():
    min_feat_array = numpy.minimum(node_feat_array, min_feat_array)
    max_feat_array = numpy.maximum(node_feat_array, max_feat_array)

  
  load_time = time.time() - load_start_time
  
  print ' Load graph feature values into a dictionary. Took %.3f seconds' %load_time
  print ' ', auxiliary.get_memory_usage()
  print
  
  return graph_feat_dict, min_feat_array, max_feat_array, feat_name_list

# =============================================================================

class MinHashLSH():
  """A class that implements a min-hashing locality sensitive hashing (LSH)
     approach to be used for blocking the plain-text q-grams sets in order to
     prevent a full-pair-wise comparison of all q-gram set pairs.
  """

  def __init__(self, lsh_band_size, lsh_num_band, random_seed=None):
    """Initialise the parameters for min-hashing LSH including generating
       random values for hash functions.

       Input arguments:
         - lsh_band_size  The length of the min-hash bands.
         - lsh_num_band   The number of LSH bands.
         - random_seed    If not None then initalise the random number
                          generator with this seed value.

       Output:
         - This method does not return anything.

       LSH min-hashing follows the code provided here:
        https://github.com/chrisjmccormick/MinHash/blob/master/ \
              runMinHashExample.py

       The probability for a pair of sets with Jaccard sim 0 < s <= 1 to be
       included as a candidate pair is (with b = lsh_num_band and
       r = lsh_band_size, i.e. the number of rows/hash functions per band) is
       (Leskovek et al., 2014, page 89):

         p_cand = 1- (1 - s^r)^b

       Approximation of the 'threshold' of the S-curve (Leskovek et al., 2014,
       page 90) is: t = (1/k)^(1/r).
    """

    if (random_seed != None):
      random.seed(random_seed)

    # Calculate error probabilities for given parameter values
    #
    assert lsh_num_band >  1, lsh_num_band
    assert lsh_band_size > 1, lsh_band_size

    self.lsh_num_band =   lsh_num_band
    self.lsh_band_size =  lsh_band_size
    self.num_hash_funct = lsh_band_size*lsh_num_band  # Total number needed

    b = float(lsh_num_band)
    r = float(lsh_band_size)
    t = (1.0/b)**(1.0/r)

    s_p_cand_list = []
    for i in range(1,10):
      s = 0.1*i
      p_cand = 1.0-(1.0-s**r)**b
      assert 0.0 <= p_cand <= 1.0
      s_p_cand_list.append((s, p_cand))

    print 'Initialise LSH blocking using Min-Hash'
    print '  Number of hash functions: %d' % (self.num_hash_funct)
    print '  Number of bands:          %d' % (lsh_num_band)
    print '  Size of bands:            %d' % (lsh_band_size)
    print '  Threshold of s-curve:     %.3f' % (t)
    print '  Probabilities for candidate pairs:'
    print '   Jacc_sim | prob(cand)'
    for (s,p_cand) in s_p_cand_list:
      print '     %.2f   |   %.5f' % (s, p_cand)
    print

    max_hash_val = 2**31-1  # Maximum possible value a CRC hash could have

    # We need the next largest prime number above 'maxShingleID'.
    # From here:
    # http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
    #
    self.next_prime = 4294967311

    # Random hash function will take the form of: h(x) = (a*x + b) % c
    # where 'x' is the input value, 'a' and 'b' are random coefficients, and
    # 'c' is a prime number just greater than max_hash_val
    #
    # Generate 'num_hash_funct' coefficients
    #
    coeff_a_set = set()
    coeff_b_set = set()

    while (len(coeff_a_set) < self.num_hash_funct): 
      coeff_a_set.add(random.randint(0, max_hash_val))
    while (len(coeff_b_set) < self.num_hash_funct): 
      coeff_b_set.add(random.randint(0, max_hash_val))
    self.coeff_a_list = sorted(coeff_a_set)
    self.coeff_b_list = sorted(coeff_b_set)
    assert self.coeff_a_list != self.coeff_b_list

  # ---------------------------------------------------------------------------

  def hash_q_gram_set(self, q_gram_set):
    """Min-hash the given set of q-grams and return a list of hash signatures
       depending upon the Min-hash parameters set during the class
       initialisation.

       Input arguments:
         - q_gram_set  The q-gram set to be hashed.

       Output:
         -  band_hash_sig_list  A list with the min-hash signatures for the
                                input q-gram set.
    """

    next_prime =    self.next_prime
    coeff_a_list =  self.coeff_a_list
    coeff_b_list =  self.coeff_b_list
    lsh_band_size = self.lsh_band_size
    lsh_num_band =  self.lsh_num_band

    crc_hash_set = set()

    for q_gram in q_gram_set:  # Hash the q-grams into 32-bit integers
      crc_hash_set.add(binascii.crc32(q_gram) & 0xffffffff)

    assert len(q_gram_set) == len(crc_hash_set)  # Check no collision

    # Now generate all the min-hash values for this q-gram set
    #
    min_hash_sig_list = []

    for h in range(self.num_hash_funct):
 
      # For each CRC hash value (q-gram) in the q-gram set calculate its Min-
      # hash value for all 'num_hash_funct' functions
      #
      min_hash_val = next_prime + 1  # Initialise to value outside range

      for crc_hash_val in crc_hash_set:
        hash_val = (coeff_a_list[h]*crc_hash_val + coeff_b_list[h]) % \
                   next_prime
        min_hash_val = min(min_hash_val, hash_val)

      min_hash_sig_list.append(min_hash_val)

    # Now split hash values into bands and generate the list of
    # 'lsh_num_band' hash values used for blocking
    #
    band_hash_sig_list = []

    start_ind = 0
    end_ind =   lsh_band_size
    for band_num in range(lsh_num_band):
      band_hash_sig = min_hash_sig_list[start_ind:end_ind]
      assert len(band_hash_sig) == lsh_band_size
      start_ind = end_ind
      end_ind +=  lsh_band_size
      band_hash_sig_list.append(band_hash_sig)

    return band_hash_sig_list

# =============================================================================

class CosineLSH():
  """A class that implements a random vector based Cosine similarity locality
     sensitive hashing (LSH) approach.

     As described in the paper:
       A. Andoni and P. Indyk: Near-optimal hashing algorithms for approximate
       nearest neighbor in high dimensions. In FOCS. IEEE, 2006.

     Based on code from: https://www.bogotobogo.com/Algorithms/ \
         Locality_Sensitive_Hashing_LSH_using_Cosine_Distance_Similarity.php
  """
  
  def __init__(self, vec_len, num_hash_sign_len, random_seed=None):
    """Initialise the LSH approach by generating a certain number of random
       vectors.

       Input arguments:
         - vec_len            The dimensionality of the feature vectors to be
                              hashed.
         - num_hash_sign_len  The length of the bit arrays to be generated
                              (i.e. the number of random hyper-planes to be
                              generated).
         - random_seed        If not None then initalise the random number
                              generator with this seed value

       Output:
         - This method does not return anything.
    """

    if (random_seed != None):
      numpy.random.seed(random_seed)

    self.vec_len =           vec_len
    self.num_hash_sign_len = num_hash_sign_len
    
    # Generate the required number of random hyper-planes (num_hash_sign_len),
    # each a vector of length 'vec_len'
    #
    self.ref_planes = numpy.random.randn(num_hash_sign_len, vec_len)

    print 'Initialised CosineLSH by generating %d hyper-planes each of ' % \
          (num_hash_sign_len) + 'dimensionality %d' % (vec_len)
    print

  # ---------------------------------------------------------------------------

  def gen_sim_hash(self, feature_dict):
    """Encode each of the feature vectors in the given dictionary using the
       random hyper-planes resulting a bit-array for each feature vector.

       Input arguments:
         - feature_dict  The feature dictionary with keys being node values
                         and values being lists of numerical feature values.

       Output:
         - sim_hash_dict  A dictionary with the calculated hash bit arrays,
                          with keys being node values and values bit arrays.
    """

    num_hash_sign_len = self.num_hash_sign_len
    vec_len =           self.vec_len

    sim_hash_dict = {}

    ref_planes = self.ref_planes

    # To allow calculation of statistics, keep a counter of how many bits are
    # set to 1 in each bit position
    #
    bit_pos_1_count_list = [0]*num_hash_sign_len

    # Also get the Hamming weight (number of 1-bits) distribution over all
    # bit arrays
    #
    hw_list = []

    for (val, feat_array) in feature_dict.iteritems():
      assert len(feat_array) == vec_len

      hash_bit_array = bitarray.bitarray(num_hash_sign_len)
      hash_bit_array.setall(0)

      # Generate the hash bit signature based on the given random hyper-planes
      #
      for (bit_pos, rand_vec) in enumerate(ref_planes):
        if (numpy.dot(feat_array, rand_vec) >= 0):
          hash_bit_array[bit_pos] = 1
          bit_pos_1_count_list[bit_pos] += 1

      sim_hash_dict[val] = hash_bit_array
      hw_list.append(int(hash_bit_array.count(1)))

    # Check the number of 0/1 bits for each position in the bit arrays and
    # print statistics of these distributions
    #
    print 'Statistics for sim hash encoding of %d feature vectors:' % \
          (len(sim_hash_dict))
    print '  Minimum, average (std-dev), median and maximum number of 1-' + \
          'bits per position:  %d / %.3f (%.3f) / %d / %d (from %d in total)' \
          % (min(bit_pos_1_count_list), numpy.mean(bit_pos_1_count_list),
             numpy.std(bit_pos_1_count_list),
             numpy.median(bit_pos_1_count_list), max(bit_pos_1_count_list),
             len(sim_hash_dict))
    print '  Minimum, average (std-dev), median and maximum number of 1-' + \
          'bits per bit array: %d / %.3f (%.3f) / %d / %d (from %d in total)' \
          % (min(hw_list), numpy.mean(hw_list), numpy.std(hw_list),
             numpy.median(hw_list), max(hw_list),num_hash_sign_len)
    print

    return sim_hash_dict

  # ---------------------------------------------------------------------------

  def all_in_one_blocking(self, plain_sim_hash_dict, encode_sim_hash_dict):
    """Generate two blocking dictionaries, one per input similarity hash
       dictionary, where all records from an input dictionary are put into
       one block. Returns these two blocking dictionaries.

       Input arguments:
         - plain_sim_hash_dict   A dictionary with keys being node values and
                                 values being hash bit arrays.
         - encode_sim_hash_dict  A dictionary with keys being node values and
                                 values being hash bit arrays.

       Output:
         - plain_sim_block_dict    A dictionary with one key being 'all' and
                                   values being being node key values from the
                                   plain-text similarity hash dictionary.
         - encode_sim_block_dict   A dictionary with one key being 'all' and
                                   values being being node key values from the
                                   encoded similarity hash dictionary.
    """

    plain_sim_block_dict =  {'all':set(plain_sim_hash_dict.keys())}
    encode_sim_block_dict = {'all':set(encode_sim_hash_dict.keys())}

    print 'Number of blocks for the all-in-one index: 1 / 1'
    print '  Block sizes: %d (plain) / %d (encoded)' % \
          (len(plain_sim_block_dict['all']),
           len(encode_sim_block_dict['all']))
    print

    return plain_sim_block_dict, encode_sim_block_dict

  # ---------------------------------------------------------------------------

  def hlsh_blocking(self, plain_sim_hash_dict, encode_sim_hash_dict,
                    hlsh_sample_size, hlsh_num_sample, bit_array_len,
                    random_seed=None):
    """Generate blocks from the similarity hash dictionaries using Hamming
       locality sensitive hashing (HLSH) on the bit arrays representing nodes
       to limit comparison only of those nodes with a certain Hamming
       similarity. Returns these two blocking dictionaries.

       Input arguments:
         - plain_sim_hash_dict   A dictionary with keys being node values and
                                 values being hash bit arrays.
         - encode_sim_hash_dict  A dictionary with keys being node values and
                                 values being hash bit arrays.
         - hlsh_sample_size      The number of bits to sample for each HLSH
                                 block.
         - hlsh_num_sample       The number of times to sample from each bit
                                 array.
         - bit_array_len         The total length of the bit arrays in the
                                 similarity hash dictionaries.
         - random_seed           If not None then initialise the random number
                                 generator with this seed value

       Output:
         - plain_sim_block_dict    A dictionary with keys being HLSH values
                                   and values being being node key values from
                                   the plain-text similarity hash dictionary.
         - encode_sim_block_dict   A dictionary with keys being HLSH values
                                   and values being being node key values from
                                   the encoded similarity hash dictionary.
    """

    start_time = time.time()

    plain_sim_block_dict =  {}  # The dictionaries to be returned
    encode_sim_block_dict = {}

    if (random_seed != None):
      numpy.random.seed(random_seed)

    # First generate the required list of bit position arrays to be used for
    # sampling ('hlsh_num_sample' arrays each of length 'hlsh_sample_size')
    #
    bit_sample_list = []

    for sample_num in xrange(hlsh_num_sample):
      bit_sample_list.append(random.sample(xrange(bit_array_len),
                                           hlsh_sample_size))
      if (len(bit_sample_list) > 1):
        assert bit_sample_list[-1] != bit_sample_list[-2]  # Check uniqueness

    # Now apply HLSH on both similarity hash dictionaries
    #
    for (hlsh_block_dict, sim_hash_dict, dict_name) in \
      [(plain_sim_block_dict, plain_sim_hash_dict, 'plain-text'),
       (encode_sim_block_dict, encode_sim_hash_dict, 'encoded')]:

      # Loop over all similarity hash dictionary keys and bit arrays
      #
      for (key_val, dict_bit_array) in sim_hash_dict.iteritems():

        # Loop over all HSLH bit position sample lists
        #
        for (sample_num, bit_pos_list) in enumerate(bit_sample_list):
          sample_bit_array = bitarray.bitarray(hlsh_sample_size)
          sample_num_str = str(sample_num)+'-'

          for (i, bit_pos) in enumerate(bit_pos_list):
            sample_bit_array[i] = dict_bit_array[bit_pos]

          # Generate the HLSH block (dictionary) key
          #
          hlsh_dict_key = sample_num_str+sample_bit_array.to01()

          # Add the current key value into the corresponding HLSH block
          #
          hlsh_block_key_set = hlsh_block_dict.get(hlsh_dict_key, set())
          hlsh_block_key_set.add(key_val)
          hlsh_block_dict[hlsh_dict_key] = hlsh_block_key_set

      # Print summary statistics about the generated LSH blocks
      #
      print 'Number of blocks for the %s HLSH index: %d' % \
            (dict_name, len(hlsh_block_dict)) + \
            '  (with sample size: %d, and number of samples: %d' % \
            (hlsh_sample_size, hlsh_num_sample)
      hlsh_block_size_list = []
      for hlsh_block_key_set in hlsh_block_dict.itervalues():
        hlsh_block_size_list.append(len(hlsh_block_key_set))
      print '  Minimum, average, median and maximum block sizes: ' + \
            '%d / %.2f / %d / %d' % (min(hlsh_block_size_list),
                                 numpy.mean(hlsh_block_size_list),
                                 numpy.median(hlsh_block_size_list),
                                 max(hlsh_block_size_list))

    # Get the HLSH keys that do not occur in both dictionaries are remove
    # them from the block dictionaries
    #
    plain_hlsh_key_set =  set(plain_sim_block_dict.keys())
    encode_hlsh_key_set = set(encode_sim_block_dict.keys())

    plain_only_hlsh_key_set =  plain_hlsh_key_set - encode_hlsh_key_set
    encode_only_hlsh_key_set = encode_hlsh_key_set - plain_hlsh_key_set
    print
    print '  Number of HLSH blocks that occur in plain-text only: %d' % \
          (len(plain_only_hlsh_key_set))
    print '  Number of HLSH blocks that occur in encoded only: %d' % \
          (len(encode_only_hlsh_key_set))

    # Remove all not common blocks
    #
    for plain_only_block_key in plain_only_hlsh_key_set:
      del plain_sim_block_dict[plain_only_block_key]
    for encode_only_block_key in encode_only_hlsh_key_set:
      del encode_sim_block_dict[encode_only_block_key]

    print '    Final numbers of blocks: %d (plain) / %d (encoded)' % \
          (len( plain_sim_block_dict), len(encode_sim_block_dict))
    print '  Time used: %.3f sec' % (time.time() - start_time)
    print

    return plain_sim_block_dict, encode_sim_block_dict


  # ---------------------------------------------------------------------------

  def calc_cos_sim_all_pairs(self, plain_sim_hash_dict,
                             encode_sim_hash_dict, plain_sim_block_dict,
                             encode_sim_block_dict, hash_min_sim):
    """Calculate the Cosine similarity of all node pairs in the same blocks
       between the two graphs based on the LSH values in the given two
       block dictionaries. Only keep those pairs with a similarity above the
       given threshold.

       Input arguments:
         - plain_sim_hash_dict     A dictionary with keys being node key
                                   values and values being hash bit arrays.
         - encode_sim_hash_dict    A dictionary with keys being node key
                                   values and values being hash bit arrays.
         - plain_sim_block_dict    A dictionary with keys being block key
                                   values and values being being node key
                                   values from the plain-text similarity hash
                                   dictionary.
         - encode_sim_block_dict   A dictionary with keys being block key
                                   values and values being being node key
                                   values from the encoded similarity hash
                                   dictionary.
          - hash_min_sim           The minimum similarity required for a pair
                                   of identifiers to be stored in the
                                   dictionary to be returned.
       Output:
         - hash_sim_dict  A dictionary with node key value pairs as keys and
                          their hash similarities as values.
    """

    start_time = time.time()
    
    hash_sim_dict = {}

    num_pairs_compared = 0
    comp_pairs_dict = {}  # For each key from the plain-text dictionary a set
                          # of keys from the encoded dictionary

    # Loop over all blocks in both block dictionaries
    #
    for (plain_block_key, plain_key_set) in plain_sim_block_dict.iteritems():
      encode_key_set = encode_sim_block_dict.get(plain_block_key, set())

      for key_val1 in plain_key_set:
        bit_array1 =     plain_sim_hash_dict[key_val1]
        bit_array1_len = len(bit_array1)

        encode_comp_key_set = comp_pairs_dict.get(key_val1, set())

        for key_val2 in encode_key_set:
          if (key_val2 in encode_comp_key_set):  # An already compared pair
            continue

          encode_comp_key_set.add(key_val2)  # Mark as being compared

          bit_array2 = encode_sim_hash_dict[key_val2]

          num_pairs_compared += 1

          # Get number of differing 1-bits
          #
          ba_xor_hw = (bit_array1 ^ bit_array2).count(1)

          # Calculate the angle difference using LSH based on Cosine distance
          #
          cos_hash_sim = 1.0 - float(ba_xor_hw) / bit_array1_len

          if (cos_hash_sim >= hash_min_sim):
            hash_sim_dict[(key_val1, key_val2)] = cos_hash_sim

        comp_pairs_dict[key_val1] = encode_comp_key_set

    print 'Compared %d record pairs based on their Cosine LSH hash values' % \
          (num_pairs_compared) + ' (using all-pair comparison)'
    print '  %d of these pairs had a similarity of at least %.2f' % \
          (len(hash_sim_dict), hash_min_sim)

    print '  Time used: %.3f sec' % (time.time() - start_time)
    print

    comp_pairs_dict = {}  # Not needed anymore

    return hash_sim_dict

  # ---------------------------------------------------------------------------

  def calc_cos_sim_edge_comp(self, plain_sim_hash_dict, encode_sim_hash_dict,
                             plain_sim_block_dict, encode_sim_block_dict,
                             hash_min_sim, plain_sim_graph, encode_sim_graph,
                             low_sim_tol, up_sim_tol, edge_min_sim,
                             min_sim_tol):
    """Calculate the Cosine similarity of node pairs based on the similarities
       of the most similar edge of each node (assuming the most similar edges
       of two matching nodes should be highly similar). The nodes whose
       highest similar edge have comparable similarities (within the interval
       [low_sim_tol, up_sim_tol] tolerances) and a comparable number of edges
       are being compared and their similarities calculated based on the LSH
       hash values in the given two feature dictionaries. Only keep those
       pairs with a similarity above the given threshold.

       Input arguments:
         - plain_sim_hash_dict     A dictionary with keys being node key
                                   values and values being hash bit arrays.
         - encode_sim_hash_dict    A dictionary with keys being node key
                                   values and values being hash bit arrays.
         - plain_sim_block_dict    A dictionary with keys being block key
                                   values and values being being node key
                                   values from the plain-text similarity hash
                                   dictionary.
         - encode_sim_block_dict   A dictionary with keys being block key
                                   values and values being being node key
                                   values from the encoded similarity hash
                                   dictionary.
          - hash_min_sim           The minimum similarity required for a pair
                                   of identifiers to be stored in the
                                   dictionary to be returned.
         - plain_sim_graph         The plain-text similarity graph.
         - encode_sim_graph        The encoded similarity graph.
         - low_sim_tol             The lower similarity tolerance between a
                                   plain-text similarity and an encoded
                                   similarity (i.e. how much an encoded edge
                                   similarity can be lower than a plain-text
                                   edge similarity to be considered).
         - up_sim_tol              The upper similarity tolerance between a
                                   plain-text similarity and an encoded
                                   similarity (i.e. how much an encoded edge
                                   similarity can be higher than a plain-text
                                   edge similarity to be considered).
         - edge_min_sim            The assumed minimum similarity of any edge
                                   in the two graphs.
         - min_sim_tol             The tolerance to be used to get the number
                                   of edges for each node in the range
                                   [edge_min_sim, edge_min_sim+min_sim_tol].

       Output:
         - hash_sim_dict  A dictionary with node key value pairs as keys and
                          their hash similarities as values.
    """

    assert low_sim_tol >= 0.0, low_sim_tol
    assert up_sim_tol >=  0.0, up_sim_tol

    start_time = time.time()

    hash_sim_dict = {}

    # Generate two dictionaries where for each node we keep its edge with the
    # highest similarity, as well as its total number of edges and the
    # number of edges with a similarity of at least edge_min_sim+min_sim_tol.
    #
    plain_edge_sim_dict =  {}
    encode_edge_sim_dict = {}

    min_high_sim_val = edge_min_sim + min_sim_tol

    for key_val1 in plain_sim_hash_dict:
      max_edge_sim =      0.0
      num_all_sim_edge =  0  # Number of all edges
      num_high_sim_edge = 0  # Similarity of at least edge_min_sim+min_sim_tol
      for key_val2 in plain_sim_graph.neighbors(key_val1):
        edge_sim = plain_sim_graph.edge[key_val1][key_val2]['sim']
        max_edge_sim = max(edge_sim, max_edge_sim)
        if (edge_sim >= min_high_sim_val):
          num_high_sim_edge += 1
        num_all_sim_edge += 1
      plain_edge_sim_dict[key_val1] = (max_edge_sim, num_all_sim_edge,
                                       num_high_sim_edge)

    for key_val1 in encode_sim_hash_dict:
      max_edge_sim =      0.0
      num_all_sim_edge =  0  # Number of all edges
      num_high_sim_edge = 0  # Similarity of at least edge_min_sim+min_sim_tol
      for key_val2 in encode_sim_graph.neighbors(key_val1):
        edge_sim = encode_sim_graph.edge[key_val1][key_val2]['sim']
        max_edge_sim = max(edge_sim, max_edge_sim)
        if (edge_sim >= min_high_sim_val):
          num_high_sim_edge += 1
        num_all_sim_edge += 1
      encode_edge_sim_dict[key_val1] = (max_edge_sim, num_all_sim_edge,
                                        num_high_sim_edge)

    num_pairs_compared =   0
    num_pair_sim_filter =  0
    num_pair_edge_filter = 0

    comp_pairs_dict = {}  # For each key from the plain-text dictionary a set
                          # of keys from the encoded dictionary

    # Loop over all blocks in both block dictionaries
    #
    for (plain_block_key, plain_key_set) in plain_sim_block_dict.iteritems():
      encode_key_set = encode_sim_block_dict.get(plain_block_key, set())

      for key_val1 in plain_key_set:
        bit_array1 = plain_sim_hash_dict[key_val1]
        bit_array1_len = len(bit_array1)

        encode_comp_key_set = comp_pairs_dict.get(key_val1, set())

        plain_max_edge_sim, plain_num_all_sim_edge, \
                     plain_num_high_sim_edge = plain_edge_sim_dict[key_val1]
        
        for key_val2 in encode_key_set:
          if (key_val2 in encode_comp_key_set):  # An already compared pair
            continue

          encode_comp_key_set.add(key_val2)  # Mark as being compared

          encode_max_edge_sim, encode_num_all_sim_edge, \
                   encode_num_high_sim_edge = encode_edge_sim_dict[key_val2]

          # First check the maximum edge similarities between the two nodes
          #
          edge_sim_diff = encode_max_edge_sim - plain_max_edge_sim

          # Don't compare the node pair if their highest similarities are too
          # different
          # - Encoding similarity is too high compared to plain similarity
          # - Encoding similarity is too low compared to plain similarity
          #
          if (edge_sim_diff > up_sim_tol) or (edge_sim_diff < -low_sim_tol):
            num_pair_sim_filter += 1
            continue
          
          # Second check if the number of edges between the two nodes are
          # similar as otherwise they unlikely refer to the same value and
          # should not be compared
          #
          if (encode_num_all_sim_edge < plain_num_high_sim_edge) or \
             (plain_num_all_sim_edge < encode_num_high_sim_edge):
            num_pair_edge_filter += 1
            continue

          bit_array2 = encode_sim_hash_dict[key_val2]

          num_pairs_compared += 1

          # Get number of differing 1-bits
          #
          ba_xor_hw = (bit_array1 ^ bit_array2).count(1)

          # Calculate the angle difference using LSH based on Cosine distance
          #
          cos_hash_sim = 1.0 - float(ba_xor_hw) / bit_array1_len

          if (cos_hash_sim >= hash_min_sim):
            hash_sim_dict[(key_val1, key_val2)] = cos_hash_sim

        comp_pairs_dict[key_val1] = encode_comp_key_set

    print 'Compared %d record pairs based on their Cosine LSH hash values' % \
          (num_pairs_compared) + ' (using sim-edge comparison)'
    print '  %d of these pairs had a similarity of at least %.2f' % \
          (len(hash_sim_dict), hash_min_sim)
    print '  %d pairs filtered due to their different maximum edge ' % \
          (num_pair_sim_filter) + 'similarity'
    print '  %d pairs filtered due to their different number of edges' % \
          (num_pair_edge_filter)
    
    print '  Time used: %.3f sec' % (time.time() - start_time)
    print

    comp_pairs_dict = {}  # Not needed anymore

    return hash_sim_dict

  # ---------------------------------------------------------------------------

  def calc_cosine_sim(self, hash_sim_dict, feat_dict1, feat_dict2, calc_type):
    """Calculate the actual Cosine similarity for all record pairs in the given
       hash dictionary based on their actual feature vectors, and return a new
       dictionary with these Cosine similarities.

       Input arguments:
         - hash_sim_dict  A dictionary with node value pairs as keys and their
                          hash similarities as values.
         - feat_dict1     A dictionary with keys being node values and values
                          being feature vectors.
         - feat_dict2     A dictionary with keys being node values and values
                          being feature vectors.
         - calc_type      The way Cosine similarity is calculated. This can be
                          'scipy' or 'numpy'.

       Output:
         - cos_sim_dict  A dictionary with node value pairs as keys and their
                         actual Cosine similarities as values.
    """

    assert calc_type in ['scipy', 'numpy'], calc_type

    cos_sim_dict = {}

    for (val1,val2) in hash_sim_dict.iterkeys():
      feat_vec1 = feat_dict1[val1]
      feat_vec2 = feat_dict2[val2]

      if (calc_type == 'scipy'):

        # The following does not seem to be accurate and break in certain cases
        #
        try:
          cos_sim = 1.0 - scipy.spatial.distance.cosine(feat_vec1, feat_vec2)
        except:
          cos_sim = 0.0

      elif (calc_type == 'numpy'):
        vec_len1 = math.sqrt(numpy.dot(feat_vec1, feat_vec1))
        vec_len2 = math.sqrt(numpy.dot(feat_vec2, feat_vec2))
        cosine = numpy.dot(feat_vec1,feat_vec2) / vec_len1 / vec_len2
        cos_sim = 1.0 - math.acos(min(cosine,1.0)) / math.pi

      cos_sim_dict[(val1,val2)] = cos_sim
      
    return cos_sim_dict

  # ---------------------------------------------------------------------------

  def calc_euclid_dist(self, hash_sim_dict, feat_dict1, feat_dict2):
    """Calculate the Euclidean distance for all record pairs in the given
       hash dictionary based on their actual feature vectors, and return a new
       dictionary with these Euclidean distances.

       Input arguments:
         - hash_sim_dict  A dictionary with node value pairs as keys and their
                          hash similarities as values.
         - feat_dict1     A dictionary with keys being node values and values
                          being feature vectors.
         - feat_dict2     A dictionary with keys being node values and values
                          being feature vectors.

       Output:
         - euclid_dist_dict  A dictionary with node value pairs as keys and
                             their Euclidean distances as values.
    """

    euclid_dist_dict = {}

    for (val1,val2) in hash_sim_dict.iterkeys():
      feat_vec1 = feat_dict1[val1]
      feat_vec2 = feat_dict2[val2]

      euclid_dist_dict[(val1,val2)] = numpy.linalg.norm(feat_vec1 - feat_vec2)

    return euclid_dist_dict

  # ---------------------------------------------------------------------------

  def calc_sort_edge_diff_dist(self, hash_sim_dict, plain_sim_graph,
                               encode_sim_graph):
    """Calculate a distance as the averaged absolute difference of the
       similarities between pairs of edges of two nodes, where these are
       sorted by their similarities (i.e. calculate similarity difference
       between the two edges with the highest similarities, then the second
       highest, etc.). Finally divide the sum of these distances by the
       number of compared similarity differences.

       Input arguments:
         - hash_sim_dict     A dictionary with node value pairs as keys and
                             their hash similarities as values.
         - plain_sim_graph   The plain-text similarity graph.
         - encode_sim_graph  The encoded similarity graph.

       Output:
         - edge_diff_dist_dict  A dictionary with node value pairs as keys and
                                their averaged edge similarity differences as
                                values.
    """

    edge_diff_dist_dict = {}

    for (val1,val2) in hash_sim_dict.iterkeys():

      node_sim_list1 = []  # Get the edge similarities for node 1
      for (eval1,eval2) in plain_sim_graph.edges(val1):
        node_sim_list1.append(plain_sim_graph.edge[eval1][eval2]['sim'])

      node_sim_list2 = []  # Get the edge similarities for node 2
      for (eval1,eval2) in encode_sim_graph.edges(val2):
        node_sim_list2.append(encode_sim_graph.edge[eval1][eval2]['sim'])

#      print 'pt:', node_sim_list1
#      print 'en:', node_sim_list2
      num_edge_to_comp = min(len(node_sim_list1), len(node_sim_list2))

      sim_diff_list = []
      for (i,plain_sim) in enumerate(node_sim_list1[:num_edge_to_comp]):
        sim_diff_list.append(abs(plain_sim - node_sim_list2[i]))
#      print 'sim diff:', sim_diff_list

      edge_diff_dist_dict[(val1,val2)] = sum(sim_diff_list) / num_edge_to_comp

    return edge_diff_dist_dict

# =============================================================================

def build_regre_model(ba_sim_list, qg_sim_list, qg_num_q_gram_list,
                      regre_model_str, plot_data_name, encode_method,
                      encode_sim_funct, plot_file_name, plot_title):
  
  x_val_list = qg_sim_list
  y_val_list = ba_sim_list
  q_val_list = qg_num_q_gram_list
  
  if(regre_model_str == 'linear'):
      
    print 'Traning the linear regression model'
    
    if(encode_method in ['2sh','tmh']):
      x_val_array = numpy.reshape(x_val_list,(-1,1))
      y_val_array = numpy.reshape(y_val_list,(-1,1))
    else:
      x_val_array = numpy.array(zip(x_val_list,q_val_list))
      y_val_array = numpy.reshape(y_val_list,(-1,1))
  
    # Split the data for the training and testing samples
    x_train, x_test, y_train, y_test = train_test_split(x_val_array, 
                                                        y_val_array, 
                                                        test_size=0.25, 
                                                        random_state=42)
    
    # Train the model
    reg_model = linear_model.LinearRegression()
    
    reg_model.fit(x_train, y_train)
    
    # Testing the model
    y_predict = reg_model.predict(x_test)
    
    #print y_predict
    
  elif(regre_model_str == 'isotonic'):
    
    print 'Traning the isotonic regression model'
    
    x_train, x_test, y_train, y_test = train_test_split(x_val_list, 
                                                        y_val_list, 
                                                        test_size=0.25, 
                                                        random_state=42)
  
    reg_model = IsotonicRegression()

    reg_model.fit_transform(x_train, y_train)
    
    y_predict = reg_model.predict(x_test)
    
  elif(regre_model_str == 'poly'):
    
    print 'Traning the polynomial regression model'
    
    if(encode_method in ['2sh', 'tmh']):
      x_val_array = numpy.reshape(x_val_list,(-1,1))
      y_val_array = numpy.reshape(y_val_list,(-1,1))
    else:
      x_val_array = numpy.array(zip(x_val_list,q_val_list))
      y_val_array = numpy.reshape(y_val_list,(-1,1))
  
    # Split the data for the training and testing samples
    x_train, x_test, y_train, y_test = train_test_split(x_val_array, 
                                                        y_val_array, 
                                                        test_size=0.25, 
                                                        random_state=42)
    
    poly = PolynomialFeatures(degree=2)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)
    
    # Train the model
    reg_model = linear_model.LinearRegression()
    
    reg_model.fit(x_train_poly, y_train)
    
    # Testing the model
    y_predict = reg_model.predict(x_test_poly)
  
  else:
    raise Exception, '***WARNING!, wrong regression model'   
  
  y_p_new = []
  y_t_new = []
  
  y_predict = list(zip(*y_predict)[0])
  y_test = list(zip(*y_test)[0])
  
  for i, y_p in enumerate(y_predict):
    y_t = y_test[i]
    
    if(y_p > 0):
      y_p_new.append(y_p)
      y_t_new.append(y_t)
  
  y_predict_eval = y_p_new
  y_test_eval    = y_t_new
  
  err_val_list = [abs(y_p - y_t) for y_p, y_t in zip(y_predict_eval, y_test_eval)]
  
  mean_sqrd_lg_err = numpy.mean([(math.log(1+y_p) - math.log(1+y_t))**2 
                                 for y_p, y_t in zip(y_predict_eval, y_test_eval)])
  
  # Evaluating the model
  #
  model_var_score        = metrics.explained_variance_score(y_test_eval, y_predict_eval)
  model_min_abs_err      = min(err_val_list)
  model_max_abs_err      = max(err_val_list)
  model_avrg_abs_err     = numpy.mean(err_val_list)
  model_std_abs_err      = numpy.std(err_val_list)
  model_mean_sqrd_err    = metrics.mean_squared_error(y_test_eval, y_predict_eval)
  model_mean_sqrd_lg_err = mean_sqrd_lg_err
  model_r_sqrd           = metrics.r2_score(y_test_eval, y_predict_eval)
  
  print
  print 'Evaluation of the %s regression model' %regre_model_str
  print '  Explained variance score:       %.5f' %model_var_score
  print '  Min/Max/Average absolute error: %.5f / %.5f / %.5f' %(min(err_val_list), \
                                                                 max(err_val_list), \
                                                                 numpy.mean(err_val_list))
  print '  Standard deviation error:       %.5f' %model_std_abs_err
  print '  Mean-squared error:             %.5f' %model_mean_sqrd_err
  print '  Mean-squared log error:         %.5f' %model_mean_sqrd_lg_err
  print '  R-squared value:                %.5f' %model_r_sqrd
  print
  
  eval_res_tuple = (model_var_score, model_min_abs_err, model_max_abs_err, model_avrg_abs_err,
                    model_std_abs_err, model_mean_sqrd_err, model_mean_sqrd_lg_err, 
                    model_r_sqrd)
  
  #plot_file_name = None
  if (plot_file_name != None) and (len(x_val_list) > 0):
    
    if(regre_model_str in ['linear', 'poly']):
      if(encode_method in ['2sh','tmh']):
        x_plot_train = list(zip(*x_train)[0])
        x_plot_test = list(zip(*x_test)[0])
        y_train = list(zip(*y_train)[0])
      else:
        x_plot_train, q_val_train = zip(*x_train)
        x_plot_test, q_val_test = zip(*x_test)
        y_train = list(zip(*y_train)[0])
    else:
      x_plot_train = x_train
      x_plot_test = x_test
    
    # Sampling for the plot
    x_plot_train = x_plot_train[:2000]
    y_train      = y_train[:2000]
    
    x_plot_test  = x_plot_test[:500]
    y_test       = y_test[:500]
    y_predict    = y_predict[:500]

    
    w,h = plt.figaspect(PLT_PLOT_RATIO)
    plt.figure(figsize=(w,h))

    plt.plot(x_plot_train, y_train, ".", markersize=3, color='red', label='Training set')
    #plt.plot(x_plot_test, y_test, ".", markersize=2, color='#0bb015', label="Testing set")
    #plt.plot(x_plot_test, y_predict, ".", markersize=3, color='red', label="Predictions")
    
    p = numpy.poly1d(numpy.polyfit(x_plot_train, y_train, 3))
    t = numpy.linspace(0, 1, 200)
    plt.plot(t, p(t), '-', color='#000000')
    
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)
    
    #m,b = numpy.polyfit(x_plot_train, y_train, 2)
    #print m,b
    #m = numpy.array(m)
    #b = numpy.array(b)
    #plt.plot(x_plot_train, m*x_plot_train + b, color='#000000')
    
    
    
    #fig, ax = plt.subplots(1, 1)
    #ax.set_aspect(PLT_PLOT_RATIO)
    #ax.plot(x_plot_train, y_train, ".", markersize=2, color='#144ba4', label='Training set')
    #ax.plot(x_plot_test, y_test, ".", markersize=2, color='#0bb015', label="Testing set")
    #ax.plot(x_plot_test, y_predict, ".", markersize=3, color='red', label="Predictions")
    
    #===========================================================================
    # if(regre_model_str == 'linear'):
    #   plot_title = 'Linear Regression ' + plot_data_name
    # elif(regre_model_str == 'isotonic'):
    #   plot_title = 'Isotonic Regression ' + plot_data_name
    # elif(regre_model_str == 'poly'):
    #   plot_title = 'Polynomial Regression ' + plot_data_name
    # 
    # plot_title = 'FirstName, LastName, StreetAddress'
    #===========================================================================
    
    if(encode_sim_funct == 'jacc'):
      sim_funct_str = 'Jaccard'
    elif(encode_sim_funct == 'dice'):
      sim_funct_str = 'Dice'
    else:
      sim_funct_str = 'Hamming'
    
    if(encode_method == 'bf'):
      enc_method_str = 'BF'
    elif(encode_method == 'tmh'):
      enc_method_str = 'TMH'
    else:
      enc_method_str = '2SH'
    
    #ax.set_title(plot_title, fontsize=TITLE_FONT_SIZE)
    #ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    #ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    #plot_title = 'FistName, LastName, Street'
    #plt.rc('xtick', labelsize=TICK_FONT_SIZE)
    #plt.rc('ytick', labelsize=TICK_FONT_SIZE)
    #plt.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
    #plt.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
    plt.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    plt.title(plot_title, fontsize=TITLE_FONT_SIZE)
    #plt.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    
    #ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

    plt.xlabel('Q-grams (%s)' %sim_funct_str, 
               fontsize=PLT_FONT_SIZE)
    plt.ylabel('%s encoding (%s)' %(enc_method_str, sim_funct_str), 
               fontsize=PLT_FONT_SIZE)

    plt.savefig(plot_file_name, bbox_inches='tight')

  return reg_model, eval_res_tuple
  
#----------------------------------------------------------------------

def test_sim_regre_model(regre_model, ba_sim_list, qg_sim_list, qg_num_q_gram_list,
                         regre_model_str, encode_method):
  
  sample_size = 500
  sample_indices = random.sample([i for i in xrange(len(ba_sim_list))], sample_size)
  
  text_x_val_list = []
  text_y_val_list = []
  text_q_val_list = []
  
  for list_index in sample_indices:
    text_x_val_list.append(qg_sim_list[list_index])
    text_y_val_list.append(ba_sim_list[list_index])
    text_q_val_list.append(qg_num_q_gram_list[list_index])
    
  if(encode_method in ['2sh','tmh']):
    x_test_val_array = numpy.reshape(text_x_val_list,(-1,1))
    y_test_val_array = numpy.reshape(text_y_val_list,(-1,1))
  else:
    x_test_val_array = numpy.array(zip(text_x_val_list,text_q_val_list))
    y_test_val_array = numpy.reshape(text_y_val_list,(-1,1))
    
  if(regre_model_str == 'poly'):
    
    poly = PolynomialFeatures(degree=2)
    x_test_poly = poly.fit_transform(x_test_val_array)
    
    # Testing the model
    y_test_predict = regre_model.predict(x_test_poly)
  
  else:
    # Testing the model
    y_test_predict = regre_model.predict(x_test_val_array)
    
  
  # Calculate test scores of the model
  #
  y_p_new = []
  y_t_new = []
  
  y_test_predict = list(zip(*y_test_predict)[0])
  y_test_val_array = list(zip(*y_test_val_array)[0])
  
  for i, y_p in enumerate(y_test_predict):
    y_t = y_test_val_array[i]
    
    if(y_p > 0):
      y_p_new.append(y_p)
      y_t_new.append(y_t)
  
  y_predict_eval = y_p_new
  y_test_eval    = y_t_new
  
  err_val_list = [abs(y_p - y_t) for y_p, y_t in zip(y_predict_eval, y_test_eval)]
  
  mean_sqrd_lg_err = numpy.mean([(math.log(1+y_p) - math.log(1+y_t))**2 
                                 for y_p, y_t in zip(y_predict_eval, y_test_eval)])
  
  # Evaluating the model
  #
  model_var_score        = metrics.explained_variance_score(y_test_eval, y_predict_eval)
  model_min_abs_err      = min(err_val_list)
  model_max_abs_err      = max(err_val_list)
  model_avrg_abs_err     = numpy.mean(err_val_list)
  model_std_abs_err      = numpy.std(err_val_list)
  model_mean_sqrd_err    = metrics.mean_squared_error(y_test_eval, y_predict_eval)
  model_mean_sqrd_lg_err = mean_sqrd_lg_err
  model_r_sqrd           = metrics.r2_score(y_test_eval, y_predict_eval)
  
  print
  print 'Evaluation of the loaded %s regression model' %regre_model_str
  print '  Explained variance score:       %.5f' %model_var_score
  print '  Min/Max/Average absolute error: %.5f / %.5f / %.5f' %(min(err_val_list), \
                                                                 max(err_val_list), \
                                                                 numpy.mean(err_val_list))
  print '  Standard deviation error:       %.5f' %model_std_abs_err
  print '  Mean-squared error:             %.5f' %model_mean_sqrd_err
  print '  Mean-squared log error:         %.5f' %model_mean_sqrd_lg_err
  print '  R-squared value:                %.5f' %model_r_sqrd
  
  return False
  
#----------------------------------------------------------------------

def eval_regre_model(y_test, y_predict):
  
  y_p_new = []
  y_t_new = []
  for i, y_p in enumerate(y_predict):
    y_t = y_test[i]
    
    if(y_p > 0):
      y_p_new.append(y_p)
      y_t_new.append(y_t)
  
  y_predict = y_p_new
  y_test    = y_t_new
  
  err_val_list = [abs(y_p - y_t) for y_p, y_t in zip(y_predict, y_test)]
  
  mean_sqrd_lg_err = numpy.mean([(math.log(1+y_p) - math.log(1+y_t))**2 for y_p, y_t in zip(y_predict, y_test)])
  
  # Evaluating the model
  #
  model_var_score        = metrics.explained_variance_score(y_test, y_predict)
  model_min_abs_err      = min(err_val_list)
  model_max_abs_err      = max(err_val_list)
  model_avrg_abs_err     = numpy.mean(err_val_list)
  model_std_abs_err      = numpy.std(err_val_list)
  model_mean_sqrd_err    = metrics.mean_squared_error(y_test, y_predict)
  model_mean_sqrd_lg_err = mean_sqrd_lg_err
  model_r_sqrd           = metrics.r2_score(y_test, y_predict)
  
  print
  print 'Evaluation of the regression model'
  print '  Explained variance score:       %.5f' %model_var_score
  print '  Min/Max/Average absolute error: %.5f / %.5f / %.5f' %(min(err_val_list), \
                                                                 max(err_val_list), \
                                                                 numpy.mean(err_val_list))
  print '  Standard deviation error:       %.5f' %model_std_abs_err
  print '  Mean-squared error:             %.5f' %model_mean_sqrd_err
  print '  Mean-squared log error:         %.5f' %model_mean_sqrd_lg_err
  print '  R-squared value:                %.5f' %model_r_sqrd
  print
  
  return model_var_score, model_min_abs_err, model_max_abs_err, \
    model_avrg_abs_err, model_std_abs_err, model_mean_sqrd_err, \
    model_mean_sqrd_lg_err, model_r_sqrd

#------------------------------------------------------------------------------

def plot_regre_model(x_train, x_test, y_train, y_test, y_predict):
  
  x_plot_train, q_val_train = zip(*x_train)
  x_plot_test, q_val_test = zip(*x_test)
  
  # Sampling for the plot
  x_plot_train = x_plot_train[:2000]
  y_train      = y_train[:2000]
  
  x_plot_test  = x_plot_test[:500]
  y_test       = y_test[:500]
  y_predict    = y_predict[:500]

  
  fig, ax = plt.subplots(1, 1)
  ax.plot(x_plot_train, y_train, ".", markersize=1, color='#144ba4', label='Training set')
  ax.plot(x_plot_test, y_test, ".", markersize=1, color='#0bb015', label="Testing set")
  ax.plot(x_plot_test, y_predict, ".", markersize=2, color='red', label="Predictions")

  plot_title = 'Polynomial Regression '
  
  ax.set_title(plot_title)
  ax.legend(loc="best",)

  plt.xlabel('Q-gram Jaccard similarity')
  plt.ylabel('Encoded Jaccard similarity')
  
  plot_file_name = 'compare_jacc_jacc_sim.eps'

  plt.savefig(plot_file_name, bbox_inches='tight')
  
  return 
  
#------------------------------------------------------------------------------

def graph_sim_dist_plot(qg_sim_graph_obj, ba_sim_graph_obj,
                        plain_node_val_dict, encode_node_val_dict, 
                        remove_uncommon_edges=False):
  
  qg_graph = qg_sim_graph_obj.sim_graph
  ba_graph = ba_sim_graph_obj.sim_graph
  
  qg_graph_edges = qg_graph.edges()
  ba_graph_edges = ba_graph.edges()
  
  ba_edge_dict = {}
  qg_edge_dict = {}
  
  qg_sim_val_list = []
  qg_edge_val_set = set()
  ba_edge_val_set = set()
  
  qg_key_set = set()
  ba_key_set = set()
  
  
  for qg_key1, qg_key2 in qg_graph_edges:
    qg_sim_val = qg_graph.edge[qg_key1][qg_key2]['sim']
    
    qg_edge_tuple = (qg_key1, qg_key2)
    qg_edge_tuple = tuple(sorted(qg_edge_tuple))
    
    qg_edge_val_set.add(qg_edge_tuple)
    
    qg_key_set.add(qg_key1)
    qg_key_set.add(qg_key2)
  
  for ba_key1, ba_key2 in ba_graph_edges:
    ba_sim_val = ba_graph.edge[ba_key1][ba_key2]['sim']
    
    q_set1, bit_array1 = encode_node_val_dict[ba_key1]
    q_set2, bit_array2 = encode_node_val_dict[ba_key2]
    
    ba_q_key1 = tuple(sorted(q_set1))
    ba_q_key2 = tuple(sorted(q_set1))
    
    ba_edge_tuple = (ba_q_key1, ba_q_key2)
    ba_edge_tuple = tuple(sorted(ba_edge_tuple))
    ba_edge_val_set.add(ba_edge_tuple)
    
    ba_key_set.add(ba_q_key1)
    ba_key_set.add(ba_q_key2)
    
  print list(ba_edge_val_set)[:10]
  print
  print list(qg_edge_val_set)[:10]
  
  #print ba_edge_val_set
  #print
  #print qg_edge_val_set
  
  c_nodes = ba_key_set.intersection(qg_key_set)
  print len(qg_key_set)
  print len(ba_key_set)
  print len(c_nodes)
  print
  
  common_edges = ba_edge_val_set.intersection(qg_edge_val_set)
  ba_only_edges = ba_edge_val_set - qg_edge_val_set
  qg_only_edges = qg_edge_val_set - ba_edge_val_set
  
  print '  Number of common edges:          %d' %len(common_edges)
  print '  Number of encode only edges:     %d' %len(ba_only_edges)
  print '  Number of plain-text only edges: %d' %len(qg_only_edges)
  print
  
  
  #=============================================================================
  # for qg_key1, qg_key2 in qg_graph_edges:
  #   sim_val = qg_graph.edge[qg_key1][qg_key2]['sim']
  #   qg_sim_val_list.append(sim_val)
  #   
  #   key1_ent_id_list = list(qg_graph.node[qg_key1]['ent_id_set'])
  #   key2_ent_id_list = list(qg_graph.node[qg_key2]['ent_id_set'])
  #   
  #   edge_tuple = (qg_key1, qg_key2)
  #   
  #   for id_pair in itertools.product(key1_ent_id_list, key2_ent_id_list):
  #     
  #     id_pair = tuple(sorted(id_pair))
  #     qg_edge_val_set.add(id_pair)
  #     
  #     edge_list = qg_edge_dict.get(id_pair, [])
  #     edge_list.append(edge_tuple)
  #     qg_edge_dict[id_pair] = edge_list
  #     
  # ba_sim_val_list = []
  # ba_edge_val_set = set()
  # 
  # for ba_key1, ba_key2 in ba_graph_edges:
  #   sim_val = ba_graph.edge[ba_key1][ba_key2]['sim']
  #   ba_sim_val_list.append(sim_val)
  #   
  #   key1_ent_id_list = list(ba_graph.node[ba_key1]['ent_id_set'])
  #   key2_ent_id_list = list(ba_graph.node[ba_key2]['ent_id_set'])
  #   
  #   edge_tuple = (ba_key1, ba_key2)
  #   
  #   for id_pair in itertools.product(key1_ent_id_list, key2_ent_id_list):
  #     
  #     id_pair = tuple(sorted(id_pair))
  #     ba_edge_val_set.add(id_pair)
  #     
  #     edge_list = ba_edge_dict.get(id_pair, [])
  #     edge_list.append(edge_tuple)
  #     ba_edge_dict[id_pair] = edge_list
  #     
  # common_edges = ba_edge_val_set.intersection(qg_edge_val_set)
  # ba_only_edges = ba_edge_val_set - qg_edge_val_set
  # qg_only_edges = qg_edge_val_set - ba_edge_val_set
  # 
  # print '  Number of common edges:          %d' %len(common_edges)
  # print '  Number of encode only edges:     %d' %len(ba_only_edges)
  # print '  Number of plain-text only edges: %d' %len(qg_only_edges)
  # print
  # 
  # if(remove_uncommon_edges):
  #   
  #   for qg_id_pair in qg_only_edges:
  #     qg_edge_list = qg_edge_dict[qg_id_pair]
  #     
  #     for qg_edge_key1, qg_edge_key2 in qg_edge_list:
  #       if(qg_graph.has_edge(qg_edge_key1, qg_edge_key2)):
  #         qg_graph.remove_edge(qg_edge_key1, qg_edge_key2)
  #   
  #   for ba_id_pair in ba_only_edges:
  #     ba_edge_list = ba_edge_dict[ba_id_pair]
  #     
  #     for ba_edge_key1, ba_edge_key2 in ba_edge_list:
  #       if(ba_graph.has_edge(ba_edge_key1, ba_edge_key2)):
  #         ba_graph.remove_edge(ba_edge_key1, ba_edge_key2)
  #     
  # 
  # plot_file_name = 'sim_difference_hmlsh_%d_%d.eps'
  # bins = numpy.linspace(0, 1, 50)  
  # 
  # plt.hist([qg_sim_val_list, ba_sim_val_list], bins, alpha=0.5, label=['plain-text', 'encoded'])
  # plt.legend(loc='best')
  # plt.savefig(plot_file_name, bbox_inches='tight')
  #=============================================================================
  
  return qg_sim_graph_obj, ba_sim_graph_obj, common_edges, ba_only_edges, qg_only_edges
  
  
#----------------------------------------------------------------------

def calc_uniq_ratio(feat_dict, feat_name_list, round_digit=None):
  
  unique_val_list = [[] for _ in range(len(feat_name_list))]
  
  for node_key in feat_dict.keys():
    feat_val_array = feat_dict[node_key]
    
    for i, feat_val in enumerate(feat_val_array):
      unique_val_list[i].append(feat_val)
      
  unique_ratio_dict = {}
  
  for i, feat_val_list in enumerate(unique_val_list):
    
    uniq_val_set = set(feat_val_list)
    uniq_ratio = float(len(uniq_val_set))/len(feat_val_list)
    
    unique_ratio_dict[feat_name_list[i]] = uniq_ratio
  
  return unique_ratio_dict
  
#----------------------------------------------------------------------

def random_graph_matching(edge_sim_conf_dict):
  
  edges_list = edge_sim_conf_dict.keys()
  edge_indices_list = [i for i in range(len(edges_list))]
  
  # randomly shuffle indices list 
  #
  random.shuffle(edge_indices_list)
  
  ident_ba_keys = set()
  ident_qg_keys = set()
  
  random_sim_pair_dict = {}
  
  for edge_index in edge_indices_list:
    
    qg_key, ba_key = edges_list[edge_index]
    
    if(qg_key not in ident_qg_keys and 
       ba_key not in ident_ba_keys):
      
      random_sim_pair_dict[(qg_key, ba_key)] = edge_sim_conf_dict[(qg_key, ba_key)][0]
    
    return random_sim_pair_dict

#----------------------------------------------------------------------

def symmetric_graph_matching(edge_sim_conf_dict, ba_conn_comp_dict, 
                             qg_conn_comp_dict, weight_list):
  
  weight_array = numpy.array(weight_list)
  
  sorted_sim_pair_list = sorted([(edge,sum(numpy.array(val_list)*weight_array)) 
                                 for edge,val_list in edge_sim_conf_dict.items()], 
                                 key=lambda t: sum(numpy.array(t[1])*weight_array), 
                                 reverse=True)
  
  symmetric_sim_pair_dict = {}
  
  ba_key_max_weight_dict = {}
  qg_key_max_weight_dict = {}
  
  for key_pair, weighted_sim_sum in sorted_sim_pair_list:
    qg_key, ba_key = key_pair
    
    if(qg_key not in qg_key_max_weight_dict and 
       ba_key not in ba_key_max_weight_dict):
      
      symmetric_sim_pair_dict[(qg_key, ba_key)] = weighted_sim_sum
      ba_key_max_weight_dict[ba_key] = weighted_sim_sum
      qg_key_max_weight_dict[qg_key] = weighted_sim_sum
    
    else:
      if(qg_key not in qg_key_max_weight_dict):
        qg_key_max_weight_dict[qg_key] = weighted_sim_sum
        
      if(ba_key not in ba_key_max_weight_dict):
        ba_key_max_weight_dict[ba_key] = weighted_sim_sum
  
  return symmetric_sim_pair_dict

#----------------------------------------------------------------------

def greedy_graph_matching(edge_sim_conf_dict, weight_list):
  
  weight_array = numpy.array(weight_list)
  
  sorted_sim_pair_list = sorted([(edge,sum(numpy.array(val_list)*weight_array)) 
                                 for edge,val_list in edge_sim_conf_dict.items()], 
                                 key=lambda t: sum(numpy.array(t[1])*weight_array), 
                                 reverse=True)
  
  ident_ba_keys = set()
  ident_qg_keys = set()
  
  greedy_sim_pair_dict = {}
  
  for key_pair, weighted_sim_sum in sorted_sim_pair_list:
    qg_key, ba_key = key_pair
    
    if(qg_key not in ident_qg_keys and 
       ba_key not in ident_ba_keys):
      
      greedy_sim_pair_dict[(qg_key, ba_key)] = weighted_sim_sum
      
      ident_ba_keys.add(ba_key)
      ident_qg_keys.add(qg_key)
  
  return greedy_sim_pair_dict

#---------------------------------------------------------------------- 

def shapley_graph_matching(ba_conn_comp_dict, qg_conn_comp_dict, 
                           edge_sim_conf_dict, weight_list):
  
  ba_key_pref_dict = {}
  qg_key_pref_dict = {}
  
  max_pref_list_size = 0
  
  weight_array = numpy.array(weight_list)
  
  # Getting preference lists for both plain-text and encode
  # nodes
  for ba_key, qg_edge_dict in ba_conn_comp_dict.iteritems():
    ba_key_pref_list = sorted([(qg_key,sum(numpy.array(val_list)*weight_array)) 
                                for qg_key,val_list in qg_edge_dict.items()], 
                                key=lambda t: sum(numpy.array(t[1])*weight_array), 
                                reverse=True)
    
    ba_key_pref_list = [qg_item[0] for qg_item in ba_key_pref_list]
    ba_key_pref_dict[ba_key] = ba_key_pref_list
    
    if(len(ba_key_pref_list) > max_pref_list_size):
      max_pref_list_size = len(ba_key_pref_list)
  
  for qg_key, ba_edge_dict in qg_conn_comp_dict.iteritems():
    qg_key_pref_list = sorted([(ba_key,sum(numpy.array(val_list)*weight_array)) 
                                for ba_key,val_list in ba_edge_dict.items()], 
                                key=lambda t: sum(numpy.array(t[1])*weight_array), 
                                reverse=True)
    
    qg_key_pref_list = [ba_item[0] for ba_item in qg_key_pref_list]
    qg_key_pref_dict[qg_key] = qg_key_pref_list
    
  
  iter_num = 0

  rej_ba_keys_set = set(ba_key_pref_dict.keys())
  new_rej_ba_keys_set = rej_ba_keys_set
  
  qg_assigned_ba_key_dict = {}
  
  while((iter_num == 0) or (len(rej_ba_keys_set) > 0 and rej_ba_keys_set != new_rej_ba_keys_set)):
    
    rej_ba_keys_set = new_rej_ba_keys_set
    new_rej_ba_keys_set = set()
    
    for ba_key in rej_ba_keys_set:
      ba_pref_list = ba_key_pref_dict[ba_key]
      
      if(iter_num < len(ba_pref_list)):
      
        preferred_qg_key = ba_pref_list[iter_num]
        
        if(preferred_qg_key in qg_assigned_ba_key_dict):
          assigned_ba_key = qg_assigned_ba_key_dict[preferred_qg_key]
          
          preferred_ba_key_list = qg_key_pref_dict[preferred_qg_key]
          
          old_ba_key_ind = preferred_ba_key_list.index(assigned_ba_key)
          new_ba_key_ind = preferred_ba_key_list.index(ba_key)
          
          if(new_ba_key_ind < old_ba_key_ind):
            qg_assigned_ba_key_dict[preferred_qg_key] = ba_key
            new_rej_ba_keys_set.add(assigned_ba_key)
          else:
            new_rej_ba_keys_set.add(ba_key) 
        else:
          qg_assigned_ba_key_dict[preferred_qg_key] = ba_key
      
      else:
        new_rej_ba_keys_set.add(ba_key)
    iter_num += 1
    
  
  gale_shap_sim_pair_dict = {}
  
  for qg_key, ba_key in qg_assigned_ba_key_dict.iteritems():
    sim_val_list = edge_sim_conf_dict[(qg_key, ba_key)]
    weighted_sim_sum = sum(numpy.array(sim_val_list)*weight_array)
    
    gale_shap_sim_pair_dict[(qg_key, ba_key)] = weighted_sim_sum
    
  return gale_shap_sim_pair_dict

#----------------------------------------------------------------------

def conf_graph_matching(connect_ba_sim_dict, connect_qg_sim_dict, sim_pair_list, calc_type=None):
  
  conf_sim_pair_dict = {}
              
  removed_edges_set = set()
  
  ba_sim_dict_copy = connect_ba_sim_dict.copy()
  qg_sim_dict_copy = connect_qg_sim_dict.copy()
  
  for (val_pair, cos_sim) in sim_pair_list:
    
    if(val_pair not in removed_edges_set):
    
      qg_key, ba_key = val_pair
      
      neighbour_qg_dict = ba_sim_dict_copy[ba_key]
      neighbour_ba_dict = qg_sim_dict_copy[qg_key]
      
      same_sim_edge_set = set()
      
      for neigh_qg in neighbour_qg_dict.keys():
        if(neigh_qg != qg_key):
          neigh_qg_sim = neighbour_qg_dict[neigh_qg]
          
          if(neigh_qg_sim == cos_sim):
            same_sim_edge_set.add((neigh_qg, ba_key))
      
      for neigh_ba in neighbour_ba_dict.keys():
        if(neigh_ba != ba_key):
          neigh_ba_sim = neighbour_ba_dict[neigh_ba]
          
          if(neigh_ba_sim == cos_sim):
            same_sim_edge_set.add((qg_key, neigh_ba))
            
      if(len(same_sim_edge_set) > 0):
        
        if(calc_type == 'degree'):
          max_conf_val = calc_degree_conf_val(neighbour_qg_dict,
                                              neighbour_ba_dict)
        else:
          max_conf_val = calc_sim_conf_val(ba_key, qg_key, 
                                       cos_sim, neighbour_qg_dict,
                                       neighbour_ba_dict)
        max_conf_pair = val_pair
        
        for (same_sim_qg_key, same_sim_ba_key) in same_sim_edge_set:
          other_ba_dict = qg_sim_dict_copy[same_sim_qg_key]
          other_qg_dict = ba_sim_dict_copy[same_sim_ba_key]
          if(calc_type == 'degree'):
            conf_val = calc_degree_conf_val(neighbour_qg_dict,
                                            neighbour_ba_dict)
          else:
            conf_val = calc_sim_conf_val(same_sim_ba_key, same_sim_qg_key, 
                                         cos_sim, other_qg_dict,
                                         other_ba_dict)
          
          if(conf_val > max_conf_val):
            max_conf_val = conf_val
            max_conf_pair = (same_sim_qg_key, same_sim_ba_key)
        
        # Now update dictionaries
        #
        select_qg_key, select_ba_key = max_conf_pair
        r_edge_set, new_qg_dict, new_ba_dict = \
                  graph_remove_edges(select_qg_key, select_ba_key, 
                                     qg_sim_dict_copy, 
                                     ba_sim_dict_copy)
        removed_edges_set = removed_edges_set.union(r_edge_set)
        ba_sim_dict_copy = new_ba_dict
        qg_sim_dict_copy = new_qg_dict
        
        conf_sim_pair_dict[(select_qg_key, select_ba_key)] = cos_sim
        
        
      else:
        r_edge_set, new_qg_dict, new_ba_dict = \
                  graph_remove_edges(qg_key, ba_key, 
                                     qg_sim_dict_copy, 
                                     ba_sim_dict_copy)
        removed_edges_set = removed_edges_set.union(r_edge_set)
        ba_sim_dict_copy = new_ba_dict
        qg_sim_dict_copy = new_qg_dict
        
        conf_sim_pair_dict[val_pair] = cos_sim
  
  return conf_sim_pair_dict

#----------------------------------------------------------------------

def munkress_graph_matching(conn_comp_list, edge_sim_conf_norm_dict, weight_list):
  
  # Loop over each connected component and apply Munkress
  # algorithm. Find best pairs of nodes using the algorithm
  #
  weight_array = numpy.array(weight_list)
  
  munkress_sim_pair_dict = {}
  
  num_not_pos_assign = 0
  num_total_assign   = 0
  
  proc_num_comp = 1 
  
  for qg_keys_set, ba_keys_set in conn_comp_list:
    
    if(proc_num_comp % 500 == 0):
      print '   Processed %d number of connected components' %proc_num_comp
    
    qg_keys_list = list(qg_keys_set)
    ba_keys_list = list(ba_keys_set)
    
    #max_count = max(len(qg_keys_list), len(ba_keys_list))
    
    if(len(qg_keys_list) == 1 and len(ba_keys_list) == 1):
      sim_conf_val_list = edge_sim_conf_norm_dict[(qg_keys_list[0], ba_keys_list[0])]
      weighted_sim_sum = sum(numpy.array(sim_conf_val_list)*weight_array)
      munkress_sim_pair_dict[(qg_keys_list[0], ba_keys_list[0])] = weighted_sim_sum
      continue
    
    if(len(qg_keys_list) == 1 or len(ba_keys_list) == 1):
      max_val = 0.0
      for qg_key in qg_keys_list:
        for ba_key in ba_keys_list:
          sim_conf_val_list = edge_sim_conf_norm_dict[(qg_key, ba_key)]
          weighted_sim_sum = sum(numpy.array(sim_conf_val_list)*weight_array)
          
          if(weighted_sim_sum > max_val):
            max_pair = (qg_key, ba_key)
            max_val = weighted_sim_sum
      
      munkress_sim_pair_dict[max_pair] = max_val
      continue
    
    pairs_dist_list = []
    
    for qg_key in qg_keys_list:
      dist_row = []
      for ba_key in ba_keys_list:
        if((qg_key, ba_key) in edge_sim_conf_norm_dict):
          sim_conf_val_list = edge_sim_conf_norm_dict[(qg_key, ba_key)]
          weighted_sim_sum = sum(numpy.array(sim_conf_val_list)*weight_array)
          pair_dist = 1.0 - weighted_sim_sum
        else:
          pair_dist = 1.0
        dist_row.append(pair_dist)
      
      pairs_dist_list.append(dist_row)
    
    #print len(pairs_dist_list)
    #print len(pairs_dist_list[0])
    pairs_dist_array = numpy.array(pairs_dist_list)
    
    # Applying Munkress algorithm
    qg_ind_list, ba_ind_list = linear_sum_assignment(pairs_dist_array)
    
    for qg_ind, ba_ind in zip(qg_ind_list,ba_ind_list):
      
      sel_qg_key = qg_keys_list[qg_ind]
      sel_ba_key = ba_keys_list[ba_ind]
      
      num_total_assign += 1
      
      if((sel_qg_key, sel_ba_key) not in edge_sim_conf_norm_dict):
        num_not_pos_assign += 1
        munkress_sim_pair_dict[(sel_qg_key, sel_ba_key)] = 0.1
      else:
        sim_conf_val_list = edge_sim_conf_norm_dict[(sel_qg_key, sel_ba_key)]
        weighted_sim_sum = sum(numpy.array(sim_conf_val_list)*weight_array)
        munkress_sim_pair_dict[(sel_qg_key, sel_ba_key)] = weighted_sim_sum
        
    proc_num_comp += 1
  
  return munkress_sim_pair_dict

#----------------------------------------------------------------------

def calc_sim_conf_val(encode_key, plain_key, cos_sim, enc_key_plain_dict, plain_key_enc_dict):
  
  conf_val = 0.0
  
  sim_val_list = []
  
  for other_plain_key, other_cos_sim in enc_key_plain_dict.iteritems():
    if(other_plain_key != plain_key):
      sim_val_list.append(other_cos_sim)
      
  for other_encode_key, other_cos_sim in plain_key_enc_dict.iteritems():
    if(other_encode_key != encode_key):
      sim_val_list.append(other_cos_sim)
  
  if(len(sim_val_list) > 0):
    conf_val = cos_sim / numpy.mean(sim_val_list)
  else:
    conf_val = 10.0
  
  return conf_val

#----------------------------------------------------------------------

def calc_degree_conf_val(other_qg_key_dict, other_ba_key_dict):
  
  other_qg_num_edges = len(other_ba_key_dict) - 1
  other_ba_num_edges = len(other_qg_key_dict) - 1
  
  conf_val = 1.0 /(other_qg_num_edges + other_ba_num_edges + 1.0)
  #conf_val = math.log(conf_val, 10)
  
  return conf_val

#----------------------------------------------------------------------

def get_bipart_connected_components_old(ba_graph):

  conn_comp_list = []
  conn_comp_list_edges = []
  
  for ba_key in ba_graph:
    qg_neighbour_set = ba_graph[ba_key]
    
    if(len(conn_comp_list) > 0):
      new_comp_list = []
      added_to_existing = False
      
      for ba_key_set, qg_key_set in conn_comp_list:
        qg_intersec = qg_key_set.intersection(qg_neighbour_set)
        
        if(len(qg_intersec) > 0):
          
          assert added_to_existing == False, '***Warining! cannot happen'        
  
          new_qg_key_set = qg_key_set.union(qg_neighbour_set)
          new_ba_key_set = ba_key_set.union(set([ba_key]))
          new_comp_list.append((new_ba_key_set, new_qg_key_set))
          added_to_existing = True
        else:
          new_comp_list.append((ba_key_set, qg_key_set))
      
      if(added_to_existing == False):
        new_comp_list.append((set([ba_key]), qg_neighbour_set))
      
      conn_comp_list = new_comp_list
                              
    else:
      ba_key_set1 = set([ba_key])
      qg_key_set1 = qg_neighbour_set
      conn_comp_list = [(ba_key_set1, qg_key_set1)]
  
  for ba_key_set, qg_key_set in conn_comp_list:
    edge_list = [] 
    
    for ba_key in ba_key_set:
      qg_keys = ba_graph[ba_key]
      
      for edge in itertools.product([ba_key], list(qg_keys)):
        edge_list.append(edge)
      
    conn_comp_list_edges.append(edge_list)
  
  return conn_comp_list, conn_comp_list_edges

#----------------------------------------------------------------------

def get_bipart_connected_components_recursive(ba_graph, qg_graph):
  
  traversed_qg_keys = set()
  traversed_ba_keys = set()
  
  conn_comp_list = []
  
  def get_ba_component(next_ba_key):
    next_qg_keys = ba_graph[next_ba_key]
    full_ba_re_set = set()
    full_qg_re_set = set()
    for next_qg_key in next_qg_keys:
      if(next_qg_key not in traversed_qg_keys):
        #print next_qg_key
        traversed_qg_keys.add(next_qg_key)
        full_qg_re_set.add(next_qg_key)
        return_generator = next(get_qg_component(next_qg_key))
        qg_re_set = return_generator[0]
        ba_re_set = return_generator[1]
        
        full_ba_re_set = full_ba_re_set.union(ba_re_set)
        full_qg_re_set = full_qg_re_set.union(qg_re_set)
    
    yield (full_qg_re_set, full_ba_re_set)
    
  
  def get_qg_component(next_qg_key):
    next_ba_keys = qg_graph[next_qg_key]
    full_ba_re_set = set()
    full_qg_re_set = set()
    for next_ba_key in next_ba_keys:
      if(next_ba_key not in traversed_ba_keys):
        #print next_ba_key
        traversed_ba_keys.add(next_ba_key)
        full_ba_re_set.add(next_ba_key)
        return_generator = next(get_ba_component(next_ba_key))
        qg_re_set = return_generator[0]
        ba_re_set = return_generator[1]
        
        full_ba_re_set = full_ba_re_set.union(ba_re_set)
        full_qg_re_set = full_qg_re_set.union(qg_re_set)
    
    yield (full_qg_re_set, full_ba_re_set)
        
  
  for qg_key in qg_graph:
    if(qg_key not in traversed_qg_keys):
      #print qg_key
      traversed_qg_keys.add(qg_key)
      return_generator = next(get_qg_component(qg_key))
      qg_re_set = return_generator[0]
      ba_re_set = return_generator[1]
      
      qg_re_set.add(qg_key)
      
      conn_comp_list.append((qg_re_set, ba_re_set))
  
  return conn_comp_list

#----------------------------------------------------------------------

def get_bipart_connected_components(ba_conn_comp_dict, qg_conn_comp_dict, min_sim_filter,
                                    weight_list):
  
  ba_graph = {} 
  qg_graph = {}
  
  weight_array = numpy.array(weight_list)
  
  for ba_key, qg_sim_dict in ba_conn_comp_dict.iteritems():
    qg_key_set = set([qg_key for qg_key,val_list in qg_sim_dict.items() 
                      if(sum(numpy.array(val_list)*weight_array)) >= min_sim_filter])
    
    ba_graph[ba_key] = qg_key_set
  
  for qg_key, ba_sim_dict in qg_conn_comp_dict.iteritems():
    ba_key_set = set([ba_key for ba_key,val_list in ba_sim_dict.items() 
                      if(sum(numpy.array(val_list)*weight_array)) >= min_sim_filter])
    
    qg_graph[qg_key] = ba_key_set
  
  traversed_qg_keys = set()
  
  conn_comp_list = []
  
  def get_qg_keys(next_ba_keys):
    next_qg_key_set = set()
    for ba_key in next_ba_keys:
      next_qg_key_set = next_qg_key_set.union(ba_graph[ba_key])
    
    return next_qg_key_set
      
  def get_ba_keys(next_qg_keys):
    next_ba_key_set = set()
    for qg_key in next_qg_keys:
      next_ba_key_set = next_ba_key_set.union(qg_graph[qg_key])
    
    return next_ba_key_set
  
  for qg_key in qg_graph:
    
    if(qg_key not in traversed_qg_keys):
      
      full_ba_re_set = set()
      full_qg_re_set = set()
      
      ba_keys = qg_graph[qg_key]
      
      new_qg_set = get_qg_keys(ba_keys)
      new_ba_set = get_ba_keys(new_qg_set)
      
      while(full_ba_re_set != new_ba_set or full_qg_re_set != new_qg_set):
        
        full_ba_re_set = new_ba_set
        full_qg_re_set = new_qg_set
        
        new_qg_set1 = get_qg_keys(new_ba_set)
        new_ba_set1 = get_ba_keys(new_qg_set1)
        
        new_qg_set = new_qg_set1
        new_ba_set = new_ba_set1
      
      traversed_qg_keys = traversed_qg_keys.union(full_qg_re_set)
      if(len(full_qg_re_set) > 0 and len(full_ba_re_set) > 0):
        conn_comp_list.append((full_qg_re_set, full_ba_re_set))
    
  return conn_comp_list

#---------------------------------------------------------------

def get_conn_comp_user_input(ba_conn_comp_dict, qg_conn_comp_dict,
                             weight_list):
  
  max_comp_size = 0
  
  print ' Give the initial minimum similarity threshold (float):',
  min_sim_filter = float(raw_input())
  print
  
  calc_connect_comp = True
  
  while(calc_connect_comp):
    conn_comp_list = get_bipart_connected_components(ba_conn_comp_dict, 
                                                     qg_conn_comp_dict,
                                                     min_sim_filter,
                                                     weight_list)
    print '   Found %d number of connected components' %len(conn_comp_list)
    
    for k1, k2 in conn_comp_list:
      if(max_comp_size < len(k1)):
        max_comp_size = len(k1)
      if(max_comp_size < len(k2)):
        max_comp_size = len(k2)
                       
    print '     Largest connected component for sim threhold %.5f: %d' %(min_sim_filter,
                                                                         max_comp_size)
    
    
    print
    print ' Do you want to change the the similarity threshold (y/n):',
    change_flag = raw_input()
    
    while(change_flag != 'y' and change_flag != 'n'):
      print ' Wrong input. Please enter again:', 
      change_flag = raw_input()
    
    if(change_flag == 'y'):
      max_comp_size = 0
      print ' Give the next minimum similarity threshold (float):',
      min_sim_filter = float(raw_input())
      
    elif(change_flag == 'n'):
      print ' ......Terminating the loop.....'
      calc_connect_comp = False
  
  return conn_comp_list

#---------------------------------------------------------------

def get_conn_comp_intep_search(ba_conn_comp_dict, qg_conn_comp_dict,
                               weight_list, cc_size_range_l, 
                               cc_size_range_h):
  
  min_sim_threshold = 0.0
  max_sim_threshold = 1.0
  
  min_sim_conn_comp_list = get_bipart_connected_components(ba_conn_comp_dict, 
                                                           qg_conn_comp_dict,
                                                           min_sim_threshold,
                                                           weight_list)
  max_comp_size1 = 0
  
  for k1, k2 in min_sim_conn_comp_list:
    if(max_comp_size1 < len(k1)):
      max_comp_size1 = len(k1)
    if(max_comp_size1 < len(k2)):
      max_comp_size1 = len(k2)
      
  if(len(min_sim_conn_comp_list)):
    comp_size_list = [(len(k1), len(k2)) for k1, k2 in min_sim_conn_comp_list]
    c1, c2 = zip(*comp_size_list)
    min_sim_comp_size = max(max(c1), max(c2))
  else:
    min_sim_comp_size = 0
  
  assert min_sim_comp_size == max_comp_size1, (min_sim_comp_size, max_comp_size1)
  
  print '  Largest connected component for minimum sim threhold %.5f: %d' %(min_sim_threshold,
                                                                            min_sim_comp_size)
  
      
  if(min_sim_comp_size > cc_size_range_h): # Continue with the process
    
    min_comp_size1 = 0
    
    max_sim_conn_comp_list = get_bipart_connected_components(ba_conn_comp_dict, 
                                                             qg_conn_comp_dict,
                                                             max_sim_threshold,
                                                             weight_list)
    
    for k1, k2 in max_sim_conn_comp_list:
      if(min_comp_size1 < len(k1)):
        min_comp_size1 = len(k1)
      if(min_comp_size1 < len(k2)):
        min_comp_size1 = len(k2)
        
    if(len(max_sim_conn_comp_list) > 0):
      comp_size_list = [(len(k1), len(k2)) for k1, k2 in max_sim_conn_comp_list]
      c1, c2 = zip(*comp_size_list)
      max_sim_comp_size = max(max(c1), max(c2))
    else:
      max_sim_comp_size = 0
    
    assert max_sim_comp_size == min_comp_size1
    
    print '  Largest connected component for maximum sim threhold %.5f: %d' %(max_sim_threshold,
                                                                              max_sim_comp_size)
    print
    
    new_max_comp_size = min_sim_comp_size
    prev_sim_threshold = 2.0
    
    while (new_max_comp_size > cc_size_range_h or new_max_comp_size < cc_size_range_l):
      
      new_threshold = min_sim_threshold + (float(cc_size_range_h - min_sim_comp_size)/\
                      (max_sim_comp_size - min_sim_comp_size))*(max_sim_threshold - min_sim_threshold)
                          
      conn_comp_list = get_bipart_connected_components(ba_conn_comp_dict, 
                                                       qg_conn_comp_dict,
                                                       new_threshold,
                                                       weight_list)
      
      if(len(conn_comp_list)> 0):
        comp_size_list = [(len(k1), len(k2)) for k1, k2 in conn_comp_list]
        c1, c2 = zip(*comp_size_list)
        new_max_comp_size = max(max(c1), max(c2))
      else:
        new_max_comp_size = 0
      
      print '    New threshold:              ', new_threshold
      print '    Largest connected component:', new_max_comp_size
      print
        
      if(new_threshold == prev_sim_threshold):
        if(new_max_comp_size <= cc_size_range_h):
          break
        else:
          min_sim_threshold = new_threshold + 0.01
          min_sim_comp_size = new_max_comp_size
          continue 
      
      if(new_max_comp_size > cc_size_range_h):
        min_sim_threshold = new_threshold
        min_sim_comp_size = new_max_comp_size
        
      elif(new_max_comp_size < cc_size_range_l):
        max_sim_threshold = new_threshold
        max_sim_comp_size = new_max_comp_size
        
      prev_sim_threshold = new_threshold
    
    print '  Found an optimal threhold of %.4f with largest connected component size %d' %(new_threshold, new_max_comp_size)
  
  else:
    print '  ***Largest connected component is below the expected range...'
    conn_comp_list = min_sim_conn_comp_list
      
  
  return conn_comp_list

#---------------------------------------------------------------

def graph_remove_edges(qg_key, ba_key, qg_sim_dict, ba_sim_dict):
  
  removed_ba_keys = set() 
  removed_qg_keys = set()
  
  removed_edges_set = set()
  
  other_ba_dict = qg_sim_dict[qg_key]
  other_qg_dict = ba_sim_dict[ba_key]
  
  for other_ba_key in other_ba_dict:
    if(other_ba_key != ba_key):
      removed_ba_keys.add(other_ba_key)
      
  for other_qg_key in other_qg_dict:
    if(other_qg_key != qg_key):
      removed_qg_keys.add(other_qg_key)
      
  
  for remove_ba_key in removed_ba_keys:
    del other_ba_dict[remove_ba_key]
    del ba_sim_dict[remove_ba_key][qg_key]
    
    removed_edges_set.add((qg_key,remove_ba_key))
    
  for remove_qg_key in removed_qg_keys:
    del other_qg_dict[remove_qg_key]
    del qg_sim_dict[remove_qg_key][ba_key]
    
    removed_edges_set.add((remove_qg_key,ba_key))
  
  ba_sim_dict[ba_key] = other_qg_dict
  qg_sim_dict[qg_key] = other_ba_dict
  
  return removed_edges_set, qg_sim_dict, ba_sim_dict

#---------------------------------------------------------------------------

def learn_embeddings(walks, num_dimensions, window_size, num_workers, num_iter):
  '''
  Learn embeddings by optimizing the Skipgram objective using SGD.
  '''
  
  start_time = time.time()
  
  walks = [map(str, walk) for walk in walks]
  model = Word2Vec(walks, size=num_dimensions, window=window_size, min_count=0, sg=1, workers=num_workers, iter=num_iter)
  
  learn_time = time.time() - start_time
  print '    Learning node embeddings took %.3f seconds' %learn_time
  print
  
  return model

def check_sim_class(sim_val):
  
  if(sim_val <= 1.0 and sim_val > 0.9):
    sim_class = '9-10'
  elif(sim_val <= 0.9 and sim_val > 0.8):
    sim_class = '8-9'
  elif(sim_val <= 0.8 and sim_val > 0.7):
    sim_class = '7-8'
  elif(sim_val <= 0.7 and sim_val > 0.6):
    sim_class = '6-7'
  elif(sim_val <= 0.6 and sim_val > 0.5):
    sim_class = '5-6'
  elif(sim_val <= 0.5 and sim_val > 0.4):
    sim_class = '4-5'
  elif(sim_val <= 0.4 and sim_val > 0.3):
    sim_class = '3-4'
  elif(sim_val <= 0.3 and sim_val > 0.2):
    sim_class = '2-3'
  elif(sim_val <= 0.2 and sim_val > 0.1):
    sim_class = '1-2'
  elif(sim_val <= 0.1 and sim_val > 0.0):
    sim_class = '0-1'
  else:
    sim_class = 'None'
    
  return sim_class


# =============================================================================
# Main program

# Parse command line parameters
#
encode_data_set_name =    sys.argv[1]
encode_ent_id_col =       int(sys.argv[2])
encode_col_sep_char =     sys.argv[3]
encode_header_line_flag = eval(sys.argv[4])
encode_attr_list =        eval(sys.argv[5])
encode_num_rec =          int(sys.argv[6])
#
plain_data_set_name =    sys.argv[7]
plain_ent_id_col =       int(sys.argv[8])
plain_col_sep_char =     sys.argv[9]
plain_header_line_flag = eval(sys.argv[10])
plain_attr_list =        eval(sys.argv[11])
plain_num_rec =          int(sys.argv[12])
#
q =           int(sys.argv[13])
padded_flag = eval(sys.argv[14])
#
plain_sim_funct_name =  sys.argv[15].lower()

sim_diff_adjust_flag =  eval(sys.argv[16])

encode_sim_funct_name = sys.argv[-1].lower()  # The last command line parameter
#
encode_method = sys.argv[17].lower()
assert encode_method in ['bf','tmh','2sh'], encode_method
#
if (encode_method == 'bf'):
  bf_hash_type =      sys.argv[18].lower()
  bf_num_hash_funct = sys.argv[19]
  bf_len =            int(sys.argv[20])
  bf_encode =         sys.argv[21].lower()
  bf_harden =         sys.argv[22].lower()
  bf_enc_param =      eval(sys.argv[23])
elif (encode_method == 'tmh'):
  tmh_num_hash_bits = int(sys.argv[18])
  tmh_hash_funct =    sys.argv[19].lower()
  tmh_num_tables =    int(sys.argv[20])
  tmh_key_len =       int(sys.argv[21])
  tmh_val_len =       int(sys.argv[22])
elif (encode_method == '2sh'):
  cmh_num_hash_funct = int(sys.argv[18])
  cmh_num_hash_col =   int(sys.argv[19])
  cmh_max_rand_int =   int(sys.argv[20])

# Check validity of parameters
#
assert encode_ent_id_col >= 0, encode_ent_id_col
assert encode_header_line_flag in [True, False], encode_header_line_flag
assert isinstance(encode_attr_list, list), encode_attr_list
assert (encode_num_rec == -1) or (encode_num_rec >= 1), encode_num_rec
#
assert plain_ent_id_col >= 0, plain_ent_id_col
assert plain_header_line_flag in [True, False], plain_header_line_flag
assert isinstance(plain_attr_list, list), plain_attr_list
assert (plain_num_rec == -1) or (plain_num_rec >= 1), plain_num_rec
#
assert q >= 1, q
assert padded_flag in [True,False], padded_flag
#
assert plain_sim_funct_name in ['dice', 'jacc'], plain_sim_funct_name
assert encode_sim_funct_name in ['dice', 'jacc', 'hamm'], encode_sim_funct_name
#
sim_diff_adjust_flag in [True, False], sim_diff_adjust_flag
#
if (encode_method == 'bf'):
  assert bf_hash_type in ['dh', 'rh'], bf_hash_type
  if bf_num_hash_funct.isdigit():
    bf_num_hash_funct = int(bf_num_hash_funct)
    assert bf_num_hash_funct >= 1, bf_num_hash_funct
  else:
    assert bf_num_hash_funct == 'opt', bf_num_hash_funct
  assert bf_len > 1, bf_len
  assert bf_encode in ['abf','clk', 'rbf-s', 'rbf-d', 'clkrbf'], bf_encode
  
  assert bf_harden in ['none', 'balance', 'fold'], bf_harden
  if (bf_harden == 'fold'):
    if (bf_len%2 != 0):
      raise Exception, 'BF hardening approach "fold" needs an even BF length'

elif (encode_method == 'tmh'):
  assert tmh_num_hash_bits > 1, tmh_num_hash_bits
  assert tmh_hash_funct in ['md5', 'sha1', 'sha2'], tmh_hash_funct
  assert tmh_num_tables > 1, tmh_num_tables
  assert tmh_key_len > 1, tmh_key_len
  assert tmh_val_len > 1,  tmh_val_len
  
elif(encode_method == '2sh'):
  assert cmh_num_hash_funct > 1, cmh_num_hash_funct
  assert cmh_num_hash_col > 1, cmh_num_hash_col
  assert cmh_max_rand_int > 1, cmh_max_rand_int

if (len(encode_col_sep_char) > 1):
  if (encode_col_sep_char == 'tab'):
    encode_col_sep_char = '\t'
  elif (encode_col_sep_char[0] == '"') and (encode_col_sep_char[-1] == '"') \
       and (len(encode_col_sep_char) == 3):
    encode_col_sep_char = encode_col_sep_char[1]
  else:
    print 'Illegal encode data set column separator format:', \
          encode_col_sep_char

if (len(plain_col_sep_char) > 1):
  if (plain_col_sep_char == 'tab'):
    plain_col_sep_char = '\t'
  elif (plain_col_sep_char[0] == '"') and \
     (plain_col_sep_char[-1] == '"') and \
     (len(plain_col_sep_char) == 3):
    plain_col_sep_char = plain_col_sep_char[1]
  else:
    print 'Illegal plain text data set column separator format:', \
          plain_col_sep_char

# Check if same data sets and same attributes were given
#
if ((encode_data_set_name == plain_data_set_name) and \
    (encode_attr_list == plain_attr_list)):
  same_data_attr_flag = True
else:
  same_data_attr_flag = False

# Get base names of data sets (remove directory names) for summary output
#
encode_base_data_set_name = encode_data_set_name.split('/')[-1]
encode_base_data_set_name = encode_base_data_set_name.replace('.csv', '')
encode_base_data_set_name = encode_base_data_set_name.replace('.gz', '')
assert ',' not in encode_base_data_set_name

plain_base_data_set_name = plain_data_set_name.split('/')[-1]
plain_base_data_set_name = plain_base_data_set_name.replace('.csv', '')
plain_base_data_set_name = plain_base_data_set_name.replace('.gz', '')
assert ',' not in plain_base_data_set_name

res_file_name = 'pprl-graph-attack-results-%s-%s-%s.csv' % \
                (encode_base_data_set_name, plain_base_data_set_name, \
                 today_str)
print
print 'Write results into file:', res_file_name
print
print '-'*80
print

# -----------------------------------------------------------------------------
# Step 1: Load the data sets and extract q-grams for selected attributes
#
start_time = time.time()

encode_rec_attr_val_dict, encode_attr_name_list, encode_num_rec_loaded, encode_soundex_val_dict = \
               load_data_set(encode_data_set_name, encode_attr_list,
                             encode_ent_id_col, soundex_attr_list, encode_num_rec,
                             encode_col_sep_char, encode_header_line_flag)

if (same_data_attr_flag == False):  # The two data sets are different

  plain_rec_attr_val_dict, plain_attr_name_list, plain_num_rec_loaded, plain_soundex_val_dict = \
               load_data_set(plain_data_set_name, plain_attr_list,
                             plain_ent_id_col, soundex_attr_list, plain_num_rec,
                             plain_col_sep_char, plain_header_line_flag)

  if (encode_attr_name_list != plain_attr_name_list):
    print
    print '*** Warning: Different attributes used in encoded and ' + \
          'plain-text files:'
    print '***   Encoded file:   ', encode_attr_name_list
    print '***   Plain-text file:', plain_attr_name_list

else:  # Set to same as encode
  plain_rec_attr_val_dict = encode_rec_attr_val_dict
  plain_attr_name_list =    encode_attr_name_list
  plain_num_rec_loaded =    encode_num_rec_loaded

# Generate q-gram sets for records
#
encode_q_gram_dict = gen_q_gram_sets(encode_rec_attr_val_dict, q, padded_flag)
plain_q_gram_dict =  gen_q_gram_sets(plain_rec_attr_val_dict, q, padded_flag)

load_q_gram_time = time.time() - start_time

print
print 'Time for loading data sets and generating q-grams sets: %.2f sec' % \
      (load_q_gram_time)
print

enc_id_set = set(encode_q_gram_dict.keys())
plain_id_set = set(plain_q_gram_dict.keys())

common_rec_id_set = enc_id_set.intersection(plain_id_set)

print 'The two datasets have %d overlapping records (%.1f %%)' % \
      (len(common_rec_id_set), 200*float(len(common_rec_id_set))/(len(enc_id_set)+len(plain_id_set)))
print

# -----------------------------------------------------------------------------
# Step 2: If both graphs are available as pickle files then load and use them,
#         otherwise encode the first data set according to the encoding
#         settings
#
QG_sim_graph = SimGraph()  # Initialise the two graphs
BA_sim_graph = SimGraph()

if(sim_diff_adjust_flag == True):
  # Define the regression model
  if(encode_method == '2sh'):
    regre_model_str = 'poly' # linear, poly, isotonic
  else:
    regre_model_str = 'linear' # linear, poly, isotonic
else:
  regre_model_str = 'none'

# Create the graph pickle file names, and if they both are available then
# load the graphs from files
#
plain_attr_list_str =  str(plain_attr_list).replace(', ','_')[1:-1]
encode_attr_list_str = str(encode_attr_list).replace(', ','_')[1:-1]

if (encode_method == 'bf'):
  encode_method_str = 'bf-%s-%s-%d-%s-%s' % (bf_hash_type, bf_num_hash_funct,
                      bf_len, bf_encode, bf_harden)
elif (encode_method == 'tmh'):
  encode_method_str = 'tmh-%d-%s-%d-%d-%d' % (tmh_num_hash_bits,
                      tmh_hash_funct, tmh_num_tables, tmh_key_len, tmh_val_len)
elif (encode_method == '2sh'):
  encode_method_str = '2sh-%d-%d-%d' % (cmh_max_rand_int,
                                        cmh_num_hash_funct, cmh_num_hash_col)
  

# Define encode and plaintext blocking methods
#
encode_blck_method = 'minhash' # hmlsh, minhash, soundex, none
plain_blck_method  = 'minhash' # minhash, soundex, none

# The generated graph file names
#          
encode_str = 'encode-sim-graph-%s-%s-%d-%d-%s-%s-%s-%s-%.3f-%s' \
             % (encode_base_data_set_name, encode_attr_list_str,
                encode_num_rec_loaded, q, str(padded_flag).lower(),
                regre_model_str.lower(),
                encode_sim_funct_name, encode_blck_method, min_sim,
                encode_method_str)
             
plain_str = 'plain-sim-graph-%s-%s-%d-%d-%s-%s-%s-%s-%.3f-%s' % \
             (plain_base_data_set_name, plain_attr_list_str,
              plain_num_rec_loaded, q, str(padded_flag).lower(),
              regre_model_str.lower(),
              plain_sim_funct_name, plain_blck_method, min_sim,
              encode_method_str)

encode_graph_file_name = encode_str + '.pickle'
plain_graph_file_name = plain_str + '.pickle'

graph_path = 'graphs/'

encode_graph_file_name = graph_path + encode_graph_file_name
plain_graph_file_name  = graph_path + plain_graph_file_name

#parei aqui

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Two dictionaries that can be usedt to get similar nodes in both graphs
# since graph node values cannot be compared.
#
qg_graph_node_id_dict = {}
ba_graph_node_id_dict = {}

if (os.path.isfile(plain_graph_file_name) and \
    os.path.isfile(encode_graph_file_name)):
  print 'Load graphs from pickle files:'
  print '  Plain-text graph file:', plain_graph_file_name
  print '  Encoded graph file:   ', encode_graph_file_name
  print

  generated_graph_flag = False  # Graphs were not generated in this run

  start_time = time.time()

  QG_sim_graph.sim_graph = networkx.read_gpickle(plain_graph_file_name)
  BA_sim_graph.sim_graph = networkx.read_gpickle(encode_graph_file_name)

  print '  Time for loading the q-gram and bit-array similarity graphs: ' + \
        '%.2f secs' % (time.time() - start_time)
  print ' ', auxiliary.get_memory_usage()
  print
    
  for node_key_val in QG_sim_graph.sim_graph.nodes():
    qg_id_set = QG_sim_graph.sim_graph.node[node_key_val]['ent_id_set']
    
    for qg_id in qg_id_set:
      qg_graph_node_id_dict[qg_id] = node_key_val
      
  for node_key_val in BA_sim_graph.sim_graph.nodes():
    ba_id_set = BA_sim_graph.sim_graph.node[node_key_val]['ent_id_set']
    
    for ba_id in ba_id_set:
      ba_graph_node_id_dict[ba_id] = node_key_val

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

else:  # Encode the second data set and then generate the two graphs

  generated_graph_flag = True  # Graphs were generated in this run

  start_time = time.time()

  if (encode_method == 'bf'): # Bloom filter encoding

    if (bf_num_hash_funct == 'opt'):
      bf_num_hash_funct_str = 'opt'

      # Get the average number of q-grams in the string values
      #
      total_num_q_gram = 0
      total_num_val =    0

      for q_gram_set in encode_q_gram_dict.itervalues():
        total_num_q_gram += len(q_gram_set)
        total_num_val +=    1

      avrg_num_q_gram = float(total_num_q_gram) / total_num_val

      # Set number of hash functions to have in average 50% of bits set to 1
      #
      bf_num_hash_funct = int(round(math.log(2.0)*float(bf_len) / \
                              avrg_num_q_gram))

      print ' Set optimal number of BF hash functions to: %d' % \
            (bf_num_hash_funct)
      print

    else:
      bf_num_hash_funct_str = str(bf_num_hash_funct)

    # Set up the Bloom filter hashing
    #
    if (bf_hash_type == 'dh'):
      BF_hashing = hashing.DoubleHashing(BF_HASH_FUNCT1, BF_HASH_FUNCT2,
                                         bf_len, bf_num_hash_funct)
    elif (bf_hash_type == 'rh'):
      BF_hashing = hashing.RandomHashing(BF_HASH_FUNCT1, bf_len,
                                         bf_num_hash_funct)
    else:
      raise Exception, 'This should not happen'
    
    
    # Check the BF encoding method
    #
    if(bf_encode == 'clk'): # Cryptographic Long-term Key
      rec_tuple_list = []
      
      for att_num in range(len(encode_attr_list)):
        rec_tuple_list.append([att_num, q, padded_flag, BF_hashing])
    
      BF_Encoding = encoding.CryptoLongtermKeyBFEncoding(rec_tuple_list)
    
    elif(bf_encode.startswith('rbf')): # Record-level Bloom filter
      
      num_bits_list = bf_enc_param # List of percentages of number of bits
      
      rec_tuple_list = []
      
      for att_num in range(len(encode_attr_list)):
        rec_tuple_list.append([att_num, q, padded_flag, BF_hashing, 
                               int(num_bits_list[att_num]*bf_len)])
      
      BF_Encoding = encoding.RecordBFEncoding(rec_tuple_list)
      
      if(bf_encode == 'rbf-d'): # AFB length set to dynamic
        
        rec_val_list = encode_rec_attr_val_dict.values()
        avr_num_q_gram_dict = BF_Encoding.get_avr_num_q_grams(rec_val_list)
        abf_len_dict = BF_Encoding.get_dynamic_abf_len(avr_num_q_gram_dict, 
                                                      num_hash_funct)
        BF_Encoding.set_abf_len(abf_len_dict)
    

    encode_hash_dict = {}  # Generate one bit-array hash code per record

    # Keep the generated BF for each q-gram set so we only generate it once
    #
    bf_hash_cache_dict = {}

    num_enc = 0  # Count number of encodings, print progress report
    
    for (ent_id, q_gram_set) in encode_q_gram_dict.iteritems():
      
      attr_val_list = encode_rec_attr_val_dict[ent_id]
      
      num_enc += 1
      if (num_enc % 10000 == 0):
        time_used = time.time() - start_time
        print '  Encoded %d of %d q-gram sets in %d sec (%.2f msec average)' \
              % (num_enc, len(encode_q_gram_dict), time_used,
                 1000.0*time_used/num_enc)
        print '   ', auxiliary.get_memory_usage()
      
      q_gram_str = ''.join(sorted(q_gram_set))
      if (q_gram_str in bf_hash_cache_dict):
        q_gram_str_bf = bf_hash_cache_dict[q_gram_str]
      else:
        q_gram_str_bf = BF_Encoding.encode(attr_val_list)
        q_gram_str_bf_test = BF_hashing.hash_q_gram_set(q_gram_set)
        
        if(bf_encode == 'clk'):
          assert q_gram_str_bf == q_gram_str_bf_test
        
        bf_hash_cache_dict[q_gram_str] = q_gram_str_bf
      encode_hash_dict[ent_id] = q_gram_str_bf

    print
    print '  Encoded %d unique Bloom filters for %d q-gram sets' % \
          (len(bf_hash_cache_dict), len(encode_hash_dict))

    bf_hash_cache_dict.clear()  # Not needed anymore

  elif (encode_method == 'tmh'):  # Tabulation min-hash encoding

    if (tmh_hash_funct == 'md5'):
      tmh_hash_funct_obj = hashlib.md5
    elif (tmh_hash_funct == 'sha1'):
      tmh_hash_funct_obj = hashlib.sha1
    elif (tmh_hash_funct == 'sha2'):
      tmh_hash_funct_obj = hashlib.sha256
    else:
      raise Exception, 'This should not happen'

    TMH_hashing = tabminhash.TabMinHashEncoding(tmh_num_hash_bits,
                                                tmh_num_tables, tmh_key_len,
                                                tmh_val_len, tmh_hash_funct_obj)

    encode_hash_dict = {}  # Generate one bit-array hash code per record

    # Keep the generated BA for each q-gram set so we only generate it once
    #
    ba_hash_cache_dict = {}

    num_enc = 0  # Count number of encodings, print progress report

    for (ent_id, q_gram_set) in encode_q_gram_dict.iteritems():

      num_enc += 1
      if (num_enc % 10000 == 0):
        time_used = time.time() - start_time
        print '  Encoded %d of %d q-gram sets in %d sec (%.2f msec average)' \
              % (num_enc, len(encode_q_gram_dict), time_used,
                 1000.0*time_used/num_enc)
        print '   ', auxiliary.get_memory_usage()

      q_gram_str = ''.join(sorted(q_gram_set))
      if (q_gram_str in ba_hash_cache_dict):
        q_gram_str_ba = ba_hash_cache_dict[q_gram_str]
      else:
        q_gram_str_ba = TMH_hashing.encode_q_gram_set(q_gram_set)
        ba_hash_cache_dict[q_gram_str] = q_gram_str_ba
      encode_hash_dict[ent_id] = q_gram_str_ba

    print
    print '  Encoded %d unique bit-arrays for %d q-gram sets' % \
          (len(ba_hash_cache_dict), len(encode_hash_dict))

    ba_hash_cache_dict.clear()  # Not needed anymore

  elif(encode_method == '2sh'): # Two-step hash encoding             
    
    CMH_hashing = colminhash.ColMinHashEncoding(cmh_num_hash_funct,
                                                cmh_num_hash_col)
    
    encode_hash_dict = {}  # Generate one column hash code per record
    
    # Keep the generated column hash codes for each q-gram set so we only generate it once
    #
    col_hash_cache_dict = {}

    num_enc = 0  # Count number of encodings, print progress report
    
    for (ent_id, q_gram_set) in encode_q_gram_dict.iteritems():
      
      num_enc += 1
      if (num_enc % 10000 == 0):
        time_used = time.time() - start_time
        print '  Encoded %d of %d q-gram sets in %d sec (%.2f msec average)' \
              % (num_enc, len(encode_q_gram_dict), time_used,
                 1000.0*time_used/num_enc)
        print '   ', auxiliary.get_memory_usage()

      q_gram_str = ''.join(sorted(q_gram_set))
      if (q_gram_str in col_hash_cache_dict):
        q_gram_str_col_hash_set = col_hash_cache_dict[q_gram_str]
      else:
        q_gram_str_col_hash_set = CMH_hashing.encode_q_gram_set(q_gram_set)
        col_hash_cache_dict[q_gram_str] = q_gram_str_col_hash_set
      encode_hash_dict[ent_id] = q_gram_str_col_hash_set

    print
    print '  Encoded %d unique col hash sets for %d q-gram sets' % \
          (len(col_hash_cache_dict), len(encode_hash_dict))
    
    col_hash_cache_dict.clear()  # Not needed anymore  
  
  else:
    raise Exception, 'This should not happen'

  hashing_time = time.time() - start_time

  print
  print 'Time for hashing the encode data set: %.2f sec' % (hashing_time)
  print '  Number of records hashed:', len(encode_hash_dict)
  print

  # Check which entity identifiers occur in both data sets
  #
  common_ent_id_set = set(plain_rec_attr_val_dict.keys()) & \
                      set(encode_hash_dict.keys())
  common_num_ent = len(common_ent_id_set)

  plain_num_ent =  len(plain_rec_attr_val_dict)
  encode_num_ent = len(encode_hash_dict)

  print '  Number of entities in the two data sets (plain-text / encoded): ' \
        + '%d / %d' % (plain_num_ent, encode_num_ent)
  print '    Number of entities that occur in both data sets: %d' % \
        (common_num_ent)
  print

  # ---------------------------------------------------------------------------
  # Step 3: Generate the two graphs by calculating similarities between records
  #         (only needed if graphs have not been loaded from pickle files)

  # Initialise the actual similarity functions to be used
  #
  if (plain_sim_funct_name == 'dice'):
    plain_sim_funct = simcalc.q_gram_dice_sim
  elif (plain_sim_funct_name == 'jacc'):
    plain_sim_funct = simcalc.q_gram_jacc_sim
  else:
    raise Exception, 'This should not happen'

  if(encode_method == '2sh'):
    if (encode_sim_funct_name == 'dice'):
      encode_sim_funct = simcalc.q_gram_dice_sim
    elif (encode_sim_funct_name == 'jacc'):
      encode_sim_funct = simcalc.q_gram_jacc_sim
    else:
      raise Exception, 'This should not happen'
  else:
    if (encode_sim_funct_name == 'dice'):
      encode_sim_funct = simcalc.bit_array_dice_sim
    elif (encode_sim_funct_name == 'hamm'):
      encode_sim_funct = simcalc.bit_array_hamm_sim
    elif (encode_sim_funct_name == 'jacc'):
      encode_sim_funct = simcalc.bit_array_jacc_sim
    else:
      raise Exception, 'This should not happen' 

  # First add all records as nodes to the two graphs, and also generate two
  # dictionaries where keys are the values added as nodes into the graphs while
  # values are the original sets or encodings (that cannot be used as nodes)
  # This ensures we compare each unique pair for q-grams / bit-arrays only once
  #
  plain_node_val_dict =  {}
  encode_node_val_dict = {}

  # A dictionary where keys are node key values based on q-grams (i.e.
  # plain-text values) and values are their corresponding bit arrays (these
  # are the true matching q-gram sets to bit arrays to be used for the
  # similarity difference calculations)
  #
  encode_plain_node_val_dict = {}
  encode_key_q_gram_key_dict = {}
  
  include_only_common = False
  
  if(include_only_common):
    print '  Remove non overlapping records from the plain-text graph...'
    print

  for (ent_id, q_gram_set) in plain_q_gram_dict.iteritems():
    
    if(include_only_common):
      if(ent_id not in common_rec_id_set):
        continue
    
    node_key_val = tuple(sorted(q_gram_set))  # Sets cannot be dictionary keys
    plain_node_val_dict[node_key_val] = q_gram_set

    # Only keep none empty attribute values
    #
    QG_sim_graph.add_rec(node_key_val, ent_id,
                         filter(None, plain_rec_attr_val_dict[ent_id]))
    
    if(ent_id not in qg_graph_node_id_dict):
      qg_graph_node_id_dict[ent_id] = node_key_val

  for (ent_id, bit_array) in encode_hash_dict.iteritems():
    q_gram_set = encode_q_gram_dict[ent_id]
    
    if(encode_method == '2sh'):
      node_key_val = [str(val) for val in bit_array]
      node_key_val = tuple(sorted(node_key_val))  # Sets cannot be dictionary keys
    else:
      node_key_val = str(bit_array)  # Bit arrays cannot be dictionary keys  
    
    encode_node_val_dict[node_key_val] = (q_gram_set, bit_array)
    
    if(node_key_val in encode_key_q_gram_key_dict):
      assert encode_key_q_gram_key_dict[node_key_val] == tuple(sorted(q_gram_set))
    else:
      encode_key_q_gram_key_dict[node_key_val] = tuple(sorted(q_gram_set))

    # Only keep none empty attribute values
    #
    BA_sim_graph.add_rec(node_key_val, ent_id,
                         filter(None, encode_rec_attr_val_dict[ent_id]))
    
    if(ent_id not in ba_graph_node_id_dict):
      ba_graph_node_id_dict[ent_id] = node_key_val

    plain_node_key_val = tuple(sorted(q_gram_set))
    encode_plain_node_val_dict[plain_node_key_val] = bit_array

  plain_graph_num_node =  len(plain_node_val_dict)
  encode_graph_num_node = len(encode_node_val_dict)

  print 'Number of nodes in the two graphs: %d in plain-text / %d in encoded' \
        % (plain_graph_num_node, encode_graph_num_node)
  print

  start_time = time.time() 

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # Initialise the LSH based indexing methods (to prevent full pair-wise
  # comparisons of all pairs of q-grams and bit-arrays)
  # Aim to include all pairs with a similarity of at least 0.6 with high
  # probability
  # Band size / Number of bands / Threshold of s-curve / Prob. at .5/.6/.7
  #     3           40                  0.292                0.99994 / 1.00000
  #     4           40                  0.398                0.99612 / 0.99998
  #     5           40                  0.478                0.96076 / 0.99936
  #     6           40                  0.541                0.85209 / 0.99331
  #
  #     4           50                  0.376      0.96032 / 0.99903 / 1.00000
  #     4           60                  0.359      0.97919 / 0.99976 / 1.00000

  #     3           80                  0.232                1.00000 / 1.00000
  #     4           80                  0.334                0.99998 / 1.00000
  #     5           80                  0.416                0.99846 / 1.00000
  #     6           80                  0.482                0.97812 / 0.99996
  #
  plain_sample_size = 4
  plain_num_samples = 50
  QG_min_hash = MinHashLSH(plain_sample_size, plain_num_samples, random_seed)
  
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # We use min-hash for the bit array hashing based on the q-gram sets of
  # the encoded values in order to generate a similar blocking structure
  # (i.e. same q-gram sets have the same probability to be compared or not)
  #
  
  ba_blck_dict = {} # A dictionary to store blocked values
  q_gram_min_hash = False
  
  if(encode_blck_method == 'hmlsh'): # Hamming LSH blocking
    enc_sample_size = 4
    enc_num_samples = 70
    
    if(encode_method == 'bf'):
      bit_array_len    = bf_len
    elif(encode_method == 'tmh'):
      bit_array_len    = tmh_num_hash_bits
    
    ba_blck_dict = encode_hlsh_blocking(encode_node_val_dict, 
                                        enc_sample_size, 
                                        enc_num_samples, 
                                        bit_array_len)
    
  elif(encode_blck_method == 'minhash'): # Min-hash LSH blocking
    
    q_gram_min_hash = True
    
    if(q_gram_min_hash):
      enc_sample_size = plain_sample_size
      enc_num_samples = plain_num_samples
      for (node_key_val, q_gram_set_bit_array) in encode_node_val_dict.iteritems():
        q_gram_set, bit_array = q_gram_set_bit_array
        ba_band_hash_sig_list = QG_min_hash.hash_q_gram_set(q_gram_set)
        for min_hash_val in ba_band_hash_sig_list:
          min_hash_val_tuple = tuple(min_hash_val)
          min_hash_key_val_set = ba_blck_dict.get(min_hash_val_tuple, set())
          min_hash_key_val_set.add(node_key_val)
          ba_blck_dict[min_hash_val_tuple] = min_hash_key_val_set
    else:
      if(encode_method == '2sh'):
        enc_sample_size = 2
        enc_num_samples = 100
        QG_enc_min_hash = MinHashLSH(enc_sample_size, enc_num_samples, random_seed)
        
        for (node_key_val, q_gram_set_bit_array) in encode_node_val_dict.iteritems():
          q_gram_set, bit_array = q_gram_set_bit_array
          rand_int_str_set = set()
          for rand_int in bit_array:
            rand_int_str_set.add(str(rand_int))
          
          ba_band_hash_sig_list = QG_enc_min_hash.hash_q_gram_set(rand_int_str_set)
          for min_hash_val in ba_band_hash_sig_list:
            min_hash_val_tuple = tuple(min_hash_val)
            min_hash_key_val_set = ba_blck_dict.get(min_hash_val_tuple, set())
            min_hash_key_val_set.add(node_key_val)
            ba_blck_dict[min_hash_val_tuple] = min_hash_key_val_set
     
      elif(encode_method in ['tmh', 'bf']):
        enc_sample_size = 2
        enc_num_samples = 200
        QG_enc_min_hash = MinHashLSH(enc_sample_size, enc_num_samples, random_seed)
       
        for (node_key_val, q_gram_set_bit_array) in encode_node_val_dict.iteritems():
          q_gram_set, bit_array = q_gram_set_bit_array
       
          ba_1_bit_pos_set = set()
       
          for pos in range(len(bit_array)):
            if(bit_array[pos] == 1):
              ba_1_bit_pos_set.add(str(pos))
       
          ba_band_hash_sig_list = QG_enc_min_hash.hash_q_gram_set(ba_1_bit_pos_set)
          for min_hash_val in ba_band_hash_sig_list:
            min_hash_val_tuple = tuple(min_hash_val)
            min_hash_key_val_set = ba_blck_dict.get(min_hash_val_tuple, set())
            min_hash_key_val_set.add(node_key_val)
            ba_blck_dict[min_hash_val_tuple] = min_hash_key_val_set
    
  elif(encode_blck_method == 'soundex'): # Soundex blocking
    ba_blck_dict = rec_soundex_blocking(encode_soundex_val_dict, 
                                        ba_graph_node_id_dict)
    
  elif(encode_blck_method == 'none'): # No blocking.
    enc_sample_size = 0
    enc_num_samples = 0
    ba_blck_dict['all'] = set(encode_node_val_dict.keys())
    
  else:
    raise Exception, '***Wrong blocking method for encoded text!'
  
  
  num_blocks = len(ba_blck_dict)
  print 'Bit array %s indexing contains %d blocks' % (encode_blck_method, num_blocks)

  num_rec_pair = 0
  min_hash_block_size_list = []
  for min_hash_key_val_set in ba_blck_dict.itervalues():
    block_size = len(min_hash_key_val_set)
    min_hash_block_size_list.append(block_size)
    num_rec_pair += block_size*(block_size-1) / 2

  num_all_rec_pair = len(encode_node_val_dict)*(len(encode_node_val_dict)-1)/2

  print '  Minimum, average, median and maximum block sizes: ' + \
        '%d / %.2f / %d / %d' % (min(min_hash_block_size_list),
                                 numpy.mean(min_hash_block_size_list),
                                 numpy.median(min_hash_block_size_list),
                                 max(min_hash_block_size_list))

  print '  Compare %d record pairs from %s blocking:' % (num_rec_pair, encode_blck_method)
  print '    (total number of all record pairs: %d)' % (num_all_rec_pair)
  print

  min_hash_block_size_list = []  # Not needed any more

  # Now compare all encoded value pairs in the same block (loop over all blocks)
  #
  ba_sim_list        = []
  qg_sim_list        = []
  qg_num_q_gram_list = []
  sim_abs_diff_list  = []
  sim_sample_dict = {'0-1': 0, '1-2': 0, '2-3': 0, '3-4': 0, '4-5': 0, '5-6': 0, 
                     '6-7': 0, '7-8': 0, '8-9': 0, '9-10': 0}
  
  for (bn, min_hash_key_val_set) in enumerate(ba_blck_dict.itervalues()):

    if ((bn > 0) and (bn % 10000 == 0)):
      print '    Processed %d of %d blocks' % (bn, num_blocks)

    if (len(min_hash_key_val_set) > 1):
      min_hash_key_val_list = sorted(min_hash_key_val_set)

      for (i, node_key_val1) in enumerate(min_hash_key_val_list[:-1]):
        q_gram_set1, bit_array1 = encode_node_val_dict[node_key_val1]

        for node_key_val2 in min_hash_key_val_list[i+1:]:
          q_gram_set2, bit_array2 = encode_node_val_dict[node_key_val2]

          plain_pair_sim = plain_sim_funct(q_gram_set1, q_gram_set2)
          encode_pair_sim = encode_sim_funct(bit_array1, bit_array2)
          
          if(sim_diff_adjust_flag == True):
            sim_class = check_sim_class(plain_pair_sim)
            if(sim_class != 'None'):
              if(sim_sample_dict[sim_class] < 500):
                full_q_set = q_gram_set1.union(q_gram_set2)
                ba_sim_list.append(encode_pair_sim)
                qg_sim_list.append(plain_pair_sim)
                qg_num_q_gram_list.append(len(full_q_set))
                sim_sample_dict[sim_class]+=1
                
                abs_sim_diff = abs(encode_pair_sim - plain_pair_sim)
                sim_abs_diff_list.append(abs_sim_diff)

          enc_min_sim = min_sim
          #if (encode_pair_sim >= enc_min_sim):
          if (encode_pair_sim >= min_sim):
            BA_sim_graph.add_edge_sim(node_key_val1, node_key_val2, \
                                      encode_pair_sim)
  print
  
  
#  if(sim_diff_adjust_flag == True):
    
#===============================================================================
#     # Write evaluation resuls to a csv file
#     #
#     sim_diff_eval_res_file = 'graph_attack_similarity_differences_res.csv'
#   
#     min_abs_diff    = min(sim_abs_diff_list)
#     max_abs_diff    = max(sim_abs_diff_list)
#     mean_abs_diff   = numpy.mean(sim_abs_diff_list)
#     median_abs_diff = numpy.median(sim_abs_diff_list)
#     std_abs_diff    = numpy.std(sim_abs_diff_list)
#     var_abs_diff    = numpy.var(sim_abs_diff_list)
#   
#     sim_diff_header_list = ['date','time','plain_dataset_name','plain_attr_list','plain_num_rec',
#                             'encode_dataset_name','encode_attr_list','encode_num_rec','q_length',
#                             'padded','sim_adjust_flag','regression_model','plain_sim_funct',
#                             'enc_sim_func','min_sim',
#                             'encode_method','bf_hash_type/tmh_num_hash_bits/cmh_max_rand_int','num_hash_funct',
#                             'bf_len/tmh_num_hash_tables/cmh_num_hash_col','bf_encode/tmh_key_len', 
#                             'bf_harden/tmh_val_len',
#                             'min_abs_diff','max_abs_diff','mean_abs_diff','median_abs_diff','std_abs_diff',
#                             'var_abs_diff']
# 
#     sim_diff_val_list = [today_str, now_str, plain_base_data_set_name, plain_attr_list_str, 
#                          plain_num_rec_loaded,
#                          encode_base_data_set_name, encode_attr_list_str, encode_num_rec_loaded,
#                          q, str(padded_flag).lower(), sim_diff_adjust_flag, regre_model_str.lower(),
#                          plain_sim_funct_name, encode_sim_funct_name,
#                          min_sim, encode_method]
#   
#   
#     if (encode_method == 'bf'):
#       sim_diff_val_list += [bf_hash_type, bf_num_hash_funct, bf_len, bf_encode, bf_harden]
#     
#     elif (encode_method == 'tmh'):
#       sim_diff_val_list += [tmh_num_hash_bits, tmh_hash_funct, tmh_num_tables, tmh_key_len, tmh_val_len]
#     
#     elif(encode_method == '2sh'):
#       sim_diff_val_list += [cmh_max_rand_int, cmh_num_hash_funct, cmh_num_hash_col, '', '']
#   
#     sim_diff_val_list += [min_abs_diff, max_abs_diff, mean_abs_diff, median_abs_diff, 
#                           std_abs_diff, var_abs_diff]
#===============================================================================
  

  ba_graph_gen_time = time.time() - start_time

  ba_graph_num_edges = networkx.number_of_edges(BA_sim_graph.sim_graph)

  ba_graph_num_singleton = BA_sim_graph.remove_singletons()

  print 'Time for generating the bit-array similarity graph: %.2f sec' % \
        (ba_graph_gen_time)
  print '  Number of edges in graph: %d' % (ba_graph_num_edges)
  print ' ', auxiliary.get_memory_usage()
  print
  
  #--------------------------------------------------------------------------
  # If the similarity adjustment flag is set to True, then assuming we have
  # a sample dataset with ground truth availble to us build a regression model
  # to predict the the encoded similarity given the q-gram similarity. Here
  # we use two features for the model q-gram similarity and the total number
  # of unique q-grams of the two records.
  
  regre_file_str = 'regre-model-%s-%s-%s-%d-%s-%s-%s-%s' %\
                   (regre_model_str.lower(), plain_base_data_set_name, 
                    plain_attr_list_str, q, str(padded_flag).lower(),
                    plain_sim_funct_name, encode_sim_funct_name,
                    encode_method_str)
  
  regre_file_name = regre_file_str + '.sav'
  regre_file_path = 'regre-models/'
  regre_file_name = graph_path + regre_file_name
  
  if (os.path.isfile(regre_file_name)):
    
    print 'Load regression model from saved file:', regre_file_name
    print
    
    regre_model = pickle.load(open(regre_file_name, 'rb'))
    
    # Test the model with some sample data
    #
    test_sim_regre_model(regre_model, ba_sim_list, qg_sim_list, qg_num_q_gram_list,
                         regre_model_str, encode_method)

  else:
    num_samples = 20000  # Ensure there are enough samples
  
    if(sim_diff_adjust_flag == True):
    
      regres_file_name = 'plain-encode-sim-%s-regression-scatter-%s-%s-%s-%d-%s-' % \
                       (regre_model_str, plain_base_data_set_name, encode_base_data_set_name,
                        plain_attr_list_str, plain_num_rec_loaded,
                        encode_attr_list_str) + \
                       '%d-%d-%s-%s-%s-%.3f-%s-%s.eps' % \
                       (encode_num_rec_loaded, q, str(padded_flag).lower(),
                        plain_sim_funct_name, encode_sim_funct_name,
                        min_sim, encode_method_str, today_str)
    
      num_attr = len(encode_attr_list_str.split('_'))
      if(num_attr == 1):
        plot_title = 'FirstName'
      elif(num_attr == 2):
        plot_title = 'FirstName, LastName'
      elif(num_attr == 3):
        plot_title = 'FirstName, LastName, Street'
      
      if('ncvoter' in encode_base_data_set_name):
        plot_data_name = '(NCVR)'
        if(num_attr == 4):
          plot_title = 'FirstName, LastName, Street, City'
      elif('titanic' in encode_base_data_set_name):
        plot_data_name = '(TITANIC)'
      elif('passenger' in encode_base_data_set_name):
        plot_data_name = '(TITANIC)'
      elif('euro' in encode_base_data_set_name):
        plot_data_name = '(EURO)'
        if(num_attr == 4):
          plot_title = 'FirstName, LastName, Street, Postcode'
    
      regre_model, eval_res_tuple = build_regre_model(ba_sim_list, qg_sim_list,
                                                    qg_num_q_gram_list,
                                                    regre_model_str, plot_data_name,
                                                    encode_method, encode_sim_funct_name,
                                                    regres_file_name, plot_title)
      
      # Save the regression model in a pickle file
      #
      #save_file = open(regre_file_name, 'wb')
      #pickle.dump(regre_model, save_file)
    
      # Write evaluation resuls to a csv file
      #
      regre_model_eval_res_file = 'graph_attack_regression_model_eval_res.csv'
    
    
      res_header_list = ['date','time','plain_dataset_name','plain_attr_list','plain_num_rec',
                         'encode_dataset_name','encode_attr_list','encode_num_rec','q_length',
                         'padded','sim_adjust_flag','regression_model','plain_sim_funct',
                         'enc_sim_func','min_sim',
                         'encode_method','bf_hash_type/tmh_num_hash_bits/cmh_max_rand_int','num_hash_funct',
                         'bf_len/tmh_num_hash_tables/cmh_num_hash_col','bf_encode/tmh_key_len', 'bf_harden/tmh_val_len',
                         'explained_var','min_abs_err','max_abs_err','avrg_abs_err','std_abs_err',
                         'mean_sqrd_err','mean_sqrd_lg_err','r_sqrd']

      res_val_list = [today_str, now_str, plain_base_data_set_name, plain_attr_list_str, plain_num_rec_loaded,
                      encode_base_data_set_name, encode_attr_list_str, encode_num_rec_loaded,
                      q, str(padded_flag).lower(), sim_diff_adjust_flag, regre_model_str.lower(),
                      plain_sim_funct_name, encode_sim_funct_name,
                      min_sim, encode_method]
    
    
      if (encode_method == 'bf'):
        res_val_list += [bf_hash_type, bf_num_hash_funct, bf_len, bf_encode, bf_harden]
      
      elif (encode_method == 'tmh'):
        res_val_list += [tmh_num_hash_bits, tmh_hash_funct, tmh_num_tables, tmh_key_len, tmh_val_len]
      
      elif(encode_method == '2sh'):
        res_val_list += [cmh_max_rand_int, cmh_num_hash_funct, cmh_num_hash_col, '', '']
    
    
      res_val_list += list(eval_res_tuple)
    
      if (not os.path.isfile(regre_model_eval_res_file)):
        write_reg_file = open(regre_model_eval_res_file, 'w')
        csv_writer = csv.writer(write_reg_file)
        csv_writer.writerow(res_header_list)
       
        print 'Created new result file:', regre_model_eval_res_file
       
      else:  # Append results to an existing file
        write_reg_file = open(regre_model_eval_res_file, 'a')
        csv_writer = csv.writer(write_reg_file)
       
      csv_writer.writerow(res_val_list)
      write_reg_file.close()
  
  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  # We also use min-hash for the q-gram similarities in order to create a
  # similar blocking structure
  # (i.e. same q-gram sets have the same probability to be compared or not)
  #
  start_time = time.time()
  
  qg_blck_dict = {}
  
  if(plain_blck_method == 'minhash'):
    
    for (node_key_val, q_gram_set) in plain_node_val_dict.iteritems():
      qg_band_hash_sig_list = QG_min_hash.hash_q_gram_set(q_gram_set)
      for min_hash_val in qg_band_hash_sig_list:
        min_hash_val_tuple = tuple(min_hash_val)
        min_hash_key_val_set = qg_blck_dict.get(min_hash_val_tuple, set())
        min_hash_key_val_set.add(node_key_val)
        qg_blck_dict[min_hash_val_tuple] = min_hash_key_val_set
    
  elif(plain_blck_method == 'soundex'):
    qg_blck_dict = rec_soundex_blocking(plain_soundex_val_dict, 
                                        qg_graph_node_id_dict)
    
  elif(plain_blck_method == 'none'):
    qg_blck_dict['all'] = set(plain_node_val_dict.keys())
    
  else:
    raise Exception, '***Wrong blocking method for plain-text!'
  

  num_blocks = len(qg_blck_dict)
  print 'Q-gram %s indexing contains %d blocks' % (plain_blck_method, num_blocks)

  num_rec_pair = 0
  min_hash_block_size_list = []
  for min_hash_key_val_set in qg_blck_dict.itervalues():
    block_size = len(min_hash_key_val_set)
    min_hash_block_size_list.append(block_size)
    num_rec_pair += block_size*(block_size-1) / 2

  num_all_rec_pair = len(plain_node_val_dict)*(len(plain_node_val_dict)-1) / 2

  print '  Minimum, average, median and maximum block sizes: ' + \
        '%d / %.2f / %d / %d' % (min(min_hash_block_size_list),
                                 numpy.mean(min_hash_block_size_list),
                                 numpy.median(min_hash_block_size_list),
                                 max(min_hash_block_size_list))

  print '  Compare %d record pairs from %s blocking:' % (num_rec_pair, plain_blck_method)
  print '    (total number of all record pairs: %d)' % (num_all_rec_pair)
  print

  min_hash_block_size_list = []  # Not needed any more
  
  same_ba_blck = False
  
  if(same_ba_blck):
    for (bn, min_hash_key_val_set) in enumerate(ba_blck_dict.itervalues()):
      
      if ((bn > 0) and (bn % 10000 == 0)):
        print '    Processed %d of %d blocks' % (bn, num_blocks)

      if (len(min_hash_key_val_set) > 1):
        min_hash_key_val_list = sorted(min_hash_key_val_set)

        for (i, node_key_val1) in enumerate(min_hash_key_val_list[:-1]):
          plain_node_key_val1 = encode_key_q_gram_key_dict[node_key_val1]
          q_gram_set1 = plain_node_val_dict[plain_node_key_val1]
        
          assert q_gram_set1 == set(plain_node_key_val1)
          
          for node_key_val2 in min_hash_key_val_list[i+1:]:
            plain_node_key_val2 = encode_key_q_gram_key_dict[node_key_val2]
            q_gram_set2 = plain_node_val_dict[plain_node_key_val2]
          
            assert q_gram_set2 == set(plain_node_key_val2)
            
            plain_pair_sim = plain_sim_funct(q_gram_set1, q_gram_set2)
          
            # If needed adjust the edge similarity - - - - - - - - - - - - - - -
            #
            if (sim_diff_adjust_flag == True):
            
              if(regre_model_str == 'linear'):
                full_q_gram_set = q_gram_set1.union(q_gram_set2)
                plain_pair_sim = regre_model.predict([[plain_pair_sim, len(full_q_gram_set)]])[0]
              elif(regre_model_str == 'poly'):
                full_q_gram_set = q_gram_set1.union(q_gram_set2)
                poly = PolynomialFeatures(degree=2)
                ori_sim_val = poly.fit_transform([[plain_pair_sim, len(full_q_gram_set)]])
                plain_pair_sim = regre_model.predict(ori_sim_val)[0]
              else: # isotonic
                plain_pair_sim = regre_model.predict(plain_pair_sim)[0]
          
            if (plain_pair_sim >= min_sim):
              QG_sim_graph.add_edge_sim(plain_node_key_val1, plain_node_key_val2,
                                        plain_pair_sim)
  
  

  # Now compare all q-gram sets in the same block (loop over all blocks)
  #
  for (bn, min_hash_key_val_set) in enumerate(qg_blck_dict.itervalues()):
 
    if ((bn > 0) and (bn % 10000 == 0)):
      print '    Processed %d of %d blocks' % (bn, num_blocks)
 
    if (len(min_hash_key_val_set) > 1):
      min_hash_key_val_list = sorted(min_hash_key_val_set)
 
      for (i, node_key_val1) in enumerate(min_hash_key_val_list[:-1]):
        #print node_key_val1
        q_gram_set1 = plain_node_val_dict[node_key_val1]
         
        assert q_gram_set1 == set(node_key_val1)
 
        for node_key_val2 in min_hash_key_val_list[i+1:]:
          q_gram_set2 = plain_node_val_dict[node_key_val2]
           
          assert q_gram_set2 == set(node_key_val2)
 
          plain_pair_sim = plain_sim_funct(q_gram_set1, q_gram_set2)
           
          # If needed adjust the edge similarity - - - - - - - - - - - - - - -
          #
          if (sim_diff_adjust_flag == True):
            
            if(encode_method == 'bf'):
              full_q_gram_set = q_gram_set1.union(q_gram_set2)
              input_array = [[plain_pair_sim, len(full_q_gram_set)]]
            if(encode_method in ['tmh', '2sh']):
              input_array = [[plain_pair_sim]]
             
            if(regre_model_str == 'linear'):
              plain_pair_sim = regre_model.predict(input_array)[0][0]
            elif(regre_model_str == 'poly'):
              poly = PolynomialFeatures(degree=2)
              ori_sim_val = poly.fit_transform(input_array)
              plain_pair_sim = regre_model.predict(ori_sim_val)[0][0]
            else: # isotonic
              plain_pair_sim = regre_model.predict(plain_pair_sim)[0]
           
          if (plain_pair_sim >= min_sim):
            QG_sim_graph.add_edge_sim(node_key_val1, node_key_val2,
                                      plain_pair_sim)

  qg_graph_gen_time = time.time() - start_time

  qg_graph_num_edges = networkx.number_of_edges(QG_sim_graph.sim_graph)

  print 'Time for generating the q-gram similarity graph: %.2f sec' % \
        (qg_graph_gen_time)
  print '  Number of edges in graph: %d' % (qg_graph_num_edges)
  print ' ', auxiliary.get_memory_usage()
  print

  qg_graph_num_singleton = QG_sim_graph.remove_singletons()
  

  # For randomly sampled edges that are the same across the two graphs get
  # their similarity differences and calculate average such differences for
  # different similarity intervals
  #
  sim_diff_plot_file_name = 'plain-encode-sim-diff-scatter-%s-%s-%s-%d-%s-' % \
                            (plain_base_data_set_name, encode_base_data_set_name,
                             plain_attr_list_str, plain_num_rec_loaded,
                             encode_attr_list_str) + \
                            '%d-%d-%s-%s-%s-%.3f-%s-%s.eps' % \
                            (encode_num_rec_loaded, q, str(padded_flag).lower(),
                             plain_sim_funct_name, encode_sim_funct_name,
                             min_sim, encode_method_str, today_str)
  
  sim_diff_plot_file_name_adj = sim_diff_plot_file_name.replace('.eps',
                                                        '-adjusted.eps')
  
  sim_diff_interval_size = 0.05

  QG_sim_graph.comp_sim_differences(encode_plain_node_val_dict,
                                    BA_sim_graph.sim_graph,
                                    sim_diff_interval_size, num_samples,
                                    sim_diff_plot_file_name_adj)
  
  # Save graphs into pickled (binary) files for later use
  #
  networkx.write_gpickle(QG_sim_graph.sim_graph, plain_graph_file_name)
  networkx.write_gpickle(BA_sim_graph.sim_graph, encode_graph_file_name)

  print 'Wrote graphs into pickle files:'
  print '  Plain-text graph file:', plain_graph_file_name
  print '  Encoded graph file:   ', encode_graph_file_name
  print ' ', auxiliary.get_memory_usage()
  print

## ESTOU AQUI
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Back to common code (graphs either loaded from files or generated)

calc_common_edges = False
                     
if(calc_common_edges):
  
  remove_uncommon_edges = False
  
  QG_sim_graph, BA_sim_graph, common_edges, ba_only_edges, qg_only_edges = \
                     graph_sim_dist_plot(QG_sim_graph, BA_sim_graph, 
                                         plain_node_val_dict,
                                         encode_node_val_dict,
                                         remove_uncommon_edges)
                     
  write_res_flag = False
  
  if(write_res_flag):
    
    blocking_alignment_res_file = 'graph_attack_minhash_hamming_align_res.csv'
    
    blck_res_header_list = ['date','time','plain_dataset_name','plain_attr_list','plain_num_rec',
                       'encode_dataset_name','encode_attr_list','encode_num_rec','q_length',
                       'padded','sim_adjust_flag','regression_model','plain_sim_funct',
                       'enc_sim_func','min_sim',
                       'encode_method','encode_blocking','plain_blocking',
                       'encode_num_samples(d)', 'encode_sample_size(r)',
                       'plain_num_samples(d)', 'plain_sample_size(r)',
                       'bf_hash_type/tmh_num_hash_bits/cmh_max_rand_int','num_hash_funct',
                       'bf_len/tmh_num_hash_tables/cmh_num_hash_col','bf_encode/tmh_key_len', 
                       'bf_harden/tmh_val_len','num_common_edges','num_enc_only_edges',
                       'num_plain_only_edges']

    blck_res_val_list = [today_str, now_str, plain_base_data_set_name, plain_attr_list_str, plain_num_rec_loaded,
                    encode_base_data_set_name, encode_attr_list_str, encode_num_rec_loaded,
                    q, str(padded_flag).lower(), sim_diff_adjust_flag, regre_model_str.lower(),
                    plain_sim_funct_name, encode_sim_funct_name,
                    min_sim, encode_method, encode_blck_method, plain_blck_method,
                    enc_num_samples, enc_sample_size,
                    plain_num_samples, plain_sample_size]
    
    
    if (encode_method == 'bf'):
      blck_res_val_list += [bf_hash_type, bf_num_hash_funct, bf_len, bf_encode, bf_harden]
      
    elif (encode_method == 'tmh'):
      blck_res_val_list += [tmh_num_hash_bits, tmh_hash_funct, tmh_num_tables, tmh_key_len, tmh_val_len]
      
    elif(encode_method == '2sh'):
      blck_res_val_list += [cmh_max_rand_int, cmh_num_hash_funct, cmh_num_hash_col, '', '']
      
    blck_res_val_list += [len(common_edges), len(ba_only_edges), len(qg_only_edges)]
    
    assert len(blck_res_header_list) == len(blck_res_val_list)
    
    if (not os.path.isfile(blocking_alignment_res_file)):
      write_blck_file = open(blocking_alignment_res_file, 'w')
      csv_writer = csv.writer(write_blck_file)
      csv_writer.writerow(blck_res_header_list)
       
      print 'Created new result file:', blocking_alignment_res_file
       
    else:  # Append results to an existing file
      write_blck_file = open(blocking_alignment_res_file, 'a')
      csv_writer = csv.writer(write_blck_file)
       
    csv_writer.writerow(blck_res_val_list)
    write_blck_file.close()


QG_sim_graph.show_graph_characteristics(all_sim_list,
                                        'Plain-text q-gram graph')

BA_sim_graph.show_graph_characteristics(all_sim_list,
                                        'Encoded bit-array graph')

# Generate a plot of the similarity histograms of the graphs
#
if (encode_method == 'bf'):
  encode_method_str = 'bf-%s-%s-%d-%s' % (bf_hash_type, bf_num_hash_funct,
                      bf_len, bf_harden)
elif (encode_method == 'tmh'):
  encode_method_str = 'tmh-%d-%s-%d-%d-%d' % (tmh_num_hash_bits,
                      tmh_hash_funct, tmh_num_tables, tmh_key_len, tmh_val_len)
elif (encode_method == '2sh'):
  encode_method_str = '2sh-%d-%d-%d' % (cmh_max_rand_int, cmh_num_hash_funct, 
                                        cmh_num_hash_col)

sim_histo_plot_file_name = 'sim-graph-histograms-%s-%s-%s-%d-%s-%d-' % \
                           (plain_base_data_set_name, encode_base_data_set_name,
                            plain_attr_list_str, plain_num_rec_loaded,
                            encode_attr_list_str, encode_num_rec_loaded) + \
                           '%d-%s-%s-%s-%s-%.3f-%s-%s.eps' % \
                           (q, str(padded_flag).lower(), plain_sim_funct_name,
                            str(sim_diff_adjust_flag).lower(),
                            encode_sim_funct_name, min_sim, encode_method_str,
                            today_str)

# Generate a list of similarities of all edges between records from the same
# entities (not just q-gram or bit array values) and count the number of
# such edges not in the intersection
#
qg_edge_dict = {}
ba_edge_dict = {}

plain_sim_graph =  QG_sim_graph.sim_graph  # Short cuts
encode_sim_graph = BA_sim_graph.sim_graph

for (node_key_val1, node_key_val2) in plain_sim_graph.edges():
  ent_id_set1 = plain_sim_graph.node[node_key_val1]['ent_id_set']
  ent_id_set2 = plain_sim_graph.node[node_key_val2]['ent_id_set']
  edge_sim =    plain_sim_graph.edge[node_key_val1][node_key_val2]['sim']

  for ent_id1 in ent_id_set1:
    for ent_id2 in ent_id_set2:
      ent_id_pair = tuple(sorted([ent_id1,ent_id2]))
      assert ent_id_pair not in qg_edge_dict, ent_id_pair
      qg_edge_dict[ent_id_pair] = edge_sim

for (node_key_val1, node_key_val2) in encode_sim_graph.edges():
  ent_id_set1 = encode_sim_graph.node[node_key_val1]['ent_id_set']
  ent_id_set2 = encode_sim_graph.node[node_key_val2]['ent_id_set']
  edge_sim =    encode_sim_graph.edge[node_key_val1][node_key_val2]['sim']

  for ent_id1 in ent_id_set1:
    for ent_id2 in ent_id_set2:
      ent_id_pair = tuple(sorted([ent_id1,ent_id2]))
      assert ent_id_pair not in ba_edge_dict, ent_id_pair
      ba_edge_dict[ent_id_pair] = edge_sim

common_edge_set =       set(qg_edge_dict.keys()) & set(ba_edge_dict.keys())
num_all_edges =         len(set(qg_edge_dict.keys()) | set(ba_edge_dict.keys()))
num_common_edges =      len(common_edge_set)
num_edges_plain_only =  len(qg_edge_dict) - num_common_edges
num_edges_encode_only = len(ba_edge_dict) - num_common_edges

print '#### Identified %d (%.2f%%) edges between entities common to both ' % \
      (num_common_edges, 100.0*num_common_edges / num_all_edges) + \
      'similarity graphs'
print '####   %d (%.2f%%) edges only occur in plain-text similarity graph' % \
      (num_edges_plain_only, 100.0*num_edges_plain_only / num_all_edges)
print '####   %d (%.2f%%) edges only occur in encode similarity graph' % \
      (num_edges_encode_only, 100.0*num_edges_encode_only / num_all_edges)
print

# Calculate edge similarity differences for all given minimum similarities
#
print '#### Similarity differences across the two graphs between true ' + \
      'matching record pairs:'
print '####  (negative means plain-tex similarity < encoded similarity)' 

# For plotting the similarity differences keep one list of differences per
# minimum similarity (with minimum similarities as keys, each with a list
# of similarity differences)
#
plot_list_dict = {}

for check_min_sim in all_sim_list:
  edge_sim_diff_list = []

  for ent_id_pair in common_edge_set:
    qg_sim = qg_edge_dict[ent_id_pair]
    ba_sim = ba_edge_dict[ent_id_pair]

    # Check at least one of the two similarities is high enough
    #
    if (qg_sim >= check_min_sim) or  (ba_sim >= check_min_sim):
      sim_diff = qg_edge_dict[ent_id_pair] - ba_edge_dict[ent_id_pair]
      edge_sim_diff_list.append(sim_diff)

  sim_diff_min = min(edge_sim_diff_list)
  sim_diff_avr = numpy.mean(edge_sim_diff_list)
  sim_diff_std = numpy.std(edge_sim_diff_list)
  sim_diff_med = numpy.median(edge_sim_diff_list)
  sim_diff_max = max(edge_sim_diff_list)

  plot_list_dict[check_min_sim] = edge_sim_diff_list

  print '####   For a minimum similarity of %.3f:' % (check_min_sim) + \
        '(with %d similarity pairs)' % (len(edge_sim_diff_list))
  print '####     min=%.3f / avr=%.3f (std=%.3f) / med=%.3f / max=%.3f' % \
        (sim_diff_min, sim_diff_avr, sim_diff_std, sim_diff_med,sim_diff_max)
print

del qg_edge_dict  # Not needed anymore
del ba_edge_dict

# Generate a plot for the edge similarity differences
#
if (encode_method == 'bf'):
  encode_method_str = 'bf-%s-%s-%d-%s' % (bf_hash_type, bf_num_hash_funct,
                      bf_len, bf_harden)
elif (encode_method == 'tmh'):
  encode_method_str = 'tmh-%d-%s-%d-%d-%d' % (tmh_num_hash_bits,
                      tmh_hash_funct, tmh_num_tables, tmh_key_len, tmh_val_len)
elif (encode_method == '2sh'):
  encode_method_str = '2sh-%d-%d-%d' % (cmh_max_rand_int, cmh_num_hash_funct, 
                                        cmh_num_hash_col)

sim_diff_plot_file_name = 'plain-encode-sim-diff-%s-%s-%s-%d-%s-%d-' % \
                           (plain_base_data_set_name, encode_base_data_set_name,
                            plain_attr_list_str, plain_num_rec_loaded,
                            encode_attr_list_str, encode_num_rec_loaded) + \
                           '%d-%s-%s-%s-%s-%.3f-%s-%s.eps' % \
                           (q, str(padded_flag).lower(), plain_sim_funct_name,
                            str(sim_diff_adjust_flag).lower(),
                            encode_sim_funct_name, min_sim, encode_method_str,
                            today_str)

#plot_save_sim_diff(plot_list_dict, sim_diff_plot_file_name,2,False)

memo_use = auxiliary.get_memory_usage_val()
print
print '   ', auxiliary.get_memory_usage()
print

# Print a summary result line for the encoding and graph generation steps
#
print '#### Graph matching PPRL attack result summary'
print '####   Experiment started on %s at %s' % (today_str, now_str)
print '####'
print '####   Plain-text data set: %s' % (plain_base_data_set_name)
print '####     Number of records: %d' % (plain_num_rec_loaded)
print '####     Attributes used:   %s' % (str(plain_attr_list))
print '####   Encoded data set:    %s' % (encode_base_data_set_name)
print '####     Number of records: %d' % (encode_num_rec_loaded)
print '####     Attributes used:   %s' % (str(encode_attr_list))
print '####'
print '####  Q-gram encoding: q=%d, padding:   %s' % (q, str(padded_flag))
print '####    Plain-text similarity function: %s' % (plain_sim_funct_name)
print '####    Encode similarity function:     %s' % (encode_sim_funct_name)
print '####    Overall minimum similarity:     %.2f' % (min_sim)
print '####'
print '####  Similarity difference adjustment: %s' % \
      (str(sim_diff_adjust_flag))
print '####'
print '####  Encoding method: %s' % (encode_method)
if (encode_method == 'bf'):
  print '####    Hash type:                %s' % (bf_hash_type)
  print '####    BF length:                %d bits' % (bf_len)
  print '####    Number of hash functions: %s' % (str(bf_num_hash_funct))
  print '####    BF hardening method:      %s' % (bf_harden)
elif (encode_method == 'tmh'):
  print '####    Hash function:            %s' % (tmh_hash_funct)
  print '####    Bit array length:         %d bits' % (tmh_num_hash_bits)
  print '####    Number of tables:         %d' % (tmh_num_tables)
  print '####    Key length:               %d bits' % (tmh_key_len)
  print '####    Value length:             %d bits' % (tmh_val_len)
elif(encode_method == '2sh'):
  print '####    Maximum random integer:   %d' % (cmh_max_rand_int)
  print '####    Number of hash functions: %d' % (cmh_num_hash_funct)
  print '####    Number of hash columns:   %d' % (cmh_num_hash_col)
  
print '####'

# Generate the header (column names) and result lines for the summary output
#
header_line = 'run_date,run_time,plain_file_name,plain_num_rec,' + \
              'plain_attr_list,encode_file_name,encode_num_rec,' + \
              'encode_attr_list,q,padded,plain_sim_funct_name,' + \
              'sim_diff_adjust_flag,regression_model,encode_sim_funct_name,min_sim,'
result_line = '%s,%s,%s,%d,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%.2f,' % \
              (today_str, now_str, plain_base_data_set_name, \
               plain_num_rec_loaded, plain_attr_list_str, \
               encode_base_data_set_name, encode_num_rec_loaded, \
               encode_attr_list_str, q, str(padded_flag).lower(), \
               plain_sim_funct_name, str(sim_diff_adjust_flag).lower(), \
               regre_model_str.lower(), encode_sim_funct_name, min_sim)
if (encode_method == 'bf'):
  header_line = header_line + 'encoding_method,bf_hash_type,bf_len,' + \
                'bf_num_hash_funct,bf_harden,bf_encode'
  result_line = result_line + 'bf,%s,%d,%s,%s,%s' % (bf_hash_type, bf_len, \
                str(bf_num_hash_funct), bf_harden,bf_encode)
elif (encode_method == 'tmh'):
  header_line = header_line + 'encoding_method,tmh_hash_funct,' + \
                'tmh_num_hash_bits,tmh_num_tables,tmh_key_len,tmh_val_len'
  result_line = result_line + 'tmh,%s,%d,%d,%d,%d' % (tmh_hash_funct, \
                tmh_num_hash_bits, tmh_num_tables, tmh_key_len, tmh_val_len)
elif (encode_method == '2sh'):
  header_line = header_line + 'encoding_method,cmh_max_rand_int,' + \
                'cmh_num_hash_funct,cmh_num_hash_col'
  result_line = result_line + '2sh,%d,%d,%d' % (cmh_max_rand_int, \
                                                cmh_num_hash_funct, \
                                                cmh_num_hash_col)

print '######## load:', header_line
print '######## load:', result_line
print '########'

# Generate header line and result line for result file
#
attack_res_file_name = 'pprl_graph_attack_res_%s.csv' %today_str


attck_res_header_list = ['date','time','plain_dataset_name','plain_attr_list','plain_num_rec',
                         'encode_dataset_name','encode_attr_list','encode_num_rec','q_length',
                         'padded','sim_adjust_flag','regression_model','plain_sim_funct','enc_sim_func',
                         'min_sim','encode_method','bf_hash_type/tmh_num_hash_bits/cmh_max_rand_int','num_hash_funct',
                         'bf_len/tmh_num_hash_tables/cmh_num_hash_col','bf_encode/tmh_key_len', 'bf_harden/tmh_val_len']

attck_res_val_list = [today_str, now_str, plain_base_data_set_name, plain_attr_list_str, plain_num_rec_loaded,
                      encode_base_data_set_name, encode_attr_list_str, encode_num_rec_loaded,
                      q, str(padded_flag).lower(), sim_diff_adjust_flag, regre_model_str.lower(), 
                      plain_sim_funct_name, encode_sim_funct_name,
                      min_sim, encode_method]


if (encode_method == 'bf'):
  attck_res_val_list += [bf_hash_type, bf_num_hash_funct, bf_len, bf_encode, bf_harden]
  
elif (encode_method == 'tmh'):
  attck_res_val_list += [tmh_num_hash_bits, tmh_hash_funct, tmh_num_tables, tmh_key_len, tmh_val_len]
  
elif (encode_method == '2sh'):
  attck_res_val_list += [cmh_max_rand_int, cmh_num_hash_funct, cmh_num_hash_col, '', '']
  
assert len(attck_res_header_list) == len(attck_res_val_list)

# Summary line with timing and graph generation results (if graph was
# generated)
#
if (generated_graph_flag == True):
  header_line = 'load_q_gram_time,hashing_time,plain_text_num_ent,' + \
                'encode_num_ent,common_num_ent,plain_graph_num_node,' + \
                'encode_graph_num_node,plain_graph_gen_time,' + \
                'encode_graph_gen_time,plain_graph_num_edges,' + \
                'encode_graph_num_edges,plain_graph_num_singleton,' + \
                'encode_num_singleton'
  result_line = '%.2f,%.2f,%d,%d,%d,%d,%d,%.2f,%.2f,%d,%d,%d,%d' % \
                (load_q_gram_time, hashing_time, plain_num_ent, \
                 encode_num_ent, common_num_ent, plain_graph_num_node, \
                 encode_graph_num_node, qg_graph_gen_time, ba_graph_gen_time, \
                 qg_graph_num_edges, ba_graph_num_edges, \
                 qg_graph_num_singleton, ba_graph_num_singleton)
  print '######## load:', header_line
  print '######## load:', result_line
  print '########'

# -----------------------------------------------------------------------------
# Loop over the different experimental parameter settings
#
print '######## match: graph_node_min_degr_name,graph_sim_list_name,' + \
      'graph_feat_list_name,graph_feat_select_name,' + \
      'sim_hash_num_bit_name,sim_hash_block_funct,' + \
      'sim_comp_funct,sim_hash_match_sim,num_feat,' + \
      'use_num_feat,num_sim_pair, corr_top_10,corr_top_20,' + \
      'corr_top_50,corr_top_100,corr_all, feat_gen_time,' + \
      'hash_sim_gen_time,hash_sim_block_time,hash_sim_comp_time'

# Loop over different minimum node degrees
#
for (graph_node_min_degr_name, graph_node_min_degr) in \
       graph_node_min_degr_list:

  # ---------------------------------------------------------------------------
  # Step 4: Generate the features from the two graphs

  # First remove all connected components smaller than given minimum size
  #
  min_conn_comp_size = graph_node_min_degr+1  # Smaller connected components
                                              # cannot have nodes with the
                                              # required minimum degree

  QG_sim_graph.gen_min_conn_comp_graph(min_conn_comp_size)
  BA_sim_graph.gen_min_conn_comp_graph(min_conn_comp_size)
  QG_conn_sim_graph = QG_sim_graph
  BA_conn_sim_graph = BA_sim_graph

  QG_conn_sim_graph.show_graph_characteristics(all_sim_list,
                                               'Plain-text q-gram graph')
  BA_conn_sim_graph.show_graph_characteristics(all_sim_list,
                                               'Encoded bit-array graph')

  # Check if the graphs do contain large enough connected components
  #
  if (len(QG_conn_sim_graph.sim_graph.nodes()) == 0) or \
     (len(BA_conn_sim_graph.sim_graph.nodes()) == 0):
    print '#### No large enough connected components (of size %d) in ' % \
          (min_conn_comp_size) + 'at least one of the two graphs' + \
          ' (need to obtain nodes with degree %d)' % (graph_node_min_degr)

    continue  # Go to next minimum connected component parameter value

  max_qg_node_degree = max(QG_conn_sim_graph.sim_graph.degree().values())
  max_ba_node_degree = max(BA_conn_sim_graph.sim_graph.degree().values())
  max_node_degree =    max(max_qg_node_degree, max_ba_node_degree)

  print '  Maximum node degrees: Q-gram graph: %d, bit array graph: %d' % \
        (max_qg_node_degree, max_ba_node_degree)
  print

  # Generate and save plots if both graphs are not too large
  #
  if (len(QG_conn_sim_graph.sim_graph.nodes()) <= MAX_DRAW_PLOT_NODE_NUM) and \
     (len(BA_conn_sim_graph.sim_graph.nodes()) <= MAX_DRAW_PLOT_NODE_NUM):

    plain_graph_plot_file_name = 'plain-sim-graph-min_degr-%d-%s-%s.eps' % \
                                 (graph_node_min_degr,
                                  plain_base_data_set_name, today_str)
    encode_graph_plot_file_name = 'encode-sim-graph-min_degr-%d-%s-%s.eps' % \
                                  (graph_node_min_degr,
                                   encode_base_data_set_name, today_str)

    plot_save_graph(QG_conn_sim_graph.sim_graph, plain_graph_plot_file_name,
                    min_sim)
    plot_save_graph(BA_conn_sim_graph.sim_graph, encode_graph_plot_file_name,
                    min_sim)
    print

  # Loop over lists with different similarities to be used to generate the
  # node features
  #
  for (graph_sim_list_name, sim_list) in graph_sim_list:

    # Loop over different sets of node features
    #
    for (graph_feat_list_name, graph_feat_list) in graph_feat_list_list:

      start_time = time.time()
      
      
      plain_graph_feat_matrix_file_name = plain_str + '-%d-%s-%s-%s.csv' %(max_node_degree,
                                          graph_node_min_degr_name, graph_feat_list_name, 
                                          graph_sim_list_name)
      encode_graph_feat_matrix_file_name = encode_str + '-%d-%s-%s-%s.csv' %(max_node_degree,
                                          graph_node_min_degr_name, graph_feat_list_name, 
                                          graph_sim_list_name)
      
      feat_path = 'feats/'

      plain_graph_feat_matrix_file_name = feat_path + plain_graph_feat_matrix_file_name
      encode_graph_feat_matrix_file_name = feat_path + encode_graph_feat_matrix_file_name
      
      print encode_graph_feat_matrix_file_name
      print plain_graph_feat_matrix_file_name
      print
      
      if(graph_feat_list[0] == 'node2vec'):
        
        directed_flag = False # define if the graph is directed or not
        
        # A hyperparamater for determining the likelihood of immediate revisiting
        p_param = 0.9
        
        # A hyperparamater for determining the inward/outward walks
        q_param = 2 
        
        num_walks = 20 # Number of walks per node
        walk_length = 30 # The length of a single walk
        
        # Word2vec parameters
        num_dimensions = 30 # Dimensions of an embedding
        window_size    = 10
        num_workers    = 5
        num_iter       = 10
      
      
      if(os.path.isfile(plain_graph_feat_matrix_file_name)):
        print 'Load plain-text graph feature matrices from csv files:'
        print '  Plain-text graph feature matrix file:', plain_graph_feat_matrix_file_name
        
        qg_feat_dict, qg_min_feat_array, qg_max_feat_array, \
                   qg_feat_name_list = \
                   load_feat_matrix_file(plain_graph_feat_matrix_file_name)
      else:
        # Generate the feature arrays for each node in the two graphs
        #
        if(graph_feat_list[0] == 'node2vec'):
          print 'Calculating node embeddings for q-gram graph using Node2Vec'
          # Defining the node2vec graph
          node2vec_qg_G = node2vec.Graph(QG_conn_sim_graph.sim_graph, directed_flag, p_param, q_param)
          
          # Calculating the transitional probabilities
          node2vec_qg_G.preprocess_transition_probs()
          
          print '  Calculating random walks'
          # Calculating the random walks
          qg_walks = node2vec_qg_G.simulate_walks(num_walks, walk_length)
          
          print '  Learn embeddings using random walks'
          # Convert the random walks to feature embeddings using word2vec skipgram
          # model
          qg_embd = learn_embeddings(qg_walks, num_dimensions, window_size, num_workers, num_iter)
          
          qg_feat_name_list = ['dimension-%d' %i for i in range(1, num_dimensions+1)]
          
          # Store enbeddings inside arrays to write them into a csv file
          #
          qg_feat_dict = {}
          qg_min_feat_array = numpy.ones(num_dimensions)
          qg_max_feat_array = numpy.zeros(num_dimensions)
          
          for qg_node_key in QG_conn_sim_graph.sim_graph.nodes():
            node_embd_array = qg_embd[str(qg_node_key)]
            qg_feat_dict[qg_node_key] = node_embd_array
            
            qg_min_feat_array = numpy.minimum(node_embd_array, qg_min_feat_array)
            qg_max_feat_array = numpy.maximum(node_embd_array, qg_max_feat_array)
          
        else:
          qg_feat_dict, qg_min_feat_array, qg_max_feat_array, \
                     qg_feat_name_list = \
                              QG_conn_sim_graph.calc_features(graph_feat_list,
                                                              sim_list,
                                                              graph_node_min_degr,
                                                              max_node_degree)
        
        # Before normalisation write feature values to a csv file
        # for future reference because generating features is very
        # time consuming
        write_feat_matrix_file(qg_feat_dict, qg_feat_name_list, \
                               plain_graph_feat_matrix_file_name)
      
      if(os.path.isfile(encode_graph_feat_matrix_file_name)):
        print 'Load encoded graph feature matrices from csv files:'
        print '  Encoded graph feature matrix file:   ', encode_graph_feat_matrix_file_name
        
        ba_feat_dict, ba_min_feat_array, ba_max_feat_array, \
                   ba_feat_name_list = \
                   load_feat_matrix_file(encode_graph_feat_matrix_file_name)  
      
      else:
        
        if(graph_feat_list[0] == 'node2vec'):
          print 'Calculating node embeddings for bit array graph using Node2Vec'
          # Defining the node2vec graph
          node2vec_ba_G = node2vec.Graph(BA_conn_sim_graph.sim_graph, directed_flag, p_param, q_param)
        
          # Calculating the transitional probabilities
          node2vec_ba_G.preprocess_transition_probs()
          
          print '  Calculating random walks'
          # Calculating the random walks
          ba_walks = node2vec_ba_G.simulate_walks(num_walks, walk_length)
          
          print '  Learn embeddings using random walks'
          # Convert the random walks to feature embeddings using word2vec skipgram
          # model
          ba_embd = learn_embeddings(ba_walks, num_dimensions, window_size, num_workers, num_iter)
          
          ba_feat_name_list = ['dimension-%d' %i for i in range(1, num_dimensions+1)]
          
          ba_feat_dict = {}
          ba_min_feat_array = numpy.ones(num_dimensions)
          ba_max_feat_array = numpy.zeros(num_dimensions)
          
          for ba_node_key in BA_conn_sim_graph.sim_graph.nodes():
            node_embd_array = ba_embd[str(ba_node_key)]
            ba_feat_dict[ba_node_key] = node_embd_array
            
            ba_min_feat_array = numpy.minimum(node_embd_array, ba_min_feat_array)
            ba_max_feat_array = numpy.maximum(node_embd_array, ba_max_feat_array)
          
        else:
          ba_feat_dict, ba_min_feat_array, ba_max_feat_array, \
                     ba_feat_name_list = \
                              BA_conn_sim_graph.calc_features(graph_feat_list,
                                                              sim_list,
                                                              graph_node_min_degr,
                                                              max_node_degree)
        
        write_feat_matrix_file(ba_feat_dict, ba_feat_name_list, \
                               encode_graph_feat_matrix_file_name)
  
      assert qg_feat_name_list == ba_feat_name_list  # Must be the same
      
      # Print all features and their minimum and maximum values
      #
      print 'Features and their minimum and maximum values ' + \
            '(for the plain-text / encoded data sets):'
      for (i,feat_name) in enumerate(qg_feat_name_list):
        print '  %22s: %.3f / %.3f  |  %.3f / %.3f' % \
              (feat_name, qg_min_feat_array[i], qg_max_feat_array[i],
               ba_min_feat_array[i], ba_max_feat_array[i])
      print
      
      calc_feat_uniq_ratio = True 
      
      if(calc_feat_uniq_ratio):
        
        qg_uniq_ratio_dict = calc_uniq_ratio(qg_feat_dict, qg_feat_name_list)
        ba_uniq_ratio_dict = calc_uniq_ratio(ba_feat_dict, ba_feat_name_list)
        
        print 'Features and their uniqueness ratio values ' + \
              '(for the plain-text / encoded data sets):'
        for feat_name in qg_feat_name_list:
          print '  %22s: %.5f  |  %.5f' % \
              (feat_name, qg_uniq_ratio_dict[feat_name], 
               ba_uniq_ratio_dict[feat_name])
      
      # Conduct an overall normalisation of features
      #
      num_feat = len(qg_min_feat_array)
      assert num_feat == len(ba_min_feat_array)

      overall_min_feat_array = numpy.minimum(qg_min_feat_array,
                                             ba_min_feat_array)
      overall_max_feat_array = numpy.maximum(qg_max_feat_array,
                                             ba_max_feat_array)
      
      normalise_feat = True
      
      if(normalise_feat):
        norm_qg_feat_dict = \
                QG_conn_sim_graph.norm_features_01(qg_feat_dict,
                                                   overall_min_feat_array,
                                                   overall_max_feat_array)
        norm_ba_feat_dict = \
                BA_conn_sim_graph.norm_features_01(ba_feat_dict,
                                                   overall_min_feat_array,
                                                   overall_max_feat_array)
                
        feat_gen_time = time.time() - start_time
        
        print 'Time for generating and normalising the features to be used:' \
              + ' %.2f sec' % (feat_gen_time)
        print
      else:  
        norm_qg_feat_dict = qg_feat_dict
        norm_ba_feat_dict = ba_feat_dict
      

      for (graph_feat_select_name, graph_feat_select) in \
                                                graph_feat_select_list:

        # Get standard deviations of features and select final set of
        # features to use
        #
        if (graph_feat_select == 'all'):
          use_feat_set = set(range(num_feat))  # All features

        else:  # Calculate standard deviations of all features
          use_feat_set = set()

          qg_feat_std_list = \
               QG_conn_sim_graph.get_std_features(norm_qg_feat_dict, num_feat)
          ba_feat_std_list = \
               BA_conn_sim_graph.get_std_features(norm_ba_feat_dict, num_feat)

          comb_feature_list = qg_feat_std_list+ba_feat_std_list

          if (graph_feat_select == 'nonzero'):
            for (feat_num, feat_std) in comb_feature_list:
              if (feat_std > 0.0):
                use_feat_set.add(feat_num)

          elif (isinstance(graph_feat_select, int)):

            # Combine and sort the two feature lists, then take the top
            # number of unique features
            #
            sorted_feat_list = sorted(comb_feature_list, key = lambda t:t[1],
                                      reverse = True)
            i = 0
            while (len(use_feat_set) < graph_feat_select) and (i < num_feat):
              use_feat_set.add(sorted_feat_list[i][0])
              i += 1

          elif (isinstance(graph_feat_select, float)):
            for (feat_num, feat_std) in comb_feature_list:
              if (feat_std >= graph_feat_select):
                use_feat_set.add(feat_num)

          else:
            raise Exception, 'This should not happen'

        use_num_feat = len(use_feat_set)

        print 'Use the following %d selected features (feature mode: %s):' \
              % (use_num_feat, str(graph_feat_select))
        if  (graph_feat_select == 'all'):
          print '  All features'
        else:
          for (feat_num, feat_name) in enumerate(qg_feat_name_list):
            if (feat_num in use_feat_set):
              gq_feat_std = qg_feat_std_list[feat_num][1]
              ba_feat_std = ba_feat_std_list[feat_num][1]
              print '  %2d: %22s (Q-gram std=%.3f / BA std=%.3f)' % \
                    (feat_num, feat_name, gq_feat_std, ba_feat_std)
        print

        sel_qg_feat_dict = \
               QG_conn_sim_graph.select_features(norm_qg_feat_dict,
                                                 use_feat_set, num_feat)
        sel_ba_feat_dict = \
               BA_conn_sim_graph.select_features(norm_ba_feat_dict,
                                                 use_feat_set, num_feat)

        # Remove all feature vectors that occur more than once because these
        # correspond to more than one node in the graph and thus do not allow
        # one-to-one matching
        #
        qg_num_org_feat_vector = len(qg_feat_dict)
        ba_num_org_feat_vector = len(ba_feat_dict)

        #sel_qg_feat_dict = \
        #       QG_conn_sim_graph.remove_freq_feat_vectors(sel_qg_feat_dict, 1)
        #sel_ba_feat_dict = \
        #       BA_conn_sim_graph.remove_freq_feat_vectors(sel_ba_feat_dict, 1)

        print '#### Reduced numbers of nodes in feature vectors: ' + \
              'QG: %d -> %d / BA: %d -> %d' % (qg_num_org_feat_vector,
              len(sel_qg_feat_dict), ba_num_org_feat_vector,
              len(sel_ba_feat_dict))
        print

        plain_sim_graph =  QG_conn_sim_graph.sim_graph  # Short cuts
        encode_sim_graph = BA_conn_sim_graph.sim_graph

        # Compare the true matching nodes based on their feature vectors as
        # well as randomly selected pairs of non-matching nodes and generate
        # a similarity histogram plot
        #
        feat_histo_plot_file_name = 'sim-feat-histograms-%s-%s-%s-%d-%s-%d-' \
                         % (plain_base_data_set_name, encode_base_data_set_name,
                            plain_attr_list_str, plain_num_rec_loaded,
                            encode_attr_list_str, encode_num_rec_loaded) + \
                           '%d-%s-%s-%s-%s-%.3f-%s-%s-%s-%d.eps' % \
                           (q, str(padded_flag).lower(), plain_sim_funct_name,
                            str(sim_diff_adjust_flag).lower(),
                            encode_sim_funct_name, min_sim, encode_method_str,
                            today_str, graph_feat_select, use_num_feat)

        calc_plot_histogram_feat_vec_cosine_sim(plain_sim_graph,
                                                encode_sim_graph,
                                                sel_qg_feat_dict,
                                                sel_ba_feat_dict,
                                                feat_histo_plot_file_name,
                                                3, True, 10)

        # ---------------------------------------------------------------------
        # Step 5: Find similar nodes accross the two graphs using SimHash
        #
        for (sim_hash_num_bit_name, sim_hash_num_bit) in \
                                              sim_hash_num_bit_list:
          start_time = time.time()

          Graph_sim_hash = CosineLSH(use_num_feat, sim_hash_num_bit,
                                     random_seed)

          qg_sim_hash_dict = Graph_sim_hash.gen_sim_hash(sel_qg_feat_dict)
          ba_sim_hash_dict = Graph_sim_hash.gen_sim_hash(sel_ba_feat_dict)

          hash_sim_gen_time = time.time() - start_time

          # Loop over the different blocking techniques for the similarity
          # hash dictionaries
          #
          for sim_hash_block_funct in sim_hash_block_funct_list:

            start_time = time.time()

            if (sim_hash_block_funct == 'all_in_one'):
              plain_sim_block_dict, encode_sim_block_dict = \
                      Graph_sim_hash.all_in_one_blocking(qg_sim_hash_dict,
                                                         ba_sim_hash_dict)

            elif (sim_hash_block_funct == 'hlsh'):

              graph_block_hlsh_sample_size = sim_hash_num_bit / \
                                             graph_block_hlsh_rel_sample_size

              plain_sim_block_dict, encode_sim_block_dict = \
                    Graph_sim_hash.hlsh_blocking(qg_sim_hash_dict,
                                                 ba_sim_hash_dict,
                                                 graph_block_hlsh_sample_size,
                                                 graph_block_hlsh_num_sample,
                                                 sim_hash_num_bit,
                                                 random_seed)

            else:
              raise Exception, 'This should not happen'

            hash_sim_block_time = time.time() - start_time

            # Loop over the different comparison hash similarity functions
            #
            for sim_comp_funct in sim_comp_funct_list:

              start_time = time.time()

              if (sim_comp_funct == 'allpairs'):

                # Compare all pairs of nodes between the two graphs
                #
                hash_sim_dict = \
                  Graph_sim_hash.calc_cos_sim_all_pairs(qg_sim_hash_dict,
                                                        ba_sim_hash_dict,
                                                        plain_sim_block_dict,
                                                        encode_sim_block_dict,
                                                        sim_hash_match_sim)

              elif (sim_comp_funct == 'simtol'):

                # Adjust the similarity tolerance based on the encoding used,
                # and also if similarity adjustment has been applied or not
                #
                if (sim_diff_adjust_flag == True):
                  hash_sim_low_tol = adj_hash_sim_low_tol
                  hash_sim_up_tol =  adj_hash_sim_up_tol
                elif (encode_method == 'bf'):
                  hash_sim_low_tol = bf_hash_sim_low_tol
                  hash_sim_up_tol =  bf_hash_sim_up_tol
                elif (encode_method == 'tmh'):
                  hash_sim_low_tol = tmh_hash_sim_low_tol
                  hash_sim_up_tol =  tmh_hash_sim_up_tol
                elif (encode_method == '2sh'):
                  hash_sim_low_tol = cmh_hash_sim_low_tol
                  hash_sim_up_tol =  cmh_hash_sim_up_tol
                else:
                  raise Exception, 'This should not happen'

                hash_sim_dict = \
                  Graph_sim_hash.calc_cos_sim_edge_comp(qg_sim_hash_dict,
                                                        ba_sim_hash_dict,
                                                        plain_sim_block_dict,
                                                        encode_sim_block_dict,
                                                        sim_hash_match_sim,
                                                        plain_sim_graph,
                                                        encode_sim_graph,
                                                        hash_sim_low_tol,
                                                        hash_sim_up_tol,
                                                        min_sim,
                                                        hash_sim_min_tol)

              else:
                raise Exception, 'This should not happen'

              hash_sim_comp_time = time.time() - start_time
              
              num_sim_pair = len(hash_sim_dict)

              print 'Identified %d node pairs across data sets with a Cosine' \
                    % (num_sim_pair) + ' LSH similarity of at least %.2f' % \
                    (sim_hash_match_sim)
              print

              # Calculate the final Cosine similarity for each pair and sort
              #
              cos_sim_dict = {}
              
              # A dictionary to connect all similarities for each node
              #
              connect_sim_dict = {}
              connect_qg_sim_dict = {}
              connect_ba_sim_dict = {}
              
              enc_key_val_set = set()
              
              # A similiarity graph to connect all encoded records with plain records
              #
              feat_sim_graph = SimGraph()  # Initialise the graph
              
              cos_sim_list = []
              
              start_time = time.time()

              for (key_val1, key_val2) in hash_sim_dict.iterkeys():
                feat_vec1 = sel_qg_feat_dict[key_val1]
                feat_vec2 = sel_ba_feat_dict[key_val2]

                # Calculate the Cosine similarity beween the featue vectors
                #
                vec_len1 = math.sqrt(numpy.dot(feat_vec1, feat_vec1))
                vec_len2 = math.sqrt(numpy.dot(feat_vec2, feat_vec2))
                cosine = numpy.dot(feat_vec1,feat_vec2) / vec_len1 / vec_len2
                cos_sim = 1.0 - math.acos(min(cosine,1.0)) / math.pi

                cos_sim_dict[(key_val1,key_val2)] = cos_sim
                
                cos_sim_list.append(cos_sim)
                
                enc_key_val_set.add(key_val2)
                
                
                # Update connected componets in encode dictionary
                qg_sim_dict = connect_ba_sim_dict.get(key_val2, {})
                
                if(key_val1 in qg_sim_dict):
                  assert qg_sim_dict[key_val1] == cos_sim, (qg_sim_dict[key_val1],cos_sim)
                else:
                  qg_sim_dict[key_val1] = cos_sim
                connect_ba_sim_dict[key_val2] = qg_sim_dict
                
                # Update connected componets in plain-text dictionary
                ba_sim_dict = connect_qg_sim_dict.get(key_val1, {})
                
                if(key_val2 in ba_sim_dict):
                  assert ba_sim_dict[key_val2] == cos_sim, (ba_sim_dict[key_val2],cos_sim)
                else:
                  ba_sim_dict[key_val2] = cos_sim
                connect_qg_sim_dict[key_val1] = ba_sim_dict
                
 
              sim_pair_list = sorted(cos_sim_dict.items(), key=lambda t: t[1],
                                     reverse = True)

              sim_cal_time = time.time() - start_time
              
              print '    Time for identifying %d similar node pairs: %.2f' \
                    % (num_sim_pair, sim_cal_time) + ' sec'
              print
              
              # Calculation of confidence values per edge
              #
              start_time = time.time()
              
              max_sim_val = max(cos_sim_list)
              min_sim_val = min(cos_sim_list)
              
              max_poss_sim_conf_val = max_sim_val/min_sim_val
              sim_conf_const = max_poss_sim_conf_val + 0.1
              
              degree_conf_val_list = []
              sim_conf_val_list = []
              
              edge_sim_conf_dict = {}
              
              for qg_key, ba_key in cos_sim_dict.keys():
                cos_sim = cos_sim_dict[(qg_key,ba_key)]
                
                qg_neigh_ba_dict = connect_qg_sim_dict[qg_key]
                ba_neigh_qg_dict = connect_ba_sim_dict[ba_key]
                
                sim_conf = calc_sim_conf_val(ba_key, qg_key, cos_sim, ba_neigh_qg_dict, qg_neigh_ba_dict)
                if(sim_conf == 10):
                  #sim_conf = sim_conf_const
                  sim_conf = cos_sim/(min_sim_val - 0.1)
                
                degree_conf = calc_degree_conf_val(ba_neigh_qg_dict, qg_neigh_ba_dict)
                
                sim_conf_val_list.append(sim_conf)
                degree_conf_val_list.append(degree_conf)
                
                edge_sim_conf_dict[(qg_key,ba_key)] = [cos_sim, sim_conf, degree_conf]
                
                
              print ' Similarity confidence distribution'
              print '  Min/Max/Average/Std: %.3f/%.3f/%.3f/%.3f' %(min(sim_conf_val_list), max(sim_conf_val_list), 
                                                                   numpy.mean(sim_conf_val_list), numpy.std(sim_conf_val_list))
              print
              
              print ' Degree confidence distribution'
              print '  Min/Max/Average/Std: %.3f/%.3f/%.3f/%.3f' %(min(degree_conf_val_list), max(degree_conf_val_list), 
                                                                   numpy.mean(degree_conf_val_list), numpy.std(degree_conf_val_list))
              print
              
              # Normalise confidence values
              #
              max_sim_conf = max(sim_conf_val_list)
              min_sim_conf = min(sim_conf_val_list)
              sim_conf_diff = max_sim_conf - min_sim_conf
              
              max_degree_conf = max(degree_conf_val_list)
              min_degree_conf = min(degree_conf_val_list)
              degree_conf_diff = max_degree_conf - min_degree_conf
              
              degree_conf_norm_val_list = []
              sim_conf_norm_val_list = []
              
              edge_sim_conf_norm_dict = {}
              
              ba_conn_comp_dict = {}
              qg_conn_comp_dict = {}
              
              for key_tuple in edge_sim_conf_dict.keys():
                
                sim_conf_val_list = edge_sim_conf_dict[key_tuple]
                
                cos_sim = sim_conf_val_list[0]
                sim_conf = sim_conf_val_list[1]
                degree_conf = sim_conf_val_list[2]
                
                norm_sim_conf = (sim_conf - min_sim_conf)/sim_conf_diff
                norm_degree_conf = (degree_conf - min_degree_conf)/degree_conf_diff
                
                degree_conf_norm_val_list.append(norm_degree_conf)
                sim_conf_norm_val_list.append(norm_sim_conf)
                
                edge_sim_conf_norm_dict[key_tuple] = [cos_sim, norm_sim_conf, norm_degree_conf]
                
                qg_key = key_tuple[0]
                ba_key = key_tuple[1]
                # Update connected componets in encode dictionary
                qg_sim_dict = ba_conn_comp_dict.get(ba_key, {})
                
                if(qg_key in qg_sim_dict):
                  assert qg_sim_dict[qg_key] == [cos_sim, norm_sim_conf, norm_degree_conf]
                else:
                  qg_sim_dict[qg_key] = [cos_sim, norm_sim_conf, norm_degree_conf]
                ba_conn_comp_dict[ba_key] = qg_sim_dict
                
                # Update connected componets in plain-text dictionary
                ba_sim_dict = qg_conn_comp_dict.get(qg_key, {})
                
                if(ba_key in ba_sim_dict):
                  assert ba_sim_dict[ba_key] == [cos_sim, norm_sim_conf, norm_degree_conf]
                else:
                  ba_sim_dict[ba_key] = [cos_sim, norm_sim_conf, norm_degree_conf]
                qg_conn_comp_dict[qg_key] = ba_sim_dict
              
              
              print ' Normalised similarity confidence distribution'
              print '  Min/Max/Average/Std: %.3f/%.3f/%.3f/%.3f' %(min(sim_conf_norm_val_list), 
                                                                   max(sim_conf_norm_val_list), 
                                                                   numpy.mean(sim_conf_norm_val_list), 
                                                                   numpy.std(sim_conf_norm_val_list))
              print
              
              print ' Normalised degree confidence distribution'
              print '  Min/Max/Average/Std: %.3f/%.3f/%.3f/%.3f' %(min(degree_conf_norm_val_list), 
                                                                   max(degree_conf_norm_val_list), 
                                                                   numpy.mean(degree_conf_norm_val_list), 
                                                                   numpy.std(degree_conf_norm_val_list))  
              
              conf_cal_time = time.time() - start_time
              
              print
              print '  Time used for calculating confidence values: %.3f seconds' %conf_cal_time
              
              
              # Assignment of plain-text nodes to encode nodes based on 
              # calculated cosine similarity can be done using multiple 
              # ways. This problem of assigining nodes can be viwed as
              # a bipartite graph matching problem where one node from
              # one graph can be linked to multiple nodes from the other
              # graph and vise versa. 
              #
              #    1.Symmetric highest match approach where only the edges with
              #      highest similarities for both of the nodes are chosen. 
              #      The highest similarity matching therfore is symmetric.
              #    2.Hungarian/ Kuhn-Munkres algorithm of finding the
              #      minimum cost based assignments for a given connected
              #      component.
              #    3.Stable marriage problem (Gale Shapley algorithm) to 
              #      find best stable edges throughout the graph.
              #    4.Randomly select edges between two sets of nodes.
              
              match_method_list = ['symmetric', 'hungarian', 'shapley', 'random']
              
              print 'Apply defined matching technique/s to find similar node pairs (node matching)'
              print '  Matching method list:', match_method_list
              
              for match_method in match_method_list:
                
                if(match_method == 'random'):
                  list_weight_lists = [[0.0, 0.0, 0.0]]
                else:
                  # Defined weight list for similarity and confidence values.
                  # structure: [cosine_sim_weight, sim_confident_weight, degree_confident_weight]
                  # 
                  list_weight_lists = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 
                                       [0.0, 0.0, 1.0], [0.6, 0.3, 0.1]]
                
                for weight_list in list_weight_lists:
                
                  match_start_time = time.time()
                  
                  if(match_method == 'random'):
                    print '  Running Random approach to find matches'
                    matched_pair_dict = random_graph_matching(edge_sim_conf_norm_dict)
                    
                  elif(match_method == 'symmetric'):
                    print '  Running Symmetric best match approach to find matches'
                    matched_pair_dict = symmetric_graph_matching(edge_sim_conf_norm_dict, 
                                                                 ba_conn_comp_dict, 
                                                                 qg_conn_comp_dict, 
                                                                 weight_list)
                    
                  elif(match_method == 'hungarian'):
                    print
                    print '  Running Hungarian algorithm to find matches'
                    print '    Weight list: ', weight_list
                    print
                    
                    # Lets user to input the similarity threshold so that certain
                    # size of connected components can be selected
                    #
                    user_input_flag = False
                    
                    if(user_input_flag):
                      
                      conn_comp_list = get_conn_comp_user_input(ba_conn_comp_dict, 
                                                                qg_conn_comp_dict,
                                                                weight_list)
                      
                    else: # Run interploation search to find the optimal threshold
                      
                      # Depending on the computational power user has,
                      # user can change the following two parameters accordingly
                      
                      # Lower limit of the accepted connected component size
                      cc_size_range_l = 2000 
                      
                      # Higher limit of the accepted connected component size
                      cc_size_range_h = 2100
                      
                      conn_comp_list = get_conn_comp_intep_search(ba_conn_comp_dict, 
                                                                  qg_conn_comp_dict,
                                                                  weight_list,
                                                                  cc_size_range_l,
                                                                  cc_size_range_h)
                    
                    matched_pair_dict = munkress_graph_matching(conn_comp_list, 
                                                                edge_sim_conf_norm_dict, 
                                                                weight_list)
                  
                  elif(match_method == 'shapley'):
                    print '  Running Gale-Shapley algorithm to find matches'
                    # In our context both of the below algorithms provide same
                    # results. However, greedy approach is faster.
                    
                    #matched_pair_dict1 = shapley_graph_matching(ba_conn_comp_dict, 
                    #                                           qg_conn_comp_dict, 
                    #                                           edge_sim_conf_norm_dict,
                    #                                           weight_list)
                    
                    matched_pair_dict = greedy_graph_matching(edge_sim_conf_norm_dict, weight_list)
                    
                    #assert matched_pair_dict == matched_pair_dict1
                    
                  else:
                    raise Exception, '***Warning! Wrong matching method...'
              
                  match_time = time.time() - match_start_time
                  
                  print '  Time used for graph matching: %.3f seconds' %match_time 
                  print 
                  print
                  
                  reident_start_time = time.time()

              
                  matched_sim_pair_list = sorted(matched_pair_dict.items(), key=lambda t: t[1],
                                                 reverse = True)   
    
                  # A list with True and False values, with True if the record
                  # values are the same and false otherwise, and a second list
                  # to see if actual values are highly similar or not (based on
                  # the sim_hash_match_sim)
                  #
                  corr_wrong_match_list = []
                  sim_diff_match_list =   []
    
                  true_match_rank_list = []  # Ranks of all true matches 
                  
                  ident_corr_id_set = set()
                  ident_wrng_id_set = set()
    
                  rank_i = 0
                  #for (val_pair, final_sim) in sim_pair_list:
                  for (val_pair, final_sim) in matched_sim_pair_list:   
                    node_key_val1, node_key_val2 = val_pair
                    ent_id_set1 = plain_sim_graph.node[node_key_val1]['ent_id_set']
                    org_val_set1 = \
                                 plain_sim_graph.node[node_key_val1]['org_val_set']
                    ent_id_set2 = \
                                 encode_sim_graph.node[node_key_val2]['ent_id_set']
                    org_val_set2 = \
                                encode_sim_graph.node[node_key_val2]['org_val_set']
                    
                    
                    if (rank_i < 5):  # Only print the first 10 pairs
    
                      print '      Node pair with %d highest similarity (or ' % \
                            (rank_i) + 'distance): %.5f' % (final_sim)
                      print '        Plain-text entity identifiers:', \
                            sorted(ent_id_set1)
                      print '          Plain-text record values:', \
                            sorted(org_val_set1)
                      print '        Encoded entity identifiers:', \
                            sorted(ent_id_set2)
                      print '          Encoded record values:', \
                            sorted(org_val_set2)
                      print '      Original feature vectors:'
                      print '       ', qg_feat_dict[node_key_val1]
                      print '       ', ba_feat_dict[node_key_val2]
                      print '      Original normalised feature vectors:'
                      print '       ', norm_qg_feat_dict[node_key_val1]
                      print '       ', norm_ba_feat_dict[node_key_val2]
                      print '      Selected feature vectors:'
                      print '       ', sel_qg_feat_dict[node_key_val1]
                      print '       ', sel_ba_feat_dict[node_key_val2]
                      print
    
                    # Get the first identifiers from these two sets
                    #
                    ent_id1 = sorted(ent_id_set1)[0]
                    ent_id2 = sorted(ent_id_set2)[0]
                    
                    rec_val1 = plain_rec_attr_val_dict[ent_id1]
                    rec_val2 = encode_rec_attr_val_dict[ent_id2]
    
                    if (rec_val1 == rec_val2):
                      corr_wrong_match_list.append(True)
                      true_match_rank_list.append(rank_i)
                      
                      ident_corr_id_set = ident_corr_id_set.union(ent_id_set2)
    
                      if (corr_wrong_match_list.count(True) < 5):
                        print '    *** True match at rank %d: (%s,%s), ' % \
                              (rank_i, ent_id1, ent_id2) + ' sim/dist = %.5f' \
                              % (final_sim)
                        print '        Encoded / plain-text value pair: %s / %s' \
                              % (rec_val1, rec_val2)
                        print '        Position: %d' % \
                              (len(corr_wrong_match_list))
                        print
    
                    else:
                      ident_wrng_id_set = ident_wrng_id_set.union(ent_id_set2)
                      corr_wrong_match_list.append(False)
    
                    # Calculate the similarity between the q-gram sets of the two
                    # records
                    #
                    encode_q_gram_set = plain_q_gram_dict[ent_id1]
                    plain_q_gram_set =  encode_q_gram_dict[ent_id2]
    
                    jacc_sim = float(len(encode_q_gram_set & \
                                         plain_q_gram_set)) / \
                                     len(encode_q_gram_set | plain_q_gram_set)
                    if (jacc_sim >= sim_hash_match_sim):
                      sim_diff_match_list.append(True)
                    else:
                      sim_diff_match_list.append(False)
    
                    rank_i += 1
                  
                  print '  Number of encode records identified correctly: %d ' %len(ident_corr_id_set)
                  print '  Number of encode records identified wrongly: %d' %len(ident_wrng_id_set)
                  print 
                  
                  print '  Ranks of %d true matches:' % \
                        (len(true_match_rank_list)), true_match_rank_list[:10], \
                        '...', true_match_rank_list[-10:]
                  print
    
                  print '  Matching accuracy (based on true matches):'
    
                  corr_top_10 =  corr_wrong_match_list[:10].count(True)
                  corr_top_20 =  corr_wrong_match_list[:20].count(True)
                  corr_top_50 =  corr_wrong_match_list[:50].count(True)
                  corr_top_100 = corr_wrong_match_list[:100].count(True)
                  corr_top_200 = corr_wrong_match_list[:200].count(True)
                  corr_top_500 = corr_wrong_match_list[:500].count(True)
                  corr_top_1000 = corr_wrong_match_list[:1000].count(True)
                  corr_all =     corr_wrong_match_list.count(True)
                  wrng_all =     corr_wrong_match_list.count(False)
    
                  print '    Top  10: %4d (%.2f%%)' % (corr_top_10,
                                                       10.0* corr_top_10)
                  print '    Top  20: %4d (%.2f%%)' % (corr_top_20,
                                                       5.0* corr_top_20)
                  print '    Top  50: %4d (%.2f%%)' % (corr_top_50,
                                                       2.0* corr_top_50)
                  print '    Top 100: %4d (%.2f%%)' % (corr_top_100,
                                                       float(corr_top_100))
                  print '    Top 200: %4d (%.2f%%)' % (corr_top_200,
                                                       float(corr_top_200)/2.0)
                  print '    Top 500: %4d (%.2f%%)' % (corr_top_500,
                                                       float(corr_top_500)/5.0)
                  print '    Top 1000: %4d (%.2f%%)' % (corr_top_1000,
                                                       float(corr_top_1000)/10.0)
                  if (len(corr_wrong_match_list) > 0):
                    print '    All:     %4d (%.2f%%) from %d' % (corr_all,
                                                     100.0*float(corr_all) / \
                                                     len(corr_wrong_match_list),
                                                     len(corr_wrong_match_list))
    
                  print '  Matching accuracy (based on Jaccard value ' + \
                        'similarity of at least %.3f):' % (sim_hash_match_sim)
    
                  scorr_top_10 =  sim_diff_match_list[:10].count(True)
                  scorr_top_20 =  sim_diff_match_list[:20].count(True)
                  scorr_top_50 =  sim_diff_match_list[:50].count(True)
                  scorr_top_100 = sim_diff_match_list[:100].count(True)
                  scorr_top_200 = sim_diff_match_list[:200].count(True)
                  scorr_top_500 = sim_diff_match_list[:500].count(True)
                  scorr_top_1000 = sim_diff_match_list[:1000].count(True)
                  scorr_all =     sim_diff_match_list.count(True)
                  swrng_all =     sim_diff_match_list.count(False)
    
                  print '    Top  10: %4d (%.2f%%)' % (scorr_top_10,
                                                       10.0* scorr_top_10)
                  print '    Top  20: %4d (%.2f%%)' % (scorr_top_20,
                                                       5.0* scorr_top_20)
                  print '    Top  50: %4d (%.2f%%)' % (scorr_top_50,
                                                       2.0* scorr_top_50)
                  print '    Top 100: %4d (%.2f%%)' % (scorr_top_100,
                                                       float(scorr_top_100))
                  print '    Top 200: %4d (%.2f%%)' % (scorr_top_200,
                                                       float(scorr_top_200)/2.0)
                  print '    Top 500: %4d (%.2f%%)' % (scorr_top_500,
                                                       float(scorr_top_500)/5.0)
                  print '    Top 1000: %4d (%.2f%%)' % (scorr_top_1000,
                                                       float(scorr_top_1000)/10.0)
                  if (len(sim_diff_match_list) > 0):
                    print '    All:     %4d (%.2f%%) from %d' % (scorr_all,
                                                     100.0*float(scorr_all) / \
                                                     len(sim_diff_match_list),
                                                     len(sim_diff_match_list))
                    print
    
                  reident_time = time.time() - reident_start_time
                  
                  # Write evaluation resuls to a csv file
                  #
                  extra_res_header = ['graph_node_min_degree','graph_sim_list','graph_feat_list',
                                      'graph_selected_feat','sim_hash_num_bit','sim_hash_block_funct',
                                      'sim_comp_funct','sim_hash_match_sim','num_feat','use_num_feat',
                                      'total_num_sim_edges','match_method','weight_list',
                                      'num_corr_ident_rec','num_wrng_ident_rec','corr_top_10',
                                      'corr_top_20','corr_top_50','corr_top_100','corr_top_200',
                                      'corr_top_500','corr_top_1000','num_all_corr','num_all_wrng',
                                      'scorr_top_10','scorr_top_20','scorr_top_50','scorr_top_100',
                                      'scorr_top_200','scorr_top_500','scorr_top_1000','num_all_scorr',
                                      'num_all_swrng','sim_threshold','feat_gen_time',
                                      'hash_sim_gen_time','hash_sim_block_time','hash_sim_cal_time',
                                      'cos_sim_cal_time','conf_cal_time','node_match_time',
                                      'both_q_gram_minhash','encode_blocking','plain_blocking',
                                      'encode_num_samples(d)', 'encode_sample_size(r)',
                                      'plain_num_samples(d)', 'plain_sample_size(r)',
                                      'enc_min_sim', 'plain_min_sim','plain_graph_num_node',
                                      'encode_graph_num_node', 'hashing_time', 
                                      'plain_graph_gen_time', 'encode_graph_gen_time',
                                      'plain_graph_num_edges', 'encode_graph_num_edges',
                                      'plain_num_singleton', 'encode_num_singleton',
                                      'reident_time']
                  
                  
                  extra_res_val = [graph_node_min_degr_name, graph_sim_list_name, graph_feat_list_name,
                                   graph_feat_select_name, sim_hash_num_bit_name, sim_hash_block_funct,
                                   sim_comp_funct, sim_hash_match_sim, num_feat, use_num_feat, 
                                   num_sim_pair, match_method, weight_list, len(ident_corr_id_set), 
                                   len(ident_wrng_id_set), corr_top_10, corr_top_20, corr_top_50, 
                                   corr_top_100, corr_top_200, corr_top_500, corr_top_1000, corr_all, 
                                   wrng_all, scorr_top_10, scorr_top_20, scorr_top_50, scorr_top_100,
                                   scorr_top_200, scorr_top_500, scorr_top_1000, scorr_all, swrng_all,
                                   sim_hash_match_sim, feat_gen_time, hash_sim_gen_time, hash_sim_block_time,
                                   hash_sim_comp_time, sim_cal_time, conf_cal_time, match_time,
                                   q_gram_min_hash, encode_blck_method, plain_blck_method,
                                   enc_num_samples, enc_sample_size,
                                   plain_num_samples, plain_sample_size,
                                   enc_min_sim, min_sim, plain_graph_num_node, encode_graph_num_node,
                                   hashing_time, qg_graph_gen_time, 
                                   ba_graph_gen_time, qg_graph_num_edges, 
                                   ba_graph_num_edges, qg_graph_num_singleton, 
                                   ba_graph_num_singleton, reident_time]
                  
                  assert len(extra_res_header) == len(extra_res_val)
                  
                  new_attck_res_header_list = attck_res_header_list + extra_res_header
                  new_attck_res_val_list = attck_res_val_list + extra_res_val
                  
                  
                  if (not os.path.isfile(attack_res_file_name)):
                    attck_write_file = open(attack_res_file_name, 'w')
                    csv_writer = csv.writer(attck_write_file)
                    csv_writer.writerow(new_attck_res_header_list)
                     
                    print 'Created new result file:', attack_res_file_name
                     
                  else:  # Append results to an existing file
                    attck_write_file = open(attack_res_file_name, 'a')
                    csv_writer = csv.writer(attck_write_file)
                     
                  csv_writer.writerow(new_attck_res_val_list)
                  attck_write_file.close()
                  
                  print ' Wrote result line:', new_attck_res_val_list

# End.
