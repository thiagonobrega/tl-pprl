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

# import binascii
import csv
import gzip
# import hashlib
# import math
# import os.path
# import random
# import sys
# import time

# import bitarray
# import itertools
# import numpy
# import numpy.linalg
# import scipy.stats
# import scipy.spatial.distance
# import sklearn.tree
# import pickle

# import networkx
# import networkx.drawing

# import matplotlib
# matplotlib.use('Agg')  # For running on adamms4 (no display)
# import matplotlib.pyplot as plt

# import auxiliary
# import hashing   # Bloom filter based PPRL functions
# import encoding
# import hardening
# import tabminhash  # Tabulation min-hash PPRL functions
# import colminhash # Two step column hashing for PPRL
# import simcalc  # Similarity functions for q-grams and bit-arrays
# import node2vec # Node2vec module for generating features using random walks

import time
import math
import hashlib
import logging, sys

from utils import auxiliary
from utils.anonimization import encoding, hashing , tabminhash , colminhash

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

BF_HASH_FUNCT1 = hashlib.sha1
BF_HASH_FUNCT2 = hashlib.md5

# logging.debug('A debug message!')
# logging.info('We processed %d records', 3)

def load_data_set(data_set_name, attr_num_list, ent_id_col, soundex_attr_val_list,
                  num_rec=-1, col_sep_char=',', header_line_flag=False,
                  debug=False):
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
    header_list = next(csv_reader) # testar
    # csv_reader.next()

    logging.debug('File header line:', header_list)
    logging.debug('  Attributes to be used:')
    for attr_num in attr_num_list:
      logging.debug(header_list[attr_num])
      attr_name_list.append(header_list[attr_num])
    logging.debug('')

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
  
    for attr_num in range(max_attr_num):

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

  logging.debug('  Loaded %d records from file' % (rec_num))
  logging.debug('    Stored %d values' % (len(rec_attr_val_dict)))

  # Get the frequency distribution of values
  #
  rec_tuple_count_dict = {}
  for use_rec_val_list in rec_attr_val_dict.values():
    rec_tuple = tuple(use_rec_val_list)
    rec_tuple_count_dict[rec_tuple] = \
                       rec_tuple_count_dict.get(rec_tuple, 0) + 1
  count_dist_dict = {}
  for rec_tuple_count in rec_tuple_count_dict.values():
    count_dist_dict[rec_tuple_count] = \
                       count_dist_dict.get(rec_tuple_count, 0) + 1

  logging.debug('  Count distribution of value occurrences:', \
        sorted(count_dist_dict.items())
  )

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

  for (ent_id, attr_val_list) in rec_attr_val_dict.items():
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

### Step 2 methods

# -----------------------------------------------------------------------------

def encode_ds(encode_q_gram_dict,encode_attr_list,encode_rec_attr_val_dict,
                plain_rec_attr_val_dict,
                encode_method,bf_len,q,bf_encode,
                random_seed, # definir
                bf_hash_type='rh',bf_num_hash_funct='opt',
                padded_flag='t', #clk
                bf_enc_param='t', #rbf
                # tmh param
                tmh_hash_funct='md5',tmh_num_hash_bits=200,
                tmh_num_tables=200,tmh_key_len=200,tmh_val_len=200,
                # 2sg two hash encoding
                cmh_num_hash_funct=0,cmh_num_hash_col=0
            ):
    
    start_time = time.time()
    assert bf_hash_type in ['rh' , 'dh']
    assert bf_hash_type in ['clk' , 'rbf' , 'rbf-d']

    
    if (encode_method == 'bf'): # Bloom filter encoding

        if (bf_num_hash_funct == 'opt'):
            bf_num_hash_funct_str = 'opt' # qual o problema?

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

            logging.debug(' Set optimal number of BF hash functions to: %d' % \
                    (bf_num_hash_funct)
            )

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
            raise Exception('This should not happen')
    
    
        # Check the BF encoding method
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
            abf_len_dict = BF_Encoding.get_dynamic_abf_len(avr_num_q_gram_dict,bf_num_hash_funct)
            BF_Encoding.set_abf_len(abf_len_dict)
    

        encode_hash_dict = {}  # Generate one bit-array hash code per record

        # Keep the generated BF for each q-gram set so we only generate it once
        #
        bf_hash_cache_dict = {}

        num_enc = 0  # Count number of encodings, print progress report
    
        for (ent_id, q_gram_set) in encode_q_gram_dict.items():
            attr_val_list = encode_rec_attr_val_dict[ent_id]
            num_enc += 1
            # comentar isso qlq coisa
            if (num_enc % 10000 == 0):
                time_used = time.time() - start_time
                logging.debug('  Encoded %d of %d q-gram sets in %d sec (%.2f msec average)' \
                                % (num_enc, len(encode_q_gram_dict), time_used,
                                1000.0*time_used/num_enc)
                )
                logging.debug('   ', auxiliary.get_memory_usage())
        
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
        # fim do for

        logging.debug('')
        logging.debug('  Encoded %d unique Bloom filters for %d q-gram sets' % \
                    (len(bf_hash_cache_dict), len(encode_hash_dict))
        )

        bf_hash_cache_dict.clear()  # Not needed anymore

    # Tabulation min-hash encoding
    elif (encode_method == 'tmh'):

        if (tmh_hash_funct == 'md5'):
            tmh_hash_funct_obj = hashlib.md5
        elif (tmh_hash_funct == 'sha1'):
            tmh_hash_funct_obj = hashlib.sha1
        elif (tmh_hash_funct == 'sha2'):
            tmh_hash_funct_obj = hashlib.sha256
        else:
            raise Exception('This should not happen')

        TMH_hashing = tabminhash.TabMinHashEncoding(tmh_num_hash_bits,
                                                    tmh_num_tables, tmh_key_len,
                                                    tmh_val_len, tmh_hash_funct_obj)

        encode_hash_dict = {}  # Generate one bit-array hash code per record

        # Keep the generated BA for each q-gram set so we only generate it once
        #
        ba_hash_cache_dict = {}

        num_enc = 0  # Count number of encodings, print progress report

        for (ent_id, q_gram_set) in encode_q_gram_dict.items():
            num_enc += 1
            if (num_enc % 10000 == 0): # alterar
                time_used = time.time() - start_time
                logging.debug('  Encoded %d of %d q-gram sets in %d sec (%.2f msec average)' \
                                % (num_enc, len(encode_q_gram_dict), time_used,
                                1000.0*time_used/num_enc)
                )
                logging.debug('   ', auxiliary.get_memory_usage())

            q_gram_str = ''.join(sorted(q_gram_set))
            if (q_gram_str in ba_hash_cache_dict):
                q_gram_str_ba = ba_hash_cache_dict[q_gram_str]
            else:
                q_gram_str_ba = TMH_hashing.encode_q_gram_set(q_gram_set)
                ba_hash_cache_dict[q_gram_str] = q_gram_str_ba
            
            encode_hash_dict[ent_id] = q_gram_str_ba

        logging.debug('')
        logging.debug('  Encoded %d unique bit-arrays for %d q-gram sets' % \
            (len(ba_hash_cache_dict), len(encode_hash_dict))
        )

        ba_hash_cache_dict.clear()  # Not needed anymore

    # Two-step hash encoding
    elif(encode_method == '2sh'):

        CMH_hashing = colminhash.ColMinHashEncoding(cmh_num_hash_funct,cmh_num_hash_col)

        encode_hash_dict = {}  # Generate one column hash code per record

        # Keep the generated column hash codes for each q-gram set so we only generate it once
        #
        col_hash_cache_dict = {}

        num_enc = 0  # Count number of encodings, print progress report
    
        for (ent_id, q_gram_set) in encode_q_gram_dict.items():
      
            num_enc += 1
            if (num_enc % 10000 == 0):
                time_used = time.time() - start_time
                logging.debug('  Encoded %d of %d q-gram sets in %d sec (%.2f msec average)' \
                    % (num_enc, len(encode_q_gram_dict), time_used,
                    1000.0*time_used/num_enc)
                )
                logging.debug('   ', auxiliary.get_memory_usage())

            q_gram_str = ''.join(sorted(q_gram_set))

            if (q_gram_str in col_hash_cache_dict):
                q_gram_str_col_hash_set = col_hash_cache_dict[q_gram_str]
            else:
                q_gram_str_col_hash_set = CMH_hashing.encode_q_gram_set(q_gram_set)
                col_hash_cache_dict[q_gram_str] = q_gram_str_col_hash_set
      
            encode_hash_dict[ent_id] = q_gram_str_col_hash_set

        logging.debug('')
        logging.debug('  Encoded %d unique col hash sets for %d q-gram sets' % \
            (len(col_hash_cache_dict), len(encode_hash_dict))
        )
    
        col_hash_cache_dict.clear()  # Not needed anymore  
  
    else:
        raise Exception('This should not happen')

    #novo alinhamneto
    hashing_time = time.time() - start_time

    logging.debug('')
    logging.debug('Time for hashing the encode data set: %.2f sec' % (hashing_time) )
    logging.debug('  Number of records hashed:', len(encode_hash_dict) )
    logging.debug('')

    # Check which entity identifiers occur in both data sets

    common_ent_id_set = set(plain_rec_attr_val_dict.keys()) & \
                      set(common_num_ent.keys())
    common_num_ent = len(common_ent_id_set)

    plain_num_ent =  len(plain_rec_attr_val_dict)
    encode_num_ent = len(encode_hash_dict)

    return encode_hash_dict,hashing_time,plain_num_ent, encode_num_ent # colocar outros retornos se necessario

# -----------------------------------------------------------------------------

### Step 3 methods

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------