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

# Set the graph generation parameters depending upon data sets used
#

# import binascii
import sys
import random
import os
import os.path
import csv
import gzip
import time
import math
import hashlib
import logging

import bitarray
# import itertools
import numpy
from scipy.optimize import linear_sum_assignment
# import numpy.linalg
# import scipy.stats
# import scipy.spatial.distance
# import sklearn.tree
import pickle

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures

import networkx
# import networkx.drawing
# import node2vec
from graph_attack.utils import node2vec
from gensim.models import Word2Vec


import matplotlib
matplotlib.use('Agg')  # For running on adamms4 (no display)
import matplotlib.pyplot as plt

from graph_attack.utils import auxiliary
from graph_attack.utils.anonimization import encoding, hashing , tabminhash , colminhash
from graph_attack.utils.indexing import MinHashLSH, CosineLSH
from graph_attack.utils import simcalc
from graph_attack.utils.graph import SimGraph

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

BF_HASH_FUNCT1 = hashlib.sha1
BF_HASH_FUNCT2 = hashlib.md5

#
PLT_PLOT_RATIO = 1.0
FILE_FORMAT  = '.png'#'.eps' #'.png'
PLT_FONT_SIZE    = 20# 28 # used for axis lables and ticks
LEGEND_FONT_SIZE = 20 # 28 # used for legends
TITLE_FONT_SIZE  = 19 # 30 # used for plt title
TICK_FONT_SIZE   = 18
MAX_DRAW_PLOT_NODE_NUM = 200

# logging.debug('A debug message!')
# logging.info('We processed %d records', 3)

def load_data_set(data_set_name, attr_num_list, ent_id_col, soundex_attr_val_list,
                  num_rec=-1, col_sep_char=',', header_line_flag=False,
                ):
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

    logging.debug('File header line: ' str(header_list))
    logging.debug('  Attributes to be used:')
    for attr_num in attr_num_list:
      logging.debug(header_list[attr_num])
      attr_name_list.append(header_list[attr_num])
    logging.debug('')

  max_attr_num = numpy.max(attr_num_list)+1

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

  logging.debug('  Count distribution of value occurrences: ' + str(sorted(count_dist_dict.items()) )
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

#TODO: RETORNAR PARAMENTROS DE ANONIMIZACAO DO BF
def encode_ds(encode_q_gram_dict,encode_attr_list,encode_rec_attr_val_dict,
                plain_rec_attr_val_dict,
                encode_method,bf_len,q,bf_encode,
                random_seed, # definir
                padded_flag='t', #clk
                #bf
                bf_hash_type='rh',bf_num_hash_funct='opt',
                bf_enc_param='t', #rbf
                # tmh param
                tmh_hash_funct='md5',tmh_num_hash_bits=200,
                tmh_num_tables=200,tmh_key_len=200,tmh_val_len=200,
                # 2sg two hash encoding
                cmh_num_hash_funct=0,cmh_num_hash_col=0
            ):
    
    start_time = time.time()
    assert bf_hash_type in ['rh' , 'dh']
    assert bf_encode in ['clk' , 'rbf' , 'rbf-d']

    
    if (encode_method == 'bf'): # Bloom filter encoding

        if (bf_num_hash_funct == 'opt'):
            bf_num_hash_funct_str = 'opt' # qual o problema?

            # Get the average number of q-grams in the string values
            #
            total_num_q_gram = 0
            total_num_val =    0

            for q_gram_set in encode_q_gram_dict.values():
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
                # logging.debug('   ', auxiliary.get_memory_usage())
        
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
                # logging.debug('   ', auxiliary.get_memory_usage())

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
                # logging.debug('   ', auxiliary.get_memory_usage())

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
    logging.debug('  Number of records hashed: %.2f' % len(encode_hash_dict) )
    logging.debug('')

    # Check which entity identifiers occur in both data sets

    common_ent_id_set = set(plain_rec_attr_val_dict.keys()) & set(encode_hash_dict.keys())
    common_num_ent = len(common_ent_id_set)

    plain_num_ent =  len(plain_rec_attr_val_dict)
    encode_num_ent = len(encode_hash_dict)

    return encode_hash_dict,hashing_time,plain_num_ent, encode_num_ent,common_num_ent  # colocar outros retornos se necessario

# -----------------------------------------------------------------------------

### Step 3 methods:
#                   Generate the two graphs by calculating similarities between records
#                   (only needed if graphs have not been loaded from pickle files)

# ::: auxiliary methods 
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

  for sample_num in range(hlsh_num_sample):
    bit_sample_list.append(random.sample(range(bit_array_len),
                                         hlsh_sample_size))
    if (len(bit_sample_list) > 1):
      assert bit_sample_list[-1] != bit_sample_list[-2]  # Check uniqueness

  # Now apply HLSH on both similarity hash dictionaries
  #
  for (node_key_val, q_gram_set_bit_array) in encode_data_dict.items():
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
  logging.debug('Number of blocks for the Encoded HLSH index: %d' % \
        (len(encode_block_dict)) + \
        '  (with sample size: %d, and number of samples: %d' % \
        (hlsh_sample_size, hlsh_num_sample)
  )
  hlsh_block_size_list = []
  for hlsh_block_key_set in encode_block_dict.values():
    hlsh_block_size_list.append(len(hlsh_block_key_set))
    
  logging.debug('  Minimum, average, median and maximum block sizes: ' + \
        '%d / %.2f / %d / %d' % (numpy.min(hlsh_block_size_list),
                             numpy.mean(hlsh_block_size_list),
                             numpy.median(hlsh_block_size_list),
                             numpy.max(hlsh_block_size_list))
                             )
  
  logging.debug('  Time used: %.3f sec' % (time.time() - start_time))
  logging.debug('')

  return encode_block_dict

#------------------------------------------------------------------

def rec_soundex_blocking(soundex_val_dict, graph_node_key_id_dict):
  
  block_dict = {}
  
  for ent_id, soundex_val_list in soundex_val_dict.items():
    
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

#--------------------------------------------------------------------------
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
#--------------------------------------------------------------------------
def build_regre_model(ba_sim_list, qg_sim_list, qg_num_q_gram_list,
                      regre_model_str, plot_data_name, encode_method,
                      encode_sim_funct, plot_file_name, plot_title):  
  
  x_val_list = qg_sim_list
  y_val_list = ba_sim_list
  q_val_list = qg_num_q_gram_list
  
  if(regre_model_str == 'linear'):
      
    logging.debug('Traning the linear regression model')
    
    
    if(encode_method in ['2sh','tmh']):
      x_val_array = numpy.reshape(x_val_list,(-1,1))
      y_val_array = numpy.reshape(y_val_list,(-1,1))
    else:
      x_val_array = numpy.array(list(zip(x_val_list,q_val_list)))
      y_val_array = numpy.reshape(y_val_list,(-1,1))
  
    # Split the data for the training and testing samples
    x_train, x_test, y_train, y_test = train_test_split(x_val_array, 
                                                        y_val_array, 
                                                        test_size=0.25,
                                                        # stratify = y_val_array,
                                                        random_state=42)
    
    # Train the model
    reg_model = linear_model.LinearRegression()
    
    reg_model.fit(x_train, y_train)
    
    # Testing the model
    y_predict = reg_model.predict(x_test)
  elif(regre_model_str == 'isotonic'):
    logging.debug('Traning the isotonic regression model')
    
    x_train, x_test, y_train, y_test = train_test_split(x_val_list, 
                                                        y_val_list, 
                                                        test_size=0.25, 
                                                        random_state=42)
  
    reg_model = IsotonicRegression()

    reg_model.fit_transform(x_train, y_train)
    
    y_predict = reg_model.predict(x_test)
  elif(regre_model_str == 'poly'):
    
    logging.debug('Traning the polynomial regression model')
    
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
    raise Exception('***WARNING!, wrong regression model')
  
  y_p_new = []
  y_t_new = []
  
  #ALTERADO alterado
  # y_predict = list(zip(*y_predict)[0])
  # y_test = list(zip(*y_test)[0])
  y_predict = list(zip(*y_predict))[0]
  y_test = list(zip(*y_test))[0]
  
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
  model_min_abs_err      = numpy.min(err_val_list)
  model_max_abs_err      = numpy.max(err_val_list)
  model_avrg_abs_err     = numpy.mean(err_val_list)
  model_std_abs_err      = numpy.std(err_val_list)
  model_mean_sqrd_err    = metrics.mean_squared_error(y_test_eval, y_predict_eval)
  model_mean_sqrd_lg_err = mean_sqrd_lg_err
  model_r_sqrd           = metrics.r2_score(y_test_eval, y_predict_eval)
  
  logging.debug('')
  logging.debug('Evaluation of the %s regression model' %regre_model_str)
  logging.debug('  Explained variance score:       %.5f' %model_var_score)
  logging.debug('  Min/Max/Average absolute error: %.5f / %.5f / %.5f' %(numpy.min(err_val_list), \
                                                                 numpy.max(err_val_list), \
                                                                 numpy.mean(err_val_list))
                )
  logging.debug('  Standard deviation error:       %.5f' %model_std_abs_err)
  logging.debug('  Mean-squared error:             %.5f' %model_mean_sqrd_err)
  logging.debug('  Mean-squared log error:         %.5f' %model_mean_sqrd_lg_err)
  logging.debug('  R-squared value:                %.5f' %model_r_sqrd)
  logging.debug('')
  
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
        y_train = list(zip(*y_train))[0]
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
#--------------------------------------------------------------------------
def test_sim_regre_model(regre_model, ba_sim_list, qg_sim_list, qg_num_q_gram_list,
                         regre_model_str, encode_method):
  
  sample_size = 500
  sample_indices = random.sample([i for i in range(len(ba_sim_list))], sample_size)
  
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
  model_min_abs_err      = numpy.min(err_val_list)
  model_max_abs_err      = numpy.max(err_val_list)
  model_avrg_abs_err     = numpy.mean(err_val_list)
  model_std_abs_err      = numpy.std(err_val_list)
  model_mean_sqrd_err    = metrics.mean_squared_error(y_test_eval, y_predict_eval)
  model_mean_sqrd_lg_err = mean_sqrd_lg_err
  model_r_sqrd           = metrics.r2_score(y_test_eval, y_predict_eval)
  
  logging.debug('')
  logging.debug('Evaluation of the loaded %s regression model' %regre_model_str)
  logging.debug('  Explained variance score:       %.5f' %model_var_score)
  logging.debug('  Min/Max/Average absolute error: %.5f / %.5f / %.5f' %(numpy.min(err_val_list), \
                                                                 numpy.max(err_val_list), \
                                                                 numpy.mean(err_val_list))
                )
  logging.debug('  Standard deviation error:       %.5f' %model_std_abs_err)
  logging.debug('  Mean-squared error:             %.5f' %model_mean_sqrd_err)
  logging.debug('  Mean-squared log error:         %.5f' %model_mean_sqrd_lg_err)
  logging.debug('  R-squared value:                %.5f' %model_r_sqrd)
  
  return False
#--------------------------------------------------------------------------
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
    qg_sim_val = qg_graph.edges[qg_key1,qg_key2]['sim']
    
    qg_edge_tuple = (qg_key1, qg_key2)
    qg_edge_tuple = tuple(sorted(qg_edge_tuple))
    
    qg_edge_val_set.add(qg_edge_tuple)
    
    qg_key_set.add(qg_key1)
    qg_key_set.add(qg_key2)
  
  for ba_key1, ba_key2 in ba_graph_edges:
    ba_sim_val = ba_graph.edges[ba_key1,ba_key2]['sim']
    
    q_set1, bit_array1 = encode_node_val_dict[ba_key1]
    q_set2, bit_array2 = encode_node_val_dict[ba_key2]
    
    ba_q_key1 = tuple(sorted(q_set1))
    ba_q_key2 = tuple(sorted(q_set1))
    
    ba_edge_tuple = (ba_q_key1, ba_q_key2)
    ba_edge_tuple = tuple(sorted(ba_edge_tuple))
    ba_edge_val_set.add(ba_edge_tuple)
    
    ba_key_set.add(ba_q_key1)
    ba_key_set.add(ba_q_key2)
    
  logging.debug(list(ba_edge_val_set)[:10])
  logging.debug('')
  logging.debug(list(qg_edge_val_set)[:10])
  
  #print ba_edge_val_set
  #print
  #print qg_edge_val_set
  
  c_nodes = ba_key_set.intersection(qg_key_set)
  logging.debug(len(qg_key_set))
  logging.debug(len(ba_key_set))
  logging.debug(len(c_nodes))
  logging.debug('')
  
  common_edges = ba_edge_val_set.intersection(qg_edge_val_set)
  ba_only_edges = ba_edge_val_set - qg_edge_val_set
  qg_only_edges = qg_edge_val_set - ba_edge_val_set
  
  logging.debug('  Number of common edges:          %d' %len(common_edges))
  logging.debug('  Number of encode only edges:     %d' %len(ba_only_edges))
  logging.debug('  Number of plain-text only edges: %d' %len(qg_only_edges))
  logging.debug('')
  
  
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

#--------------------------------------------------------------------------

# If the similarity adjustment flag is set to True, then assuming we have
# a sample dataset with ground truth availble to us build a regression model
# to predict the the encoded similarity given the q-gram similarity. Here
# we use two features for the model q-gram similarity and the total number
# of unique q-grams of the two records.    

def genG(plain_num_ent,encode_num_ent,
        encode_method, # encoding
        # encode
        encode_sim_funct_name,
        encode_hash_dict,
        encode_q_gram_dict,
        encode_rec_attr_val_dict,
        encode_blck_method, #'hmlsh': Hamming LSH blocking 'minhash' :  Min-hash LSH blocking
        #plain
        plain_q_gram_dict,
        plain_rec_attr_val_dict,
        plain_sim_funct_name,
        plain_blck_method,
        #soundex
        encode_soundex_val_dict,
        plain_soundex_val_dict,
        # encoded_plain_vars
        # qg_graph_node_id_dict, #TODO: VERIFICAR O QUE É ISSO
        # ba_graph_node_id_dict,
        # regression
        regre_model_str,
        plain_base_data_set_name,
        plain_attr_list_str,
        q,
        padded_flag,
        encode_method_str,
        encode_base_data_set_name,plain_num_rec_loaded,encode_attr_list_str,encode_num_rec_loaded,
        #outros
        min_sim, # vem la de cima
        all_sim_list,
        #
        num_samples = 20000, # regression num samples
        random_seed=101,
        ## anonimizacao
        bf_hash_type='clk',
        bf_num_hash_funct=0,
        bf_len=200,
        bf_encode='xpto',
        bf_harden='None',
        # 2sh
        cmh_max_rand_int=0,
        cmh_num_hash_funct=0,
        cmh_num_hash_col=0,
        #tmh
        tmh_hash_funct=0,
        tmh_num_tables=0,
        tmh_key_len=0,
        tmh_val_len=0,
        tmh_num_hash_bits=0, #num bits tmh
        # leitura dos grafos
        graph_path = '.',
        regre_file_path = 'regre-models/', # pode sair
        plain_graph_file_name='plain_graph',
        encode_graph_file_name='encoded_graph',
        generated_graph_flag = False,
        # utilizar apenas atributos em comum
        include_only_common = False,
        common_rec_id_set={},#setar caso seja utilizado
        #nao sei o que e isso mas estava hardoced
        same_ba_blck = False,
        #ploting
        plot_diffsim_graph=False,
        #ajueste de similaridade
        sim_diff_interval_size = 0.05,
        sim_diff_adjust_flag=True,
        #outros 2
        calc_common_edges = False):
    #TODO:colocar assert tmh,bf,2sg
    # tmh_hash_funct=0,
    # tmh_num_tables=0,
    # tmh_key_len=0,
    # tmh_val_len=0,
    # tmh_num_hash_bits=0, #num bits tmh
    

    today_str = time.strftime("%Y%m%d", time.localtime())
    now_str =   time.strftime("%H%M", time.localtime())
    today_time_str = time.strftime("%Y%m%d %H:%M:%S", time.localtime())
    
    # Initialise the two graphs
    QG_sim_graph = SimGraph()
    BA_sim_graph = SimGraph()

    # Two dictionaries that can be usedt to get similar nodes in both graphs
    # since graph node values cannot be compared.
    #
    qg_graph_node_id_dict = {}
    ba_graph_node_id_dict = {}

    #parametros de blocagem
    plain_sample_size = 4
    plain_num_samples = 50
    q_gram_min_hash = False
    
    
    
    
    # Initialise the actual similarity functions to be used
    #
    if (plain_sim_funct_name == 'dice'):
        plain_sim_funct = simcalc.q_gram_dice_sim
    elif (plain_sim_funct_name == 'jacc'):
        plain_sim_funct = simcalc.q_gram_jacc_sim
    else:
        raise Exception('This should not happen')

    if(encode_method == '2sh'):
        if (encode_sim_funct_name == 'dice'):
            encode_sim_funct = simcalc.q_gram_dice_sim
        elif (encode_sim_funct_name == 'jacc'):
            encode_sim_funct = simcalc.q_gram_jacc_sim
        else:
            raise Exception('This should not happen')
  
    else:
        if (encode_sim_funct_name == 'dice'):
            encode_sim_funct = simcalc.bit_array_dice_sim
        elif (encode_sim_funct_name == 'hamm'):
            encode_sim_funct = simcalc.bit_array_hamm_sim
        elif (encode_sim_funct_name == 'jacc'):
            encode_sim_funct = simcalc.bit_array_jacc_sim
        else:
            raise Exception('This should not happen') 
    
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
  
    
    
    if(include_only_common):
        logging.debug('  Remove non overlapping records from the plain-text graph...')
        logging.debug('')
    
    for (ent_id, q_gram_set) in plain_q_gram_dict.items():
        if(include_only_common):
            if(ent_id not in common_rec_id_set):
                continue
    
        node_key_val = tuple(sorted(q_gram_set))  # Sets cannot be dictionary keys
        plain_node_val_dict[node_key_val] = q_gram_set

        # Only keep none empty attribute values
        #
        QG_sim_graph.add_rec(node_key_val, ent_id,filter(None, plain_rec_attr_val_dict[ent_id]))
    
        if(ent_id not in qg_graph_node_id_dict):
            qg_graph_node_id_dict[ent_id] = node_key_val
    #fim do for 01 (plano)

    for (ent_id, bit_array) in encode_hash_dict.items():
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
    # fim do for 02 (encode_hash_dict)

    plain_graph_num_node =  len(plain_node_val_dict)
    encode_graph_num_node = len(encode_node_val_dict)

    logging.debug('Number of nodes in the two graphs: %d in plain-text / %d in encoded' \
            % (plain_graph_num_node, encode_graph_num_node)
    )
    logging.debug('')
    
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

    QG_min_hash = MinHashLSH(plain_sample_size, plain_num_samples, random_seed)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # We use min-hash for the bit array hashing based on the q-gram sets of
    # the encoded values in order to generate a similar blocking structure
    # (i.e. same q-gram sets have the same probability to be compared or not)
    #
    ba_blck_dict = {} # A dictionary to store blocked values

    if(encode_blck_method == 'hmlsh'): # Hamming LSH blocking
        enc_sample_size = 4
        enc_num_samples = 70
    
        if(encode_method == 'bf'):
            bit_array_len    = bf_len
        elif(encode_method == 'tmh'):
            bit_array_len = tmh_num_hash_bits
    
        ba_blck_dict = encode_hlsh_blocking(encode_node_val_dict,
                                            enc_sample_size, 
                                            enc_num_samples, 
                                            bit_array_len)
    #fim do hmlsh if
    elif(encode_blck_method == 'minhash'): # Min-hash LSH blocking
        q_gram_min_hash = True
        if(q_gram_min_hash):
            enc_sample_size = plain_sample_size
            enc_num_samples = plain_num_samples
            for (node_key_val, q_gram_set_bit_array) in encode_node_val_dict.items():
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
        
                for (node_key_val, q_gram_set_bit_array) in encode_node_val_dict.items():
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

                for (node_key_val, q_gram_set_bit_array) in encode_node_val_dict.items():
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
    #fim do minhash if
    elif(encode_blck_method == 'soundex'): # Soundex blocking
        ba_blck_dict = rec_soundex_blocking(encode_soundex_val_dict, ba_graph_node_id_dict)
    elif(encode_blck_method == 'none'): # No blocking.
        enc_sample_size = 0
        enc_num_samples = 0
        ba_blck_dict['all'] = set(encode_node_val_dict.keys())
    else:
        raise Exception('***Wrong blocking method for encoded text!')

    ###
    ### avaliação da blocagem
    ###
    num_blocks = len(ba_blck_dict)
    logging.debug('Bit array %s indexing contains %d blocks' % (encode_blck_method, num_blocks))

    num_rec_pair = 0
    min_hash_block_size_list = []
    for min_hash_key_val_set in ba_blck_dict.values():
        block_size = len(min_hash_key_val_set)
        min_hash_block_size_list.append(block_size)
        num_rec_pair += block_size*(block_size-1) / 2

    num_all_rec_pair = len(encode_node_val_dict)*(len(encode_node_val_dict)-1)/2

    logging.debug('  Minimum, average, median and maximum block sizes: ' + \
        '%d / %.2f / %d / %d' % (numpy.min(min_hash_block_size_list),
        numpy.mean(min_hash_block_size_list),
        numpy.median(min_hash_block_size_list),
        numpy.max(min_hash_block_size_list))
    )

    logging.debug('  Compare %d record pairs from %s blocking:' % (num_rec_pair, encode_blck_method))
    logging.debug('    (total number of all record pairs: %d)' % (num_all_rec_pair))
    logging.debug('')
    min_hash_block_size_list = []  # Not needed any more

    # Now compare all encoded value pairs in the same block (loop over all blocks)
    #
    ba_sim_list        = []
    qg_sim_list        = []
    qg_num_q_gram_list = []
    sim_abs_diff_list  = []
    sim_sample_dict = {'0-1': 0, '1-2': 0, '2-3': 0, '3-4': 0, '4-5': 0, '5-6': 0, 
                        '6-7': 0, '7-8': 0, '8-9': 0, '9-10': 0}
    
    for (bn, min_hash_key_val_set) in enumerate(ba_blck_dict.values()):

        if ((bn > 0) and (bn % 10000 == 0)):
            logging.debug('    Processed %d of %d blocks' % (bn, num_blocks))

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
                    #fim do for adjust_flag
                    enc_min_sim = min_sim
                    if (encode_pair_sim >= min_sim):
                        BA_sim_graph.add_edge_sim(node_key_val1, node_key_val2,encode_pair_sim)
    logging.debug('')
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

    logging.debug('Time for generating the bit-array similarity graph: %.2f sec' % \
        (ba_graph_gen_time)
    )

    logging.debug('  Number of edges in graph: %d' % (ba_graph_num_edges))
    # logging.debug(' ', auxiliary.get_memory_usage())
    logging.debug('')

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
    regre_file_name = graph_path + regre_file_name
    if (os.path.isfile(regre_file_name)):
        logging.debug('Load regression model from saved file:', regre_file_name)
        logging.debug('')
        regre_model = pickle.load(open(regre_file_name, 'rb'))
    
        # Test the model with some sample data
        #
        test_sim_regre_model(regre_model, ba_sim_list, qg_sim_list, qg_num_q_gram_list,
                            regre_model_str, encode_method)
    else:
        regre_model_eval_res_file = 'graph_attack_regression_model_eval_res.csv'
        if(sim_diff_adjust_flag == True):
            regres_file_name = 'plain-encode-sim-%s-regression-scatter-%s-%s-%s-%d-%s-' % \
                       (regre_model_str, plain_base_data_set_name, encode_base_data_set_name,
                        plain_attr_list_str, plain_num_rec_loaded,
                        encode_attr_list_str) + \
                       '%d-%d-%s-%s-%s-%.3f-%s-%s'% \
                       (encode_num_rec_loaded, q, str(padded_flag).lower(),
                        plain_sim_funct_name, encode_sim_funct_name,
                        min_sim, encode_method_str, today_str)
            regres_file_name = regres_file_name+FILE_FORMAT
            
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
            else:
              #TODO AJUSTAR COLOCAR NA CHAMADA DA FUNCAO
              plot_data_name = '(Outro)'
            
            regre_model, eval_res_tuple = build_regre_model(ba_sim_list, qg_sim_list,
                                                    qg_num_q_gram_list,
                                                    regre_model_str, plot_data_name,
                                                    encode_method, encode_sim_funct_name,
                                                    regres_file_name, plot_title)

            # Save the regression model in a pickle file
            #
            save_file = open(regre_file_name, 'wb')
            pickle.dump(regre_model, save_file)
            
            # Write evaluation resuls to a csv file
            #
            
    

            #TODO: REVER VARIAVEIS
            res_header_list = ['date','time','plain_dataset_name','plain_attr_list','plain_num_rec','encode_dataset_name','encode_attr_list','encode_num_rec','q_length','padded','sim_adjust_flag','regression_model','plain_sim_funct','enc_sim_func','min_sim','encode_method','bf_hash_type/tmh_num_hash_bits/cmh_max_rand_int','num_hash_funct','bf_len/tmh_num_hash_tables/cmh_num_hash_col','bf_encode/tmh_key_len', 'bf_harden/tmh_val_len','explained_var','min_abs_err','max_abs_err','avrg_abs_err','std_abs_err','mean_sqrd_err','mean_sqrd_lg_err','r_sqrd']
            #TODO: REVER SAIDA
            res_val_list = [today_str, now_str, plain_base_data_set_name, plain_attr_list_str, plain_num_rec_loaded,encode_base_data_set_name, encode_attr_list_str, encode_num_rec_loaded,q, str(padded_flag).lower(), sim_diff_adjust_flag, regre_model_str.lower(),plain_sim_funct_name, encode_sim_funct_name,min_sim, encode_method]
            if (encode_method == 'bf'):
                res_val_list += [bf_hash_type, bf_num_hash_funct, bf_len, bf_encode, bf_harden]
            
            elif (encode_method == 'tmh'):
                res_val_list += [tmh_num_hash_bits, tmh_hash_funct, tmh_num_tables, tmh_key_len, tmh_val_len]
            
            elif(encode_method == '2sh'):
                res_val_list += [cmh_max_rand_int, cmh_num_hash_funct, cmh_num_hash_col, '', '']
            
            res_val_list += list(eval_res_tuple)
    
            #REIDENTADO
            if (not os.path.isfile(regre_model_eval_res_file)):
              write_reg_file = open(regre_model_eval_res_file, 'w')
              csv_writer = csv.writer(write_reg_file)
              csv_writer.writerow(res_header_list)
              logging.debug('Created new result file:', regre_model_eval_res_file)
       
            else:  # Append results to an existing file
              write_reg_file = open(regre_model_eval_res_file, 'a')
              csv_writer = csv.writer(write_reg_file)
        
            csv_writer.writerow(res_val_list)
            write_reg_file.close()
    #fim do else
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # We also use min-hash for the q-gram similarities in order to create a
    # similar blocking structure
    # (i.e. same q-gram sets have the same probability to be compared or not)
    #
    start_time = time.time()
    qg_blck_dict = {}
    if(plain_blck_method == 'minhash'):
        for (node_key_val, q_gram_set) in plain_node_val_dict.items():
          qg_band_hash_sig_list = QG_min_hash.hash_q_gram_set(q_gram_set)
          for min_hash_val in qg_band_hash_sig_list:
              min_hash_val_tuple = tuple(min_hash_val)
              min_hash_key_val_set = qg_blck_dict.get(min_hash_val_tuple, set())
              min_hash_key_val_set.add(node_key_val)
              qg_blck_dict[min_hash_val_tuple] = min_hash_key_val_set
    elif(plain_blck_method == 'soundex'):
        qg_blck_dict = rec_soundex_blocking(plain_soundex_val_dict,qg_graph_node_id_dict)
    elif(plain_blck_method == 'none'):
        qg_blck_dict['all'] = set(plain_node_val_dict.keys())
    else:
        raise Exception('***Wrong blocking method for plain-text!')
    
    num_blocks = len(qg_blck_dict)
    logging.debug('Q-gram %s indexing contains %d blocks' % (plain_blck_method, num_blocks))
    num_rec_pair = 0
    min_hash_block_size_list = []

    for min_hash_key_val_set in qg_blck_dict.values():
        block_size = len(min_hash_key_val_set)
        min_hash_block_size_list.append(block_size)
        num_rec_pair += block_size*(block_size-1) / 2

    num_all_rec_pair = len(plain_node_val_dict)*(len(plain_node_val_dict)-1) / 2
    logging.debug('  Minimum, average, median and maximum block sizes: ' + \
        '%d / %.2f / %d / %d' % (numpy.min(min_hash_block_size_list),
                                 numpy.mean(min_hash_block_size_list),
                                 numpy.median(min_hash_block_size_list),
                                 numpy.max(min_hash_block_size_list))
    )

    logging.debug('  Compare %d record pairs from %s blocking:' % (num_rec_pair, plain_blck_method))
    logging.debug('    (total number of all record pairs: %d)' % (num_all_rec_pair))
    logging.debug('')

    min_hash_block_size_list = []  # Not needed any more
  
    if(same_ba_blck):
        for (bn, min_hash_key_val_set) in enumerate(ba_blck_dict.values()):
            if ((bn > 0) and (bn % 10000 == 0)):
                logging.debug('    Processed %d of %d blocks' % (bn, num_blocks))
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
                    #fim if sim_diff_adjust_flag
                    if (plain_pair_sim >= min_sim):
                        QG_sim_graph.add_edge_sim(plain_node_key_val1, plain_node_key_val2,
                                            plain_pair_sim)
            pass
    #fim do if same_ba_blck
    # Now compare all q-gram sets in the same block (loop over all blocks)
    #
    for (bn, min_hash_key_val_set) in enumerate(qg_blck_dict.values()):
        if ((bn > 0) and (bn % 10000 == 0)):
            logging.debug('    Processed %d of %d blocks' % (bn, num_blocks))
        
        if (len(min_hash_key_val_set) > 1):
            min_hash_key_val_list = sorted(min_hash_key_val_set)

            for (i, node_key_val1) in enumerate(min_hash_key_val_list[:-1]):
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
                        QG_sim_graph.add_edge_sim(node_key_val1, node_key_val2,plain_pair_sim)
    #fim do for
        
    qg_graph_gen_time = time.time() - start_time

    qg_graph_num_edges = networkx.number_of_edges(QG_sim_graph.sim_graph)

    logging.debug('Time for generating the q-gram similarity graph: %.2f sec' % \
            (qg_graph_gen_time))
    logging.debug('  Number of edges in graph: %d' % (qg_graph_num_edges))
    # logging.debug(' ', auxiliary.get_memory_usage())
    logging.debug('')

    qg_graph_num_singleton = QG_sim_graph.remove_singletons()

    # For randomly sampled edges that are the same across the two graphs get
    # their similarity differences and calculate average such differences for
    # different similarity intervals
    #
    sim_diff_plot_file_name = 'plain-encode-sim-diff-scatter-%s-%s-%s-%d-%s-' % \
                            (plain_base_data_set_name, encode_base_data_set_name,
                             plain_attr_list_str, plain_num_rec_loaded,
                             encode_attr_list_str) + \
                            '%d-%d-%s-%s-%s-%.3f-%s-%s'% \
                            (encode_num_rec_loaded, q, str(padded_flag).lower(),
                             plain_sim_funct_name, encode_sim_funct_name,
                             min_sim, encode_method_str, today_str)
    sim_diff_plot_file_name = sim_diff_plot_file_name+FILE_FORMAT
  
    if (plot_diffsim_graph):
      sim_diff_plot_file_name_adj = sim_diff_plot_file_name.replace(FILE_FORMAT,'-adjusted'+FILE_FORMAT)
    else:
      sim_diff_plot_file_name_adj=None
  
    sim_diff_interval_size = 0.05

    

    QG_sim_graph.comp_sim_differences(encode_plain_node_val_dict,
                                    BA_sim_graph.sim_graph,
                                    sim_diff_interval_size, num_samples,
                                    sim_diff_plot_file_name_adj)
  
    # Save graphs into pickled (binary) files for later use
    #
    networkx.write_gpickle(QG_sim_graph.sim_graph, plain_graph_file_name)
    networkx.write_gpickle(BA_sim_graph.sim_graph, encode_graph_file_name)

    logging.debug('Wrote graphs into pickle files:')
    # logging.debug('  Plain-text graph file:', plain_graph_file_name)
    # logging.debug('  Encoded graph file:   ', encode_graph_file_name)
    # logging.debug(' ', auxiliary.get_memory_usage())
    logging.debug('')

    # retorno 
    # ba_graph_gen_time
    # qg_graph_gen_time,qg_graph_num_edges,qg_graph_num_singleton
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    # Back to common code (graphs either loaded from files or generated)
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
          
          logging.debug('Created new result file:', blocking_alignment_res_file)
          
        else:  # Append results to an existing file
          write_blck_file = open(blocking_alignment_res_file, 'a')
          csv_writer = csv.writer(write_blck_file)
       
        csv_writer.writerow(blck_res_val_list)
        write_blck_file.close()
      #fim do write_res_flag if
      

    #fim do if calc_comomn_edges
    #########
    QG_sim_graph.show_graph_characteristics(all_sim_list,'Plain-text q-gram graph')

    BA_sim_graph.show_graph_characteristics(all_sim_list,'Encoded bit-array graph')

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
                              '%d-%s-%s-%s-%s-%.3f-%s-%s'% \
                              (q, str(padded_flag).lower(), plain_sim_funct_name,
                                str(sim_diff_adjust_flag).lower(),
                                encode_sim_funct_name, min_sim, encode_method_str,
                                today_str)
    sim_histo_plot_file_name = sim_histo_plot_file_name + FILE_FORMAT 
    
    # Generate a list of similarities of all edges between records from the same
    # entities (not just q-gram or bit array values) and count the number of
    # such edges not in the intersection
    #
    qg_edge_dict = {}
    ba_edge_dict = {}

    plain_sim_graph =  QG_sim_graph.sim_graph  # Short cuts
    encode_sim_graph = BA_sim_graph.sim_graph

    #grafo plano
    for (node_key_val1, node_key_val2) in plain_sim_graph.edges():
      ent_id_set1 = plain_sim_graph.node[node_key_val1]['ent_id_set']
      ent_id_set2 = plain_sim_graph.node[node_key_val2]['ent_id_set']
      edge_sim =    plain_sim_graph.edges[node_key_val1,node_key_val2]['sim']

      for ent_id1 in ent_id_set1:
        for ent_id2 in ent_id_set2:
          ent_id_pair = tuple(sorted([ent_id1,ent_id2]))
          assert ent_id_pair not in qg_edge_dict, ent_id_pair
          qg_edge_dict[ent_id_pair] = edge_sim
    
    #grafo codificado
    for (node_key_val1, node_key_val2) in encode_sim_graph.edges():
      ent_id_set1 = encode_sim_graph.node[node_key_val1]['ent_id_set']
      ent_id_set2 = encode_sim_graph.node[node_key_val2]['ent_id_set']
      edge_sim =    encode_sim_graph.edges[node_key_val1,node_key_val2]['sim']

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

    logging.debug('#### Identified %d (%.2f%%) edges between entities common to both ' % \
          (num_common_edges, 100.0*num_common_edges / num_all_edges) + \
          'similarity graphs'
    )
    logging.debug('####   %d (%.2f%%) edges only occur in plain-text similarity graph' % \
          (num_edges_plain_only, 100.0*num_edges_plain_only / num_all_edges)
    )
    logging.debug('####   %d (%.2f%%) edges only occur in encode similarity graph' % \
          (num_edges_encode_only, 100.0*num_edges_encode_only / num_all_edges)
    )
    logging.debug('')

    # Calculate edge similarity differences for all given minimum similarities
    #
    logging.debug('#### Similarity differences across the two graphs between true ' + \
          'matching record pairs:')
    logging.debug('####  (negative means plain-tex similarity < encoded similarity)')

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

      if len(edge_sim_diff_list) > 0:
        sim_diff_min = numpy.min(edge_sim_diff_list)
        sim_diff_avr = numpy.mean(edge_sim_diff_list)
        sim_diff_std = numpy.std(edge_sim_diff_list)
        sim_diff_med = numpy.median(edge_sim_diff_list)
        sim_diff_max = numpy.max(edge_sim_diff_list)
      else:
        sim_diff_min = 0
        sim_diff_avr = 0
        sim_diff_std = 0
        sim_diff_med = 0
        sim_diff_max = 0

      plot_list_dict[check_min_sim] = edge_sim_diff_list

      logging.debug('####   For a minimum similarity of %.3f:' % (check_min_sim) + \
            '(with %d similarity pairs)' % (len(edge_sim_diff_list))
      )
      logging.debug('####     min=%.3f / avr=%.3f (std=%.3f) / med=%.3f / max=%.3f' % \
            (sim_diff_min, sim_diff_avr, sim_diff_std, sim_diff_med,sim_diff_max)
      )
    logging.debug('')
    
    #
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
                              '%d-%s-%s-%s-%s-%.3f-%s-%s'% \
                              (q, str(padded_flag).lower(), plain_sim_funct_name,
                                str(sim_diff_adjust_flag).lower(),
                                encode_sim_funct_name, min_sim, encode_method_str,
                                today_str)
    sim_diff_plot_file_name = sim_diff_plot_file_name+FILE_FORMAT

    #plot_save_sim_diff(plot_list_dict, sim_diff_plot_file_name,2,False)

    # memo_use = auxiliary.get_memory_usage_val()
    # logging.debug('')
    # logging.debug('   ', auxiliary.get_memory_usage())
    # logging.debug('')

    # Print a summary result line for the encoding and graph generation steps
    #
    logging.debug('#### Graph matching PPRL attack result summary')
    logging.debug('####   Experiment started on %s at %s' % (today_str, now_str))
    logging.debug('####')
    logging.debug('####   Plain-text data set: %s' % (plain_base_data_set_name))
    logging.debug('####     Number of records: %d' % (plain_num_rec_loaded))
    logging.debug('####     Attributes used:   %s' % (str(plain_attr_list_str)))
    logging.debug('####   Encoded data set:    %s' % (encode_base_data_set_name))
    logging.debug('####     Number of records: %d' % (encode_num_rec_loaded))
    logging.debug('####     Attributes used:   %s' % (str(encode_attr_list_str)))
    logging.debug('####')
    logging.debug('####  Q-gram encoding: q=%d, padding:   %s' % (q, str(padded_flag)))
    logging.debug('####    Plain-text similarity function: %s' % (plain_sim_funct_name))
    logging.debug('####    Encode similarity function:     %s' % (encode_sim_funct_name))
    logging.debug('####    Overall minimum similarity:     %.2f' % (min_sim))
    logging.debug('####')
    logging.debug('####  Similarity difference adjustment: %s' % \
          (str(sim_diff_adjust_flag))
    )
    logging.debug('####')
    logging.debug('####  Encoding method: %s' % (encode_method))
    if (encode_method == 'bf'):
      logging.debug('####    Hash type:                %s' % (bf_hash_type))
      logging.debug('####    BF length:                %d bits' % (bf_len))
      logging.debug('####    Number of hash functions: %s' % (str(bf_num_hash_funct)))
      logging.debug('####    BF hardening method:      %s' % (bf_harden))
    elif (encode_method == 'tmh'):
      logging.debug('####    Hash function:            %s' % (tmh_hash_funct))
      logging.debug('####    Bit array length:         %d bits' % (tmh_num_hash_bits))
      logging.debug('####    Number of tables:         %d' % (tmh_num_tables))
      logging.debug('####    Key length:               %d bits' % (tmh_key_len))
      logging.debug('####    Value length:             %d bits' % (tmh_val_len))
    elif(encode_method == '2sh'):
      logging.debug('####    Maximum random integer:   %d' % (cmh_max_rand_int))
      logging.debug('####    Number of hash functions: %d' % (cmh_num_hash_funct))
      logging.debug('####    Number of hash columns:   %d' % (cmh_num_hash_col))
      
    logging.debug('####')

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
                    (0, 0, plain_num_ent, \
                    encode_num_ent, 0, plain_graph_num_node, \
                    encode_graph_num_node, qg_graph_gen_time, ba_graph_gen_time, \
                    qg_graph_num_edges, ba_graph_num_edges, \
                    qg_graph_num_singleton, ba_graph_num_singleton)
      logging.debug('######## load:', header_line)
      logging.debug('######## load:', result_line)
      logging.debug('########')


    # 
    return QG_sim_graph, BA_sim_graph, plain_graph_num_node, encode_graph_num_node, attck_res_header_list, attck_res_val_list

# -----------------------------------------------------------------------------
# Step 4: Generate the features from the two graphs

#---------------------------------------------------------------------------

def learn_embeddings(walks, num_dimensions, window_size, num_workers, num_iter):
  '''
  Learn embeddings by optimizing the Skipgram objective using SGD.
  '''
  
  start_time = time.time()
  
  # walks = [map(str, walk) for walk in walks]
  walks = [list(map(str, walk)) for walk in walks]
  # model = Word2Vec(walks, size=num_dimensions, window=window_size, min_count=0, sg=1, workers=num_workers, iter=num_iter)
  model = Word2Vec(walks, vector_size=num_dimensions, window=window_size, min_count=0, sg=1, workers=num_workers,epochs=num_iter)
  
  learn_time = time.time() - start_time
  logging.debug('    Learning node embeddings took %.3f seconds' %learn_time)
  logging.debug('')
  
  return model

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
  
  header_list = next(csv_reader)
  feat_name_list = header_list[1:]

  # Read each line in the file and store the required attribute values in a
  # list
  #
  vec_num = 0 

  #TODO: PROBLEMA AO CARREGAR OS DADOS
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

  for node_feat_array in graph_feat_dict.values():
    min_feat_array = numpy.minimum(node_feat_array, min_feat_array)
    max_feat_array = numpy.maximum(node_feat_array, max_feat_array)

  
  load_time = time.time() - load_start_time
  
  logging.debug(' Load graph feature values into a dictionary. Took %.3f seconds' %load_time)
  # print ' ', auxiliary.get_memory_usage()
  logging.debug('')
  
  return graph_feat_dict, min_feat_array, max_feat_array, feat_name_list

# =============================================================================

def write_feat_matrix_file(graph_feat_dict, feat_name_list, feat_matrix_file_name):
  """A funtion to write feature matrix generated for each graph to
     a csv file.
  """
  
  writing_start_time = time.time()
  
  header_list = ['node_key'] + feat_name_list
  
  for node_key_val, feat_val_array in graph_feat_dict.items():
    
    if('encode' in feat_matrix_file_name and '2sh' not in feat_matrix_file_name):
      res_list = [node_key_val] + list(feat_val_array)
    else:
      res_list = [','.join(node_key_val)] + list(feat_val_array)
  
    # Check if the result file exists, if it does append, otherwise create
    if (not os.path.isfile(feat_matrix_file_name)):
      write_file = open(feat_matrix_file_name, 'w')
      csv_writer = csv.writer(write_file)
      csv_writer.writerow(header_list)
    
      logging.debug('Created new result file:' + str(feat_matrix_file_name))
    
    else:  # Append results to an existing file
      write_file = open(feat_matrix_file_name, 'a')
      csv_writer = csv.writer(write_file)
    
    csv_writer.writerow(res_list)
    write_file.close()
  
  writing_time = time.time() - writing_start_time
  
  logging.debug(' Wrote graph feature results into the csv file. Took %.3f seconds' %writing_time)
  # logging.debug(' ', auxiliary.get_memory_usage())
  logging.debug('')


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
    edge_sim = sim_graph.edges[node_key_val1,node_key_val2]['sim']

    for (color_sim, grey_scale_color_str) in edge_color_list:
      if (edge_sim <= color_sim):
        graph_edge_color_list.append(grey_scale_color_str)
        break

  assert len(graph_edge_color_list) == len(graph_edge_list)

  networkx.draw(sim_graph, node_size=20, width=2, edgelist=graph_edge_list,
                edge_color=graph_edge_color_list, with_labels=False)

  logging.debug('Save generated graph figure with %d nodes into file: %s' % \
        (len(node_key_val_set), file_name)
  )

  plt.savefig(file_name)

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
  logging.debug('Calculated %d match Cosine similarities based on feature vectors' % \
        (num_match_sim)
  )

  # The number of non-matches we want
  num_non_match_sim_needed = num_match_sim * over_sample

  # Get all node key values from both feature vectors
  #
  node_key_val_list1 = list(feat_vec_dict1.keys())
  node_key_val_list2 = list(feat_vec_dict2.keys())

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

  if len(match_y_val_list) != 0 or len(non_match_y_val_list) != 0:
    max_count = max(max(match_y_val_list), max(non_match_y_val_list))
    min_sim =   min(min(match_x_val_list), min(non_match_x_val_list))
    max_sim =   max(max(match_x_val_list), max(non_match_x_val_list))
  else:
    max_count = 0
    min_sim =   0
    max_sim =   0

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

  logging.debug('  Written plot into file:' + str(plot_file_name) )
  logging.debug('')
  
#------------------------------------------------------------------------------
#----------------------------------------------------------------------

def calc_sim_conf_val(encode_key, plain_key, cos_sim, enc_key_plain_dict, plain_key_enc_dict):
  
  conf_val = 0.0
  
  sim_val_list = []
  
  for other_plain_key, other_cos_sim in enc_key_plain_dict.items():
    if(other_plain_key != plain_key):
      sim_val_list.append(other_cos_sim)
      
  for other_encode_key, other_cos_sim in plain_key_enc_dict.items():
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

#---> GRAPH MATCHING-------------------------------------------------------------------
####
####
#---------------------------------------------------------------
def get_bipart_connected_components(ba_conn_comp_dict, qg_conn_comp_dict, min_sim_filter,
                                    weight_list):
  
  ba_graph = {} 
  qg_graph = {}
  
  weight_array = numpy.array(weight_list)
  
  for ba_key, qg_sim_dict in ba_conn_comp_dict.items():
    qg_key_set = set([qg_key for qg_key,val_list in qg_sim_dict.items() 
                      if(sum(numpy.array(val_list)*weight_array)) >= min_sim_filter])
    
    ba_graph[ba_key] = qg_key_set
  
  for qg_key, ba_sim_dict in qg_conn_comp_dict.items():
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
  
  logging.debug('  Largest connected component for minimum sim threhold %.5f: %d' %(min_sim_threshold,
                                                                            min_sim_comp_size)
  )
  
      
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
    
    logging.debug('  Largest connected component for maximum sim threhold %.5f: %d' %(max_sim_threshold,
                                                                              max_sim_comp_size)
    )
    logging.debug('')
    
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
      
      logging.debug('    New threshold:              ' + str(new_threshold))
      logging.debug('    Largest connected component:' + str(new_max_comp_size))
      logging.debug('')
        
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
    
    logging.debug('  Found an optimal threhold of %.4f with largest connected component size %d' %(new_threshold, new_max_comp_size))
  
  else:
    logging.debug('  ***Largest connected component is below the expected range...')
    conn_comp_list = min_sim_conn_comp_list
      
  
  return conn_comp_list

#----------------------------------------------------------------------

def random_graph_matching(edge_sim_conf_dict):
  
  # edges_list = edge_sim_conf_dict.keys()
  # edge_indices_list = [i for i in range(len(edges_list))]
  edges_list = list(edge_sim_conf_dict.keys())
  edge_indices_list = [edges_list[i] for i in range(len(edges_list))]
  
  # randomly shuffle indices list 
  #
  random.shuffle(edge_indices_list)
  
  ident_ba_keys = set()
  ident_qg_keys = set()
  
  random_sim_pair_dict = {}
  
  for qg_key, ba_key in edge_indices_list:
    # qg_key, ba_key = edges_list[edge_index]    
    if(qg_key not in ident_qg_keys and 
       ba_key not in ident_ba_keys):
      
      random_sim_pair_dict[(qg_key, ba_key)] = edge_sim_conf_dict[(qg_key, ba_key)][0]
    
  #alterado a identacao
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
  for ba_key, qg_edge_dict in ba_conn_comp_dict.items():
    ba_key_pref_list = sorted([(qg_key,sum(numpy.array(val_list)*weight_array)) 
                                for qg_key,val_list in qg_edge_dict.items()], 
                                key=lambda t: sum(numpy.array(t[1])*weight_array), 
                                reverse=True)
    
    ba_key_pref_list = [qg_item[0] for qg_item in ba_key_pref_list]
    ba_key_pref_dict[ba_key] = ba_key_pref_list
    
    if(len(ba_key_pref_list) > max_pref_list_size):
      max_pref_list_size = len(ba_key_pref_list)
  
  for qg_key, ba_edge_dict in qg_conn_comp_dict.items():
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
  
  for qg_key, ba_key in qg_assigned_ba_key_dict.items():
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
      logging.debug('   Processed %d number of connected components'%proc_num_comp)
    
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
      
      # alterado a posicao
      # munkress_sim_pair_dict[max_pair] = max_val
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
####
####
# -----------------------------------------------------------------------------
def step04(QG_sim_graph,BA_sim_graph,
            graph_node_min_degr_list,
            min_sim,
            all_sim_list,
            plain_rec_attr_val_dict,
            encode_rec_attr_val_dict,
            plain_q_gram_dict,
            encode_q_gram_dict,
            #
            sim_comp_funct_list,
            sim_hash_num_bit_list,
            sim_hash_block_funct_list,
            # sim_hash_match_sim,
            #
            hash_sim_min_tol,
            adj_hash_sim_low_tol,
            adj_hash_sim_up_tol,
            anon_hash_sim_low_tol,
            anon_hash_sim_up_tol,
            #
            plain_str,
            encode_str,
            #
            graph_sim_list,
            graph_feat_list_list,
            graph_feat_select_list,
            #
            plain_base_data_set_name,
            encode_base_data_set_name,
            #
            encode_method,
            #
            attck_res_header_list, attck_res_val_list,
            #
            sim_hash_match_sim=0, #all blocagem
            # blocagem
            graph_block_hlsh_rel_sample_size=1,
            graph_block_hlsh_num_sample=1,
            #
            plain_graph_num_node=1,
            encode_graph_num_node=1,
            attack_res_file_name='ataque.csv',
            random_seed=101,
            feat_path = 'feats' + os.sep,
            sim_diff_adjust_flag=False,
            exp_params=[]
            ):
  
  bf_hash_sim_low_tol = anon_hash_sim_low_tol
  bf_hash_sim_up_tol  = anon_hash_sim_up_tol
  tmh_hash_sim_low_tol = anon_hash_sim_low_tol
  tmh_hash_sim_up_tol = anon_hash_sim_up_tol
  cmh_hash_sim_low_tol = anon_hash_sim_low_tol
  cmh_hash_sim_up_tol = anon_hash_sim_up_tol
  #atributos
  ba_graph_num_edges = networkx.number_of_edges(BA_sim_graph.sim_graph)
  qg_graph_num_edges = networkx.number_of_edges(QG_sim_graph.sim_graph)

  # QUICKFIX PARAMS
  #TODO: FIX THIS PARAMETRES
  plain_attr_list_str,plain_num_rec_loaded, encode_attr_list_str,encode_num_rec_loaded,\
    padded_flag,q,plain_sim_funct_name,encode_sim_funct_name,encode_method_str,\
      encode_blck_method, plain_blck_method = exp_params

  q_gram_min_hash = True
  hashing_time = qg_graph_gen_time = ba_graph_gen_time = -1
  plain_num_samples = plain_sample_size = enc_sample_size = enc_num_samples = -1
  ba_graph_num_singleton = qg_graph_num_singleton = -1
  enc_min_sim = min_sim
  
  today_str = time.strftime("%Y%m%d", time.localtime())
  now_str =   time.strftime("%H%M", time.localtime())
  today_time_str = time.strftime("%Y%m%d %H:%M:%S", time.localtime())

  # Loop over different minimum node degrees
  #
  for (graph_node_min_degr_name, graph_node_min_degr) in \
        graph_node_min_degr_list:
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
      logging.debug('#### No large enough connected components (of size %d) in ' % \
            (min_conn_comp_size) + 'at least one of the two graphs' + \
            ' (need to obtain nodes with degree %d)' % (graph_node_min_degr)
      )
      continue  # Go to next minimum connected component parameter value

    # max_qg_node_degree = max(QG_conn_sim_graph.sim_graph.degree().values())
    max_qg_node_degree = 0
    for n,d in list(QG_conn_sim_graph.sim_graph.degree()):
      max_qg_node_degree = max(max_qg_node_degree,d)
    # max_ba_node_degree = max(BA_conn_sim_graph.sim_graph.degree().values())
    # max_ba_node_degree = max(list(BA_conn_sim_graph.sim_graph.degree()))
    max_ba_node_degree = 0
    for n,d in list(BA_conn_sim_graph.sim_graph.degree()):
      max_ba_node_degree = max(max_qg_node_degree,d)
    
    max_node_degree =    max(max_qg_node_degree, max_ba_node_degree)

    logging.debug('  Maximum node degrees: Q-gram graph: %d, bit array graph: %d' % \
          (max_qg_node_degree, max_ba_node_degree)
    )
    logging.debug('')

    # Generate and save plots if both graphs are not too large
    #
    if (len(QG_conn_sim_graph.sim_graph.nodes()) <= MAX_DRAW_PLOT_NODE_NUM) and \
      (len(BA_conn_sim_graph.sim_graph.nodes()) <= MAX_DRAW_PLOT_NODE_NUM):

      plain_graph_plot_file_name = 'plain-sim-graph-min_degr-%d-%s-%s'% \
                                  (graph_node_min_degr,
                                    plain_base_data_set_name, today_str)
      plain_graph_plot_file_name = plain_graph_plot_file_name+FILE_FORMAT
      encode_graph_plot_file_name = 'encode-sim-graph-min_degr-%d-%s-%s'% \
                                    (graph_node_min_degr,
                                    encode_base_data_set_name, today_str)
      encode_graph_plot_file_name = encode_graph_plot_file_name+FILE_FORMAT

      plot_save_graph(QG_conn_sim_graph.sim_graph, plain_graph_plot_file_name,
                      min_sim)
      plot_save_graph(BA_conn_sim_graph.sim_graph, encode_graph_plot_file_name,
                      min_sim)
      logging.debug('')

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
        
        

        plain_graph_feat_matrix_file_name = feat_path + plain_graph_feat_matrix_file_name
        encode_graph_feat_matrix_file_name = feat_path + encode_graph_feat_matrix_file_name
        
        logging.debug(encode_graph_feat_matrix_file_name)
        logging.debug(plain_graph_feat_matrix_file_name)
        logging.debug('')
        
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
        
        
        #carrega se existir features
        if(os.path.isfile(plain_graph_feat_matrix_file_name)):
          logging.debug('Load plain-text graph feature matrices from csv files:')
          logging.debug('  Plain-text graph feature matrix file:' + str(plain_graph_feat_matrix_file_name) )
          
          qg_feat_dict, qg_min_feat_array, qg_max_feat_array, \
                    qg_feat_name_list = \
                    load_feat_matrix_file(plain_graph_feat_matrix_file_name)
        else:
          # Generate the feature arrays for each node in the two graphs
          #
          if(graph_feat_list[0] == 'node2vec'):
            logging.debug('Calculating node embeddings for q-gram graph using Node2Vec')
            # Defining the node2vec graph
            node2vec_qg_G = node2vec.Graph(QG_conn_sim_graph.sim_graph, directed_flag, p_param, q_param)
            
            # Calculating the transitional probabilities
            node2vec_qg_G.preprocess_transition_probs()
            
            logging.debug('  Calculating random walks')
            # Calculating the random walks
            qg_walks = node2vec_qg_G.simulate_walks(num_walks, walk_length)
            
            logging.debug('  Learn embeddings using random walks')
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
              node_embd_array = qg_embd.wv[str(qg_node_key)]
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
          os.makedirs(feat_path,exist_ok=True)
          write_feat_matrix_file(qg_feat_dict, qg_feat_name_list, \
                                plain_graph_feat_matrix_file_name)
        
        if(os.path.isfile(encode_graph_feat_matrix_file_name)):
          logging.debug('Load encoded graph feature matrices from csv files:')
          logging.debug('  Encoded graph feature matrix file:   ', encode_graph_feat_matrix_file_name)
          
          ba_feat_dict, ba_min_feat_array, ba_max_feat_array, \
                    ba_feat_name_list = \
                    load_feat_matrix_file(encode_graph_feat_matrix_file_name)  
        
        else:
          
          if(graph_feat_list[0] == 'node2vec'):
            logging.debug('Calculating node embeddings for bit array graph using Node2Vec')
            # Defining the node2vec graph
            node2vec_ba_G = node2vec.Graph(BA_conn_sim_graph.sim_graph, directed_flag, p_param, q_param)
          
            # Calculating the transitional probabilities
            node2vec_ba_G.preprocess_transition_probs()
            
            logging.debug('  Calculating random walks')
            # Calculating the random walks
            ba_walks = node2vec_ba_G.simulate_walks(num_walks, walk_length)
            
            logging.debug('  Learn embeddings using random walks')
            # Convert the random walks to feature embeddings using word2vec skipgram
            # model
            ba_embd = learn_embeddings(ba_walks, num_dimensions, window_size, num_workers, num_iter)
            
            ba_feat_name_list = ['dimension-%d' %i for i in range(1, num_dimensions+1)]
            
            ba_feat_dict = {}
            ba_min_feat_array = numpy.ones(num_dimensions)
            ba_max_feat_array = numpy.zeros(num_dimensions)
            
            for ba_node_key in BA_conn_sim_graph.sim_graph.nodes():
              #TODO CONVERVETER PARA .wv
              node_embd_array = ba_embd.wv[str(ba_node_key)]
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
        logging.debug('Features and their minimum and maximum values ' + \
              '(for the plain-text / encoded data sets):')
        for (i,feat_name) in enumerate(qg_feat_name_list):
          logging.debug('  %22s: %.3f / %.3f  |  %.3f / %.3f' % \
                (feat_name, qg_min_feat_array[i], qg_max_feat_array[i],
                ba_min_feat_array[i], ba_max_feat_array[i])
          )
        logging.debug('')
        
        calc_feat_uniq_ratio = True 
        
        if(calc_feat_uniq_ratio):
          
          qg_uniq_ratio_dict = calc_uniq_ratio(qg_feat_dict, qg_feat_name_list)
          ba_uniq_ratio_dict = calc_uniq_ratio(ba_feat_dict, ba_feat_name_list)
          
          logging.debug('Features and their uniqueness ratio values ' + \
                '(for the plain-text / encoded data sets):'
          )
          for feat_name in qg_feat_name_list:
            logging.debug('  %22s: %.5f  |  %.5f' % \
                (feat_name, qg_uniq_ratio_dict[feat_name], 
                ba_uniq_ratio_dict[feat_name])
            )
        
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
          
          logging.debug('Time for generating and normalising the features to be used:' \
                + ' %.2f sec' % (feat_gen_time)
          )
          logging.debug('')
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
              raise Exception('This should not happen')

          use_num_feat = len(use_feat_set)

          logging.debug('Use the following %d selected features (feature mode: %s):' \
                % (use_num_feat, str(graph_feat_select))
          )
          if  (graph_feat_select == 'all'):
            logging.debug('  All features')
          else:
            for (feat_num, feat_name) in enumerate(qg_feat_name_list):
              if (feat_num in use_feat_set):
                gq_feat_std = qg_feat_std_list[feat_num][1]
                ba_feat_std = ba_feat_std_list[feat_num][1]
                logging.debug('  %2d: %22s (Q-gram std=%.3f / BA std=%.3f)' % \
                      (feat_num, feat_name, gq_feat_std, ba_feat_std)
                )
          logging.debug('')

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

          logging.debug('#### Reduced numbers of nodes in feature vectors: ' + \
                'QG: %d -> %d / BA: %d -> %d' % (qg_num_org_feat_vector,
                len(sel_qg_feat_dict), ba_num_org_feat_vector,
                len(sel_ba_feat_dict))
          )
          logging.debug('')

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
                            '%d-%s-%s-%s-%s-%.3f-%s-%s-%s-%d'% \
                            (q, str(padded_flag).lower(), plain_sim_funct_name,
                              str(sim_diff_adjust_flag).lower(),
                              encode_sim_funct_name, min_sim, encode_method_str,
                              today_str, graph_feat_select, use_num_feat)
          feat_histo_plot_file_name = feat_histo_plot_file_name +FILE_FORMAT

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
                raise Exception('This should not happen')

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
                    raise Exception('This should not happen')

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
                  raise Exception('This should not happen')

                hash_sim_comp_time = time.time() - start_time
                
                num_sim_pair = len(hash_sim_dict)

                logging.debug('Identified %d node pairs across data sets with a Cosine' \
                      % (num_sim_pair) + ' LSH similarity of at least %.2f' % \
                      (sim_hash_match_sim)
                )
                logging.debug('')

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

                for (key_val1, key_val2) in hash_sim_dict.keys():
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
                
                logging.debug('    Time for identifying %d similar node pairs: %.2f' \
                      % (num_sim_pair, sim_cal_time) + ' sec')
                logging.debug('')
                
                # Calculation of confidence values per edge
                #
                start_time = time.time()
                
                try:
                  #
                  max_sim_val = max(cos_sim_list)
                  min_sim_val = min(cos_sim_list)
                  max_poss_sim_conf_val = max_sim_val/min_sim_val
                  sim_conf_const = max_poss_sim_conf_val + 0.1
                except ValueError:
                  #max() arg is an empty sequence
                  #if cos_sim_list is empty
                  max_sim_val = 0
                  min_sim_val = 100
                  max_poss_sim_conf_val = 0
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
                  
                  
                #TODO: ARRUMAR ISSO, COLOCAR APOS A NORMALIZACAO
                # logging.debug(' Similarity confidence distribution')
                # logging.debug('  Min/Max/Average/Std: %.3f/%.3f/%.3f/%.3f' %(min(sim_conf_val_list), max(sim_conf_val_list), 
                #                                                     numpy.mean(sim_conf_val_list), numpy.std(sim_conf_val_list))
                # )
                # logging.debug('')
                
                # logging.debug(' Degree confidence distribution')
                # logging.debug('  Min/Max/Average/Std: %.3f/%.3f/%.3f/%.3f' %(min(degree_conf_val_list), max(degree_conf_val_list), 
                #                                                     numpy.mean(degree_conf_val_list), numpy.std(degree_conf_val_list))
                # )
                # logging.debug('')
                
                # Normalise confidence values
                #

                try:
                  max_sim_conf = max(sim_conf_val_list)
                  min_sim_conf = min(sim_conf_val_list)
                  sim_conf_diff = max_sim_conf - min_sim_conf
                except ValueError:
                  max_sim_conf = 0
                  min_sim_conf = 0
                  sim_conf_diff = 0

                
                try:
                  max_degree_conf = max(degree_conf_val_list)
                  min_degree_conf = min(degree_conf_val_list)
                  degree_conf_diff = max_degree_conf - min_degree_conf
                except ValueError:
                  max_degree_conf = 0
                  min_degree_conf = 0
                  degree_conf_diff = 0
                
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
                  
                  try:
                    norm_sim_conf = (sim_conf - min_sim_conf)/sim_conf_diff
                  except ZeroDivisionError:
                    norm_sim_conf = 0
                  try:
                    norm_degree_conf = (degree_conf - min_degree_conf)/degree_conf_diff
                  except ZeroDivisionError:
                    norm_degree_conf = 0
                  
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
                
                
                # TODO DESCOMENTAR
                # logging.debug(' Normalised similarity confidence distribution')
                # logging.debug('  Min/Max/Average/Std: %.3f/%.3f/%.3f/%.3f' %(min(sim_conf_norm_val_list), 
                #                                                     max(sim_conf_norm_val_list), 
                #                                                     numpy.mean(sim_conf_norm_val_list), 
                #                                                     numpy.std(sim_conf_norm_val_list))
                # )
                # logging.debug('')
                
                # logging.debug(' Normalised degree confidence distribution')
                # logging.debug('  Min/Max/Average/Std: %.3f/%.3f/%.3f/%.3f' %(min(degree_conf_norm_val_list), 
                #                                                     max(degree_conf_norm_val_list), 
                #                                                     numpy.mean(degree_conf_norm_val_list), 
                #                                                     numpy.std(degree_conf_norm_val_list))
                # )
                
                conf_cal_time = time.time() - start_time
                
                logging.debug('')
                logging.debug('  Time used for calculating confidence values: %.3f seconds' %conf_cal_time)
                
                
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
                
                logging.debug('Apply defined matching technique/s to find similar node pairs (node matching)')
                logging.debug('  Matching method list: ' + str(match_method_list) )
                
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
                      logging.debug('  Running Random approach to find matches')
                      matched_pair_dict = random_graph_matching(edge_sim_conf_norm_dict)
                      
                    elif(match_method == 'symmetric'):
                      logging.debug('  Running Symmetric best match approach to find matches')
                      matched_pair_dict = symmetric_graph_matching(edge_sim_conf_norm_dict, 
                                                                  ba_conn_comp_dict, 
                                                                  qg_conn_comp_dict, 
                                                                  weight_list)
                      
                    elif(match_method == 'hungarian'):
                      logging.debug('')
                      logging.debug('  Running Hungarian algorithm to find matches')
                      logging.debug('    Weight list: ' + str(weight_list) )
                      logging.debug('')
                      
                      # Lets user to input the similarity threshold so that certain
                      # size of connected components can be selected
                      #
                      user_input_flag = False
                      
                      if(user_input_flag):
                        pass
                        # conn_comp_list = get_conn_comp_user_input(ba_conn_comp_dict, 
                        #                                           qg_conn_comp_dict,
                        #                                           weight_list)
                        
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
                      logging.debug('  Running Gale-Shapley algorithm to find matches')
                      # In our context both of the below algorithms provide same
                      # results. However, greedy approach is faster.
                      
                      #matched_pair_dict1 = shapley_graph_matching(ba_conn_comp_dict, 
                      #                                           qg_conn_comp_dict, 
                      #                                           edge_sim_conf_norm_dict,
                      #                                           weight_list)
                      
                      matched_pair_dict = greedy_graph_matching(edge_sim_conf_norm_dict, weight_list)
                      
                      #assert matched_pair_dict == matched_pair_dict1
                      
                    else:
                      raise Exception('***Warning! Wrong matching method...')
                
                    match_time = time.time() - match_start_time
                    
                    logging.debug('  Time used for graph matching: %.3f seconds'%match_time)
                    logging.debug('')
                    logging.debug('')
                    
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
      
                        logging.info('      Node pair with %d highest similarity (or ' % \
                              (rank_i) + 'distance): %.5f' % (final_sim)
                        )
                        logging.info('        Plain-text entity identifiers:' + 
                              str(sorted(ent_id_set1))
                        )
                        logging.info('          Plain-text record values:' +
                              str(sorted(org_val_set1))
                        )
                        logging.info('        Encoded entity identifiers:' +
                              str(sorted(ent_id_set2))
                        )
                        logging.info('          Encoded record values:' + 
                              str(sorted(org_val_set2))
                        )
                        logging.debug('      Original feature vectors:')
                        logging.debug('       ', str(qg_feat_dict[node_key_val1]) )
                        logging.debug('       ', str( ba_feat_dict[node_key_val2]) )
                        logging.debug('      Original normalised feature vectors:')
                        logging.debug('       ', str( norm_qg_feat_dict[node_key_val1]) )
                        logging.debug('       ', str( norm_ba_feat_dict[node_key_val2]) )
                        logging.debug('      Selected feature vectors:')
                        logging.debug('       ', str( sel_qg_feat_dict[node_key_val1]) )
                        logging.debug('       ', str( sel_ba_feat_dict[node_key_val2]) )
                        logging.debug('')
      
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
                          logging.debug('    *** True match at rank %d: (%s,%s), ' % \
                                (rank_i, ent_id1, ent_id2) + ' sim/dist = %.5f' \
                                % (final_sim)
                          )
                          logging.debug('        Encoded / plain-text value pair: %s / %s' \
                                % (rec_val1, rec_val2)
                          )
                          logging.debug('        Position: %d' % \
                                (len(corr_wrong_match_list))
                          )
                          logging.debug('')
      
                      else:
                        ident_wrng_id_set = ident_wrng_id_set.union(ent_id_set2)
                        corr_wrong_match_list.append(False)
      
                      # Calculate the similarity between the q-gram sets of the two
                      # records
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
                    
                    logging.info('  Number of encode records identified correctly: %d ' %len(ident_corr_id_set))
                    logging.info('  Number of encode records identified wrongly: %d' %len(ident_wrng_id_set))
                    logging.info('')
                    
                    logging.debug('  Ranks of %d true matches:' % \
                          (len(true_match_rank_list)), true_match_rank_list[:10], \
                          '...', true_match_rank_list[-10:]
                    )
                    logging.debug('')
      
                    logging.debug('  Matching accuracy (based on true matches):')
      
                    corr_top_10 =  corr_wrong_match_list[:10].count(True)
                    corr_top_20 =  corr_wrong_match_list[:20].count(True)
                    corr_top_50 =  corr_wrong_match_list[:50].count(True)
                    corr_top_100 = corr_wrong_match_list[:100].count(True)
                    corr_top_200 = corr_wrong_match_list[:200].count(True)
                    corr_top_500 = corr_wrong_match_list[:500].count(True)
                    corr_top_1000 = corr_wrong_match_list[:1000].count(True)
                    corr_all =     corr_wrong_match_list.count(True)
                    wrng_all =     corr_wrong_match_list.count(False)
      
                    logging.debug('    Top  10: %4d (%.2f%%)' % (corr_top_10,
                                                        10.0* corr_top_10) 
                    )
                    logging.debug('    Top  20: %4d (%.2f%%)' % (corr_top_20,
                                                        5.0* corr_top_20)
                    )
                    logging.debug('    Top  50: %4d (%.2f%%)' % (corr_top_50,
                                                        2.0* corr_top_50)
                    )
                    logging.debug('    Top 100: %4d (%.2f%%)' % (corr_top_100,
                                                        float(corr_top_100))
                    )
                    logging.debug('    Top 200: %4d (%.2f%%)' % (corr_top_200,
                                                        float(corr_top_200)/2.0) 
                    )
                    logging.debug('    Top 500: %4d (%.2f%%)' % (corr_top_500,
                                                        float(corr_top_500)/5.0)
                    )
                    logging.debug('    Top 1000: %4d (%.2f%%)' % (corr_top_1000,
                                                        float(corr_top_1000)/10.0)
                    )
                    if (len(corr_wrong_match_list) > 0):
                      logging.debug('    All:     %4d (%.2f%%) from %d' % (corr_all,
                                                      100.0*float(corr_all) / \
                                                      len(corr_wrong_match_list),
                                                      len(corr_wrong_match_list))
                      )
      
                    logging.debug('  Matching accuracy (based on Jaccard value ' + \
                          'similarity of at least %.3f):' % (sim_hash_match_sim)
                    )
      
                    scorr_top_10 =  sim_diff_match_list[:10].count(True)
                    scorr_top_20 =  sim_diff_match_list[:20].count(True)
                    scorr_top_50 =  sim_diff_match_list[:50].count(True)
                    scorr_top_100 = sim_diff_match_list[:100].count(True)
                    scorr_top_200 = sim_diff_match_list[:200].count(True)
                    scorr_top_500 = sim_diff_match_list[:500].count(True)
                    scorr_top_1000 = sim_diff_match_list[:1000].count(True)
                    scorr_all =     sim_diff_match_list.count(True)
                    swrng_all =     sim_diff_match_list.count(False)
      
                    logging.debug('    Top  10: %4d (%.2f%%)' % (scorr_top_10,
                                                        10.0* scorr_top_10)
                    )
                    logging.debug('    Top  20: %4d (%.2f%%)' % (scorr_top_20,
                                                        5.0* scorr_top_20)
                    )
                    logging.debug('    Top  50: %4d (%.2f%%)' % (scorr_top_50,
                                                        2.0* scorr_top_50)
                    )
                    logging.debug('    Top 100: %4d (%.2f%%)' % (scorr_top_100,
                                                        float(scorr_top_100))
                    )
                    logging.debug('    Top 200: %4d (%.2f%%)' % (scorr_top_200,
                                                        float(scorr_top_200)/2.0)
                    )
                    logging.debug('    Top 500: %4d (%.2f%%)' % (scorr_top_500,
                                                        float(scorr_top_500)/5.0)
                    )
                    logging.debug('    Top 1000: %4d (%.2f%%)' % (scorr_top_1000,
                                                        float(scorr_top_1000)/10.0)
                    )
                    if (len(sim_diff_match_list) > 0):
                      logging.debug('    All:     %4d (%.2f%%) from %d' % (scorr_all,
                                                      100.0*float(scorr_all) / \
                                                      len(sim_diff_match_list),
                                                      len(sim_diff_match_list))
                      )
                      logging.debug('')
      
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
                      
                      logging.info('Created new result file: ' + str(attack_res_file_name))
                      
                    else:  # Append results to an existing file
                      attck_write_file = open(attack_res_file_name, 'a')
                      csv_writer = csv.writer(attck_write_file)
                      
                    csv_writer.writerow(new_attck_res_val_list)
                    attck_write_file.close()
                    
                    logging.info(' Wrote result line:' + str(new_attck_res_val_list))

    pass

# ---------------------------------------------------------------------------


def xpto():
  # -----------------------------------------------------------------------------
  # Loop over the different experimental parameter settings
  #
  logging.debug('######## match: graph_node_min_degr_name,graph_sim_list_name,' + \
          'graph_feat_list_name,graph_feat_select_name,' + \
          'sim_hash_num_bit_name,sim_hash_block_funct,' + \
          'sim_comp_funct,sim_hash_match_sim,num_feat,' + \
          'use_num_feat,num_sim_pair, corr_top_10,corr_top_20,' + \
          'corr_top_50,corr_top_100,corr_all, feat_gen_time,' + \
          'hash_sim_gen_time,hash_sim_block_time,hash_sim_comp_time'
  )