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

from graph_attack.utils import auxiliary
from graph_attack.utils.anonimization import encoding, hashing , tabminhash , colminhash

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

    common_ent_id_set = set(plain_rec_attr_val_dict.keys()) & set(encode_hash_dict.keys())
    common_num_ent = len(common_ent_id_set)

    plain_num_ent =  len(plain_rec_attr_val_dict)
    encode_num_ent = len(encode_hash_dict)

    return encode_hash_dict,hashing_time,plain_num_ent, encode_num_ent # colocar outros retornos se necessario

# -----------------------------------------------------------------------------

### Step 3 methods

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------