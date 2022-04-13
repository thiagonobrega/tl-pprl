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

import matplotlib
matplotlib.use('Agg')  # For running on adamms4 (no display)
import matplotlib.pyplot as plt

from graph_attack.utils import auxiliary
from graph_attack.utils.anonimization import encoding, hashing , tabminhash , colminhash
from graph_attack.utils.indexing import MinHashLSH, CosineLSH
from graph_attack.utils import simcalc
from graph_attack.utils.graph import SimGraph

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

BF_HASH_FUNCT1 = hashlib.sha1
BF_HASH_FUNCT2 = hashlib.md5

#
PLT_PLOT_RATIO = 1.0
FILE_FORMAT  = '.eps' #'.png'
PLT_FONT_SIZE    = 20# 28 # used for axis lables and ticks
LEGEND_FONT_SIZE = 20 # 28 # used for legends
TITLE_FONT_SIZE  = 19 # 30 # used for plt title
TICK_FONT_SIZE   = 18
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
        '%d / %.2f / %d / %d' % (min(hlsh_block_size_list),
                             numpy.mean(hlsh_block_size_list),
                             numpy.median(hlsh_block_size_list),
                             max(hlsh_block_size_list))
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
  
  logging.debug('')
  logging.debug('Evaluation of the %s regression model' %regre_model_str)
  logging.debug('  Explained variance score:       %.5f' %model_var_score)
  logging.debug('  Min/Max/Average absolute error: %.5f / %.5f / %.5f' %(min(err_val_list), \
                                                                 max(err_val_list), \
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
  model_min_abs_err      = min(err_val_list)
  model_max_abs_err      = max(err_val_list)
  model_avrg_abs_err     = numpy.mean(err_val_list)
  model_std_abs_err      = numpy.std(err_val_list)
  model_mean_sqrd_err    = metrics.mean_squared_error(y_test_eval, y_predict_eval)
  model_mean_sqrd_lg_err = mean_sqrd_lg_err
  model_r_sqrd           = metrics.r2_score(y_test_eval, y_predict_eval)
  
  logging.debug('')
  logging.debug('Evaluation of the loaded %s regression model' %regre_model_str)
  logging.debug('  Explained variance score:       %.5f' %model_var_score)
  logging.debug('  Min/Max/Average absolute error: %.5f / %.5f / %.5f' %(min(err_val_list), \
                                                                 max(err_val_list), \
                                                                 numpy.mean(err_val_list))
                )
  logging.debug('  Standard deviation error:       %.5f' %model_std_abs_err)
  logging.debug('  Mean-squared error:             %.5f' %model_mean_sqrd_err)
  logging.debug('  Mean-squared log error:         %.5f' %model_mean_sqrd_lg_err)
  logging.debug('  R-squared value:                %.5f' %model_r_sqrd)
  
  return False
#--------------------------------------------------------------------------

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
        regre_file_path = 'regre-models/',
        plain_graph_file_name='plain_graph',
        encode_graph_file_name='encoded_graph',
        # utilizar apenas atributos em comum
        include_only_common = False,
        common_rec_id_set={},#setar caso seja utilizado
        #nao sei o que e isso mas estava hardoced
        same_ba_blck = False,
        #ajueste de similaridade
        sim_diff_interval_size = 0.05,
        sim_diff_adjust_flag=True):
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
        '%d / %.2f / %d / %d' % (min(min_hash_block_size_list),
        numpy.mean(min_hash_block_size_list),
        numpy.median(min_hash_block_size_list),
        max(min_hash_block_size_list))
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
    logging.debug(' ', auxiliary.get_memory_usage())
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
            save_file = open(regre_file_name, 'wb')
            pickle.dump(regre_model, save_file)
            
            # Write evaluation resuls to a csv file
            #
            regre_model_eval_res_file = 'graph_attack_regression_model_eval_res.csv'
    
    
            res_header_list = ['date','time','plain_dataset_name','plain_attr_list','plain_num_rec','encode_dataset_name','encode_attr_list','encode_num_rec','q_length','padded','sim_adjust_flag','regression_model','plain_sim_funct','enc_sim_func','min_sim','encode_method','bf_hash_type/tmh_num_hash_bits/cmh_max_rand_int','num_hash_funct','bf_len/tmh_num_hash_tables/cmh_num_hash_col','bf_encode/tmh_key_len', 'bf_harden/tmh_val_len','explained_var','min_abs_err','max_abs_err','avrg_abs_err','std_abs_err','mean_sqrd_err','mean_sqrd_lg_err','r_sqrd']

            #TODO: REVER VARIAVEIS
            #TODO: REVER SAIDA
            res_val_list = [today_str, now_str, plain_base_data_set_name, plain_attr_list_str, plain_num_rec_loaded,encode_base_data_set_name, encode_attr_list_str, encode_num_rec_loaded,q, str(padded_flag).lower(), sim_diff_adjust_flag, regre_model_str.lower(),plain_sim_funct_name, encode_sim_funct_name,min_sim, encode_method]
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
        '%d / %.2f / %d / %d' % (min(min_hash_block_size_list),
                                 numpy.mean(min_hash_block_size_list),
                                 numpy.median(min_hash_block_size_list),
                                 max(min_hash_block_size_list))
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
    logging.debug(' ', auxiliary.get_memory_usage())
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

    logging.debug('Wrote graphs into pickle files:')
    logging.debug('  Plain-text graph file:', plain_graph_file_name)
    logging.debug('  Encoded graph file:   ', encode_graph_file_name)
    logging.debug(' ', auxiliary.get_memory_usage())
    logging.debug('')

    # retorno 
    # ba_graph_gen_time
    # qg_graph_gen_time,qg_graph_num_edges,qg_graph_num_singleton
    return QG_sim_graph, BA_sim_graph

# -----------------------------------------------------------------------------