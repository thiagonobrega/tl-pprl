import sys
import logging
import random
import time
import math

import numpy
import scipy

import binascii
import bitarray

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
        https://github.com/chrisjmccormick/MinHash/blob/master/runMinHashExample.py

       The probability for a pair of sets with Jaccard sim 0 < s <= 1 to be
       included as a candidate pair is (with b = lsh_num_band and
       r = lsh_band_size, i.e. the number of rows/hash functions per band) is
       (Leskovek et al., 2014, page 89):

         p_cand = 1- (1 - s^r)^b

       Approximation of the 'threshold' of the S-curve (Leskovek et al., 2014,
       page 90) is: t = (1/k)^(1/r).
    """

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
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

    logging.debug('Initialise LSH blocking using Min-Hash')
    logging.debug('  Number of hash functions: %d' % (self.num_hash_funct))
    logging.debug('  Number of bands:          %d' % (lsh_num_band))
    logging.debug('  Size of bands:            %d' % (lsh_band_size))
    logging.debug('  Threshold of s-curve:     %.3f' % (t))
    logging.debug('  Probabilities for candidate pairs:')
    logging.debug('   Jacc_sim | prob(cand)')
    for (s,p_cand) in s_p_cand_list:
      logging.debug('     %.2f   |   %.5f' % (s, p_cand))
    logging.debug('')

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
      crc_hash_set.add(binascii.crc32(str.encode(q_gram)) & 0xffffffff)

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
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    if (random_seed != None):
      numpy.random.seed(random_seed)

    self.vec_len =           vec_len
    self.num_hash_sign_len = num_hash_sign_len
    
    # Generate the required number of random hyper-planes (num_hash_sign_len),
    # each a vector of length 'vec_len'
    #
    self.ref_planes = numpy.random.randn(num_hash_sign_len, vec_len)

    logging.debug('Initialised CosineLSH by generating %d hyper-planes each of ' % \
          (num_hash_sign_len) + 'dimensionality %d' % (vec_len)
    )
    logging.debug('')

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

    for (val, feat_array) in feature_dict.items():
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
    logging.debug('Statistics for sim hash encoding of %d feature vectors:' % \
          (len(sim_hash_dict))
    )
    logging.debug('  Minimum, average (std-dev), median and maximum number of 1-' + \
          'bits per position:  %d / %.3f (%.3f) / %d / %d (from %d in total)' \
          % (min(bit_pos_1_count_list), numpy.mean(bit_pos_1_count_list),
             numpy.std(bit_pos_1_count_list),
             numpy.median(bit_pos_1_count_list), max(bit_pos_1_count_list),
             len(sim_hash_dict))
    )
    logging.debug('  Minimum, average (std-dev), median and maximum number of 1-' + \
          'bits per bit array: %d / %.3f (%.3f) / %d / %d (from %d in total)' \
          % (min(hw_list), numpy.mean(hw_list), numpy.std(hw_list),
             numpy.median(hw_list), max(hw_list),num_hash_sign_len)
    )
    logging.debug('')

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

    logging.debug('Number of blocks for the all-in-one index: 1 / 1')
    logging.debug('  Block sizes: %d (plain) / %d (encoded)' % \
          (len(plain_sim_block_dict['all']),
           len(encode_sim_block_dict['all']))
    )
    logging.debug('')

    return plain_sim_block_dict, encode_sim_block_dict

  # ---------------------------------------------------------------------------
  #TODO: VERIFICAR POR PROBLEMAS

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

    for sample_num in range(hlsh_num_sample):
      bit_sample_list.append(random.sample(range(bit_array_len),hlsh_sample_size))
      if (len(bit_sample_list) > 1):
        assert bit_sample_list[-1] != bit_sample_list[-2]  # Check uniqueness

    # Now apply HLSH on both similarity hash dictionaries
    #
    for (hlsh_block_dict, sim_hash_dict, dict_name) in \
        [(plain_sim_block_dict, plain_sim_hash_dict, 'plain-text'),
        (encode_sim_block_dict, encode_sim_hash_dict, 'encoded')]:

      # Loop over all similarity hash dictionary keys and bit arrays
      #
      for (key_val, dict_bit_array) in sim_hash_dict.items(): # MECHI NA IDENTACAO DESSA FOR

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
        logging.debug('Number of blocks for the %s HLSH index: %d' % \
            (dict_name, len(hlsh_block_dict)) + \
            '  (with sample size: %d, and number of samples: %d' % \
            (hlsh_sample_size, hlsh_num_sample)
        )

        hlsh_block_size_list = []
        for hlsh_block_key_set in hlsh_block_dict.values():
            hlsh_block_size_list.append(len(hlsh_block_key_set))
      
        logging.debug('  Minimum, average, median and maximum block sizes: ' + \
            '%d / %.2f / %d / %d' % (min(hlsh_block_size_list),
                                 numpy.mean(hlsh_block_size_list),
                                 numpy.median(hlsh_block_size_list),
                                 max(hlsh_block_size_list))
        )

    
    # fim do for
    # Get the HLSH keys that do not occur in both dictionaries are remove
    # them from the block dictionaries
    #
    plain_hlsh_key_set =  set(plain_sim_block_dict.keys())
    encode_hlsh_key_set = set(encode_sim_block_dict.keys())

    plain_only_hlsh_key_set =  plain_hlsh_key_set - encode_hlsh_key_set
    encode_only_hlsh_key_set = encode_hlsh_key_set - plain_hlsh_key_set
    logging.debug('')
    logging.debug('  Number of HLSH blocks that occur in plain-text only: %d' % \
          (len(plain_only_hlsh_key_set))
    )
    logging.debug('  Number of HLSH blocks that occur in encoded only: %d' % \
          (len(encode_only_hlsh_key_set))
    )

    # Remove all not common blocks
    #
    for plain_only_block_key in plain_only_hlsh_key_set:
      del plain_sim_block_dict[plain_only_block_key]
    for encode_only_block_key in encode_only_hlsh_key_set:
      del encode_sim_block_dict[encode_only_block_key]

    logging.debug('    Final numbers of blocks: %d (plain) / %d (encoded)' % \
          (len( plain_sim_block_dict), len(encode_sim_block_dict))
    )
    logging.debug('  Time used: %.3f sec' % (time.time() - start_time))
    logging.debug('')

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
    for (plain_block_key, plain_key_set) in plain_sim_block_dict.items():
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

    logging.debug('Compared %d record pairs based on their Cosine LSH hash values' % \
          (num_pairs_compared) + ' (using all-pair comparison)'
    )
    logging.debug('  %d of these pairs had a similarity of at least %.2f' % \
          (len(hash_sim_dict), hash_min_sim)
    )

    logging.debug('  Time used: %.3f sec' % (time.time() - start_time))
    logging.debug('')

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
    for (plain_block_key, plain_key_set) in plain_sim_block_dict.items():
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

    logging.debug('Compared %d record pairs based on their Cosine LSH hash values' % \
          (num_pairs_compared) + ' (using sim-edge comparison)'
    )
    logging.debug('  %d of these pairs had a similarity of at least %.2f' % \
          (len(hash_sim_dict), hash_min_sim)
    )
    logging.debug('  %d pairs filtered due to their different maximum edge ' % \
          (num_pair_sim_filter) + 'similarity'
    )
    logging.debug('  %d pairs filtered due to their different number of edges' % \
          (num_pair_edge_filter)
    )
    
    logging.debug('  Time used: %.3f sec' % (time.time() - start_time))
    logging.debug('')

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

    for (val1,val2) in hash_sim_dict.keys():
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

    for (val1,val2) in hash_sim_dict.keys():
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

    for (val1,val2) in hash_sim_dict.keys():

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
