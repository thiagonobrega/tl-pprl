# encoding.py - Module that implements several Bloom filter encoding methods
#
# June 2018

# Peter Christen, Thilina Ranbaduge, Sirintra Vaiwsri, and Anushka Vidanage
#
# Contact: peter.christen@anu.edu.au
#
# Research School of Computer Science, The Australian National University,
# Canberra, ACT, 2601
# -----------------------------------------------------------------------------
#
# Copyright 2019 Australian National University and others.
# All Rights reserved.
#
# -----------------------------------------------------------------------------
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# =============================================================================

import random

import numpy  # For the dynamic bitposition selection in Record level BFs

import bitarray

PAD_CHAR = chr(1)   # Used for q-gram padding

# =============================================================================

class AttributeBFEncoding():
  """Attribute-level Bloom filter encoding was proposed and used by:
       - R. Schnell, T. Bachteler, and J. Reiher, Privacy-preserving record
         linkage using Bloom filters, BMC MIDM, vol. 9, no. 1, 2009.

     Note that for attribute level Bloom filters one instance of this class
     needs to be defined per attribute to be encoded.
  """

  # ---------------------------------------------------------------------------

  def __init__(self, attr_num, q, padded, hash_class):
    """Initialise the attribute level Bloom filter class by providing the
       required parameters.

       Input arguments:
         - attr_num    The number of the attribute to be encoded into a Bloom
                       filter.
         - q           The length of q-grams to be generated from the value
                       in attribute 'attr_num' in records. 
         - padded      A flag, if set to True then the attribute values to
                       be encoded are first padded before q-grams are being
                       generated.
         - hash_class  The hashing class which provides the hashing function
                       (as implemented in the hashing.py module which
                       generates one Bloom filter).

       Output:
         - This method does not return anything.
    """

    self.type = 'ABF'  # To identify the encoding method

    # Initialise the class variables
    #
    assert attr_num >= 0, attr_num
    assert q >= 1, q
    assert padded in [True, False]

    self.attr_num =   attr_num
    self.q =          q
    self.padded =     padded
    self.hash_class = hash_class

  # ---------------------------------------------------------------------------

  def encode(self, attr_val_list, salt_str=None, mc_harden_class=None):
    """Encode the value in attribute 'attr_num' in the given attribute value
       list into a Bloom filter (according to the 'hash_method' provided), and
       return this Bloom filter.

       Input arguments:
         - attr_val_list    A list of attribute values, assumed to come from a
                            record in a database. Only the attribute at index
                            number 'attr_num' will be encoded.
         - salt_str         An optional string used for salting, if provided
                            this string will be concatenated with every q-gram
                            in the given q-gram set. If set to None no salting
                            will be done.
         - mc_harden_class  An optional parameter, if given it must be a
                            reference to a Markov chain hardening class which
                            will be used to generate extra q-grams.

       Output:
         - bf               A Bloom filter with bits set according to the
                            hashing method and the value in the selected
                            attribute.
         - q_gram_pos_dict  [Only returned if 'get_q_gram_pos' is set to True
                            in the hashing method]
                            A dictionary which has q-grams as keys and where
                            values are sets with the positions these q-grams
                            are hashed to.
    """

    # Check there are enough attribute values
    #
    if (self.attr_num >= len(attr_val_list)):
        raise Exception('Not enough attributes provided')

    q = self.q  # Short-cuts
    qm1 = q - 1

    # Get attribute value and convert it into a q-gram set
    #
    attr_val = attr_val_list[self.attr_num]

    if (self.padded == True):  # Add padding start and end characters
      attr_val = PAD_CHAR*qm1+attr_val+PAD_CHAR*qm1

    attr_val_len = len(attr_val)

    q_gram_set = set([attr_val[i:i+q] for i in range(attr_val_len - qm1)])

    if (mc_harden_class != None):
      extra_q_gram_set = \
               mc_harden_class.get_other_q_grams_from_lang_model(q_gram_set)
      q_gram_set = q_gram_set | extra_q_gram_set

    # Check if q-gram position dictionary is required or not
    #
    if (self.hash_class.get_q_gram_pos == True):
      bf, q_gram_pos_dict = self.hash_class.hash_q_gram_set(q_gram_set, 
                                                            salt_str)
      return bf, q_gram_pos_dict

    else:
      bf = self.hash_class.hash_q_gram_set(q_gram_set, salt_str)
      return bf

# =============================================================================

class CryptoLongtermKeyBFEncoding():
  """Cryptographic longterm key Bloom filter encoding as proposed and used by:
       - R. Schnell, T. Bachteler, and J. Reiher, A novel error-tolerant
         anonymous linking code, German Record Linkage Center, Working Paper
         Series No. WP-GRLC-2011-02. 2011.
       - R. Schnell, Privacy-Preserving Record Linkage, book chapter in
         Methodological Developments in Data Linkage, K. Harron, H Goldstein,
         and C. Dibben, Wiley, 2015.
  """

  # ---------------------------------------------------------------------------

  def __init__(self, attr_encode_tuple_list):
    """Initialise the cryptographic long-term key Bloom filter class by
       providing the required parameters.

       Input arguments:
         - attr_encode_tuple_list  This single argument is a list made of
                                   tuples of length four, where each tuple
                                   defines how values from a certain attribute
                                   are encoded. Each tuple must contain:
                                   - attr_num    The number of the attribute
                                                 to be encoded into a Bloom
                                                 filter.
                                   - q           The length of q-grams to be
                                                 generated from the value
                                                 in attribute 'attr_num' in
                                                 records. 
                                   - padded      A flag, if set to True then
                                                 the attribute values to
                                                 be encoded are first padded
                                                 before q-grams are being
                                                 generated.
                                   - hash_class  The hashing class which
                                                 provides the hashing function
                                                 (as implemented in the
                                                 hashing.py module which
                                                 generates one Bloom filter).

       Output:
         - This method does not return anything.

       Important is that all hashing methods must generate Bloom filters of the
       same length (individual attribute level Bloom filters which will be
       combined using bit-wise OR).
    """

    self.type = 'CLK'  # To identify the encoding method

    self.attr_encode_tuple_list = attr_encode_tuple_list

  # ---------------------------------------------------------------------------

  def encode(self, attr_val_list, salt_str_list=None, mc_harden_class=None):
    """Encode values in the given 'attr_val_list' according to the settings
       provided in the 'attr_encode_tuple_list'.

       Input arguments:
         - attr_val_list    A list of attribute values, assumed to come from a
                            record in a database. Only the attributes at the
                            indexes in the 'attr_encode_tuple_list' will be
                            encoded.
         - salt_str_list    An optional list of strings used for salting. If
                            provided then for each attribute to be encoded
                            either a salt string value needs to be given which
                            will be concatenated with every q-gram in the
                            q-gram set of this attribute. If set to None for
                            a certain attribute then no salting will be done
                            for that attribute.
         - mc_harden_class  An optional parameter, if given it must be a
                            reference to a Markov chain hardening class which
                            will be used to generate extra q-grams.

       Output:
         - bf               A Bloom filter with bits set according to the
                            hashing method and the value in the selected
                            attributes.
         - q_gram_pos_dict  [Only returned if 'get_q_gram_pos' is set to True
                            in the hashing method]
                            A dictionary which has q-grams as keys and where
                            values are sets with the positions these q-grams
                            are hashed to.
    """

    clk_bf = None  # Start with a non-initialised BF

    get_bit_pos_flag = self.attr_encode_tuple_list[0][3].get_q_gram_pos
    assert (get_bit_pos_flag in [True,False])

    if (get_bit_pos_flag == True):
      all_q_gram_pos_dict = {}  # A dictionary with q-grams and their positions

    for (j, attr_encode_tuple) in enumerate(self.attr_encode_tuple_list):
      attr_num =   attr_encode_tuple[0]
      q =          attr_encode_tuple[1]
      padded =     attr_encode_tuple[2]
      hash_class = attr_encode_tuple[3]

      # Check there are enough attribute values
      #
      if (attr_num >= len(attr_val_list)):
          raise Exception('Not enough attributes provided')

      if (salt_str_list != None):
        salt_str = salt_str_list[j]
      else:
        salt_str = None

      qm1 = q - 1

      # Get attribute value and convert it into a q-gram set
      #
      attr_val = attr_val_list[attr_num]

      if (padded == True):  # Add padding start and end characters
        attr_val = PAD_CHAR*qm1+attr_val+PAD_CHAR*qm1

      attr_val_len = len(attr_val)

      q_gram_set = set([attr_val[i:i+q] for i in range(attr_val_len - qm1)])

      if (mc_harden_class != None):
        extra_q_gram_set = \
                 mc_harden_class.get_other_q_grams_from_lang_model(q_gram_set)
        q_gram_set = q_gram_set | extra_q_gram_set

      # Check if q-gram position dictionary is required or not
      #
      if (get_bit_pos_flag == True):
        bf, q_gram_pos_dict = hash_class.hash_q_gram_set(q_gram_set, salt_str)
        
        for (q_gram, pos_set) in q_gram_pos_dict.items():
          all_pos_set = all_q_gram_pos_dict.get(q_gram, set())
          all_pos_set.update(pos_set)
          all_q_gram_pos_dict[q_gram] = all_pos_set
        
      else:
        bf = hash_class.hash_q_gram_set(q_gram_set, salt_str)

      if (clk_bf == None):
        clk_bf = bf
      else:
        assert len(bf) == len(clk_bf)  # All BFs must be of same length

        clk_bf = clk_bf | bf  # Binary OR of attribute BFs.

    if (get_bit_pos_flag == True):
      return clk_bf, all_q_gram_pos_dict
    else:
      return clk_bf

# =============================================================================

class RecordBFEncoding():
  """Record-level Bloom filter encoding was proposed and used by:
       - E.A. Durham, M. Kantarcioglu, Y. Xue, C. Toth, M. Kuzu and B. Malin,
         Composite Bloom filters for secure record linkage. IEEE TKDE 26(12),
         p. 2956-2968, 2014.
  """

  # ---------------------------------------------------------------------------

  def __init__(self, attr_encode_tuple_list, random_seed=42):
    """Initialise the record level Bloom filter class by providing the
       required parameters.

       Input arguments:
         - attr_encode_tuple_list  This argument is a list made of tuples of
                                   length four, where each tuple defines how
                                   values from a certain attribute are
                                   encoded. Each tuple must contain:
                                   - attr_num    The number of the attribute
                                                 to be encoded into a Bloom
                                                 filter.
                                   - q           The length of q-grams to be
                                                 generated from the value
                                                 in attribute 'attr_num' in
                                                 records. 
                                   - padded      A flag, if set to True then
                                                 the attribute values to
                                                 be encoded are first padded
                                                 before q-grams are being
                                                 generated.
                                   - hash_class  The hashing class which
                                                 provides the hashing function
                                                 (as implemented in the
                                                 hashing.py module which
                                                 generates one Bloom filter).
                                   - num_bf_bit  The number of bits to be
                                                 sampled (from the attribute
                                                 level Bloom filter for this
                                                 attribute) to be included in
                                                 the final record level Bloom
                                                 filter.
         - random_seed             The value used to seed the random generator
                                   used to shuffle the bits in the final record
                                   level Bloom filter. If no random shuffling
                                   should be done set the value of this
                                   argument to None. Default value is set
                                   to 42.

       Output:
         - This method does not return anything.

       Note that the final length of the generated record level Bloom filter
       is the sum of the number of 'num_bf_bit' over all attributes in the
       given 'attr_encode_tuple_list'.

       Also note that Durham et al. proposed to select bit positions into
       record level Bloom filters based on the number of Bloom filters that
       have 1-bits in a certain given position. We currently do not implement
       this but rather select bit positions randomly. We leave this improvement
       as future work (as it would require to first generate all Bloom filters
       for a database, then to analyse the number of 1-bits per bit position,
       and then to select bit positions accordingly).
    """

    self.type = 'RBF'  # To identify the encoding method

    # Check if bit positions of q-grams should be collected in a dictionary
    #
    self.get_bit_pos_flag = attr_encode_tuple_list[0][3].get_q_gram_pos
    assert (self.get_bit_pos_flag in [True,False])

    rbf_bf_len = 0

    # Sum bits to be selected per attribute Bloom filter to generate record
    # level Bloom filter, and check if all bit position flags are consistent
    #
    for attr_encode_tuple in attr_encode_tuple_list:
      assert attr_encode_tuple[4] > 1, attr_encode_tuple
      rbf_bf_len += attr_encode_tuple[4]

      assert attr_encode_tuple[3].get_q_gram_pos == self.get_bit_pos_flag

    self.rbf_bf_len = rbf_bf_len

    self.attr_encode_tuple_list = attr_encode_tuple_list

    # Store the random seed so each record level Bloom filter can be permuted
    # in the same way
    #
    self.random_seed = random_seed

    if (random_seed != None):  # Generate the bit position permutation list

      random.seed(random_seed)

      # Generate a permutation list using random shuffling of bit positions
      #
      perm_pos_list = range(rbf_bf_len)
      random.shuffle(perm_pos_list)
      self.perm_pos_list = perm_pos_list

    else:
      self.perm_pos_list = None

  # ---------------------------------------------------------------------------

  def get_avr_num_q_grams(self, rec_val_list):
    """Using the given list of records, where each record contains a list of
       attribute values, this method calculates the average number of q-grams
       for each attribute which is listed in the 'attr_encode_tuple_list'. The
       method returns a dictionary where keys are attribute numbers and values
       their corresponding average number of q-grams.

       Input arguments:
         - rec_val_list  A list of records each consisting of a list of
                         attribute values.

       Output:
         - avr_num_q_gram_dict  A dictionary where keys are attribute
                                numbers and values their corresponding
                                average number of q-grams.
    """

    avr_num_q_gram_dict = {}

    print 'Calculate average number of q-grams for attributes:'

    # Loop over the attributes for which the average number of q-grams is to
    # be calculated.
    #
    for attr_encode_tuple in self.attr_encode_tuple_list:
      attr_num =    attr_encode_tuple[0]
      q =           attr_encode_tuple[1]
      padded =      attr_encode_tuple[2]

      qm1 = q - 1

      q_gram_lengh_sum =  0.0
      number_of_q_grams = 0

      # Loop over all records and extract the required attribute value
      #
      for attr_val_list in rec_val_list:
        attr_val = attr_val_list[attr_num] # Get attribute value

        if (padded == True):  # Add padding start and end characters
          attr_val = PAD_CHAR*qm1+attr_val+PAD_CHAR*qm1

        attr_val_len = len(attr_val)
  
        # Convert attribute value into its q-grams
        #
        q_gram_set = set([attr_val[i:i+q] for i in range(attr_val_len - qm1)])

        q_gram_lengh_sum +=  len(q_gram_set)
        number_of_q_grams += 1

      # Calculate the average number of q-grams for this attribute
      #
      attr_avr_num_q_gram = q_gram_lengh_sum / number_of_q_grams

      print '  Attribute number %d has an average number of q-grams of %.2f' \
            % (attr_num, attr_avr_num_q_gram)

      assert attr_num not in avr_num_q_gram_dict
      avr_num_q_gram_dict[attr_num] = attr_avr_num_q_gram

    return avr_num_q_gram_dict

  # ---------------------------------------------------------------------------

  def get_dynamic_abf_len(self, avr_num_q_gram_dict, num_hash_funct, 
                          fill_prob=0.5):
    """Using provided the average numbers of q-grams for each attribute
       (assumed to have been calculated using the 'get_avr_num_q_grams' method)
       calculate the length of the each attribute-level Bloom filter (ABF).

       According to Durham et al. (2014), using dynamic lengths for each ABF
       will improve the security of the final RBF. They calculate the length
       of each ABF such that the probability of 1-bits in the ABF should be
       0.5. Given that this probability, p, is set to 0.5, the length of the
       ABF, m, is calculated using the following;

         m = 1/(1 - p^(1/(k*g)))

       where, k is the number of hash functions and g is the average number of
       q-grams.
 
       Input arguments:
         - avr_num_q_gram_dict  A dictionary where keys are attribute
                                numbers and values their corresponding
                                average number of q-grams.
         - num_hash_funct       The number of hash functions to be used.
         - fill_prob            The probability of setting bits in the ABF
                                to 1. The default is 0.5 (equal likelihood of
                                0 and 1 bits).
       Output:
         - dyn_abf_len_dict  A dictionary of ABF lengths where keys are
                             attribute numbers and values are corresponding
                             ABF lengths.
    """

    dyn_abf_len_dict = {}

    print 'Calculate Bloom filter lengths for attributes:'

    # Loop over each attribute which will be used in encoding
    #
    for (attr_num, attr_avr_num_q_gram) in sorted(avr_num_q_gram_dict.items()):

      # Calculate the ABF length using above mentioned equation
      #
      divisor = 1 - (fill_prob**(1/(attr_avr_num_q_gram*num_hash_funct)))
      dyn_abf_size = int(round(1.0 / divisor))

      dyn_abf_len_dict[attr_num] = dyn_abf_size

      print '  Attribute number %d has an average number of q-grams of %.2f' \
          % (attr_num, attr_avr_num_q_gram) + ' resulting an Bloom filter ' + \
          'length of %d' % (dyn_abf_size)

    return dyn_abf_len_dict

  # ---------------------------------------------------------------------------

  def set_abf_len(self, abf_len_dict):
    """Set the length of the attribute level Bloom filters based on the given
       dictionary which has attribute numbers as keys and corresponding Bloom
       filter length as values.

       Input arguments:
         - abf_len_dict  A dictionary of ABF lengths where keys are attribute
                         numbers and values are corresponding ABF lengths.

       Output:
         - This method does not return anything.
    """

    for attr_encode_tuple in self.attr_encode_tuple_list:
      attr_num = attr_encode_tuple[0]

      if (attr_num in abf_len_dict):
        hash_class = attr_encode_tuple[3]
        hash_class.bf_len = abf_len_dict[attr_num]

  # ---------------------------------------------------------------------------

  def encode(self, attr_val_list, salt_str_list=None, mc_harden_class=None):
    """Encode values in the given 'attr_val_list' according to the settings
       provided in the 'attr_encode_tuple_list', and the optional salting
       values list and Markov chain hardening class.

       Input arguments:
         - attr_val_list    A list of attribute values, assumed to come from
                            a record in a database. Only the attributes at
                            the indexes in the 'attr_encode_tuple_list' will
                            be encoded.
         - salt_str_list    An optional list of strings used for salting. If
                            provided then for each attribute to be encoded
                            either a salt string value needs to be given
                            which will be concatenated with every q-gram in
                            the q-gram set of this attribute. If set to None
                            for a certain attribute then no salting will be
                            done for that attribute.
         - mc_harden_class  An optional parameter, if given it must be a
                            reference to a Markov chain hardening class which
                            will be used to generate extra q-grams.

       Output:
         - bf               A record Bloom filter with bits set according to
                            the hashing method and the value in the selected
                            attributes.
         - q_gram_pos_dict  [Only returned if 'get_q_gram_pos' is set to True
                            in the hashing method]
                            A dictionary which has q-grams as keys and where
                            values are sets with the positions these q-grams
                            are hashed to.
    """

    rbf_bf_len = self.rbf_bf_len

    attr_encode_tuple_list = self.attr_encode_tuple_list

    get_bit_pos_flag = self.get_bit_pos_flag

    if (get_bit_pos_flag == True):
      rbf_q_gram_pos_dict = {}  # A dictionary with q-grams and their positions

    # Initalise the record level Bloom filter
    #
    rbf_bf = bitarray.bitarray(rbf_bf_len)

    # A pointer into the record level Bloom filter for setting bit positions
    # from the different attributes
    #
    rbf_bit_pos = 0

    for (j, attr_encode_tuple) in enumerate(self.attr_encode_tuple_list):
      attr_num =   attr_encode_tuple[0]
      q =          attr_encode_tuple[1]
      padded =     attr_encode_tuple[2]
      hash_class = attr_encode_tuple[3]
      num_bf_bit = attr_encode_tuple[4]

      # Check there are enough attribute values
      #
      if (attr_num >= len(attr_val_list)):
          raise Exception('Not enough attributes provided')

      qm1 = q - 1

      if (salt_str_list != None):
        salt_str = salt_str_list[j]
      else:
        salt_str = None

      # Get attribute value and convert it into a q-gram set
      #
      attr_val = attr_val_list[attr_num]

      if (padded == True):  # Add padding start and end characters
        attr_val = PAD_CHAR*qm1+attr_val+PAD_CHAR*qm1

      attr_val_len = len(attr_val)

      q_gram_set = set([attr_val[i:i+q] for i in range(attr_val_len - qm1)])

      if (mc_harden_class != None):
        extra_q_gram_set = \
                 mc_harden_class.get_other_q_grams_from_lang_model(q_gram_set)
        q_gram_set = q_gram_set | extra_q_gram_set

      # Check if q-gram position dictionary is required or not
      #
      if (get_bit_pos_flag == True):
        abf, q_gram_pos_dict = hash_class.hash_q_gram_set(q_gram_set, salt_str)

        # Convert into a position / q-gram dictionary for merging into RBF
        #
        pos_q_gram_dict = {}
        for (q_gram, pos_set) in q_gram_pos_dict.items():
          for pos in pos_set:
            q_gram_set = pos_q_gram_dict.get(pos, set())
            q_gram_set.add(q_gram)
            pos_q_gram_dict[pos] = q_gram_set

      else:
        abf = hash_class.hash_q_gram_set(q_gram_set, salt_str)

      abf_len = len(abf)  # Get the length of this attribute level Bloom filter

      # When ABF length is set to dynamic the output ABF length may not always
      # be higher or equal to the number of bits that need to be sampled from
      # that ABF. This is because the number of bits that need to be sampled
      # are defined by the user prior to the dynamic ABF length calculation.
      #
      # assert abf_len >= num_bf_bit, (abf_len, num_bf_bit)

      # Sample a desired number of bit positions (set the random seed to the
      # attribute number to make sure for the same attribute the same bit
      # positions are sampled each time)
      #
      if(abf_len >= num_bf_bit):
        random.seed(attr_num)
        use_bit_pos_list = random.sample(range(abf_len), num_bf_bit)

      else:  # Sampling with replacement
        numpy.random.seed(attr_num)
        use_bit_pos_list = range(abf_len)  # Make sure all bits are included
        more_sample_bits_needed = num_bf_bit - len(use_bit_pos_list)
        use_bit_pos_list += list(numpy.random.choice(range(abf_len),
                                 more_sample_bits_needed))

      assert len(use_bit_pos_list) == num_bf_bit

      for abf_bit_pos in use_bit_pos_list:
        rbf_bf[rbf_bit_pos] = abf[abf_bit_pos]

        if (get_bit_pos_flag == True):

          # Get all q-grams at this bit position in the ABF
          #
          pos_q_gram_set = pos_q_gram_dict.get(abf_bit_pos, set())

          # Keep the RBF bit positions for this q-gram
          #
          for q_gram in pos_q_gram_set:
            q_gram_pos_set = rbf_q_gram_pos_dict.get(q_gram, set())
            q_gram_pos_set.add(rbf_bit_pos)
            rbf_q_gram_pos_dict[q_gram] = q_gram_pos_set

        rbf_bit_pos += 1

    assert rbf_bit_pos == rbf_bf_len

    # Do final permutation of the RBF if required
    #
    if (self.random_seed != None):
      perm_rbf_bf = bitarray.bitarray(rbf_bf_len)

      perm_pos_list = self.perm_pos_list

      for pos in range(rbf_bf_len):
        perm_rbf_bf[pos] = rbf_bf[perm_pos_list[pos]]

      # If needed also change the bit positions of all q-grams
      #
      if (get_bit_pos_flag == True):

        for q_gram in rbf_q_gram_pos_dict:
          org_q_gram_set =  rbf_q_gram_pos_dict.get(q_gram, set())
          perm_q_gram_set = set()

          for pos in org_q_gram_set:
            perm_q_gram_set.add(perm_pos_list[pos])

          rbf_q_gram_pos_dict[q_gram] = perm_q_gram_set

    else:
      perm_rbf_bf = rbf_bf  # No permutation to be done

    if (get_bit_pos_flag == True):
      return perm_rbf_bf, rbf_q_gram_pos_dict
    else:
      return perm_rbf_bf

# -----------------------------------------------------------------------------
#
# The CLK-RBF approach, as proposed in:
#
#  D. Vatsalan, P. Christen, C.M. O'Keefe and V.S. Verykios, An evaluation
#  framework for privacy-preserving record linkage, In Journal of Privacy and
#  Confidentiality 6 (1), 3, 2014.
#
# can easily be simulated using the above implemented
# CryptoLongtermKeyBFEncoding() class by setting different numbers of hash
# functions for the different attribute hashing methods.
#
# Therefore no specific implementation is required.

# =============================================================================
# Some testing code if called from the command line

# if (__name__ == '__main__'):

#   print 'Running some tests:'
#   print

#   import hashlib    # A standard Python library
#   import hashing    # Bloom filter hashing module
#   import hardening  # Bloom filter hardening module

#   # Define two hash functions
#   #
#   bf_hash_funct1 = hashlib.sha1
#   bf_hash_funct2 = hashlib.md5
#   bf_hash_funct3 = hashlib.sha224

#   # Define Bloom filter hashing parameters
#   #
#   bf_len = 1000
#   k =      10

#   # Define four hashing methods
#   #
#   DH =  hashing.DoubleHashing(bf_hash_funct1, bf_hash_funct2, bf_len, k)
#   RH =  hashing.RandomHashing(bf_hash_funct1, bf_len, 2*k)
#   EDH = hashing.EnhancedDoubleHashing(bf_hash_funct1, bf_hash_funct2,
#                                       bf_len, 3*k)
#   TH =  hashing.TripleHashing(bf_hash_funct1, bf_hash_funct2, bf_hash_funct3,
#                               bf_len, 2*k)

#   # Define several records
#   #
#   rec_list = [['sean', 'smith',  'sydney',     '2000', 'nsw'],
#               ['mary', 'miller', 'melbourne',  '3000', 'vic'],
#               ['tony', 'tinker', 'townsville', '7123', 'qld']]

#   print '  Testing attribute level Bloom filter encoding...',  # - - - - - - -

#   # Initialise the attribute level Bloom filter encoding class

#   # Setup encoding of first names with q=2 using double hashing)
#   #
#   ABFencode_first_name2 = AttributeBFEncoding(0, 2, False, DH)

#   # Setup encoding of first names with q=4, padded, and random hashing
#   #
#   ABFencode_first_name4 = AttributeBFEncoding(0, 4, True, RH)

#   # Setup encoding of town names with q=3 and random hashing
#   #
#   ABFencode_town_name = AttributeBFEncoding(2,3, False, RH)

#   # Encode the three records
#   #
#   for attr_val_list in rec_list:
#     ss = attr_val_list[3]  # Use postcode as salting value

#     first_name_bf1 = ABFencode_first_name2.encode(attr_val_list)
#     print first_name_bf1
#     first_name_bf2 = ABFencode_first_name2.encode(attr_val_list, salt_str=ss)
#     assert len(first_name_bf1) == bf_len
#     assert len(first_name_bf2) == bf_len

#     assert first_name_bf1 != first_name_bf2  # Very unlikely they are the same

#     first_name_bf3 = ABFencode_first_name4.encode(attr_val_list)
#     first_name_bf4 = ABFencode_first_name4.encode(attr_val_list, salt_str=ss)
#     assert len(first_name_bf3) == bf_len
#     assert len(first_name_bf4) == bf_len

#     assert first_name_bf3 != first_name_bf4  # Very unlikely they are the same

#     town_name_bf1 = ABFencode_town_name.encode(attr_val_list)
#     town_name_bf2 = ABFencode_town_name.encode(attr_val_list, salt_str=ss)
#     assert len(town_name_bf1) == bf_len
#     assert len(town_name_bf2) == bf_len

#     assert town_name_bf1 != town_name_bf2  # Very unlikely they are the same

#   # Check g-gram position dictionary
#   #
#   DH2 = hashing.DoubleHashing(bf_hash_funct1, bf_hash_funct2, bf_len, k, True)

#   ABFencode_first_name3 = AttributeBFEncoding(0, 2, False, DH2)

#   for attr_val_list in rec_list:
#     first_name_bf1, abf_q_gram_pos_dict = \
#                                  ABFencode_first_name3.encode(attr_val_list)
#     assert len(abf_q_gram_pos_dict) > 0

#     for (q_gram, pos_set) in abf_q_gram_pos_dict.items():
#       assert q_gram in attr_val_list[0]  # Q-gram must be in attribute number 0
#       assert len(pos_set) > 0 and len(pos_set) <= k
#       for pos in pos_set:
#         assert pos >= 0 and pos < bf_len

#   print 'OK'
#   print

#   print '  Testing cryptographic long-term key encoding',  # - - - - - - - - -

#   # Define the hashing method used (traditional CLK uses the same hashing
#   # method for all attributes to be hashed into a CLK BF)
#   #
#   DH =  hashing.DoubleHashing(bf_hash_funct1, bf_hash_funct2, bf_len, k)

#   attr_tuple_list1 = [(0, 2, False, DH), (1, 3, True, DH)]

#   attr_tuple_list2 = [(1, 2, False, DH), (0, 2, True, DH), (2, 4, False, DH)]

#   attr_tuple_list3 = [(0, 2, True, DH), (2, 3, False, DH), (3, 2, True, DH),
#                       (4, 2, False, DH)]

#   CLKtuple1 = CryptoLongtermKeyBFEncoding(attr_tuple_list1)

#   CLKtuple2 = CryptoLongtermKeyBFEncoding(attr_tuple_list2)

#   CLKtuple3 = CryptoLongtermKeyBFEncoding(attr_tuple_list3)

#   salt_str_list = ['ab', '43', None, '#']

#   for attr_val_list in rec_list:

#     rec_bf1 = CLKtuple1.encode(attr_val_list)
#     print rec_bf1
#     rec_bf2 = CLKtuple1.encode(attr_val_list, salt_str_list)

#     assert len(rec_bf1) == bf_len
#     assert len(rec_bf2) == bf_len

#     assert rec_bf1 != rec_bf2  # Very unlikely they are the same
#     assert rec_bf1.count(1) > 0
#     assert rec_bf2.count(1) > 0

#     rec_bf3 = CLKtuple2.encode(attr_val_list)
#     rec_bf4 = CLKtuple2.encode(attr_val_list, salt_str_list)

#     assert len(rec_bf3) == bf_len
#     assert len(rec_bf4) == bf_len

#     assert rec_bf3 != rec_bf4  # Very unlikely they are the same
#     assert rec_bf3.count(1) > 0
#     assert rec_bf4.count(1) > 0

#     rec_bf5 = CLKtuple3.encode(attr_val_list)
#     rec_bf6 = CLKtuple3.encode(attr_val_list, salt_str_list)

#     assert len(rec_bf5) == bf_len
#     assert len(rec_bf6) == bf_len

#     assert rec_bf5 != rec_bf6  # Very unlikely they are the same
#     assert rec_bf5.count(1) > 0
#     assert rec_bf6.count(1) > 0

#   # Check g-gram position dictionary
#   #
#   DH2 = hashing.DoubleHashing(bf_hash_funct1, bf_hash_funct2, bf_len, k, True)

#   attr_tuple_list4 = [(0, 2, False, DH2), (1, 3, False, DH2)]

#   CLKtuple4 = CryptoLongtermKeyBFEncoding(attr_tuple_list4)

#   for attr_val_list in rec_list:
#     rec_bf4, clk_q_gram_pos_dict = CLKtuple4.encode(attr_val_list)
#     assert len(clk_q_gram_pos_dict) > 0

#     for (q_gram, pos_set) in clk_q_gram_pos_dict.items():
#       assert q_gram in attr_val_list[0]+attr_val_list[1]

#       assert len(pos_set) > 0 and len(pos_set) <= k
#       for pos in pos_set:
#         assert pos >= 0 and pos < bf_len

#   print 'OK'
#   print

#   print '  Testing record-level Bloom filter encoding...',  # - - - - - - - - -

#   # Test static length RBF first
#   #
#   rec_tuple_list1 = [(0, 2, False, DH, 400), (1, 3, True, DH, 600)]

#   rec_tuple_list2 = [(1, 2, False, RH, 500), (0, 2, True, DH, 200),
#                      (2, 4, False, RH, 300)]

#   rec_tuple_list3 = [(0, 2, False, EDH, 400), (2, 3, True, RH, 200),
#                      (3, 2, False, DH,  200), (4, 2, True, TH, 200)]

#   RBFtuple1 = RecordBFEncoding(rec_tuple_list1)

#   RBFtuple2 = RecordBFEncoding(rec_tuple_list2)

#   RBFtuple3 = RecordBFEncoding(rec_tuple_list3)

#   salt_str_list = ['ab', '43', None, '#']

#   for attr_val_list in rec_list:

#     rec_bf1 = RBFtuple1.encode(attr_val_list)
#     rec_bf2 = RBFtuple1.encode(attr_val_list, salt_str_list)

#     assert len(rec_bf1) == bf_len, (len(rec_bf1), bf_len)
#     assert len(rec_bf2) == bf_len, (len(rec_bf2), bf_len)

#     assert rec_bf1 != rec_bf2  # Very unlikely they are the same
#     assert rec_bf1.count(1) > 0
#     assert rec_bf2.count(1) > 0

#     rec_bf3 = RBFtuple2.encode(attr_val_list)
#     rec_bf4 = RBFtuple2.encode(attr_val_list, salt_str_list)

#     assert len(rec_bf3) == bf_len, (len(rec_bf3), bf_len)
#     assert len(rec_bf4) == bf_len, (len(rec_bf4), bf_len)

#     assert rec_bf3 != rec_bf4  # Very unlikely they are the same
#     assert rec_bf3.count(1) > 0
#     assert rec_bf4.count(1) > 0

#     rec_bf5 = RBFtuple3.encode(attr_val_list)
#     rec_bf6 = RBFtuple3.encode(attr_val_list, salt_str_list)

#     assert len(rec_bf5) == bf_len, (len(rec_bf5), bf_len)
#     assert len(rec_bf6) == bf_len, (len(rec_bf6), bf_len)

#     assert rec_bf5 != rec_bf6  # Very unlikely they are the same
#     assert rec_bf5.count(1) > 0
#     assert rec_bf6.count(1) > 0

#   # Test dynamic length RBF next (need different number of bits to select)
#   #
#   bf_len1 = 140
#   rec_tuple_list1 = [(0, 2, False, DH, 40), (1, 3, True, DH, 100)]

#   RBFtuple1 = RecordBFEncoding(rec_tuple_list1)
#   avr_num_q_gram_dict1 = RBFtuple1.get_avr_num_q_grams(rec_list)
#   abf_len_dict1 = RBFtuple1.get_dynamic_abf_len(avr_num_q_gram_dict1, k)
#   RBFtuple1.set_abf_len(abf_len_dict1)

#   for attr_val_list in rec_list:
#     rec_bf7 = RBFtuple1.encode(attr_val_list)
#     rec_bf8 = RBFtuple1.encode(attr_val_list, salt_str_list)

#     assert len(rec_bf7) == bf_len1, (len(rec_bf7), bf_len1)
#     assert len(rec_bf8) == bf_len1, (len(rec_bf8), bf_len1)

#     assert rec_bf7 != rec_bf8  # Very unlikely they are the same
#     assert rec_bf7.count(1) > 0
#     assert rec_bf8.count(1) > 0

#   bf_len2 = 1000
#   #rec_tuple_list2 = [ (0, 2, True, DH, 70), (1, 2, False, RH, 60),
#   #                   (2, 4, False, RH, 70)]
#   rec_tuple_list2 = [ (0, 2, True, DH, 500), (1, 2, False, RH, 200),
#                      (2, 4, False, RH, 300)]

#   RBFtuple2 = RecordBFEncoding(rec_tuple_list2)
#   avr_num_q_gram_dict2 = RBFtuple2.get_avr_num_q_grams(rec_list)
#   abf_len_dict2 = RBFtuple1.get_dynamic_abf_len(avr_num_q_gram_dict2, k) 
#   RBFtuple2.set_abf_len(abf_len_dict2)

#   for attr_val_list in rec_list:
#     rec_bf9 = RBFtuple2.encode(attr_val_list)
#     rec_bf10 = RBFtuple2.encode(attr_val_list, salt_str_list)

#     assert len(rec_bf9) == bf_len2, (len(rec_bf9), bf_len2)
#     assert len(rec_bf10) == bf_len2, (len(rec_bf10), bf_len2)

#     assert rec_bf9 != rec_bf10  # Very unlikely they are the same
#     assert rec_bf9.count(1) > 0
#     assert rec_bf10.count(1) > 0

#   bf_len3 = 200
#   rec_tuple_list3 = [(0, 2, False, EDH, 40), (2, 3, True, RH, 80),
#                      (3, 2, False, DH,  30), (4, 2, True, TH, 50)]

#   RBFtuple3 = RecordBFEncoding(rec_tuple_list3)
#   avr_num_q_gram_dict3 = RBFtuple3.get_avr_num_q_grams(rec_list)
#   abf_len_dict3 = RBFtuple1.get_dynamic_abf_len(avr_num_q_gram_dict3, k)
#   RBFtuple3.set_abf_len(abf_len_dict3)

#   for attr_val_list in rec_list:
#     rec_bf11 = RBFtuple3.encode(attr_val_list)
#     rec_bf12 = RBFtuple3.encode(attr_val_list, salt_str_list)

#     assert len(rec_bf11) == bf_len3, (len(rec_bf11), bf_len3)
#     assert len(rec_bf12) == bf_len3, (len(rec_bf12), bf_len3)

#     assert rec_bf11 != rec_bf12  # Very unlikely they are the same
#     assert rec_bf11.count(1) > 0
#     assert rec_bf12.count(1) > 0

#   # Check g-gram position dictionary
#   #
#   DH3 = hashing.DoubleHashing(bf_hash_funct1, bf_hash_funct2, bf_len, k, True)

#   rec_tuple_list4 = [(0, 2, False, DH2, 400), (1, 3, False, DH2, 600)]

#   RBFtuple4 = RecordBFEncoding(rec_tuple_list4)

#   for attr_val_list in rec_list:

#     rec_bf4, rbf_q_gram_pos_dict = RBFtuple4.encode(attr_val_list)
#     assert len(rbf_q_gram_pos_dict) > 0

#     for (q_gram, pos_set) in rbf_q_gram_pos_dict.items():
#       assert q_gram in attr_val_list[0]+attr_val_list[1]

#       assert len(pos_set) > 0 and len(pos_set) <= k
#       for pos in pos_set:
#         assert pos >= 0 and pos < bf_len

#   print 'OK'
#   print

#   print '  Testing CLK-RBF encoding...',  # - - - - - - - - - - - - - - - - - -

#   # Define three hashing methods with different numbers of hash functions to
#   # simulate differeent weights
#   #
#   DHw1 = hashing.DoubleHashing(bf_hash_funct1, bf_hash_funct2, bf_len, k)
#   DHw2 = hashing.DoubleHashing(bf_hash_funct1, bf_hash_funct2, bf_len, 2*k)
#   DHw3 = hashing.DoubleHashing(bf_hash_funct1, bf_hash_funct2, bf_len, 3*k)

#   attr_tuple_list1 = [(0, 2, False, DHw2), (1, 3, True, DHw3)]

#   attr_tuple_list2 = [(0, 2, False, DHw3), (1, 2, True, DHw2),
#                       (2, 4, False, DHw1)]

#   attr_tuple_list3 = [(0, 2, True, DHw3), (2, 3, False, DHw2),
#                       (3, 2, True, DHw1), (4, 2, False, DHw1)]

#   CLKRBFtuple1 = CryptoLongtermKeyBFEncoding(attr_tuple_list1)

#   CLKRBFtuple2 = CryptoLongtermKeyBFEncoding(attr_tuple_list2)

#   CLKRBFtuple3 = CryptoLongtermKeyBFEncoding(attr_tuple_list3)

#   salt_str_list = ['ab', '43', None, '#']

#   for attr_val_list in rec_list:

#     rec_bf1 = CLKRBFtuple1.encode(attr_val_list)
#     rec_bf2 = CLKRBFtuple1.encode(attr_val_list, salt_str_list)

#     assert len(rec_bf1) == bf_len
#     assert len(rec_bf2) == bf_len

#     assert rec_bf1 != rec_bf2  # Very unlikely they are the same
#     assert rec_bf1.count(1) > 0
#     assert rec_bf2.count(1) > 0

#     rec_bf3 = CLKRBFtuple2.encode(attr_val_list)
#     rec_bf4 = CLKRBFtuple2.encode(attr_val_list, salt_str_list)

#     assert len(rec_bf3) == bf_len
#     assert len(rec_bf4) == bf_len

#     assert rec_bf3 != rec_bf4  # Very unlikely they are the same
#     assert rec_bf3.count(1) > 0
#     assert rec_bf4.count(1) > 0

#     rec_bf5 = CLKRBFtuple3.encode(attr_val_list)
#     rec_bf6 = CLKRBFtuple3.encode(attr_val_list, salt_str_list)

#     assert len(rec_bf5) == bf_len
#     assert len(rec_bf6) == bf_len

#     assert rec_bf5 != rec_bf6  # Very unlikely they are the same
#     assert rec_bf5.count(1) > 0
#     assert rec_bf6.count(1) > 0

#   print 'OK'
#   print

# # =============================================================================
# # End.
