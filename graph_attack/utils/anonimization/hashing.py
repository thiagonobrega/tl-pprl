# hashing.py - Module that implements several Bloom filter hashing methods
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

import hashlib  # A standard Python library
import random   # For random hashing

import bitarray  # Efficient bit-arrays, available from:
                 # https://pypi.org/project/bitarray/

# =============================================================================

class DoubleHashing():
  """Double-hashing for Bloom filters was proposed and used by:
       - A. Kirsch and M. Mitzenmacher, Less hashing, same performance:
         building a better Bloom filter, ESA, 2006, pp. 456-467.
       - R. Schnell, T. Bachteler, and J. Reiher, Privacy-preserving record
         linkage using Bloom filters, BMC MIDM, vol. 9, no. 1, 2009.
  """

  # ---------------------------------------------------------------------------

  def __init__(self, hash_funct1, hash_funct2, bf_len, num_hash_funct,
               get_q_gram_pos=False):
    """Initialise the double-hashing class by providing the required
       parameters.

       Input arguments:
         - hash_funct1, hash_funct2  Two hash functions.
         - bf_len                    The length in bits of the Bloom filters
                                     to be generated.
         - num_hash_funct            The number of hash functions to be used.
         - get_q_gram_pos            A flag, if set to True then the bit
                                     positions of where q-grams are hash into
                                     are returned in a dictionary.

       Output:
         - This method does not return anything.
    """

    # Initialise the class variables
    #
    self.hash_funct1 = hash_funct1
    self.hash_funct2 = hash_funct2

    assert bf_len > 1, bf_len
    self.bf_len = bf_len

    assert num_hash_funct > 0
    self.num_hash_funct = num_hash_funct

    assert get_q_gram_pos in [True, False]
    self.get_q_gram_pos = get_q_gram_pos

  # ---------------------------------------------------------------------------

  def hash_q_gram_set(self, q_gram_set, salt_str=None):
    """Hash the given q-gram set according to the parameter using when
       initialising the class.

       Input arguments:
         - q_gram_set  The set of q-grams (strings) to be hashed into a Bloom
                       filter.
         - salt_str    An optional string used for salting, if provided this
                       string will be concatenated with every q-gram in the
                       given q-gram set. If set to None no salting will be
                       done.

       Output:
         - bf               A Bloom filter with bits set according to the input
                            q-gram set and double hashing parameters.
         - q_gram_pos_dict  [Only returned if 'get_q_gram_pos' is set to True]
                            A dictionary which has q-grams as keys and where
                            values are sets with the positions these q-grams
                            are hashed to.
    """

    bf_len = self.bf_len  # Short-cuts
    k =      self.num_hash_funct

    get_q_gram_pos = self.get_q_gram_pos
    if (get_q_gram_pos == True):
      q_gram_pos_dict = {}

    # Initialise the Bloom filter to have only 0-bits
    #
    bf = bitarray.bitarray(bf_len)
    bf.setall(0)

    # Hash the q-grams into the Bloom filter
    #
    for q_gram in q_gram_set:

      if (get_q_gram_pos == True):
        q_gram_pos_set = set()

      if (salt_str != None):  # If a salt is given concatenate with q-gram
        q_gram = q_gram + salt_str

      hex_str1 = self.hash_funct1(q_gram).hexdigest()
      int1 =     int(hex_str1, 16)

      hex_str2 = self.hash_funct2(q_gram).hexdigest()
      int2 =     int(hex_str2, 16)

      for i in range(1, k+1):
        pos_i = int1 + i*int2
        pos_i = int(pos_i % bf_len)
        bf[pos_i] = 1

        if (get_q_gram_pos == True):
          q_gram_pos_set.add(pos_i)

      if (get_q_gram_pos == True):
        q_gram_pos_dict[q_gram] = q_gram_pos_set

    if (get_q_gram_pos == True):
      return bf, q_gram_pos_dict
    else:
      return bf

# =============================================================================

class EnhancedDoubleHashing():
  """Enhanced Double-hashing for Bloom filters was proposed and used by:
       - P.C. Dillinger and P. Manolios, Bloom filters in probabilistic
         verification, International Conference on Formal Methods in
         Computer-Aided Design, 2004, pp. 367-381.
       - P.C. Dillinger and P. Manolios, Fast and accurate bitstate
         verification for SPIN, International SPIN Workshop on Model
         Checking of Software, 2004, pp. 57-75.
  """

  # ----------------------------------------------------------------------------

  def __init__(self, hash_funct1, hash_funct2, bf_len, num_hash_funct,
               get_q_gram_pos=False):
    """Initialise the enhanced double-hashing class by providing the required
       parameters.

       Input arguments:
         - hash_funct1, hash_funct2  Two hash functions.
         - bf_len                    The length in bits of the Bloom filters
                                     to be generated.
         - num_hash_funct            The number of hash functions to be used.
         - get_q_gram_pos            A flag, if set to True then the bit
                                     positions of where q-grams are hash into
                                     are returned in a dictionary.

       Output:
         - This method does not return anything.
    """

    # Initialise the class variables
    #
    self.hash_funct1 = hash_funct1
    self.hash_funct2 = hash_funct2

    assert bf_len > 1, bf_len
    self.bf_len = bf_len

    assert num_hash_funct > 0
    self.num_hash_funct = num_hash_funct

    assert get_q_gram_pos in [True, False]
    self.get_q_gram_pos = get_q_gram_pos

  # ---------------------------------------------------------------------------

  def hash_q_gram_set(self, q_gram_set, salt_str=None):
    """Hash the given q-gram set according to the parameter using when
       initialising the class.

       Input arguments:
         - q_gram_set  The set of q-grams (strings) to be hashed into a Bloom
                       filter.
         - salt_str    An optional string used for salting, if provided this
                       string will be concatenated with every q-gram in the
                       given q-gram set. If set to None no salting will be
                       done.

       Output:
         - bf               A Bloom filter with bits set according to the input
                            q-gram set and enhanced double hashing parameters.
         - q_gram_pos_dict  [Only returned if 'get_q_gram_pos' is set to True]
                            A dictionary which has q-grams as keys and where
                            values are sets with the positions these q-grams
                            are hashed to.
    """

    bf_len = self.bf_len  # Short-cuts
    k =      self.num_hash_funct

    get_q_gram_pos = self.get_q_gram_pos
    if (get_q_gram_pos == True):
      q_gram_pos_dict = {}

    # Initialise the Bloom filter to have only 0-bits
    #
    bf = bitarray.bitarray(bf_len)
    bf.setall(0)

    # Hash the q-grams into the Bloom filter
    #
    for q_gram in q_gram_set:

      if (get_q_gram_pos == True):
        q_gram_pos_set = set()

      if (salt_str != None):  # If a salt is given concatenate with q-gram
        q_gram = q_gram + salt_str

      hex_str1 = self.hash_funct1(q_gram).hexdigest()
      int1 =     int(hex_str1, 16)

      hex_str2 = self.hash_funct2(q_gram).hexdigest()
      int2 =     int(hex_str2, 16)

      for i in range(1, k+1):
        pos_i = int1 + i*int2 + int((float(i*i*i) - i) / 6.0)
        pos_i = int(pos_i % bf_len)
        bf[pos_i] = 1

        if (get_q_gram_pos == True):
          q_gram_pos_set.add(pos_i)

      if (get_q_gram_pos == True):
        q_gram_pos_dict[q_gram] = q_gram_pos_set

    if (get_q_gram_pos == True):
      return bf, q_gram_pos_dict
    else:
      return bf

# =============================================================================

class TripleHashing():
  """Triple-hashing for Bloom filters was proposed and used by:
       - P.C. Dillinger and P. Manolios, Bloom filters in probabilistic
         verification, International Conference on Formal Methods in
         Computer-Aided Design, 2004, pp. 367-381.
       - P.C. Dillinger and P. Manolios, Fast and accurate bitstate
         verification for spin, International SPIN Workshop on Model
         Checking of Software, 2004, pp. 57-75.
  """

  # ---------------------------------------------------------------------------

  def __init__(self, hash_funct1, hash_funct2, hash_funct3, bf_len,
               num_hash_funct, get_q_gram_pos=False):
    """Initialise the triple-hashing class by providing the required
       parameters.

       Input arguments:
         - hash_funct1, hash_funct2, hash_funct3  Three hash functions.
         - bf_len                                 The length in bits of the
                                                  Bloom filters to be
                                                  generated.
         - num_hash_funct                         The number of hash functions
                                                  to be used.
         - get_q_gram_pos                         A flag, if set to True then
                                                  the bit positions of where
                                                  q-grams are hash into are
                                                  returned in a dictionary.

       Output:
         - This method does not return anything.
    """

    # Initialise the class variables
    #
    self.hash_funct1 = hash_funct1
    self.hash_funct2 = hash_funct2
    self.hash_funct3 = hash_funct3

    assert bf_len > 1, bf_len
    self.bf_len = bf_len

    assert num_hash_funct > 0
    self.num_hash_funct = num_hash_funct

    assert get_q_gram_pos in [True, False]
    self.get_q_gram_pos = get_q_gram_pos

  # ---------------------------------------------------------------------------

  def hash_q_gram_set(self, q_gram_set, salt_str=None):
    """Hash the given q-gram set according to the parameter using when
       initialising the class.

       Input arguments:
         - q_gram_set  The set of q-grams (strings) to be hashed into a Bloom
                       filter.
         - salt_str    An optional string used for salting, if provided this
                       string will be concatenated with every q-gram in the
                       given q-gram set. If set to None no salting will be
                       done.

       Output:
         - bf               A Bloom filter with bits set according to the input
                            q-gram set and triple hashing parameters.
         - q_gram_pos_dict  [Only returned if 'get_q_gram_pos' is set to True]
                            A dictionary which has q-grams as keys and where
                            values are sets with the positions these q-grams
                            are hashed to.
    """

    bf_len = self.bf_len  # Short-cuts
    k =      self.num_hash_funct

    get_q_gram_pos = self.get_q_gram_pos
    if (get_q_gram_pos == True):
      q_gram_pos_dict = {}

    # Initialise the Bloom filter to have only 0-bits
    #
    bf = bitarray.bitarray(bf_len)
    bf.setall(0)

    # Hash the q-grams into the Bloom filter
    #
    for q_gram in q_gram_set:

      if (get_q_gram_pos == True):
        q_gram_pos_set = set()

      if (salt_str != None):  # If a salt is given concatenate with q-gram
        q_gram = q_gram + salt_str

      hex_str1 = self.hash_funct1(q_gram).hexdigest()
      int1 =     int(hex_str1, 16)

      hex_str2 = self.hash_funct2(q_gram).hexdigest()
      int2 =     int(hex_str2, 16)

      hex_str3 = self.hash_funct3(q_gram).hexdigest()
      int3 =     int(hex_str3, 16)

      for i in range(1, k+1):
        pos_i = int1 + i*int2 + int(float(i*(i-1)/2.0))*int3
        pos_i = int(pos_i % bf_len)
        bf[pos_i] = 1

        if (get_q_gram_pos == True):
          q_gram_pos_set.add(pos_i)

      if (get_q_gram_pos == True):
        q_gram_pos_dict[q_gram] = q_gram_pos_set

    if (get_q_gram_pos == True):
      return bf, q_gram_pos_dict
    else:
      return bf

# =============================================================================

class RandomHashing():
  """Random-hashing for Bloom filters was proposed and used by:
       - R. Schnell and C. Borgs, Randomized response and balanced Bloom
         filters for privacy preserving record linkage, Workshop on Data
         Integration and Applications, held at ICDM, Barcelona, 2016.
  """

  # ---------------------------------------------------------------------------

  def __init__(self, hash_funct, bf_len, num_hash_funct,
               get_q_gram_pos=False):
    """Initialise the random-hashing class by providing the required
       parameters.

       Input arguments:
         - hash_funct      The hash function to be used to encode q-grams.
         - bf_len          The length in bits of the Bloom filters to be
                           generated.
         - num_hash_funct  The number of hash functions to be used.
         - get_q_gram_pos  A flag, if set to True then the bit positions of
                           where q-grams are hash into are returned in a
                           dictionary.

       Output:
         - This method does not return anything.
    """

    # Initialise the class variables
    #
    self.hash_funct = hash_funct

    assert bf_len > 1, bf_len
    self.bf_len = bf_len

    assert num_hash_funct > 0
    self.num_hash_funct = num_hash_funct

    assert get_q_gram_pos in [True, False]
    self.get_q_gram_pos = get_q_gram_pos

  # ---------------------------------------------------------------------------

  def hash_q_gram_set(self, q_gram_set, salt_str=None):
    """Hash the given q-gram set according to the parameter using when
       initialising the class.

       Input arguments:
         - q_gram_set  The set of q-grams (strings) to be hashed into a Bloom
                       filter.
         - salt_str    An optional string used for salting, if provided this
                       string will be concatenated with every q-gram in the
                       given q-gram set. If set to None no salting will be
                       done.

       Output:
         - bf               A Bloom filter with bits set according to the input
                            q-gram set and random hashing parameters.
         - q_gram_pos_dict  [Only returned if 'get_q_gram_pos' is set to True]
                            A dictionary which has q-grams as keys and where
                            values are sets with the positions these q-grams
                            are hashed to.
    """

    bf_len = self.bf_len  # Short-cuts
    k =      self.num_hash_funct

    get_q_gram_pos = self.get_q_gram_pos
    if (get_q_gram_pos == True):
      q_gram_pos_dict = {}

    # Initialise the Bloom filter to have only 0-bits
    #
    bf = bitarray.bitarray(bf_len)
    bf.setall(0)

    # Bloom filter length minus 1 using for position index in Bloom filter
    #
    bf_len_m1 = bf_len - 1

    # Hash the q-grams into the Bloom filter
    #
    for q_gram in q_gram_set:

      if (get_q_gram_pos == True):
        q_gram_pos_set = set()

      if (salt_str != None):  # If a salt is given concatenate with q-gram
        q_gram = q_gram + salt_str

      # Use q-gram itself to see random number generator
      #
      hex_str = self.hash_funct(q_gram).hexdigest()
      random_seed = random.seed(int(hex_str, 16))

      for i in range(k):
        pos_i = random.randint(0, bf_len_m1)
        bf[pos_i] = 1

        if (get_q_gram_pos == True):
          q_gram_pos_set.add(pos_i)

      if (get_q_gram_pos == True):
        q_gram_pos_dict[q_gram] = q_gram_pos_set

    if (get_q_gram_pos == True):
      return bf, q_gram_pos_dict
    else:
      return bf

# =============================================================================
# Some testing code if called from the command line

if (__name__ == '__main__'):

  print 'Running some tests:'
  print

  # Define three hash functions
  #
  bf_hash_funct1 = hashlib.sha1
  bf_hash_funct2 = hashlib.md5
  bf_hash_funct3 = hashlib.sha256

  # Define Bloom filter hashing parameters
  #
  bf_len = 1000
  k =      10

  test_q_gram_set = set(['he','el','ll','lo','o ',' w','wo','or','rl','ld'])

  print '  Test q-gram set:', test_q_gram_set
  print

  print '  Testing double hashing...',  # - - - - - - - - - - - - - - - - - - -

  # Initialise the double hashing class
  #
  DH = DoubleHashing(bf_hash_funct1, bf_hash_funct2, bf_len, k)

  dh_bf1 = DH.hash_q_gram_set(test_q_gram_set)
  assert len(dh_bf1) == bf_len
  assert dh_bf1.count(1) > 0

  dh_bf2 = DH.hash_q_gram_set(test_q_gram_set)
  assert len(dh_bf2) == bf_len
  assert dh_bf2.count(1) > 0

  assert dh_bf1 == dh_bf2

  dh_bf3 = DH.hash_q_gram_set(test_q_gram_set, 'test salt str')
  assert len(dh_bf3) == bf_len
  assert dh_bf3.count(1) > 0

  assert dh_bf1 != dh_bf3

  # Check g-gram position dictionary
  #
  DH2 = DoubleHashing(bf_hash_funct1, bf_hash_funct2, bf_len, k, True)

  dh_bf, dh_q_gram_pos_dict = DH2.hash_q_gram_set(test_q_gram_set)

  assert len(dh_q_gram_pos_dict) == len(test_q_gram_set)
  for (q_gram, pos_set) in dh_q_gram_pos_dict.iteritems():
    assert q_gram in test_q_gram_set
    assert len(pos_set) > 0 and len(pos_set) <= k
    for pos in pos_set:
      assert pos >= 0 and pos < bf_len

  print 'OK'
  print

  print '  Testing enchanced double hashing...',  # - - - - - - - - - - - - - -

  # Initialise the enhanced double hashing class
  #
  EDH = EnhancedDoubleHashing(bf_hash_funct1, bf_hash_funct2, bf_len, k)

  edh_bf1 = EDH.hash_q_gram_set(test_q_gram_set)
  assert len(edh_bf1) == bf_len
  assert edh_bf1.count(1) > 0

  edh_bf2 = EDH.hash_q_gram_set(test_q_gram_set)
  assert len(edh_bf2) == bf_len
  assert edh_bf2.count(1) > 0

  assert edh_bf1 == edh_bf2

  edh_bf3 = EDH.hash_q_gram_set(test_q_gram_set, 'test salt str')
  assert len(edh_bf3) == bf_len
  assert edh_bf3.count(1) > 0

  assert edh_bf1 != edh_bf3

  assert dh_bf1 != edh_bf1  # Highly unlikely the two hashing approaches will
  assert dh_bf2 != edh_bf2  # generate the same Bloom filters
  assert dh_bf3 != edh_bf3

  # Check g-gram position dictionary
  #
  EDH2 = EnhancedDoubleHashing(bf_hash_funct1, bf_hash_funct2, bf_len, k, True)

  edh_bf, edh_q_gram_pos_dict = EDH2.hash_q_gram_set(test_q_gram_set)

  assert len(edh_q_gram_pos_dict) == len(test_q_gram_set)
  for (q_gram, pos_set) in edh_q_gram_pos_dict.iteritems():
    assert q_gram in test_q_gram_set
    assert len(pos_set) > 0 and len(pos_set) <= k
    for pos in pos_set:
      assert pos >= 0 and pos < bf_len

  print 'OK'
  print

  print '  Testing triple hashing...',  # - - - - - - - - - - - - - - - - - - -

  # Initialise the triple hashing class
  #
  TH = TripleHashing(bf_hash_funct1, bf_hash_funct2, bf_hash_funct3, bf_len, k)

  th_bf1 = TH.hash_q_gram_set(test_q_gram_set)
  assert len(th_bf1) == bf_len
  assert th_bf1.count(1) > 0

  th_bf2 = TH.hash_q_gram_set(test_q_gram_set)
  assert len(th_bf2) == bf_len
  assert th_bf2.count(1) > 0

  assert th_bf1 == th_bf2

  th_bf3 = TH.hash_q_gram_set(test_q_gram_set, 'test salt str')
  assert len(th_bf3) == bf_len
  assert th_bf3.count(1) > 0

  assert th_bf1 != th_bf3

  assert dh_bf1 != th_bf1  # Highly unlikely the two hashing approaches will
  assert dh_bf2 != th_bf2  # generate the same Bloom filters
  assert dh_bf3 != th_bf3

  # Check g-gram position dictionary
  #
  TH2 = TripleHashing(bf_hash_funct1, bf_hash_funct2, bf_hash_funct3, bf_len,
                      k, True)

  th_bf, th_q_gram_pos_dict = TH2.hash_q_gram_set(test_q_gram_set)

  assert len(th_q_gram_pos_dict) == len(test_q_gram_set)
  for (q_gram, pos_set) in th_q_gram_pos_dict.iteritems():
    assert q_gram in test_q_gram_set
    assert len(pos_set) > 0 and len(pos_set) <= k
    for pos in pos_set:
      assert pos >= 0 and pos < bf_len

  print 'OK'
  print

  print '  Testing random hashing...',  # - - - - - - - - - - - - - - - - - - -

  # Initialise the random hashing class
  #
  RH = RandomHashing(bf_hash_funct1, bf_len, k)

  rh_bf1 = RH.hash_q_gram_set(test_q_gram_set)
  assert len(rh_bf1) == bf_len
  assert rh_bf1.count(1) > 0

  rh_bf2 = RH.hash_q_gram_set(test_q_gram_set)
  assert len(rh_bf2) == bf_len
  assert rh_bf2.count(1) > 0

  assert rh_bf1 == rh_bf2

  rh_bf3 = RH.hash_q_gram_set(test_q_gram_set, 'test salt str')
  assert len(rh_bf3) == bf_len
  assert rh_bf3.count(1) > 0

  assert rh_bf1 != rh_bf3

  assert dh_bf1 != rh_bf1  # Highly unlikely the two hashing approaches will
  assert dh_bf2 != rh_bf2  # generate the same Bloom filters
  assert dh_bf3 != rh_bf3

  # Check g-gram position dictionary
  #
  RH2 = RandomHashing(bf_hash_funct1, bf_len, k, True)

  rh_bf, rh_q_gram_pos_dict = RH2.hash_q_gram_set(test_q_gram_set)

  assert len(rh_q_gram_pos_dict) == len(test_q_gram_set)
  for (q_gram, pos_set) in rh_q_gram_pos_dict.iteritems():
    assert q_gram in test_q_gram_set
    assert len(pos_set) > 0 and len(pos_set) <= k
    for pos in pos_set:
      assert pos >= 0 and pos < bf_len

  print 'OK'
  print

# =============================================================================
# End.
