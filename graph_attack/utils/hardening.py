# hardening.py - Module that implements several Bloom filter hardening methods
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

import numpy.random  # For probability choice function used in Markov chain
                     # hardening

import bitarray  # Efficient bit-arrays, available from:
                 # https://pypi.org/project/bitarray/

PAD_CHAR = chr(1)   # Used for q-gram padding

# =============================================================================

class Balancing():
  """Balancing Bloom filter hardening was proposed and used by:
       - R. Schnell and C. Borgs, Randomized response and balanced Bloom
         filters for privacy preserving record linkage, Workshop on Data
         Integration and Applications, held at ICDM, Barcelona, 2016.
  """

  # ---------------------------------------------------------------------------

  def __init__(self, get_q_gram_pos=False, random_seed=42):
    """Initialise the Bloom filter balancing hardening class by providing the
       required parameters.

       Input arguments:
         - get_q_gram_pos  A flag, if set to True then the bit positions of
                           where q-grams are hash into are returned in a
                           dictionary.
         - random_seed     The value used to seed the random generator used to
                           shuffle the balanced Bloom filter. If no random
                           shuffling should be done set the value of this
                           argument to None. Default value is set to 42.

       Output:
         - This method does not return anything.
    """

    self.type = 'BAL'  # To identify the hardening method

    # Store the random seed so each balanced Bloom filter can be shuffled in
    # the same way
    #
    self.random_seed =    random_seed
    self.get_q_gram_pos = get_q_gram_pos

    self.perm_pos_list = None

  # ---------------------------------------------------------------------------

  def harden_bf(self, bf, org_q_gram_pos_dict=None):
    """Harden the provided Bloom filter by balancing it.

       Note that the returned balanced Bloom filter has double the length as
       the input Bloom filter.

       The bits of the balanced Bloom filter are randomly shuffled using the
       provided random seed (so all Bloom filters are shuffled in the same
       way), unless the  value of the random_shuffle argument was set to
       None.

       Input arguments:
         - bf                   A Bloom filter assumed to have its bits set
                                from an encoded q-gram set.
         - org_q_gram_pos_dict  The q-gram dictionary generated when hashing
                                q-grams into the BF.

       Output:
         - bal_bf               The balanced Bloom filter.
         - org_q_gram_pos_dict  [Only returned if 'get_q_gram_pos' is set to
                                 True]
                                A dictionary which has q-grams as keys and
                                where values are sets with the positions these
                                q-grams are hashed to after balancing has been
                                applied.
    """

    # Add complement of the original Bloom filter to itself
    #
    bal_bf = bf + ~bf
    bf_len = len(bf)
    bal_bf_len = len(bal_bf)

    # If the q-gram position flag is set to True get the new positions
    #
    if (self.get_q_gram_pos == True):
      q_gram_pos_dict = {}

      for (q_gram, pos_set) in org_q_gram_pos_dict.iteritems():
        new_pos_set = set([pos+bf_len for pos in pos_set])
        new_pos_set.update(pos_set)

        q_gram_pos_dict[q_gram] = new_pos_set

    if (self.random_seed != None):  # Permutate the bit positions

      # Generate a permutation list using random shuffling of bit positions
      #
      if (self.perm_pos_list == None):
        perm_pos_list = range(bal_bf_len)
        random.shuffle(perm_pos_list)
        self.perm_pos_list = perm_pos_list

      else:
        perm_pos_list = self.perm_pos_list

      # Permute the bits in the balanced Bloom filter
      #
      perm_bal_bf = bitarray.bitarray(bal_bf_len)

      for pos in xrange(bal_bf_len):
        perm_bal_bf[pos] = bal_bf[perm_pos_list[pos]]

      # If needed also change the bit positions of all q-grams
      #
      if (self.get_q_gram_pos == True):

        for (q_gram, org_pos_set) in q_gram_pos_dict.iteritems():
          perm_pos_set = set()
          for pos in org_pos_set:
            perm_pos_set.add(perm_pos_list[pos])
          q_gram_pos_dict[q_gram] = perm_pos_set

    else:
      perm_bal_bf = bal_bf

    if (self.get_q_gram_pos == True):
      return perm_bal_bf, q_gram_pos_dict
    else:
      return perm_bal_bf

    return bal_bf

# =============================================================================

class Folding():
  """XOR folding Bloom filter hardening was proposed and used by:
       - R. Schnell and C. Borgs, Randomized response and balanced Bloom
         filters for privacy preserving record linkage, Workshop on Data
         Integration and Applications, held at ICDM, Barcelona, 2016.
  """

  # ---------------------------------------------------------------------------
  
  def __init__(self, get_q_gram_pos=False):
    """Initialise the Bloom filter XOR folding hardening class by providing
       the required parameters.

       Input arguments:
         - get_q_gram_pos  A flag, if set to True then the bit positions of
                           where q-grams are hash into are returned in a
                           dictionary.

       Output:
         - This method does not return anything.

      This class requires the length of a Bloom filter to be even.
    """

    self.type = 'XOR'  # To identify the hardening method

    self.get_q_gram_pos = get_q_gram_pos

  # ---------------------------------------------------------------------------

  def harden_bf(self, bf, org_q_gram_pos_dict=None):
    """Harden the provided Bloom filter by XOR folding it.

       Note that the returned folded Bloom filter has half the length as the
       input Bloom filter.

       Input arguments:
         - bf                   A Bloom filter assumed to have its bits set
                                from an encoded q-gram set.
         - org_q_gram_pos_dict  The q-gram dictionary generated when hashing
                                q-grams into the BF.

       Output:
         - fold_bf              The XOR folded Bloom filter.
         - org_q_gram_pos_dict  [Only returned if 'get_q_gram_pos' is set to
                                 True]
                                A dictionary which has q-grams as keys and
                                where values are sets with the positions these
                                q-grams are hashed to after folding has been
                                applied.
    """

    # Check if the length of the given Bloom filter is even
    #
    bf_len = len(bf)
    half_len = int(float(bf_len) / 2)
    assert 2*half_len == bf_len

    fold_bf = bf[:half_len] ^ bf[half_len:]  # XOR folding

    # If the q-gram position flag is set to True get the new positions
    #
    if (self.get_q_gram_pos == True):
      q_gram_pos_dict = {}

      for (q_gram, pos_set) in org_q_gram_pos_dict.iteritems():
        new_pos_set = set()

        for pos in pos_set:
          if(pos >= half_len):
            new_pos_set.add(pos - half_len)
          else:
            new_pos_set.add(pos)

        q_gram_pos_dict[q_gram] = new_pos_set

      return fold_bf, q_gram_pos_dict

    else:
      return fold_bf

# =============================================================================

class Rule90():
  """Rule 90 was proposed and used by:
       - S. Wolfram, Statistical mechanics of cellular automata, Reviews of
         modern physics, 1983, 55(3), p.601.
       - R. Schell and C. Borgs, Protecting record linkage identifiers using
         cellular automata and language models for patient names, 2018.
  """

  # ---------------------------------------------------------------------------

  def __init__(self):
    """Initialise the Bloom filter Rule 90 hardening class by providing
       the required parameters.

       Input arguments:
         - This method does not require any input arguments.

       Output:
         - This method does not return anything.
    """

    self.type = 'R90R'  # To identify the hardening method

    # Initialise a dictionary with the patterns of Rule 90
    #
    self.rule90_dict = {'111':0, '110':1, '101':0, '100':1, \
                        '011':1, '010':0, '001':1, '000':0}

  # ---------------------------------------------------------------------------

  def harden_bf(self, bf):
    """Harden the provided Bloom filter by applying Wolfram's rule 90.

       Input arguments:
         - bf  A Bloom filter assumed to have its bits set from an encoded
               q-gram set.

       Output:
         - rule90_bf  The new Bloom filter after rule 90 has been applied.
    """

    bf_len = len(bf)
    
    # Initialise bitarray for a new Bloom filter
    #
    rule90_bf = bitarray.bitarray(bf_len)

    for pos in xrange(bf_len):
      if (pos == 0):  # Wrap around of the first bit
        org_bit_triple = bf[-1:].to01()+bf[:2].to01()
        
      elif (pos == (bf_len-1)):  # Wrap around of the last bit
        org_bit_triple = bf[-2:].to01()+bf[:1].to01()        

      else:  # All other bits
        org_bit_triple = bf[pos-1:pos+2].to01()

      assert len(org_bit_triple) == 3

      # Set the current bit position according to Rule 90
      #
      new_bit = self.rule90_dict[org_bit_triple]
      rule90_bf[pos] = new_bit

    assert len(bf) == len(rule90_bf)

    return rule90_bf

# =============================================================================

class MarkovChain():
  """Markov chain Bloom filter (MCBF) hardening was proposed and used by:
       - R. Schell and C. Borgs, Protecting record linkage identifiers using
         cellular automata and language models for patient names, 2018.

     Note that the methods of this class are NOT applied on already
     generated Bloom filters, rather a q-gram set needs to be provided
     which is expanded with other q-grams according to the generated
     probabilistic language model.
  """

  # ---------------------------------------------------------------------------

  def __init__(self, q, padded, chain_len, sel_method):
    """Initialise the Markov chain Bloom filter hardening class by providing
       the required parameters.

       Input arguments:
         - q           Length of the q-grams to be used when generating the
                       language model.
         - padded      A flag, if set to True then the values provided to
                       generate the language model are first padded before
                       q-grams are being extracted.
         - chain_len   The length of the Markov chain to use (number of q-grams
                       to add for each original q-gram).
         - sel_method  The method of how q-grams are being selected, which can
                       either be 'prob' (probabilistic, based on the transition
                       probabilities of q-grams) or 'freq' (the most frequent
                       q-grams for a given q-gram).

       Output:
         - This method does not return anything.
    """

    self.type = 'MC'  # To identify the hardening method

    # Initialise the class variables
    #
    assert q >= 1, q
    assert padded in [True, False]
    assert chain_len >= 1, chain_len
    assert sel_method in ['prob','freq'], sel_method

    self.q =          q
    self.padded =     padded
    self.chain_len =  chain_len
    self.sel_method = sel_method

  # ---------------------------------------------------------------------------

  def calc_trans_prob(self, val_list):
    """Calculate transition probabilities of pairs of consecutive q-grams as
       extracted from the list of string values provided.

       Input arguments:
         - val_list  A list of string values from which q-grams will be
                     extracted to generate a transition probability matrix.

       Output:
         - This method does not return anything.
    """

    q = self.q  # Short-cuts
    qm1 = q - 1

    # Initialise the transition probability dictionary, where keys are q-grams
    # and values are dictionaries with other q-grams and their probabilities
    # of co-occurrence with the key q-gram.
    #
    trans_prob_dict = {}

    for str_val in val_list:

      # Generate the q-grams from this value
      #
      if (self.padded == True):  # Add padding start and end characters
        str_val = PAD_CHAR*qm1+str_val+PAD_CHAR*qm1

      str_val_len = len(str_val)

      q_gram_list = [str_val[i:i+q] for i in xrange(str_val_len - qm1)]

      # Generate the pairs of consecutive q-grams and add them into the
      # transition dictionary
      #
      for (i, q_gram1) in enumerate(q_gram_list[:-1]):
        q_gram2 = q_gram_list[i+1]

        # Insert the q-gram pair into the transition dictionary
        #
        q_gram2_dict = trans_prob_dict.get(q_gram1, {})
        q_gram2_dict[q_gram2] = q_gram2_dict.get(q_gram2, 0) + 1
        trans_prob_dict[q_gram1] = q_gram2_dict

    print 'Transition probability dictionary contains %d q-grams' % \
          len(trans_prob_dict)

    # Convert the count of co-occurences into probabilities (sum must be 1.0
    # for each q-gram)
    #
    for q_gram in trans_prob_dict:
      other_q_gram_dict =      trans_prob_dict[q_gram]
      other_q_gram_count_sum = sum(other_q_gram_dict.values())

      for (other_q_gram, count) in other_q_gram_dict.iteritems():
        other_q_gram_dict[other_q_gram] = float(count) / other_q_gram_count_sum

      # Make sure the probabilities sum to 1.0
      #
      while (sum(other_q_gram_dict.values()) != 1.0):

        # Add the probability to a random q-gram
        #
        rand_q_gram = random.choice(other_q_gram_dict.keys())

        sum_diff = sum(other_q_gram_dict.values()) - 1.0
        other_q_gram_dict[rand_q_gram] -= sum_diff

      assert sum(other_q_gram_dict.values()) == 1.0, \
             (most_prob_q_gram, sum(other_q_gram_dict.values()) - 1.0)

      trans_prob_dict[q_gram] = other_q_gram_dict

    # For the 'freq' selection method we only need the 'chain_len' most likely
    # other q-grams for each key q-gram, therefore find for each key q-gram
    # its 'chain_len' other q-grams with highest probabilities
    #
    if (self.sel_method == 'freq'):
      for q_gram in trans_prob_dict:
        other_q_gram_list_sorted = sorted(trans_prob_dict[q_gram].items(),
                                          key=lambda x: x[1], reverse=True)
        other_greq_q_gram_list = []
        for (other_q_gram, count) in other_q_gram_list_sorted[:self.chain_len]:
          other_greq_q_gram_list.append(other_q_gram)

        trans_prob_dict[q_gram] = other_greq_q_gram_list

    # For the probabilistc approach we need for each q-gram its list of other
    # q-grams and their probabilities
    #
    else:
      for q_gram in trans_prob_dict:
        other_q_gram_list_sorted = sorted(trans_prob_dict[q_gram].items(),
                                          key = lambda x: x[1], reverse=True)
        other_q_gram_val_list =  []
        other_q_gram_prob_list = []
        for (other_q_gram, q_gram_prob) in other_q_gram_list_sorted:
          other_q_gram_val_list.append(other_q_gram)
          other_q_gram_prob_list.append(q_gram_prob)

        trans_prob_dict[q_gram] = (other_q_gram_val_list, \
                                   other_q_gram_prob_list)

    self.trans_prob_dict = trans_prob_dict

  # ---------------------------------------------------------------------------

  def get_other_q_grams_from_lang_model(self, q_gram_set):
    """For each q-gram in the given set of q-grams, get the 'chain_len' most
       commonly co-occurring other q-grams according to the built probabilistic
       language  model.

       Input arguments:
         - q_gram_set  A set of q-grams for which  we want to get other
                       frequently co-occurring q-grams.

       Output:
         - other_q_gram_set  The set of additional q-grams to be encoded into
                             a Bloom filter for the given input q-gram set.
    """

    sel_method =      self.sel_method  # Short-cuts
    trans_prob_dict = self.trans_prob_dict
    chain_len =       self.chain_len

    other_q_gram_set = set()

    # For each q-gram select other q-grams according to the selection method
    #
    for q_gram in q_gram_set:

      if q_gram in trans_prob_dict:

        if (sel_method == 'freq'):

          # For each given q-gram retrieve its 'chain_len' most frequent other
          # q-grams from the language model
          #
          other_q_gram_list = trans_prob_dict.get(q_gram, [])
          other_q_gram_set = other_q_gram_set | set(other_q_gram_list)

        else:
          other_q_gram_val_list, other_q_gram_prob_list = \
            trans_prob_dict[q_gram]

          # Make sure no endless loop if there are not enough other q-grams
          #
          if (len(other_q_gram_val_list) <= chain_len):

            # If there are not (or just) enough other q-grams add all of them
            #
            other_q_gram_set = other_q_gram_set | set(other_q_gram_val_list)

          else:  # Randomly select other q-grams based on their probabilities
                 # until we have enough
            this_q_gram_other_set = set() 
            while (len(this_q_gram_other_set) < chain_len):
              rand_other_q_gram = numpy.random.choice(other_q_gram_val_list,
                                                      p=other_q_gram_prob_list)
              this_q_gram_other_set.add(rand_other_q_gram)

            other_q_gram_set = other_q_gram_set | this_q_gram_other_set

    return other_q_gram_set

# =============================================================================
# Some testing code if called from the command line

if ( __name__ == '__main__' ):

  print 'Running some tests:'
  print

  import hashlib  # A standard Python library
  import hashing  # Bloom filter hashing module

  # Define two hash functions
  #
  bf_hash_funct1 = hashlib.sha1
  bf_hash_funct2 = hashlib.md5

  # Define Bloom filter hashing parameters
  #
  bf_len = 1000
  k =      10

  test_q_gram_set = set(['he','el','ll','lo','o ',' w','wo','or','rl','ld'])

  print '  Test q-gram set:', test_q_gram_set
  print

  # Initialise the double hashing class
  #
  DH = hashing.DoubleHashing(bf_hash_funct1, bf_hash_funct2, bf_len, k)

  # Generate a Bloom filter
  #
  bf1 = DH.hash_q_gram_set(test_q_gram_set)
  assert len(bf1) == bf_len
  assert bf1.count(1) > 0

  print 'Test balancing hardening...',  # - - - - - - - - - - - - - - - - - - -

  BFBalHard1 = Balancing()

  bf_bal_hardened1 = BFBalHard1.harden_bf(bf1)

  assert len(bf_bal_hardened1) == 2*bf_len
  assert bf_bal_hardened1.count(1) == bf_len

  # Check permutation is the same if called again
  #
  bf_bal_hardened2 = BFBalHard1.harden_bf(bf1)

  assert len(bf_bal_hardened2) == 2*bf_len
  assert bf_bal_hardened2.count(1) == bf_len

  assert bf_bal_hardened1 == bf_bal_hardened2

  # And test with a different random seed
  #
  BFBalHard2 = Balancing(random_seed=82)

  bf_bal_hardened3 = BFBalHard2.harden_bf(bf1)
  bf_bal_hardened4 = BFBalHard2.harden_bf(bf1)

  assert bf_bal_hardened3 == bf_bal_hardened4

  # It is very unlikely to obtain the same permutation
  #
  assert bf_bal_hardened1 != bf_bal_hardened3

  # Check the q-gram bit positions dictionary
  #
  # Initialise the double hashing class
  #
  RH = hashing.RandomHashing(bf_hash_funct1, bf_len, k, True)

  # Generate a Bloom filter
  #
  bf2, q_gram_pos_dict = RH.hash_q_gram_set(test_q_gram_set)

  assert len(bf2) == bf_len
  assert bf2.count(1) > 0

  for (q_gram, pos_set) in q_gram_pos_dict.iteritems():
    assert q_gram in test_q_gram_set
    for pos in pos_set:
      assert pos >= 0 and pos < bf_len

  # Test balancing with q-gram dictionaries
  #
  BFBalHard3 = Balancing(True, None)

  bf_bal_hardened5, new_q_gram_pos_dict = BFBalHard3.harden_bf(bf2, 
                                                               q_gram_pos_dict)

  assert sorted(q_gram_pos_dict.keys()) == sorted(new_q_gram_pos_dict.keys())

  for (q_gram, pos_set) in q_gram_pos_dict.iteritems():
    new_pos_set = new_q_gram_pos_dict[q_gram]
    assert pos_set == new_pos_set.intersection(pos_set)
    assert len(new_pos_set) == 2*len(pos_set), (len(new_pos_set), len(pos_set))

  print 'OK'
  print

  print 'Test XOR folding hardening...',  # - - - - - - - - - - - - - - - - - -

  BFFoldHard1 = Folding()

  bf_fold_hardened1 = BFFoldHard1.harden_bf(bf1)

  assert 2*len(bf_fold_hardened1) == bf_len

  # Check permutation is the same if called again
  #
  bf_fold_hardened2 = BFFoldHard1.harden_bf(bf1)

  assert 2*len(bf_fold_hardened2) == bf_len

  assert bf_fold_hardened1 == bf_fold_hardened2

  # Test folsing with q-gram dictionaries
  #
  BFFoldHard2 = Folding(True)

  bf_fold_hardened3, new_q_gram_pos_dict2 = \
                              BFFoldHard2.harden_bf(bf2, q_gram_pos_dict)

  for q_gram, pos_set in q_gram_pos_dict.items():
    new_pos_set = new_q_gram_pos_dict2[q_gram]
    assert pos_set != new_pos_set
    assert len(new_pos_set) == len(pos_set), (len(new_pos_set), len(pos_set))

  print 'OK'
  print

  print 'Test Rule 90 hardening...',  # - - - - - - - - - - - - - - - - - - - -

  BFRule90Hard = Rule90()

  bf_rule90_hardened1 = BFRule90Hard.harden_bf(bf1)

  assert len(bf_rule90_hardened1) == bf_len

  # Check if applied twice gives same result
  #
  bf_rule90_hardened2 = BFRule90Hard.harden_bf(bf1)

  assert len(bf_rule90_hardened2) == bf_len

  bf_rule90_hardened1 = bf_rule90_hardened2

  print 'OK'
  print

  print 'Test Markov chain hardening...',  # - - - - - - - - - - - - - - - - -

  lang_model_val_list = ['hello world', 'hello', 'world', 'world', 'hallo',
                         'hall', 'worry', 'holden', 'halo', 'heli']

  # Test both methods of how other q-grams are selected:
  # (1) The most frequent ones
  # (2) Randomly based on their probabilities of co-occurring
  #
  for cl in [1,2,3,4]:
    for sm in ['freq','prob']:

      BFMarkovChainHard = MarkovChain(q=2, padded=False, chain_len=cl,
                                      sel_method=sm)

      BFMarkovChainHard.calc_trans_prob(lang_model_val_list)

      extra_q_gram_set1 = \
         BFMarkovChainHard.get_other_q_grams_from_lang_model(test_q_gram_set)

      extra_q_gram_set2 = \
         BFMarkovChainHard.get_other_q_grams_from_lang_model(set(['he','el']))

      extra_q_gram_set3 = \
        BFMarkovChainHard.get_other_q_grams_from_lang_model(set(['he','el',
                                                                 'wo']))

#      print 'extra q-gram set 1:', sorted(extra_q_gram_set1)
#      print 'extra q-gram set 2:', sorted(extra_q_gram_set2)
#      print 'extra q-gram set 3:', sorted(extra_q_gram_set3)

      if (sm == 'freq'):
        assert extra_q_gram_set2.issubset(extra_q_gram_set1)
        assert extra_q_gram_set2.issubset(extra_q_gram_set3)

  print 'OK'
  print

# =============================================================================
# End.

