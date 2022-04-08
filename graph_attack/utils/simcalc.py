# simcalc.py - Implementation of various similarity functions, both q-gram and
# bit-array based.
#
# Peter Christen, September 2018
# -----------------------------------------------------------------------------

def q_gram_dice_sim(q_gram_set1, q_gram_set2):
  """Calculate the Dice similarity between the two given sets of q-grams.

     Dice similarity is calculated between two sets A and B as 

        Dice similarity (A,B) = 2 x number of common elements of A and B
                                -------------------------------------------
                                number of elems in A + number of elems in B
  
     Returns a similarity value between 0 and 1.
  """

  num_common_q_gram = len(q_gram_set1 & q_gram_set2)

  q_gram_dice_sim = (2.0 * num_common_q_gram) / \
                     (len(q_gram_set1) + len(q_gram_set2))

  assert 0 <= q_gram_dice_sim and q_gram_dice_sim <= 1.0

  return q_gram_dice_sim

# -----------------------------------------------------------------------------

def q_gram_jacc_sim(q_gram_set1, q_gram_set2):
  """Calculate the Jaccard similarity between the two given sets of q-grams.

     Jaccard similarity is calculated between two sets A and B as 

        Jaccard similarity (A,B) = |A intersection B|
                                   ------------------
                                      | A union B|

     Returns a similarity value between 0 and 1.
  """

  q_gram_intersection_set = q_gram_set1 & q_gram_set2
  q_gram_union_set =        q_gram_set1 | q_gram_set2

  q_gram_jacc_sim = float(len(q_gram_intersection_set)) / len(q_gram_union_set)

  assert 0 <= q_gram_jacc_sim and q_gram_jacc_sim <= 1.0

  return q_gram_jacc_sim

# -----------------------------------------------------------------------------

def q_gram_hamm_sim(q_gram_set1, q_gram_set2):
  """Calculate the Hamming similarity between the two given sets of q-grams.

     Hamming similarity is calculated between two sets A and B as 

        Hamming similarity (A,B) = 1.0 - 
              |(elements in A only) union (elements in B only)|
              -------------------------------------------------
              | (elements in A) union (elements in B)|
  
     Returns a similarity value between 0 and 1.
  """

  # Set of q-grams that only occur in one input q-gram set
  #
  q_gram_symm_diff_set = q_gram_set1.symmetric_difference(q_gram_set2)

  q_gram_union_set =        q_gram_set1 | q_gram_set2

  q_gram_hamm_sim =  1.0 - float(len(q_gram_symm_diff_set)) / \
                                 len(q_gram_union_set)

  assert 0 <= q_gram_hamm_sim and q_gram_hamm_sim <= 1.0

  return q_gram_hamm_sim

# -----------------------------------------------------------------------------

def bit_array_dice_sim(ba1, ba2):
  """Calculate the Dice similarity between the two given bit arrays only
     considering 1-bits.

     Returns a similarity value between 0 and 1.
  """

  num_ones_ba1 = ba1.count(1)
  num_ones_ba2 = ba2.count(1)

  ba_common =       ba1 & ba2
  num_ones_common = ba_common.count(1)

  ba_dice_sim = (2.0 * num_ones_common) / (num_ones_ba1 + num_ones_ba2)

  assert 0 <= ba_dice_sim and ba_dice_sim <= 1.0

  return ba_dice_sim

# -----------------------------------------------------------------------------

def bit_array_hamm_sim(ba1, ba2):
  """Calculate the Hamming similarity between the two given bit arrays
     considering all differing bit positions.

     Returns a similarity value between 0 and 1.
  """

  diff_bit_pos_ba =  ba1 ^ ba2  # XOR of the two input bit arrays
  num_diff_bit_pos = diff_bit_pos_ba.count(1)

  ba_hamm_sim = 1.0 - float(num_diff_bit_pos) / len(ba1)

  assert 0 <= ba_hamm_sim and ba_hamm_sim <= 1.0

  return ba_hamm_sim

# -----------------------------------------------------------------------------

def bit_array_jacc_sim(ba1, ba2):
  """Calculate the Jaccard similarity between the two given bit arrays
     considering all differing bit positions.

     Returns a similarity value between 0 and 1.
  """

  diff_bit_pos_ba =  ba1 ^ ba2  # XOR of the two input bit arrays
  num_diff_bit_pos = diff_bit_pos_ba.count(1)

  ba_jacc_sim = 1.0 - 2.0*float(num_diff_bit_pos) / len(ba1)

  # Possibly a similarity below 0 if more than half of the bit positions differ
  #
  ba_jacc_sim = max(0.0, ba_jacc_sim)

  assert ba_jacc_sim <= 1.0, (num_diff_bit_pos, ba_jacc_sim)

  return ba_jacc_sim

# -----------------------------------------------------------------------------

def bit_array_one_bit_jacc_sim(ba1, ba2):
  """Calculate the Jaccard similarity between the two given bit arrays
     considering only 1 bit positions.

     Returns a similarity value between 0 and 1.
  """

  common_bit_pos =  ba1 & ba2  # XOR of the two input bit arrays
  num_comm_bit_pos = common_bit_pos.count(1)
  
  all_1_bit_pos = ba1 | ba2
  num_all_1_bit_pos = all_1_bit_pos.count(1)

  ba_jacc_sim = float(num_comm_bit_pos) / num_all_1_bit_pos

  # Possibly a similarity below 0 if more than half of the bit positions differ
  #
  ba_jacc_sim = max(0.0, ba_jacc_sim)

  assert ba_jacc_sim <= 1.0, (num_diff_bit_pos, ba_jacc_sim) #thiago: possible bug?

  return ba_jacc_sim

# -----------------------------------------------------------------------------

# End.
