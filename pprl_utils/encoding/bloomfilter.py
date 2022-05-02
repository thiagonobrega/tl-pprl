from __future__ import division
import math
import xxhash
import bitarray
import numpy as np
from scipy.stats import entropy

class BloomFilter:

    # BloomFilter(1000, 0.01)
    # A bloom filter with 1000-element capacity and a 1% false positive rate
    def __init__(self, bit_size, num_hash, bfpower=8):


        self.number_of_elements = 0
        self.potencia = bfpower
        self.bit_size = bit_size

        # Make sure the bloom filter is not too large for the 64-bit hash
        # to fill up.
        if self.bit_size > 18446744073709551616:
            raise Exception(
                'BloomFilter is too large for supported hash functions. Make it smaller by reducing capacity or increasing the false positive rate.')
            return

        # Calculate the optimal number of hash functions
        self.hash_functions = num_hash

        # Build the empty bloom filter
        self.filter = bitarray.bitarray(self.bit_size)
        self.filter.setall(False)


    # Add an element to the bloom filter
    def add(self, element):
        # Run the hashes and add at the hashed index
        for seed in range(self.hash_functions):
            self.filter[ xxhash.xxh64(element, seed=seed).intdigest() % self.bit_size ] = True
        self.number_of_elements += 1


    # Check the filter for an element
    # False means the element is definitely not in the filter
    # True means the element is PROBABLY in the filter
    def check(self, element):
        # Check each hash location. The seed for each hash is 
        # just incremented from the previous seed starting at 0
        for seed in range(self.hash_functions):
            if not self.filter[ xxhash.xxh64(element, seed=seed).intdigest() % self.bit_size ]:
                return False
        # Probably in the filter if it was at each hashed location
        return True
    
    """
        Implentei intercecao e union
    """
    def intersection(self, other):
        """ Calculates the intersection of the two underlying bitarrays and returns
        a new bloom filter object."""
        if self.bit_size != other.bit_size:
            raise ValueError("Intersecting filters requires both filters to have equal capacity and error rate")
        new_bloom = self.copy()
        new_bloom.filter = new_bloom.filter & other.filter
        return new_bloom
    
    def __and__(self, other):
        return self.intersection(other)
    
    def union(self, other):
        """ Calculates the union of the two underlying bitarrays and returns
        a new bloom filter object."""
        if self.bit_size != other.bit_size:
            raise ValueError("Unioning filters requires both filters to have both the same capacity and error rate")
        new_bloom = self.copy()
        new_bloom.filter = new_bloom.filter | other.filter
        return new_bloom

    def __or__(self, other):
        return self.union(other)

    def copy(self):
        """
            Return a copy of this bloom filter.
        """
        new_filter = BloomFilter(self.bit_size, self.hash_functions)
        new_filter.filter = self.filter.copy()
        return new_filter

    def __str__(self):
        return str(self.filter.to01())

    # For testing purposes
    def print_stats(self):
        # print('Capacity '+str(self.capacity))
        # print('Expected Probability of False Positive '+str(self.false_positive_rate))
        print('Bit Size '+str(self.bit_size))
        print('Number of Hash Functions '+str(self.hash_functions))
        print('Number of Elements '+str(self.number_of_elements))

    ##################################################################################
    ###
    ### SPLITING BLOOM FILTER
    ###
    ##################################################################################

    def split(self,n=2,p=256):
        # for i in range(0,n):
        lbs = round(self.bit_size/n)
        a = BloomFilter(self.bit_size, self.hash_functions)
        b = BloomFilter(self.bit_size, self.hash_functions)
        a.filter = self.filter[0:lbs]
        b.filter = self.filter[lbs:]
        return a,b


    ##################################################################################
    ###
    ### BLOOM FILTER HARDENING
    ###
    ##################################################################################
    def xor_folding(self):
        """
            Returns a XOR-Folding BloomFilter with one folding

            Schnell, R., Borgs, C., & Encryptions, F. (2016). XOR-Folding for Bloom Encryptions for Record Linkage.
        """

        fold_pos = round( len(self.filter) / 2 )

        a = self.filter[0:fold_pos]
        b = self.filter[fold_pos:]
        # print('======>',self.bit_size,'#',len(a),len(b))

        r = BloomFilter(self.bit_size, self.hash_functions)
        r.filter = a.__ixor__(b)
        return r

    def blip(self,f=0.02):
        """
            BLoom-and-flI (BLIP)

            Schnell, R., & Borgs, C. (2017). Randomized Response and Balanced Bloom Filters for Privacy Preserving Record Linkage.
            IEEE International Conference on Data Mining Workshops, ICDMW, 218–224. https://doi.org/10.1109/ICDMW.2016.0038
        """

        pf = 0.5 * f
        a = self.filter.copy()

        for i in range(0,len(a)):
            if np.random.random() < pf:
                a[i] = not a[i]


        r = BloomFilter(self.bit_size, self.hash_functions)
        r.filter = a
        return r

    def bblip(self,f=0.02):
        """
            Balanced BLoom-and-flI (BBLIP)

            Schnell, R., & Borgs, C. (2017). Randomized Response and Balanced Bloom Filters for Privacy Preserving Record Linkage.
            IEEE International Conference on Data Mining Workshops, ICDMW, 218–224. https://doi.org/10.1109/ICDMW.2016.0038

            The deufault value was chosend by the best result presented by the author
        """
        a = self.filter.copy()
        b = self.filter.copy()
        b.invert()
        c = a + b

        pf = 0.5 * f


        for i in range(0, len(c)):
            if np.random.random() < pf:
                c[i] = not c[i]

        # r = BloomFilter(cap=cap, fpr=self.false_positive_rate, bfpower=self.potencia, num_hash=lbs)
        r = BloomFilter(int(self.bit_size * 2), int(self.hash_functions*2))
        

        r.filter = c
        return r


########
########
########

class NullField():
    """
        Classe utilizada quando o valor  é nulo
    """

    def __init__(self, nome):
        self.nome = nome
        self.sim = 0


def dice_coefficient(filter1, filter2):
    """
        Calculates the overlap coefficient,or Szymkiewicz–Simpson coefficient, of the two underlyng bloom filter.



        filter1 : crypto.mybloom.bloomfilter
        filter2 : crypto.mybloom.bloomfilter

        return : number between 0 and 1
    """

    if (type(filter1) is NullField) or (type(filter2) is NullField):
        return 0

    h = filter1.intersection(filter2).filter.count(True)
    a = filter1.filter.count(True)
    b = filter2.filter.count(True)
    return 2 * h / (a + b)


def jaccard_coefficient(filter1, filter2):
    """
        Calculates the jaccard index between 2 bloom filters

        filter1 : crypto.mybloom.bloomfilter
        filter2 : crypto.mybloom.bloomfilter

        return : number between 0 and 1
    """

    if (type(filter1) is NullField) or (type(filter2) is NullField):
        return 0

        # Brasileiro, i've coded this part using this website [1]
    # Please check this, chunk of code. It may contain error :p
    # 1 -http://blog.demofox.org/2015/02/08/estimating-set-membership-with-a-bloom-filter/

    inter = filter1.intersection(filter2).filter.count(True)
    union = filter1.union(filter2).filter.count(True)

    if union == 0:
        return 0

    return inter / union

def entropy_coefficient(filter1, filter2, base=2):
    """
        Calculates the entropy coeficiente of the two underlyng bloom filter.

        Entropy is a formula to calculate the homogeneity of a sample data.
        If the value of entropy is 0 it is a completely homogenous sample data.
        If the value of entropy is 1 than it is an equally divided sample data.
        The entropy formula for negative and positive element is,

        Entropy (p, n) = -p log2 (p) – n log2 (n)

        from: https://pdfs.semanticscholar.org/102c/413ee7a354a3acb59af8ebb624114b198dcb.pdf
        Entropy Based Measurement of Text Dissimilarity for Duplicate Detection
        Venkatesh Kumar (Corresponding author), G. Rajendran
        Modern Applied Science Vol. 4, No. 9; September 2010

        filter1 : crypto.mybloom.bloomfilter
        filter2 : crypto.mybloom.bloomfilter

        return : number between 0 and 1
    """

    assert filter1.bit_size == filter2.bit_size
    if (type(filter1) is NullField) or (type(filter2) is NullField):
        return 0

    h1 = entropy(filter1.filter)
    h2 = entropy(filter2.filter)
    hsim = abs(h1 - h2)

    assert hsim >= 0

    return 1 - hsim


def overlap_coefficient(filter1, filter2):
    """
        Calculates the Szymkiewicz–Simpson coefficient (or overlap coeficiente) of the two underlyng bloom filter.

        $overlap(X,Y) = \frac{| X \cap Y |}{min(|X|,|Y|)}$
        If set X is a subset of Y or the converse then the overlap coefficient is equal to 1.

         Vijaymeena, M. K.; Kavitha, K. (March 2016). "A Survey on Similarity Measures in Text Mining" (PDF). Machine Learning and Applications: An International Journal. 3 (1): 19–28. doi:10.5121/mlaij.2016.3103.

        filter1 : crypto.mybloom.bloomfilter
        filter2 : crypto.mybloom.bloomfilter

        return : number between 0 and 1
    """

    if (type(filter1) is NullField) or (type(filter2) is NullField):
        return 0

    xy = filter1.intersection(filter2).filter.count(True)
    x = filter1.filter.count(True)
    y = filter2.filter.count(True)
    if min(x, y) == 0:
        return 0

    return xy / min(x, y)


def hamming_coefficient(filter1, filter2):
    """
        Calculates the Hamming coeficiente of the two underlyng bloom filter.

        In information theory, the Hamming distance between two strings of equal length is the number of positions at which the corresponding symbols are different. In other words, it measures the minimum number of substitutions required to change one string into the other, or the minimum number of errors that could have transformed one string into the other.

         Vijaymeena, M. K.; Kavitha, K. (March 2016). "A Survey on Similarity Measures in Text Mining" (PDF). Machine Learning and Applications: An International Journal. 3 (1): 19–28. doi:10.5121/mlaij.2016.3103.

        filter1 : crypto.mybloom.bloomfilter
        filter2 : crypto.mybloom.bloomfilter

        return : number between 0 and 1
    """

    if (type(filter1) is NullField) or (type(filter2) is NullField):
        return 0

    xy = filter1.intersection(filter2).filter.count(True)
    x = filter1.filter.count(True)
    y = filter2.filter.count(True)

    changes = min(x - xy, y - xy)
    if changes == 0:
        return 1
    else:
        return 1 - (changes / int(filter1.bit_size))


# The seed should be a string
# def test_bloom_filter_performance(number_of_elements = 1000000,false_positive_rate = 0.01, number_of_false_positive_tests = 10000, seed = 'Test'):
#     # Let the user know it might take a bit
#     print('Performing test, this might take a few seconds.')

#     # Make a big bloom filter
#     bf = BloomFilter(number_of_elements, false_positive_rate)

#     # Fill it right up
#     for i in range (number_of_elements):
#         bf.add(seed+i)

#     # Try things we know aren't in the filter
#     false_positives = 0
#     for i in range (number_of_false_positive_tests):
#         if bf.check(seed+'!'+i):
#             false_positives += 1

#     # Calculate the tested rate of false positives
#     tested_false_positive_rate = false_positives/number_of_false_positive_tests

#     # Show the results
#     bf.print_stats()
#     print('')
#     print('Number of False Positive Tests '+number_of_false_positive_tests)
#     print('False Positives Detected '+false_positives)
#     print('Tested False Positive Rate '+tested_false_positive_rate)

