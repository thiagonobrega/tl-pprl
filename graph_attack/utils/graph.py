import math
import random
import networkx
import logging, sys
import numpy
import time
import matplotlib
matplotlib.use('Agg')  # For running on adamms4 (no display)
import matplotlib.pyplot as plt
numpy.set_printoptions(precision=4, linewidth=120)

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

PLT_PLOT_RATIO = 1.0
FILE_FORMAT  = '.eps' #'.png'

PLT_FONT_SIZE    = 20# 28 # used for axis lables and ticks
LEGEND_FONT_SIZE = 20 # 28 # used for legends
TITLE_FONT_SIZE  = 19 # 30 # used for plt title
TICK_FONT_SIZE   = 18


class SimGraph():
  """A class to implement a similarity graph, where nodes correspond to records
     and edges to the similarities calculated between them.

     The node identifiers are record values, such as strings, q-gram sets, or
     bit arrays (encodings).

     The attributes to be stored in the graph are:
     - For edges, their similarity as 'sim' (assumed to be normalised into
       the range 0..1)
     - For nodes, a set off record identifiers (that have the node's value).
  """

  def __init__(self):
    """Initialise a similarity graph.

       Input arguments:
         - This method has no input.

       Output:
         - This method does not return anything.
    """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    # Initialise an empty similarity graph
    #
    self.sim_graph = networkx.Graph()

  # ---------------------------------------------------------------------------

  def add_rec(self, node_key_val, ent_id, org_val):
    """Add a new node to the graph, where the identifier (key) of this node
       is the object that will be compared when similarities are calculated
       between records (this can be a string or q-gram set for a plain-text
       data set, and a bit array for an encoded data set). We also store the
       entity identifier of this record with this node (as a set of entity
       identifiers - i.e. all entities that have that node_key_val), and a set
       of the original values (attribute strings as a tuple) of these records.

       Input arguments:
         - node_key_val  The identifier for this node.
         - ent_id        The entity identifier for this record from the data
                         set.
         - org_val       The original (attribute) value of this record.

       Output:
         - This method does not return anything.
    """

    sim_graph = self.sim_graph

    # If the node key value already exists then add the entity identifier and
    # the original value to the corresponding sets
    #
    if (node_key_val in sim_graph):
      ent_id_set = sim_graph.node[node_key_val]['ent_id_set']
      ent_id_set.add(ent_id)
      org_val_set = sim_graph.node[node_key_val]['org_val_set']
      org_val_set.add(tuple(org_val))

    else:  # Create a new node
      sim_graph.add_node(node_key_val)
      ent_id_set = set([ent_id])
      org_val_set = set([tuple(org_val)])

    sim_graph.node[node_key_val]['ent_id_set'] =  ent_id_set
    sim_graph.node[node_key_val]['org_val_set'] = org_val_set

  # ---------------------------------------------------------------------------
  
  def add_feat_sim_node(self, node_key_val, node_type):
    
    feat_sim_graph = self.sim_graph
    
    if (node_key_val not in feat_sim_graph):
      feat_sim_graph.add_node(node_key_val)
      feat_sim_graph.node[node_key_val]['type'] = node_type
    else:
      assign_node_type = feat_sim_graph.node[node_key_val]['type']
      assert assign_node_type == node_type, (assign_node_type, node_type)
      
  # ---------------------------------------------------------------------------
  
  def add_feat_sim_edge(self, encode_node_key, plain_node_key, sim):
    
    feat_sim_graph = self.sim_graph
    
    if feat_sim_graph.has_edge(encode_node_key, plain_node_key):  # Check existing similarity is same
      if (feat_sim_graph.edges[encode_node_key,plain_node_key]['feat_sim'] != sim):
        logging.debug('  *** Warning: Edge between %s and %s has a different ' % \
                    (encode_node_key,plain_node_key) + \
                    'similarity to new similarity: %.3f versus %.3f' \
                    % (sim_graph.edges[encode_node_key,plain_node_key]['feat_sim'], sim) + \
                    '(old value will be overwritten)'
        )
    else:
      feat_sim_graph.add_edge(encode_node_key,plain_node_key)
    
    feat_sim_graph.edges[encode_node_key,plain_node_key]['feat_sim'] = sim  # Add or update edge
  
  # ---------------------------------------------------------------------------
  
  def add_feat_conf_edge(self):
    
    logging.debug('Adding confidence values to each edge of the feature graph')
    
    start_time = time.time()
    
    feat_sim_graph = self.sim_graph
    
    conf_val_list = [] # For basic statistics
    
    for (encode_key_val, plain_key_val) in feat_sim_graph.edges():
      
      edge_sim_val = feat_sim_graph.edges[encode_key_val,plain_key_val]['feat_sim']
      
      #encode_node_neighbors = feat_sim_graph.neighbors(encode_key_val)
      #encode_node_neighbors = feat_sim_graph.neighbors(plain_key_val)
      
      enc_neighbour_sim_sum = 0.0
      plain_neighbour_sim_sum = 0.0
      
      conf_val = 0.0
      neighbour_sim_list = []
        
      for neigh_key1, neigh_key2 in feat_sim_graph.edges(encode_key_val):
        
        if(neigh_key2 != plain_key_val):
          
          #enc_neighbour_sim_sum += feat_sim_graph[neigh_key1][neigh_key2]['feat_sim']
          neighbour_sim_list.append(feat_sim_graph.edges[neigh_key1,neigh_key2]['feat_sim'])
            
      for neigh_key1, neigh_key2 in feat_sim_graph.edges(plain_key_val):
        
        if(neigh_key2 != encode_key_val):
          
          #plain_neighbour_sim_sum += feat_sim_graph[neigh_key1][neigh_key2]['feat_sim']
          neighbour_sim_list.append(feat_sim_graph.edges[neigh_key1,neigh_key2]['feat_sim'])
      
      if(len(neighbour_sim_list) > 0):
        conf_val = edge_sim_val / numpy.mean(neighbour_sim_list)
      else:
        conf_val = 10.0
      conf_val_list.append(conf_val)
      
      feat_sim_graph.edges[encode_key_val,plain_key_val]['conf'] = conf_val
    
    conf_cal_time = time.time() - start_time
    
    logging.debug('  Minimum, average, median and maximum confidence: ' + \
        '%.3f / %.3f / %.3f / %.3f' % (min(conf_val_list),
                                 numpy.mean(conf_val_list),
                                 numpy.median(conf_val_list),
                                 max(conf_val_list))
    )
  
  
  def get_conf_val(self, encode_key, plain_key):
    
    feat_sim_graph = self.sim_graph
    
    conf_val = feat_sim_graph.edges[encode_key,plain_key]['conf']
    
    return conf_val
  
  
  # ---------------------------------------------------------------------------
  
  def get_sim_pair_list(self):
    
    feat_sim_graph = self.sim_graph
  
  # ---------------------------------------------------------------------------

  def add_edge_sim(self, node_key_val1, node_key_val2, sim):
    """Add an (undirected) edge to the similarity graph between the two given
       nodes with the given similarity. It is assumed the two node key values
       are available as nodes in the graph, if this is not the case the
       function with exit with an error.

       Input arguments:
         - node_key_val1  The first node key value - needs to be available in
                          the graph.
         - node_key_val2  The second node key value - needs to be available in
                          the graph.
         - sim            The similarity value of the edge (edge attribute).

       Output:
         - This method does not return anything.
    """

    sim_graph = self.sim_graph

    # Sort values so we only insert an edge once
    #
    val1s, val2s = sorted([node_key_val1, node_key_val2])

    if sim_graph.has_edge(val1s, val2s):  # Check existing similarity is same
      if (sim_graph.edges[val1s,val2s]['sim'] != sim):
        logging.debug('  *** Warning: Edge between %s and %s has a different ' % \
              (node_key_val1,node_key_val2) + \
              'similarity to new similarity: %.3f versus %.3f' \
              % (sim_graph.edges[val1s][val2s]['sim'], sim) + \
              '(old value will be overwritten)'
        )

    else:
      sim_graph.add_edge(val1s,val2s)
    sim_graph.edges[val1s,val2s]['sim'] = sim  # Add or update edge

  # ---------------------------------------------------------------------------
  
  def show_graph_characteristics(self, min_sim_list=None, graph_name=''):
    """Print the number of nodes and edges in the similarity graph, their
       degree distributions, number of connected components and their sizes.

       Edges, number and sizes of connected components, and number of
       singletons are calculated for different edge minimum similarities if
       the list 'min_sim_list' is provided.

       Input arguments:
         - min_sim_list  A list with similarity values, or None (default).
         - graph_name    A string to be printed with the name of the graph

       Output:
         - This method does not return anything.
    """

    sim_graph = self.sim_graph

    if (graph_name != ''):
      logging.debug('Similarity graph characteristics for "%s":' % (graph_name))
    else:
      logging.debug('Similarity graph characteristics:')
    
    logging.debug('  Number of nodes:    ', len(sim_graph.nodes()))

    # Get a distribution of the node frequencies
    # (keys will be counts, values how many nodes have that count)
    #
    node_freq_dict = {}
    for node_key_val in sim_graph.nodes():
      node_freq = len(sim_graph.node[node_key_val]['ent_id_set'])
      node_freq_dict[node_freq] = node_freq_dict.get(node_freq, 0) + 1
    logging.debug('    Distribution of node frequencies:', \
          sorted(node_freq_dict.items()))

    node_freq_dict.clear()  # Not needed anymore

    logging.debug('  Number of all edges:', len(sim_graph.edges()))

    # Count the number of singletons (identifiers with no edges)
    #
    sim_graph_degree_list = sim_graph.degree().values()
    num_singleton =         sim_graph_degree_list.count(0)
    sim_graph_degree_list = []  # Not needed anymore

    logging.debug('    Number of singleton nodes:', num_singleton)

    # Calculate number of edges and singletons for the given similarity
    # thresholds
    #
    if (min_sim_list != None):
      for min_sim in min_sim_list:
        logging.debug('  For minimum similarity threshold %.2f:' % (min_sim))

        num_min_sim_edges = 0

        for (node_key_val1,node_key_val2) in sim_graph.edges():
          if (sim_graph.edges[node_key_val1,node_key_val2]['sim'] >= min_sim):
            num_min_sim_edges += 1
        logging.debug('    Number of edges:          ', num_min_sim_edges)

        # For this minimum similarity, get the distribution of the degrees of
        # all nodes
        #
        min_sim_degree_dist_dict = {}

        for node_key_val in sim_graph.nodes():
          node_degree = 0

          # Loop over all the node's neighbours
          #
          for neigh_key_val in sim_graph.neighbors(node_key_val):

            if (sim_graph.edges[node_key_val,neigh_key_val]['sim'] >= min_sim):
              node_degree += 1
          min_sim_degree_dist_dict[node_degree] = \
                     min_sim_degree_dist_dict.get(node_degree, 0) + 1
        logging.debug('    Degree distribution:      ', \
              sorted(min_sim_degree_dist_dict.items())[:20], '.....', \
              sorted(min_sim_degree_dist_dict.items())[-20:]
        )

        min_sim_degree_dist_dict.clear()  # Not needed anymore
      
        # Generate the graph with only edges of the current minimum similarity
        # in order to find connected components
        #
        min_sim_graph = networkx.Graph()
        for node_key_val in sim_graph.nodes():
          min_sim_graph.add_node(node_key_val)
        for (node_key_val1,node_key_val2) in sim_graph.edges():
           if (sim_graph.edges[node_key_val1,node_key_val2]['sim'] >= min_sim):
              min_sim_graph.add_edge(node_key_val1,node_key_val2)

        conn_comp_list = list(networkx.connected_components(min_sim_graph))

        conn_comp_size_dist_dict = {}

        for conn_comp in conn_comp_list:
          conn_comp_size = len(conn_comp)
          conn_comp_size_dist_dict[conn_comp_size] = \
                   conn_comp_size_dist_dict.get(conn_comp_size, 0) + 1        
        logging.debug('    Number of connected components:', len(conn_comp_list))
        logging.debug('      Size of connected components distribution:', \
              sorted(conn_comp_size_dist_dict.items())
        )

        logging.debug('    Number of singletons: %d' % \
              (conn_comp_size_dist_dict.get(1, 0))
        )

        min_sim_graph.clear()  # Not needed anymore
        conn_comp_list = []
        conn_comp_size_dist_dict.clear()
 
        logging.debug('')

  # ---------------------------------------------------------------------------

  def remove_singletons(self):
    """Remove all singleton nodes (nodes that have no edge to any other node).

       Input arguments:
         - This method has no input.

       Output:
         - num_singleton  The number of singletones removed
    """

    sim_graph = self.sim_graph

    num_singleton = 0

    for (node_key_val, node_degree) in sim_graph.degree().items():
      if (node_degree == 0):
        sim_graph.remove_node(node_key_val)
        num_singleton += 1

    logging.debug('Removed %d singleton nodes' % (num_singleton))
    logging.debug('')

    return num_singleton

  # ---------------------------------------------------------------------------

# Improved version of the function below: do piece-wise linear regression
#  See for example: https://datascience.stackexchange.com/questions/8457/python-library-for-segmented-regression-a-k-a-piecewise-regression

  # ---------------------------------------------------------------------------
  
  def get_sim_differences(self, other_node_val_dict, other_sim_funct,
                          sim_diff_inter_size, num_samples,
                          plot_file_name=None):
    """Sample edges of the similarity graph and calculate similarities between
       the corresponding same pair of nodes in the given other node dictionary
       (assumed to contain q-gram sets and bit arrays), and then calculate the
       similarity differences between the two edges. For the given different
       similarity intervals calculate and return an average similarity
       difference.

       Input arguments:
         - other_node_val_dict  A dictionary with plain-text node key values
                                as keys and bit arrays (encoded values) as
                                values.
         - other_sim_funct      The similarity function to be used to compare
                                the encoded bit arrays.
         - sim_diff_inter_size  The size of the similarity difference interval
                                to consider
         - num_samples          The total number of edges to be sampled.
         - plot_file_name       If provided then a plot will be generated with
                                the similarity differences as well as their
                                averages over the different similarity
                                intervals.

       Output:
         - sim_diff_dict  For each minimum similarity (key) given in the input
                          'sim_interval_list' list return a pair with the
                          average similarity difference in this interval and
                          the number of samples in the interval. Note that if
                          there are no samples in an interval then the
                          similarity difference will be set to 0.0 (no
                          difference).
    """

    sim_graph = self.sim_graph

    # Generate a list of all similarity intervals downwards then reverse
    #
    sim_interval_list = [1.0]
    while ((sim_interval_list[-1]-sim_diff_inter_size) > 0.0):
      sim_interval_list.append(sim_interval_list[-1] - sim_diff_inter_size)
    sim_interval_list[-1] = 0.0  # Ensure lowest similarity is not negative

    sim_interval_list.reverse()  # Make 1.0 last

    # Initialise an empty similarity difference dictionary
    #
    sim_diff_dict = {}
    for sim_interval in sim_interval_list:
      sim_diff_dict[sim_interval] = []

    all_edge_list = list(sim_graph.edges())

    # If a plot is to be generated create lists of x- and y values for plotting
    #
    if (plot_file_name != None):
      x_val_list = []  # The similarity differences
      y_val_list = []  # The plain-text similarities

    # Reduce number of samples is there are not enough edges
    #
    if (len(all_edge_list) < num_samples):
      num_samples = len(all_edge_list)-1

    logging.debug('Sample %d from %d edges in graph to calculate similarity ' % \
          (num_samples, len(all_edge_list)) + 'differences'
    )
    logging.debug('  Using similarity intervals:', sim_interval_list)

    for s in range(num_samples):

      sample_edge = random.choice(all_edge_list)

      # Get the two key values of this edge and check they both also occur in
      # in the encoded data set
      #
      node_key_val1, node_key_val2 = sample_edge

      if ((node_key_val1 in other_node_val_dict) and \
          (node_key_val2 in other_node_val_dict)):

        bit_array1 = other_node_val_dict[node_key_val1]
        bit_array2 = other_node_val_dict[node_key_val2]

        ba_sim = other_sim_funct(bit_array1, bit_array2)

        edge_sim = sim_graph.edges[node_key_val1,node_key_val2]['sim']

        # If sim_diff is positive: encoded sim is larger than plain-text sim
        # If sim_diff is negative: encoded sim is smaller than plain-text sim
        #
        sim_diff = ba_sim - edge_sim

        if (plot_file_name != None):
          x_val_list.append(sim_diff)
          y_val_list.append(ba_sim)

        # Add the similarity difference to the dictionary to be returned in
        # the appropriate similarity interval
        i = 0
        while (ba_sim > sim_interval_list[i]):
          i += 1
        sim_interval = sim_interval_list[i]

        assert ba_sim <= sim_interval

        # The list of similarity differences for this interval
        #
        sim_interval_sim_diff_list = sim_diff_dict[sim_interval]
        sim_interval_sim_diff_list.append(sim_diff)
        sim_diff_dict[sim_interval] = sim_interval_sim_diff_list

        all_edge_list.remove(sample_edge)  # Ensure each edge is only used once
  
    # Convert similarity difference lists into averages and number of samples
    # (first remove all low ones that do not contain any samples)
    #
    for sim_interval in sim_interval_list:
      if (len(sim_diff_dict[sim_interval]) == 0):
        del sim_diff_dict[sim_interval]
      else:
        break  # First non-empty interval, exit the loop

    for sim_interval in sorted(sim_diff_dict.keys()):
      sim_interval_sim_diff_list = sim_diff_dict[sim_interval]

      if (len(sim_interval_sim_diff_list) == 0):
        logging.debug('  *** Warning no samples in similarity interval >= %.2f ***' % \
              (sim_interval)
        )
        sim_diff_dict[sim_interval] = (0.0, 0)

      else:
        avr_sim_diff_inter = numpy.mean(sim_interval_sim_diff_list)
        num_samples_inter =  len(sim_interval_sim_diff_list)
        sim_diff_dict[sim_interval] = (numpy.mean(sim_interval_sim_diff_list),
                                       len(sim_interval_sim_diff_list))
        logging.debug('  Similarity interval >= %.2f has %d samples and average ' % \
              (sim_interval, num_samples_inter) + \
              'similarity difference is %.3f' % (avr_sim_diff_inter)
        )
    logging.debug('')

    all_edge_list = []  # Not needed anymore

    # Generate the plot if needed
    #
    if (plot_file_name != None) and (len(x_val_list) > 0):

      x_min = min(x_val_list)
      x_max = max(x_val_list)
      x_diff = x_max - x_min 
      plot_min_x = x_min - x_diff*0.05
      plot_max_x = x_max + x_diff*0.05

      plot_min_y = min(y_val_list) - min(y_val_list)*0.05
      plot_max_y = 1.0 + min(y_val_list)*0.05

      # Plot the generated lists as scatter plot
      #
      w,h = plt.figaspect(PLT_PLOT_RATIO)
      plt.figure(figsize=(w,h))

      plt.title('Similarity differences of plain-text versus encoded edges')

      plt.xlabel('Encoded similarity minus plain-text similarity')
      plt.ylabel('Encoded similarity')

      plt.xlim(plot_min_x, plot_max_x)
      plt.ylim(plot_min_y, plot_max_y)

      plt.scatter(x_val_list, y_val_list, color='k')

      # Add the calculated averages as a line plot
      #
      avr_x_list = []
      avr_y_list = []

      for sim_interval in sorted(sim_diff_dict.keys()):
        sim_diff = sim_diff_dict[sim_interval][0]

        avr_x_list.append(sim_diff)
        avr_y_list.append(sim_interval)

      plt.plot(avr_x_list, avr_y_list, color='r', lw=2, label='Avr sim diff')

      plt.legend(loc="best",) #, prop={'size':14}, bbox_to_anchor=(1, 0.5))

      plt.savefig(plot_file_name, bbox_inches='tight')

    return sim_diff_dict

  # ---------------------------------------------------------------------------
  
  def comp_sim_differences(self, other_node_val_dict, other_sim_graph,
                           sim_diff_inter_size, num_samples,
                           plot_file_name=None):
    """Sample edges of the similarity graph that also occur in the other
       similarity graph, and calculate the similarity differences between
       these edges, report the average similarity difference for each interval
       based on the 'sim_diff_inter_size' and plot if a file name is given.

       Input arguments:
         - other_node_val_dict  A dictionary with plain-text node key values
                                as keys and bit arrays (encoded values) as
                                values.
         - other_sim_graph      The other similarity graph from where edges
                                are to be compared.
         - sim_diff_inter_size  The size of the similarity difference interval
                                to consider
         - num_samples          The total number of edges to be sampled.
         - plot_file_name       If provided then a plot will be generated with
                                the similarity differences as well as their
                                averages over the different similarity
                                intervals.

       Output:
         - This method does not return anything.
    """

    sim_graph = self.sim_graph

    # Generate a list of all similarity intervals downwards then reverse
    #
    sim_interval_list = [1.0]
    while ((sim_interval_list[-1]-sim_diff_inter_size) > 0.0):
      sim_interval_list.append(sim_interval_list[-1] - sim_diff_inter_size)
    sim_interval_list[-1] = 0.0  # Ensure lowest similarity is not negative

    sim_interval_list.reverse()  # Make 1.0 last

    # Initialise an empty similarity difference dictionary
    #
    sim_diff_dict = {}
    for sim_interval in sim_interval_list:
      sim_diff_dict[sim_interval] = []

    # Get pairs of actual record values from both similarity graphs an their
    # similarities
    #
    sim_graph_pair_dict = {}
    common_sim_graph_pair_dict = {} # With similarties from both graphs

    for (node_key_val1, node_key_val2) in sim_graph.edges():

      # If a node has several record value tuples take the first one only
      #
      str1 = ' '.join(sorted(sim_graph.node[node_key_val1]['org_val_set'])[0])
      str2 = ' '.join(sorted(sim_graph.node[node_key_val2]['org_val_set'])[0])

      # Only use if both the nodes have non-empty original values
      #
      if (len(str1) > 0) and (len(str2) > 0):
        rec_pair_dict_key = tuple(sorted([str1, str2]))
        assert rec_pair_dict_key not in sim_graph_pair_dict
        sim_graph_pair_dict[rec_pair_dict_key] = \
              sim_graph.edges[node_key_val1,node_key_val2]['sim']

    for (node_key_val1, node_key_val2) in other_sim_graph.edges():
      str1 = \
       ' '.join(sorted(other_sim_graph.node[node_key_val1]['org_val_set'])[0])
      str2 = \
       ' '.join(sorted(other_sim_graph.node[node_key_val2]['org_val_set'])[0])

      if (len(str1) > 0) and (len(str2) > 0):
        rec_pair_dict_key = tuple(sorted([str1, str2]))

        if (rec_pair_dict_key in sim_graph_pair_dict):
          sim_graph_sim = sim_graph_pair_dict[rec_pair_dict_key]
          common_sim_graph_pair_dict[rec_pair_dict_key] = \
            (sim_graph_sim,
             other_sim_graph.edges[node_key_val1,node_key_val2]['sim'])

          # Once we have enough pairs exit the loop
          #
          if (len(common_sim_graph_pair_dict) == num_samples):
            break  # Exit the for loop

    sim_graph_pair_dict.clear()  # Not needed anymore

    logging.debug('Using %d common edges to calculate similarity differences' % \
          (len(common_sim_graph_pair_dict)), num_samples
    )
    logging.debug('  Using similarity intervals:', sim_interval_list)

    # If a plot is to be generated create lists of x- and y values for plotting
    #
    if (plot_file_name != None):
      x_val_list = []  # The similarity differences
      y_val_list = []  # The plain-text edge similarities

    for (edge_sim, other_edge_sim) in common_sim_graph_pair_dict.values():

        # If sim_diff is positive: encoded sim is larger than plain-text sim
        # If sim_diff is negative: encoded sim is smaller than plain-text sim
        #
        sim_diff = other_edge_sim - edge_sim

        if (plot_file_name != None):
          x_val_list.append(sim_diff)
          y_val_list.append(other_edge_sim)

        # Add the similarity difference to the dictionary to be returned in
        # the appropriate similarity interval
        i = 0
        while (other_edge_sim > sim_interval_list[i]):
          i += 1
        sim_interval = sim_interval_list[i]

        assert other_edge_sim <= sim_interval

        # The list of similarity differences for this interval
        #
        sim_interval_sim_diff_list = sim_diff_dict[sim_interval]
        sim_interval_sim_diff_list.append(sim_diff)
        sim_diff_dict[sim_interval] = sim_interval_sim_diff_list

    # Convert similarity difference lists into averages and number of samples
    # (first remove all low ones that do not contain any samples)
    #
    for sim_interval in sim_interval_list:
      if (len(sim_diff_dict[sim_interval]) == 0):
        del sim_diff_dict[sim_interval]
      else:
        break  # First non-empty interval, exit the loop

    for sim_interval in sorted(sim_diff_dict.keys()):
      sim_interval_sim_diff_list = sim_diff_dict[sim_interval]

      if (len(sim_interval_sim_diff_list) == 0):
        logging.debug('  *** Warning no samples in similarity interval <= %.2f ***' % \
              (sim_interval)
        )
        sim_diff_dict[sim_interval] = (0.0, 0)

      else:
        avr_sim_diff_inter = numpy.mean(sim_interval_sim_diff_list)
        num_samples_inter =  len(sim_interval_sim_diff_list)
        sim_diff_dict[sim_interval] = (numpy.mean(sim_interval_sim_diff_list),
                                       len(sim_interval_sim_diff_list))
        logging.debug('  Similarity interval >= %.2f has %d samples and average ' % \
              (sim_interval, num_samples_inter) + \
              'similarity difference is %.3f' % (avr_sim_diff_inter)
        )
    logging.debug('')

    common_sim_graph_pair_dict.clear()  # Not needed anymore

    # Generate the plot if needed
    #
    if (plot_file_name != None):

      x_min = min(x_val_list)
      x_max = max(x_val_list)
      x_diff = x_max - x_min 
      plot_min_x = x_min - x_diff*0.05
      plot_max_x = x_max + x_diff*0.05

      plot_min_y = min(y_val_list) - min(y_val_list)*0.05
      plot_max_y = 1.0 + min(y_val_list)*0.05

      # Plot the generated lists as scatter plot
      #
      w,h = plt.figaspect(PLT_PLOT_RATIO)
      plt.figure(figsize=(w,h))

      plt.title('Adjusted similarity differences of plain-text versus ' + \
                'encoded edges')

      plt.xlabel('Encoded similarity minus plain-text similarity')
      plt.ylabel('Encoded similarity')

      plt.xlim(plot_min_x, plot_max_x)
      plt.ylim(plot_min_y, plot_max_y)

      plt.scatter(x_val_list, y_val_list, color='k')

      # Add the calculated averages as a line plot
      #
      avr_x_list = []
      avr_y_list = []

      for sim_interval in sorted(sim_diff_dict.keys()):
        sim_diff = sim_diff_dict[sim_interval][0]

        avr_x_list.append(sim_diff)
        avr_y_list.append(sim_interval)

      plt.plot(avr_x_list, avr_y_list, color='r', lw=2, label='Avr sim diff')

      plt.legend(loc="best",) #, prop={'size':14}, bbox_to_anchor=(1, 0.5))

      plt.savefig(plot_file_name, bbox_inches='tight')

  
  # ---------------------------------------------------------------------------
  
  def apply_regre(self, other_node_val_dict, other_sim_funct, 
                  num_samples, regre_model_str, plot_data_name, 
                  plot_file_name=None):
    
    """Sample edges of the similarity graph and calculate similarities between
       the corresponding same pair of nodes in the given other node dictionary
       (assumed to contain q-gram sets and bit arrays), and then calculate the
       similarity differences between the two edges. For the given different
       similarity intervals calculate and return an average similarity
       difference.

       Input arguments:
         - other_node_val_dict  A dictionary with plain-text node key values
                                as keys and bit arrays (encoded values) as
                                values.
         - other_sim_funct      The similarity function to be used to compare
                                the encoded bit arrays.
         - sim_diff_inter_size  The size of the similarity difference interval
                                to consider
         - num_samples          The total number of edges to be sampled.
         - plot_file_name       If provided then a plot will be generated with
                                the similarity differences as well as their
                                averages over the different similarity
                                intervals.

       Output:
         - bayes_regre_model  The trained Bayesian regression model
    """

    sim_graph = self.sim_graph # Encoded graph

    all_edge_list = list(sim_graph.edges())

    x_val_list = []  # The plain-text similarities (independent/predictor variable)
    q_val_list = []  # Number of unique q-grams in both records (independent/predictor variable)
    y_val_list = []  # The encoded similarities (dependent/response variable)
    

    # Reduce number of samples if there are not enough edges
    #
    if (len(all_edge_list) < num_samples):
      num_samples = len(all_edge_list)-1

    logging.debug('Sample %d from %d edges in graph to calculate similarity ' % \
          (num_samples, len(all_edge_list)) + 'differences'
    )
    logging.debug('')
    
    select_num_samples = 0
    
    #encode_node_val_dict[node_key_val] = (q_gram_set, bit_array)

    while(select_num_samples < num_samples and len(all_edge_list) > 0):
    #for s in range(num_samples):

      sample_edge = random.choice(all_edge_list)

      # Get the two key values of this edge and check they both also occur in
      # in the encoded data set
      #
      node_key_val1, node_key_val2 = sample_edge
      
      
      q_gram_set1, bit_array1 = other_node_val_dict[node_key_val1]
      q_gram_set2, bit_array2 = other_node_val_dict[node_key_val2]
      
      
      full_q_gram_set = q_gram_set1.union(q_gram_set2)

      qg_sim = other_sim_funct(q_gram_set1, q_gram_set2)

      edge_sim = sim_graph.edges[node_key_val1,node_key_val2]['sim'] # encoded similarity

      x_val_list.append(qg_sim)
      y_val_list.append(edge_sim)
      q_val_list.append(len(full_q_gram_set))

      all_edge_list.remove(sample_edge)  # Ensure each edge is only used once
      
      select_num_samples += 1
      
      if(select_num_samples%5000 == 0):
        logging.debug('  Selected %d number of samples' %select_num_samples)
      
      
      
#===============================================================================
#       if ((q_gram_key_val1 in other_node_val_dict) and \
#           (q_gram_key_val2 in other_node_val_dict)):
# 
#         q_gram_set1 = other_node_val_dict[q_gram_key_val1]
#         q_gram_set2 = other_node_val_dict[q_gram_key_val2]
#         
#         full_q_gram_set = q_gram_set1.union(q_gram_set2)
# 
#         qg_sim = other_sim_funct(q_gram_set1, q_gram_set2)
# 
#         edge_sim = sim_graph.edges[node_key_val1,node_key_val2]['sim'] # encoded similarity
# 
#         x_val_list.append(qg_sim)
#         y_val_list.append(edge_sim)
#         q_val_list.append(len(full_q_gram_set))
# 
#         all_edge_list.remove(sample_edge)  # Ensure each edge is only used once
#         
#         select_num_samples += 1
#===============================================================================

    #Reshape the X value list if there is only one feature
    #x_val_array = numpy.array(x_val_list).reshape(-1, 1)
    
    
    
    if(regre_model_str == 'linear'):
      
      logging.debug('Traning the linear regression model')
      
      x_val_array = numpy.array(zip(x_val_list,q_val_list))
    
      # Split the data for the training and testing samples
      x_train, x_test, y_train, y_test = train_test_split(x_val_array, y_val_list, test_size=0.25, random_state=42)
      
      # Train the model
      reg_model = linear_model.LinearRegression()
      
      reg_model.fit(x_train, y_train)
      
      # Testing the model
      y_predict = reg_model.predict(x_test)
      
    elif(regre_model_str == 'isotonic'):
      
      logging.debug('Traning the isotonic regression model')
      
      x_train, x_test, y_train, y_test = train_test_split(x_val_list, y_val_list, test_size=0.25, random_state=42)
    
      reg_model = IsotonicRegression()
  
      reg_model.fit_transform(x_train, y_train)
      
      y_predict = reg_model.predict(x_test)
      
    elif(regre_model_str == 'poly'):
      
      logging.debug('Traning the polynomial regression model')
      
      x_val_array = numpy.array(zip(x_val_list,q_val_list))
    
      # Split the data for the training and testing samples
      x_train, x_test, y_train, y_test = train_test_split(x_val_array, y_val_list, test_size=0.25, random_state=42)
      
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
    for i, y_p in enumerate(y_predict):
      y_t = y_test[i]
      
      if(y_p > 0):
        y_p_new.append(y_p)
        y_t_new.append(y_t)
    
    y_predict = y_p_new
    y_test    = y_t_new
    
    err_val_list = [abs(y_p - y_t) for y_p, y_t in zip(y_predict, y_test)]
    
    mean_sqrd_lg_err = numpy.mean([(math.log(1+y_p) - math.log(1+y_t))**2 for y_p, y_t in zip(y_predict, y_test)])
    
    # Evaluating the model
    #
    model_var_score        = metrics.explained_variance_score(y_test, y_predict)
    model_min_abs_err      = min(err_val_list)
    model_max_abs_err      = max(err_val_list)
    model_avrg_abs_err     = numpy.mean(err_val_list)
    model_std_abs_err      = numpy.std(err_val_list)
    model_mean_sqrd_err    = metrics.mean_squared_error(y_test, y_predict)
    model_mean_sqrd_lg_err = mean_sqrd_lg_err
    model_r_sqrd           = metrics.r2_score(y_test, y_predict)
    
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
    
    # Generate the plot if needed
    #
    
    #plot_file_name = None
    
    if (plot_file_name != None) and (len(x_val_list) > 0):
      
      if(regre_model_str in ['linear', 'poly']):
        x_plot_train, q_val_train = zip(*x_train)
        x_plot_test, q_val_test = zip(*x_test)
      else:
        x_plot_train = x_train
        x_plot_test = x_test
      
      # Sampling for the plot
      x_plot_train = x_plot_train[:2000]
      y_train      = y_train[:2000]
      
      x_plot_test  = x_plot_test[:500]
      y_test       = y_test[:500]
      y_predict    = y_predict[:500]
      

      #x_min = min(x_val_list)
      #x_max = max(x_val_list)
      #x_diff = x_max - x_min 
      #plot_min_x = x_min - x_diff*0.05
      #plot_max_x = x_max + x_diff*0.05

      #plot_min_y = min(y_val_list) - min(y_val_list)*0.05
      #plot_max_y = 1.0 + min(y_val_list)*0.05

      
      fig, ax = plt.subplots(1, 1)
      ax.plot(x_plot_train, y_train, ".", markersize=1, color='#144ba4', label='Training set')
      ax.plot(x_plot_test, y_test, ".", markersize=1, color='#0bb015', label="Testing set")
      ax.plot(x_plot_test, y_predict, ".", markersize=2, color='red', label="Predictions")
      
      if(regre_model_str == 'linear'):
        plot_title = 'Linear Regression ' + plot_data_name
      elif(regre_model_str == 'isotonic'):
        plot_title = 'Isotonic Regression ' + plot_data_name
      elif(regre_model_str == 'poly'):
        plot_title = 'Polynomial Regression ' + plot_data_name
      
      ax.set_title(plot_title)
      ax.legend(loc="best",)
      
      # Plot the generated lists as scatter plot
      #
      #w,h = plt.figaspect(PLT_PLOT_RATIO)
      #plt.figure(figsize=(w,h))

      #plt.title('Similarity variance of plain-text versus encoded edges')

      plt.xlabel('Q-gram similarity')
      plt.ylabel('Encoded similarity')

      #plt.xlim(plot_min_x, plot_max_x)
      #plt.ylim(plot_min_y, plot_max_y)
      
      #sizes = [0.1*x for x in range(len(x_val_list))]

      #plt.scatter(x_val_list, y_val_list, s=1)

#===============================================================================
#       # Add the calculated averages as a line plot
#       #
#       avr_x_list = []
#       avr_y_list = []
# 
#       for sim_interval in sorted(sim_diff_dict.keys()):
#         sim_diff = sim_diff_dict[sim_interval][0]
# 
#         avr_x_list.append(sim_diff)
#         avr_y_list.append(sim_interval)
# 
#       plt.plot(avr_x_list, avr_y_list, color='r', lw=2, label='Avr sim')
#===============================================================================

      #plt.legend(loc="best",) #, prop={'size':14}, bbox_to_anchor=(1, 0.5))

      plt.savefig(plot_file_name, bbox_inches='tight')

    return reg_model, eval_res_tuple
  
  
  # ---------------------------------------------------------------------------

  def gen_min_conn_comp_graph(self, min_conn_comp_size):
    """Remove fro mthe graph all connected components (nodes and edges) that
       aresmaller than the given minimum component size, because smaller
       components do not contain enough information to distinguishing nodes
       from each other (i.e. their neighbourhoods are too small).

       Input arguments:
         - min_conn_comp_size  The minimum size of the connected components to
                               be kept.
       Output:
         - This method does not return anything.
    """

    sim_graph = self.sim_graph

    # Loop over all connected components, and remove nodes and edges from
    # those that are too small
    #
    conn_comp_list = list(networkx.connected_components(sim_graph))
    for conn_comp in conn_comp_list:
      if (len(conn_comp) < min_conn_comp_size):
        for node_key_val in conn_comp:
          node_neighbour_list = sim_graph.neighbors(node_key_val)
          sim_graph.remove_node(node_key_val)

          for other_node_key_val in node_neighbour_list:
            if sim_graph.has_edge(node_key_val, other_node_key_val):
              sim_graph.remove_edge(node_key_val, other_node_key_val)

    conn_comp_list = []  # Not needed anymore

    logging.debug('Graph without connected components smaller than %d nodes ' % \
          (min_conn_comp_size) + 'contains %d nodes and %d edges' % \
          (len(sim_graph.nodes()), len(sim_graph.edges()))
    )
    logging.debug('')

  # ---------------------------------------------------------------------------

  def calc_features(self, calc_feature_list, min_sim_list, min_node_degree,
                    max_node_degree):
    """Calculate various features for each node based on its edges, as well as
       node and edge attributes, where some of these features are calculated
       for each of the similarity values in the provided list. Only nodes that
       have a 'min_node_degree' in the graph for the lowest similarity will be
       considered. For each such node a list of numerical features is
       calculated, returned in a dictionary with node values as keys.

       Input arguments:
         - calc_feature_list  A list with the names of the features to be
                              calculated. Possible features are are listed
                              and described below.
         - min_sim_list       A list with similarity threshold values.
         - min_node_degree    The minimum degree a node requires in order to
                              be considered.
         - max_node_degree    The maximum degree of any node in the graph.

       Output:
         - node_feat_dict  A dictionary with one entry per node of the graph,
                           with its key being the node value and its value
                           being an array of the calculated numerical feature
                           values.
         - min_feat_array  An array which for each feature contains its
                           minimum calculated value.
         - max_feat_array  An array which for each feature contains its
                           maximum calculated value.
         - feat_name_list  A list with the names of all generated features.

       Possible features that can be included in the 'calc_feature_list' are:
         - node_freq          The count of a node.
         - max_sim            The overall maximum edge similarity for a node
                              (independent of the list of similarity values
                              given).
         - min_sim            The minimum edge similarity (per minimum
                              similarity value).
         - sim_avr            The average similarity of all edges connected
                              to a node (per minimum similarity value).
         - sim_std            The similarity standard deviation of all edges
                              connected to a node (per minimum similarity
                              value).
         - degree             The number of edges connecting the node to other
                              nodes (per minimum similarity value).
         - degree_histo1      Histogram of degree distribution of neighbours
                              directly connected to a node (via one edge)
                              (per minimum similarity value).
         - degree_histo2      Histogram of degree distribution of neighbours
                              connected via two edges to a node (per minimum
                              similarity value).
         - egonet_degree      Number of edges connecting the egonet of a node
                              (i.e. all its neighbours) to other nodes outside
                              the egonet.
         - egonet_density     The density of the egonet as the number of edges
                              between nodes (including the node itself) of the
                              egonet over all possible edges.
         - pagerank           The page rank of the node (per minimum
                              similarity value).
                              *** Note very slow for large graphs. ***
         - between_central    The betweenness centrality for each node (per
                              minimum similarity value).
         - closeness_central  The closeness centrality for each node (per
                              minimum similarity value).
                              *** Note very slow for large graphs. ***
         - degree_central     The degree centrality for each node (per
                              minimum similarity value).
         - eigenvec_central   The eigenvector centrality for each node (per
                              minimum similarity value).
                              *** Does not always converge ***
    """

    t1 = time.time()

    sim_graph = self.sim_graph

    # If required calculate the required length of the degree histograms
    #
    if ('degree_histo1' in calc_feature_list) or \
       ('degree_histo2' in calc_feature_list):

       # The neighbour node degree histograms are based on logarithmic scale
       # (following "REGAL: Representation Learning-based Graph Alignment",
       # M. Heimann et al., CIKM 2018):
       #
       degree_hist_len = int(math.log(max_node_degree,2))+1
    else:
       degree_hist_len = 0

    # Generate the list of the names of all features to be generated
    #
    feat_name_list = []

    if ('node_freq' in calc_feature_list):
      feat_name_list.append('node_freq')
    if ('max_sim' in calc_feature_list):
      feat_name_list.append('max_sim')

    # The following list is ordered in the same way how features are added in
    # the code below. If the below code is changed then this list needs to be
    # updated as well.
    #
    for min_sim in min_sim_list:
      for feat_name in ['min_sim','sim_avr','sim_std','degree',
                        'degree_histo1','degree_histo2','egonet_degree',
                        'egonet_density']:
        if feat_name in calc_feature_list:
          if ('degree_histo' not in feat_name):
            feat_name_list.append(feat_name+'-%.2f' % (min_sim))
          else:  # For degree histograms one feature per histogram bucket
            for histo_bucket in range(degree_hist_len):
              feat_name_list.append(feat_name+'-b%d-%.2f' % \
                                    (histo_bucket, min_sim))
    for min_sim in min_sim_list:
      for feat_name in ['pagerank','between_central','closeness_central',
                        'degree_central', 'eigenvec_central']:
        if feat_name in calc_feature_list:
          feat_name_list.append(feat_name+'-%.2f' % (min_sim))

    num_feat = len(feat_name_list)

    logging.debug('')
    logging.debug('Generating %d features per graph node' % (num_feat) + \
          ' for %d nodes' % (sim_graph.number_of_nodes())
    )
    logging.debug('  Minimum similarities to consider:', str(min_sim_list))
    logging.debug('  Minimum node degree required:    ', min_node_degree)
    logging.debug('  Features generated:', feat_name_list)

    node_feat_dict = {}  # One list of features per node

    num_nodes_too_low_degree = 0

    # Process each node and get is neighbours, then calculate the node's
    # features for different similarity thresholds
    #
    for node_key_val in sim_graph.nodes():

      # Get the node's neighbours
      #
      node_neigh_val_list = sim_graph.neighbors(node_key_val)

      if (len(node_neigh_val_list) < min_node_degree):
        num_nodes_too_low_degree += 1
        continue

      node_feat_array = numpy.zeros(num_feat)
      next_feat =       0  # The index into the feature array for this node

      if ('node_freq' in calc_feature_list):
        node_freq = len(sim_graph.node[node_key_val]['ent_id_set'])
        assert node_freq >= 1, (node_key_val, node_freq)
        node_feat_array[next_feat] = node_freq
        next_feat += 1

      # Get the node values and similarities of all edges of this node
      #
      all_node_neigh_sim_list = []

      max_edge_sim = 0.0
      for neigh_key_val in node_neigh_val_list:
        edge_sim = sim_graph.edges[node_key_val,neigh_key_val]['sim']
        max_edge_sim = max(max_edge_sim, edge_sim)
        all_node_neigh_sim_list.append((neigh_key_val, edge_sim))

      if ('max_sim' in calc_feature_list):
        node_feat_array[next_feat] = max_edge_sim
        next_feat += 1

      for min_sim in min_sim_list:

        # First get the edges and their similarities that are at least the
        # current minimum similarity
        #
        node_neigh_val_set = set()  # Nodes connected with similar enough edges
        node_neigh_sim_list = []    # Similarities of these edges

        for (neigh_key_val, edge_sim) in all_node_neigh_sim_list:
          if (edge_sim >= min_sim):
            node_neigh_val_set.add(neigh_key_val)
            node_neigh_sim_list.append(edge_sim)

        num_sim_neigh = len(node_neigh_val_set)

        # Calculate selected features for the extracted edge similarity list
        #
        if ('min_sim' in calc_feature_list):
          if (num_sim_neigh > 0):  # Nothing to do if 0
            node_feat_array[next_feat] = min(node_neigh_sim_list)
          next_feat += 1

        if ('sim_avr' in calc_feature_list):
          if (num_sim_neigh > 0):  # Nothing to do if 0
            node_feat_array[next_feat] = numpy.mean(node_neigh_sim_list)
          next_feat += 1

        if ('sim_std' in calc_feature_list):
          if (num_sim_neigh > 0):  # Nothing to do if 0
            node_feat_array[next_feat] = numpy.std(node_neigh_sim_list)
          next_feat += 1

        if ('degree' in calc_feature_list):
          node_feat_array[next_feat] = num_sim_neigh
          next_feat += 1

        if ('degree_histo1' in calc_feature_list):
          if (num_sim_neigh > 0):  # Nothing to do if no neighbours
            for neigh_key_val in node_neigh_val_set:
              node_neigh_degree = sim_graph.degree(neigh_key_val)
              node_neigh_log_degree = int(math.log(node_neigh_degree, 2))
              assert (node_neigh_log_degree >= 0) and \
                      (node_neigh_log_degree < degree_hist_len)
              node_feat_array[next_feat + node_neigh_log_degree] += 1

          next_feat += degree_hist_len  # Advance over the degree histrogram

        if ('degree_histo2' in calc_feature_list):
          if (num_sim_neigh > 0):  # Nothing to do if no neighbours
            hop2_neigh_val_set = set()  # Get all neighbours of neighbours

            for neigh_key_val in node_neigh_val_set:
              for hop2_neigh_key_val in sim_graph.neighbors(neigh_key_val):

                # Only consider all nodes two hops away from current node
                #
                if (hop2_neigh_key_val != node_key_val) and \
                   (hop2_neigh_key_val not in node_neigh_val_set):

                  edge_sim = \
                      sim_graph.edges[neigh_key_val,hop2_neigh_key_val]['sim']

                  if (edge_sim >= min_sim):
                    hop2_neigh_val_set.add(hop2_neigh_key_val)

            for hop2_neigh_key_val in hop2_neigh_val_set:
              node_neigh_degree = sim_graph.degree(hop2_neigh_key_val)
              node_neigh_log_degree = int(math.log(node_neigh_degree, 2))
              assert (node_neigh_log_degree >= 0) and \
                      (node_neigh_log_degree < degree_hist_len)
              node_feat_array[next_feat + node_neigh_log_degree] += 1

          next_feat += degree_hist_len  # Advance over the degree histrogram

        if ('egonet_degree' in calc_feature_list) or \
           ('egonet_density' in calc_feature_list):
          if (num_sim_neigh == 0):
            if ('egonet_degree' in calc_feature_list):
              next_feat += 1
            if ('egonet_density' in calc_feature_list):
              next_feat += 1

          else:  # There are neighbours
            egonet_degree = 0.0  # Number of edges with nodes outside egonet
            egonet_edge_set = set()  # Set of edges within egonet

            for neigh_key_val in node_neigh_val_set: # Loop over all neighbours

              # Loop over all neighbours of the node's neighbours
              #
              for other_key_val in sim_graph.neighbors(neigh_key_val):

                # Check if this other node is in the egonet or not
                #
                if (other_key_val != node_key_val) and \
                   (other_key_val not in node_neigh_val_set):
                  egonet_degree += 1
                else:  # Edge is within egonet, so add to egonet edge set
                  egonet_edge = tuple(sorted([neigh_key_val,other_key_val]))
                  egonet_edge_set.add(egonet_edge)

            if ('egonet_degree' in calc_feature_list):
              node_feat_array[next_feat] = egonet_degree
              next_feat += 1
            if ('egonet_density' in calc_feature_list):
              all_egonet_num_edges = num_sim_neigh*(num_sim_neigh+1)/2.0
              egonet_density_val = len(egonet_edge_set) / all_egonet_num_edges
              assert egonet_density_val <= 1.0, egonet_density_val
              node_feat_array[next_feat] = egonet_density_val
              next_feat += 1

      node_feat_dict[node_key_val] = node_feat_array

    # Calculate pagerank and/or betweenness centrality on the similarity graph
    # according to the current minimum similarity
    #
    if ('pagerank' in calc_feature_list) or \
       ('between_central' in calc_feature_list) or \
       ('closeness_central' in calc_feature_list) or \
       ('degree_central' in calc_feature_list) or \
       ('eigenvec_central' in calc_feature_list):

      for min_sim in min_sim_list:

        # Generate the sub-graph with the current minimum similarity (pagerank
        # requires a directed graph)
        #
        di_graph = networkx.DiGraph()
        for (node_key_val1, node_key_val2) in sim_graph.edges():
          if (sim_graph.edges[node_key_val1,node_key_val2]['sim'] >= min_sim):
            di_graph.add_edge(node_key_val1,node_key_val2)
            di_graph.add_edge(node_key_val2,node_key_val1)

        if ('pagerank' in calc_feature_list):
          page_rank_dict = networkx.pagerank_numpy(di_graph)

          for (node_key_val, node_feat_array) in node_feat_dict.items():
            node_feat_array[next_feat] = page_rank_dict.get(node_key_val, 0.0)
            node_feat_dict[node_key_val] = node_feat_array
          next_feat += 1

        for (central_funct_name_str, central_funct_name) in \
            [('between_central',   networkx.betweenness_centrality),
             ('closeness_central', networkx.closeness_centrality),
             ('degree_central',    networkx.degree_centrality),
             ('eigenvec_central',  networkx.eigenvector_centrality)]:
          if (central_funct_name_str in calc_feature_list):
            central_dict = central_funct_name(di_graph)
            for (node_key_val, node_feat_array) in node_feat_dict.items():
              node_feat_array[next_feat] = \
                                  central_dict.get(node_key_val, 0.0)
              node_feat_dict[node_key_val] = node_feat_array
            next_feat += 1
        
    assert next_feat == num_feat, (next_feat, num_feat)

    # Keep track of the minimum and maximum values per feature
    #
    min_feat_array = numpy.ones(num_feat)
    max_feat_array = numpy.zeros(num_feat)

    for node_feat_array in node_feat_dict.values():
      min_feat_array = numpy.minimum(node_feat_array, min_feat_array)
      max_feat_array = numpy.maximum(node_feat_array, max_feat_array)

    logging.debug('  Feature generation for %d nodes took %.2f sec' % \
          (len(node_feat_dict), time.time() - t1)
    )
    logging.debug('    Number of nodes with a degree below %d: %d' % \
          (min_node_degree, num_nodes_too_low_degree)
    )
    logging.debug('    Minimum feature values:', min_feat_array)
    logging.debug('    Maximum feature values:', max_feat_array)
    logging.debug('')

    assert len(node_feat_dict) + num_nodes_too_low_degree == \
           len(sim_graph.nodes())

    return node_feat_dict, min_feat_array, max_feat_array, feat_name_list

  # ---------------------------------------------------------------------------

  def remove_freq_feat_vectors(self, node_feat_dict, max_count):
    """Remove from the given dictionary of feature vectors all those vectors
       that occur more than 'max_count' times, and return a new feature vector
       dictionary.

       Input arguments:
         - node_feat_dict  A dictionary with one entry per node of the graph,
                           with its key being the node value and its value
                           being an array of the calculated numerical feature
                           values.
         - max_count       An integer, where all unique feature vectors that
                           occur more than this number of times in the
                           dictionary will be removed.

      Output:
         - filter_node_feat_dict  A dictionary containing only those node
                                  identifiers as keys where their feature
                                  vectors does not occur frequently according
                                  'max_count'.
    """

    # Keys will be feature vectors and values the sets of node identifiers that
    # have this feature vector
    #
    feat_vec_count_dict = {}

    for (node_key_val, node_feat_array) in node_feat_dict.items():
      node_feat_tuple = tuple(node_feat_array)  # So it can be a dictionary key

      feat_vec_node_set = feat_vec_count_dict.get(node_feat_tuple, set())
      feat_vec_node_set.add(node_key_val)
      feat_vec_count_dict[node_feat_tuple] = feat_vec_node_set

    filter_node_feat_dict = {}

    # Now copy those feature vectors that occur for maximum 'max_count' nodes
    #
    for feat_vec_node_set in feat_vec_count_dict.values():

      if (len(feat_vec_node_set) <= max_count):
        for node_key_val in feat_vec_node_set:
          assert node_key_val not in filter_node_feat_dict

          filter_node_feat_dict[node_key_val] = node_feat_dict[node_key_val]

    logging.debug('Reduced number of nodes in feature vector dictionary from %d ' % \
          (len(node_feat_dict)) + 'to %d by removing all feature vectors ' % \
          (len(filter_node_feat_dict)) + 'that occur more than %d times' % \
          (max_count)
    )
    logging.debug('')

    return filter_node_feat_dict

  # ---------------------------------------------------------------------------

  def norm_features_01(self, node_feat_dict, min_feat_array, max_feat_array):
    """Normalise the feature vectors in the given dictionary with regard to
       the given minimum and maximum values provided such that all feature
       values will be in the 0..1 range.

       Input arguments:
         - node_feat_dict  A dictionary with one entry per node of the graph,
                           with its key being the node value and its value
                           being an array of the calculated numerical feature
                           values.
         - min_feat_array  An array which for each feature contains its
                           minimum calculated value.
         - max_feat_array  An array which for each feature contains its
                           maximum calculated value.

       Output:
         - norm_node_feat_dict  A dictionary with one entry per node of the
                                graph, with its key being the node value and
                                its value being an array of the 0 to 1
                                normalised numerical feature values.
    """

    norm_node_feat_dict = {}

    num_feat = len(min_feat_array)
    assert num_feat == len(max_feat_array)

    # Get the difference between the maximum and minimum values for each
    # feature
    #
    feat_range_array = max_feat_array - min_feat_array

    # Loop over all feature vectors and normalise them if their difference
    # between maximum and minimum values is larger than 0
    #
    for (node_key_val, node_feat_array) in node_feat_dict.items():
      norm_feat_array = numpy.zeros(num_feat)

      for i in range(num_feat):
         if (feat_range_array[i] > 0):
           norm_feat_array[i] = (node_feat_array[i] - min_feat_array[i]) / \
                                feat_range_array[i]
         else:  # No value range, but check the feature is not ouside 0..1
           assert (node_feat_array[i] >= 0.0) and (node_feat_array[i] <= 1.0),\
                  (i, node_feat_array[i], node_feat_array)

      norm_node_feat_dict[node_key_val] = norm_feat_array

    # Check minimum and maximum values per feature are now normalised
    #
    min_feat_array = numpy.ones(num_feat)
    max_feat_array = numpy.zeros(num_feat)

    for node_feat_array in norm_node_feat_dict.values():
      min_feat_array = numpy.minimum(node_feat_array, min_feat_array)
      max_feat_array = numpy.maximum(node_feat_array, max_feat_array)

    logging.debug('  Normalised minimum feature values:', min_feat_array)
    logging.debug('  Normalised maximum feature values:', max_feat_array)
    logging.debug('')

    assert min(min_feat_array) >= 0.0, min_feat_array
    assert max(max_feat_array) <= 1.0, max_feat_array

    return norm_node_feat_dict

  # ---------------------------------------------------------------------------

  def get_std_features(self, node_feat_dict, num_feat):
    """For each feature calculate its standard deviation, and return an array
       with these standard deviation values.

       Input argument:
         - node_feat_dict  A dictionary with one entry per node of the graph,
                           with its key being the node identifier and its
                           value being an array of the calculated numerical
                           feature values.
         - num_feat        The number of features for each node.

       Output:
         - feat_std_list  A list with pairs of (feature number, feature
                          standard deviation) for each feature.
    """

    feat_val_list = []  # One list of all values for each feature

    for i in range(num_feat):
      feat_val_list.append([])  # One list per feature

    for node_feat_array in node_feat_dict.values():
      for i in range(num_feat):  # All feature values to their lists
        feat_val_list[i].append(node_feat_array[i])

    feat_std_list = []

    for i in range(num_feat):
      feat_std_list.append((i, numpy.std(feat_val_list[i])))

    del feat_val_list

    return feat_std_list

  # ---------------------------------------------------------------------------

  def select_features(self, node_feat_dict, use_feat_set, org_num_feat):
    """Select certain features as provided in the given set of features to
       use.

       Input argument:
         - node_feat_dict  A dictionary with one entry per node of the graph,
                           with its key being the node identifier and its
                           value being an array of the calculated numerical
                           feature values.
         - use_feat_set    A set with the numbers of features to use (while
                           all other features  are removed from the feature
                           array of each node).
         - org_num_feat    The original number of features for each node.

       Output:
         - sel_node_feat_dict  A new dictionary with one entry per node of
                               the graph, with its key being the node
                               identifier and its value being an array of the
                               selected numerical feature values.
    """

    num_use_feat = len(use_feat_set)

    sel_node_feat_dict = {}

    for (id, node_feat_array) in node_feat_dict.items():
       use_feat_array = numpy.ones(num_use_feat)
       use_i = 0
       for i in range(org_num_feat):
         if i in use_feat_set:
           use_feat_array[use_i] = node_feat_array[i]
           use_i += 1

       sel_node_feat_dict[id] = use_feat_array

    return sel_node_feat_dict