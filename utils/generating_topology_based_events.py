'''
  https://proceedings.mlr.press/v202/shou23a.html
  About
  ------------------
  utility functions for topology based event generator

'''

import numpy as np
import math
import random
import os
import ast
import pickle
from itertools import product


### SYNTHETIC DATA GENERATION


# function generates multiple sequences using the function "gen_synth_data_from_topology"
def gen_mult_seq_from_topology_example(num_seqs, num_events_per_seq, mult_factor):
  '''
  :param num_seqs: number of sequences generated
  :param num_events_per_seq: number of events per sequence
  :param mult_factor: multiplicative factor
  :return:
  '''

  L = ['A', 'B', 'C', 'D', 'E']



  # GRAPH 2
  G = {
    'A': [],
    'B': ['A'],
    'C': ['A', 'B'],
    'D': ['A'],
    'E': ['A','B']
  }


  # GRAPH 1
#   G = {
#     'A': [],
#     'B': ['A'],
#     'C': ['A', 'B'],
#     'D': ['E'],
#     'E': ['D']
#   }

  D_dict = {}
  for k in range(1, num_seqs + 1):
    D = gen_synth_data_from_topology(L, G, num_events_per_seq, mult_factor)
    D_dict[k] = D

  return D_dict


# function for topological sequence generator
# here, the unnormalized probability of a label occurrence depends on counts of prior
# occurrences in the history of its parent labels according to some topology
def gen_synth_data_from_topology(L, G, num_events_per_seq, mult_factor):
  '''
  :param L: label set
  :param G: underlying topology/graph that guides sequence generation
  :param num_events_per_seq: number of events per sequence
  :param mult_factor: multiplicative factor
  :return:
  '''

  hist_info_dict = {} # dict that stores counts of all event labels

  D = []
  for t in range(1, num_events_per_seq + 1):
    unnormalized_prob_vec = []
    for lab in sorted(L):
      unnormalized_prob = 1 # this is the default
      # count the number of occurrences of
      counts = 0
      for par in G[lab]:
        if par in hist_info_dict:
          counts += 1
      if counts != 0:
        unnormalized_prob = mult_factor * counts
      unnormalized_prob_vec.append(unnormalized_prob)

    prob_vec = np.array(unnormalized_prob_vec) / sum(np.array(unnormalized_prob_vec))
    # generate a label and append
    this_lab = np.random.choice(L, p=prob_vec)
    D.append(this_lab)
    # update the historical information
    if this_lab in hist_info_dict:
      hist_info_dict[this_lab] += 1
    else:
      hist_info_dict[this_lab] = 1

  return D


''' testing uses 
mult_seq = gen_mult_seq_from_topology_example(3,5, 1)
print(mult_seq)
L = ['A', 'B', 'C', 'D', 'E']



  # GRAPH 2
G = {
    'A': [],
    'B': ['A'],
    'C': ['A', 'B'],
    'D': ['A'],
    'E': ['A','B']
}

synth_data = gen_synth_data_from_topology(L,G,5,1)
print(synth_data)
'''
