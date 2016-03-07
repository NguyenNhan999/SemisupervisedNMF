# Author : Hoang NT
# Date : 2016-03-07
#
# Update module to handle fractorized matrix elements update.

import tensorflow as tf
import numpy as np

# A warper of a collection of update functions 
# These functions will add the element update 
# computation node to a graph passed to it.
class UpdateElem(object) :

  # Create object with a tensor flow graph
  def __init__(self, tf_graph) :
    assert type(tf_graph) is tf.python.framework.ops.Graph
    self._graph = tf_graph 
    self._A = None
    self._W = None
    self._H = None
    self._O = None
    self._D = None
    self._semi_supervised = None
    self._lambda = None

  # Add semi supervised updating rules to the graph
  # Input: 3 numpy.ndarray(s), 1 integer, 1 float
  ## A : Adjacency matrix - np.ndarray 
  ## O, D : Matrix encoding the prior info and its diagonal matrix D - np.ndarray
  ## k : Number of latent communities, shape of W is N x k, H is N x k
  # Output: Tuple containing W and H (not transposed)
  def add_semi_supervised_rule(A, O, D, k, l) :
    # Check for data type 
    assert type(A) is np.ndarray
    assert type(O) is np.ndarray
    assert type(D) is np.ndarray
    assert type(k) is int
    assert type(l) is float
    # ... and shape
    assert O.shape == D.shape == A.shape
    # Add computational node to the graph 
    with self._graph.as_default():
      self._A = tf.constant(A, name="A")
      self._O = tf.constant(O, name="0")
      self._D = tf.constant(D, name="D")
      self._lambda = tf.constant(l, name="lambda")
      # Intialize W, H with truncated normal distribution values
      self._W = tf.Variable(tf.truncated_normal([A.shape[0],k]), name="W")
      self._H = tf.Variable(tf.truncated_normal([A.shape[0],k]), name="H")
      # Add update nodes compute denominator and numerator separately
      ## Update W
      w_nume = tf.matmul(self._A, self._H)
      w_deno_temp = tf.matmul(self._H, self._H, transpose_b = True)
      w_deno = tf.matmul(self._W, w_deno_temp)
      w_update_factor = tf.div(w_nume, w_deno)
      w_update = tf.mul(w_update_factor, self._W)
      self._W.assign(w_update) 
      ## Update H
      h_nume_a = tf.matmul(self._A, self._H, transpose_a = True)
      h_nume_b_temp = tf.matmul(self._O, self._H)
      h_nume_b = tf.mul(self._lambda, h_nume_b_temp)
      h_nume = tf.add(h_nume_a, h_nume_b)
      h_deno_a_temp = tf.matmul(self._W, self._W, transpose_b = True)
      h_deno_a = tf.matmul(self._H, h_deno_a_temp)
      h_deno_b_temp = tf.matmul(self._D, self._H)
      h_deno_b = tf.mul(self._lambda, h_deno_b_temp)
      h_deno = tf.add(h_deno_a, h_deno_b)
      h_update_factor = tf.div(h_nume, h_deno)
      h_update = tf.mul(h_update_factor, self._H)
      self._h.assign(h_update)
