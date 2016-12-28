from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.ops import rnn
import data_utils

MAX_SIZE = 50

class discriminator(object):

  def __init__(self, unit_size, batch_size, num_layers):
    self.unit_size = unit_size
    self.batch_size = batch_size
    self.num_layers = num_layers
    self.build_model()
  
    self.saver = tf.train.Saver(tf.all_variables())
  
  def build_model():
    # Default LSTM cell
    single_cell = tf.nn.rnn_cell.LSTMCell(self.unit_size, state_is_tuple = False)
    if num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
    else:
      cell = single_cell
    # Specify inputs
    self.encoder_inputs = tf.placeholder(tf.float32, shape=[None, MAX_SIZE, self.unit_size])
    # Specify sequence length
    self.seq_length = tf.placeholder(tf.int32, shape = [None])
    # Specify target, in this case, binary classification
    self.target = tf.placeholder(tf.float32, shape = [None, 2])
    # Use Dynamic rnn module
    self.outputs, self.final_state = rnn.dynamic_rnn(cell, self.encoder_inputs, self.seq_length)

    hidden_weight_1 = tf.Variable(tf.random_normal([2 * self.unit_size, 512], dtype = tf.float32))
    hidden_bias_1 = tf.Variable(tf.random_normal([512], dtype = tf.float32))

    hidden_weight_2 = tf.Variable(tf.random_normal([512, 128], dtype = tf.float32))
    hidden_bias_2= tf.Variable(tf.random_normal([128], dtype = tf.float32))

    hidden_weight_3 = tf.Variable(tf.random_normal([128, 2], dtype = tf.float32))
    hidden_bias_3= tf.Variable(tf.random_normal([2], dtype = tf.float32))

    out_layer_1 = tf.nn.relu(tf.matmul(self.final_state, hidden_weight_1) + hidden_bias_1)
    out_layer_2 = tf.nn.relu(tf.matmul(out_layer_1, hidden_weight_2) + hidden_bias_2)
    # Don't have to pass through softmax function
    out_layer_3 = tf.matmul(out_layer_2, hidden_weight_3) + hidden_bias_3
    self.output = tf.nn.softmax(out_layer_3)
    # Define loss function
    self.loss = tf.nn.softmax_cross_entropy_with_logits(out_layer_3, self.target)
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)

  def step(self, session, encoder_inputs, seq_length, target):
    # Create input_feed
    # encoder_inputs' shape should be [batch, max_length, state_num]
    # seq_length's shape should be [batch] with int32
    # target's shape should be [batch, num_class]
    input_feed = {}
    input_feed[self.encoder_inputs] = encoder_inputs
    input_feed[self.seq_length] = seq_length
    input_feed[self.target] = target

    output_feed = [self.loss]
    output_feed.append(self.output)

    # Start running
    outputs = session.run(output_feed, input_feed)
   
    # Return loss and output probabilities
    return outputs[0], outputs[1]

  # data_1 is label 1 data, and data_2 is label 0 data
  # Both are tokenized data which are already been read
  def get_batch(self, data_1, data_2):
    
    # Notice that the data structure is different from ones coming from seq2seq model
    encoder_inputs = []
    encoder_length = []
    target = []

    for _ in xrange(self.batch_size):
      if random.uniform(0, 1) < 0.5:
        encoder_input = random.choice(data_1)
        target.append([1, 0])
      else: 
        encoder_input = random.choice(data_2)
        target.append([0, 1])
      length = len(encoder_input)
      encoder_pad = [data_utils.PAD_ID] * (self.max_length - length)
    
      encoder_inputs.append(list(encoder_input + encoder_pad))
      encoder_length.append(length)

    batch_length  = np.array(encoder_length, dtype = np.int32)
    batch_input   = np.array(encoder_inputs, dtype = np.int32)
    batch_targets = np.array(target, dtype = np.float32)

  return batch_input, batch_length, batch_targets
