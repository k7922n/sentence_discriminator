from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.ops import rnn
import data_utils

VOCAB_SIZE = 60000

class discriminator(object):

  def __init__(self, unit_size, batch_size, num_layers, max_length):
    self.unit_size = unit_size
    self.batch_size = batch_size
    self.num_layers = num_layers
    self.max_length = max_length
    self.build_model()
  
    self.saver = tf.train.Saver(tf.all_variables())
  
  def build_model(self):
    # Default LSTM cell
    single_cell = tf.nn.rnn_cell.LSTMCell(self.unit_size, state_is_tuple = False)
    if self.num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
    else:
      cell = single_cell
    # Specify inputs
    #self.encoder_inputs = tf.placeholder(tf.float32, shape=[None, self.max_length, self.unit_size])
    self.encoder_inputs = tf.placeholder(tf.int32, shape=[None, self.max_length])
    # Form embedding look up
    embeddings = tf.Variable(tf.random_uniform([VOCAB_SIZE, self.unit_size]))
    embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
    #embedded = tf.one_hot(self.encoder_inputs, self.unit_size, axis = -1)

    # Specify sequence length
    self.seq_length = tf.placeholder(tf.int32, shape = [None])
    # Specify target, in this case, binary classification
    self.target = tf.placeholder(tf.float32, shape = [None, 2])
    # Use Dynamic rnn module
    self.outputs, self.final_state = rnn.dynamic_rnn(cell, embedded, self.seq_length, dtype = tf.float32)
    # for debug
    '''
    print(self.encoder_inputs.get_shape())
    print(self.unit_size) 
    print(self.outputs.get_shape())
    print(self.final_state.get_shape())
    exit()
    '''   
    # It seems like dynamic_rnn final_state has 2 * num_layers * num_unit dimension
    hidden_weight_1 = tf.Variable(tf.random_normal([2 * self.num_layers * self.unit_size, 512], dtype = tf.float32))
    hidden_bias_1 = tf.Variable(tf.random_normal([512], dtype = tf.float32))

    hidden_weight_2 = tf.Variable(tf.random_normal([512, 256], dtype = tf.float32))
    hidden_bias_2= tf.Variable(tf.random_normal([256], dtype = tf.float32))

    hidden_weight_3 = tf.Variable(tf.random_normal([256, 2], dtype = tf.float32))
    hidden_bias_3= tf.Variable(tf.random_normal([2], dtype = tf.float32))

    out_layer_1 = tf.nn.relu(tf.matmul(self.final_state, hidden_weight_1) + hidden_bias_1)
    out_layer_2 = tf.nn.relu(tf.matmul(out_layer_1, hidden_weight_2) + hidden_bias_2)
    # Don't have to pass through softmax function
    out_layer_3 = tf.matmul(out_layer_2, hidden_weight_3) + hidden_bias_3
    self.pre_loss = out_layer_3
    self.output = tf.nn.softmax(out_layer_3)
    # Define loss function
    loss = tf.nn.softmax_cross_entropy_with_logits(out_layer_3, self.target)
    
    self.loss = tf.reduce_mean(loss)
    #optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)
    self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.loss)

  def step(self, session, encoder_inputs, seq_length, target, predict):
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

    # For test 
    output_feed.append(self.encoder_inputs)
    output_feed.append(self.seq_length)
    output_feed.append(self.target)
    output_feed.append(self.pre_loss)

    # Start running
    if predict:  # predicting process
      outputs = session.run(output_feed, input_feed)
    else:        # training process
      output_feed.append(self.optimizer)
      outputs = session.run(output_feed, input_feed)
    # Return loss and output probabilities
    return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]

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
        target.append([1., 0.])
      else: 
        encoder_input = random.choice(data_2)
        target.append([0., 1.])
      
      length = len(encoder_input)
   
      if length > self.max_length:
        encoder_inputs.append(encoder_input[:self.max_length])
        encoder_length.append(self.max_length)
      else:
        encoder_pad = [data_utils.PAD_ID] * (self.max_length - length)
        encoder_inputs.append(list(encoder_input + encoder_pad)) 
        encoder_length.append(length)

    # For test

    batch_length  = np.array(encoder_length, dtype = np.int32)
    batch_input   = np.array(encoder_inputs, dtype = np.int32)
    batch_targets = np.array(target, dtype = np.float32)

    return batch_input, batch_length, batch_targets
