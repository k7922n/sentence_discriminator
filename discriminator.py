from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

import numpy as np
from six.moves import xrange
import tensorflow as tf

import data_utils
import model

MAX_SENTENCE_LENGTH = 50
BATCH_SIZE = 32
UNIT_SIZE = 128
NUM_LAYER = 2
train_dir = 'save/'

def create_model(session):
  
  model = model.discriminator(UNIT_SIZE, BATCH_SIZE, NUM_LAYER)
  ckpt = tf.train.get_checkpoint_state(train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Creating model with fresh parameters.")

  return model

def train():
  # FIXME
  data_utils.prepare_whole_data('corpus/movie_lines_cleaned.txt', '60000')
  
  with tf.Session() as sess:
    print("Creating model containing %s layers and %s units" % (NUM_LAYER, UNIT_SIZE)) 
    model = create_model(session) 
    # data_set contains the whole training data stored in a list
    data_set = data_utils.read_token_data('corpus/movie_lines_cleaned.txt')
  
    # TODO prepare batch 
