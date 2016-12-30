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
STEP_PER_CHECKPOINT = 100
train_dir = 'save/'

def create_model(session):
  
  dis = model.discriminator(UNIT_SIZE, BATCH_SIZE, NUM_LAYER, MAX_SENTENCE_LENGTH)
  ckpt = tf.train.get_checkpoint_state(train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    dis.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Creating model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return dis

def train():
  # FIXME  need to read and process two files at the same time
  # TODO   path_2
  data_utils.prepare_whole_data('corpus/movie_lines_cleaned.txt', 'corpus/movie_lines_cleaned.txt', '60000')

  with tf.Session() as sess:
    print("Creating model containing %s layers and %s units" % (NUM_LAYER, UNIT_SIZE)) 
    model = create_model(sess) 
    # data_set contains the whole training data stored in a list
    data_set_1 = data_utils.read_token_data('corpus/movie_lines_cleaned.txt')
    data_set_2 = data_utils.read_token_data('corpus/movie_lines_cleaned.txt')
 
    current_step = 0
    loss = 0

    while True: 
      # TODO prepare batch 
      batch_input, batch_length, batch_target = model.get_batch(data_set_1, data_set_2)
      step_loss, _ = model.step(sess, batch_input, batch_length, batch_target)
      current_step += 1
      loss += step_loss / STEP_PER_CHECKPOINT
  
      if current_step % STEP_PER_CHECKPOINT == 0:
        # save check point
        checkpoint_path = train_dir + "discriminator.ckpt"
        model.saver.save(sess, checkpoint_path, global_step = current_step)
        print("Loss: %s" % loss)
        loss = 0
  
if __name__ == '__main__':
  train()        
