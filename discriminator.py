from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

import sys
import numpy as np
from six.moves import xrange
import tensorflow as tf

import data_utils
import model

MAX_SENTENCE_LENGTH = 40
BATCH_SIZE = 32
UNIT_SIZE = 256
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
  name_1 = sys.argv[1]
  name_2 = sys.argv[2]

  # you can choose which tokenizer to use, NLTK will take much more time to process...
  data_utils.prepare_whole_data(name_1, name_2, 60000, nltk_tokenizer = True)

  with tf.Session() as sess:
    print("Creating model containing %s layers and %s units" % (NUM_LAYER, UNIT_SIZE)) 
    model = create_model(sess) 
    # data_set contains the whole training data stored in a list
    data_set_1 = data_utils.read_token_data(name_1)
    data_set_2 = data_utils.read_token_data(name_2)

    # divide it into training set and validation set
    set_1_train, set_1_valid = data_utils.split(data_set_1, 0.9)
    set_2_train, set_2_valid = data_utils.split(data_set_2, 0.9)
 
    current_step = 0
    loss = 0

    # training part
    while True: 
      # TODO prepare batch 
      batch_input, batch_length, batch_target = model.get_batch(set_1_train, set_2_train)
      step_loss, step_output, en_input, en_seq, target, l = model.step(sess,\
                                                         batch_input, batch_length, batch_target, False)
      current_step += 1
      loss += step_loss / STEP_PER_CHECKPOINT
  
      if current_step % STEP_PER_CHECKPOINT == 0:
        # save check point
        checkpoint_path = train_dir + "discriminator.ckpt"
        model.saver.save(sess, checkpoint_path, global_step = current_step)
        print("Loss: %s" % loss)
        #print(step_output)
        #print(target)
        #print("Loss: %s, Output: %s" % (step_loss, step_output))
        #print("Length: %s, Target: %s" % (en_seq, target))
        #print("Pre_loss: %s" % l)
        loss = 0

        # testing part, default 10 times...
        valid_count = 0
        train_count = 0
        for _ in xrange(10):  
          valid_input, valid_length, valid_target = model.get_batch(set_1_valid, set_2_valid)
          train_input, train_length, train_target = model.get_batch(set_1_train, set_2_train)
          _, valid_output, _, _, valid_answer, _ = model.step(sess, valid_input, valid_length, valid_target, True)
          _, train_output, _, _, train_answer, _ = model.step(sess, train_input, train_length, train_target, True)
          valid_predict = np.argmax(valid_output, axis = 1)
          train_predict = np.argmax(train_output, axis = 1)
          valid_truth   = np.argmax(valid_answer, axis = 1)
          train_truth   = np.argmax(train_answer, axis = 1)
          valid_number = np.sum(valid_predict == valid_truth)
          train_number = np.sum(train_predict == train_truth)
          valid_count += valid_number
          train_count += train_number
        
        print("Valid set accuracy: %s" % str(float(valid_count) / (10 * BATCH_SIZE)))
        print("Train set accuracy: %s" % str(float(train_count) / (10 * BATCH_SIZE)))
   

  
if __name__ == '__main__':
  train()        
