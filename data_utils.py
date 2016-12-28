from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys

from tensorflow.python.platform import gfile

WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_UNK = b"UNK"
_PAD = b"PAD"
UNK_ID = 0
PAD_ID = 1

# Tokenize a sentence into a word list
def tokenizer(sentence):
  words = []
  for split_sen in sentence.lower().strip().split():
    words.extend(WORD_SPLIT.split(split_sen))
  return [word for word in words if word]

# Form vocab map (vocab to index) according to maxsize
def form_vocab_mapping(filename_1, filename_2, max_size):
  
  output_path = filenamei_1 + str(max_size) + '.mapping'
  
  if gfile.Exists(output_path):
    print('Mapping of %s has already been formed!' % filename)
  else:
    print('Forming mapping file according to %s and %s' % (filename_1, filename_2))  
    print('Max vocabulary size : %s' % max_size)

    vocab = {}
    with gfile.GFile(filename_1, mode = 'rb') as f_1:
      with gfile.GFile(filename_2, mode = 'rb') as f_2:
        f = [f_1, f_2]
        for fil in f:
          for line in fil:
            counter = 0
            for line in f:
              counter += 1
              if counter % 100000 == 0:
                print("  Processing to line %s" % counter)
              tokens = tokenizer(line)   
       
              for word in tokens:
                if word in vocab:
                  vocab[word] += 1
                else:
                  vocab[word] = 1
      
        vocab_list = [_UNK, _PAD] + sorted(vocab, key = vocab.get, reverse = True)
        if len(vocab_list) > max_size:
          vocab_list = vocab_list[:max_size]

        with gfile.GFile(output_path, 'wb') as vocab_file:
          for w in vocab_list:
            vocab_file.write(w + b'\n')

# Read mapping file from map_path
# Return mapping dictionary
def read_map(map_path):

  if gfile.Exists(map_path):
    vocab_list = []
    with gfile.GFile(map_path, mode = 'rb') as f:
      vocab_list.extend(f.readlines())
    vocab_list = [line.strip() for line in vocab_list]
    vocab_dict = dict([(x, y) for (y, x) in enumerate(vocab_list)])
    
    return vocab_dict

  else:
    raise ValueError("Vocabulary file %s not found!", map_path)

def convert_to_token(sentence, vocab_map):
  
  words = tokenizer(sentence)  
 
  return [vocab_map.get(w, UNK_ID) for w in words]

def file_to_token(file_path, vocab_map):
  output_path = file_path + ".token"
  if gfile.Exists(output_path):
    print("Token file %s has already existed!" % output_path)
  else:
    print("Tokenizing data according to %s" % file_path)

    with gfile.GFile(file_path, 'rb') as input_file:
      with gfile.GFile(output_path, 'w') as output_file:
        counter = 0
        for line in input_file:
          counter += 1
          if counter % 100000 == 0:
            print("  Tokenizing line %s" % counter)
          token_ids = convert_to_token(line, vocab_map)

          output_file.write(" ".join([str(tok) for tok in token_ids]) + '\n')

def prepare_whole_data(input_path, max_size):
  form_vocab_mapping(input_path_1, input_path_2, max_size)
  map_path = input_path_1 + str(max_size) + '.mapping'  
  vocab_map = read_map(map_path)
  file_to_token(input_path_1, vocab_map)
  file_to_token(input_path_2, vocab_map)

# Read token data from tokenized data
def read_token_data(file_path):
  token_path = file_path + '.token'
  if gfile.Exists(token_path):
    data_set = []
    with gfile.GFile(token_path, mode = 'r') as t_file:
      counter = 0
      token_file = t_file.readline()
      while token_file:
        counter += 1
        if counter % 100000 == 0:
          print("  Reading data line %s" % counter)
          sys.stdout.flush()
        token_ids = [int(x) for x in token_file.split()]
        data_set.append(token_ids)
        token_file = t_file.readline()

    return data_set

  else:
    raise ValueError("Can not find token file %s" % token_path)

if __name__ == "__main__":
  prepare_whole_data('corpus/movie_lines_cleaned.txt', 60000)
  data_set = read_token_data('corpus/movie_lines_cleaned.txt')
  print(len(data_set))
  print(data_set[0])

