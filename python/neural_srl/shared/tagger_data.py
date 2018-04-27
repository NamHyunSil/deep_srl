from constants import UNKNOWN_TOKEN

import numpy as np
import random

def tensorize(sentence, max_length):
  """ Input:
      - sentence: The sentence is a tuple of lists (s1, s2, ..., sk)
            s1 is always a sequence of word ids.
            sk is always a sequence of label ids.
            s2 ... sk-1 are sequences of feature ids,
              such as predicate or supertag features.
      - max_length: The maximum length of sequences, used for padding.
  """
  #sentence = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]] 전체 n개  [1,2,3] = s1   [10,11,12] = sk
  #max_length = 10
  #x = [[1,4,7],[2,5,8],[3,6,9]]  n-1*n-1   3*3
  x = np.array([t for t in zip(*sentence[:-1])])
  #y = [[10,11,12]]
  y = np.array(sentence[-1])
  weights = (y >= 0).astype(float)
  #x = [[1,4,7],[2,5,8], [3,6,9],['','',''],['','',''],['','',''],['','',''],['','',''],['','',''],['','','']]        10*3
  x.resize([max_length, x.shape[1]])
  #y = [[10,11,12],['','',''],['','',''],['','',''],['','',''],['','',''],['','',''],['','',''],['','',''],['','','']] 10
  y.resize([max_length])
  weights.resize([max_length])
  return x, np.absolute(y), len(sentence[0]), weights
  #TODO: y(라벨)값에 왜 절대값을 시키는가
  
class TaggerData(object):
  def __init__(self, config, train_sents, dev_sents, word_dict, label_dict, embeddings, embedding_shapes,
         feature_dicts=None):
    ''' 
    '''
    self.max_train_length = config.max_train_length
    self.max_dev_length = max([len(s[0]) for s in dev_sents]) if len(dev_sents) > 0 else 0
    self.batch_size = config.batch_size
    self.use_se_marker = config.use_se_marker
    self.unk_id = word_dict.str2idx[UNKNOWN_TOKEN]
    
    self.train_sents = [s for s in train_sents if len(s[0]) <= self.max_train_length]
    self.dev_sents = dev_sents
    self.word_dict = word_dict
    self.label_dict = label_dict
    self.embeddings = embeddings
    self.embedding_shapes = embedding_shapes
    self.feature_dicts = feature_dicts
    
    self.train_tensors = [tensorize(s, self.max_train_length) for s in self.train_sents]
    self.dev_tensors =  [tensorize(s, self.max_dev_length) for s in self.dev_sents]
    
  def get_training_data(self, include_last_batch=False):
    """ Get shuffled training samples. Called at the beginning of each epoch.
    """
    # TODO: Speed up: Use variable size batches (different max length).  
    #train_sents 리스트의 인덱스 list를 만듦
    train_ids = range(len(self.train_sents))
    random.shuffle(train_ids)
    #train_ids 배치사이즈 만큼 잘랐을 때 마지막 배치가 남았을 경우 마지막 배치를 제거1
    if not include_last_batch:
      num_batches = len(train_ids) // self.batch_size
      train_ids = train_ids[:num_batches * self.batch_size]
      
    num_samples = len(self.train_sents)
    #train_ids에 maapping된 train_tensors(train_sents를 텐서화) 리스트화
    tensors = [self.train_tensors[t] for t in train_ids]
    #tensors를 배치만큼씩 잘라 리스트화
    batched_tensors = [tensors[i: min(i+self.batch_size, num_samples)]  #마지막 배치가 남았을 경우 마지막 배치를 제거2
               for i in xrange(0, num_samples, self.batch_size)]
    #batched_tensors의 원소를 하나씩 꺼내어 zip시킨 후 정렬
    results = [zip(*t) for t in batched_tensors]
    
    print("Extracted {} samples and {} batches.".format(num_samples, len(batched_tensors)))
    return results
  
  def get_development_data(self, batch_size=None):
    if batch_size is None:
      return [np.array(v) for v in zip(*self.dev_tensors)]
    
    num_samples = len(self.dev_sents)
    batched_tensors = [self.dev_tensors[i: min(i+ batch_size, num_samples)]
               for i in xrange(0, num_samples, batch_size)]
    return [zip(*t) for t in batched_tensors]
  
  def get_test_data(self, test_sentences, batch_size = None):
    max_len = max([len(s[0]) for s in test_sentences])
    num_samples = len(test_sentences)
    #print("Max sentence length: {} among {} samples.".format(max_len, num_samples))
    test_tensors =  [tensorize(s, max_len) for s in test_sentences]
    if batch_size is None:
      return [np.array(v) for v in zip(*test_tensors)]
    batched_tensors = [test_tensors[i: min(i+ batch_size, num_samples)]
               for i in xrange(0, num_samples, batch_size)]
    return [zip(*t) for t in batched_tensors]
  
