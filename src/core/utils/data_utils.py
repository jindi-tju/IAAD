# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""

import json
import re
import random
import io
import torch
import numpy as np
import spacy

from scipy.sparse import *
from collections import Counter, defaultdict

from .text_data import data_utils as text_data_utils
from .timer import Timer
from . import padding_utils
from . import constants
from .generic_utils import  batch_normalize_adj

nlp = spacy.load('en_core_web_sm')

def vectorize_input(batch, config, training=True, device=None):  # 将id 转成张量
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    batch_size = len(batch.sent1_word)

    context = torch.LongTensor(batch.sent1_word)
    context_lens = torch.LongTensor(batch.sent1_length)


    sen_adj = torch.FloatTensor([item.cpu().detach().numpy() for item in batch.s_adj]).cuda()
    dep_adj = torch.FloatTensor([item.cpu().detach().numpy() for item in batch.d_adj]).cuda()
    sen_adj_norm = batch_normalize_adj(sen_adj)
    dep_adj_norm = batch_normalize_adj(dep_adj)

    if config['task_type'] == 'regression':
        targets = torch.Tensor(batch.labels)
    elif config['task_type'] == 'classification':
        targets = torch.LongTensor(batch.labels)
    else:
        raise ValueError('Unknwon task_type: {}'.format(config['task_type']))

    if batch.has_sent2:
        context2 = torch.LongTensor(batch.sent2_word)
        context2_lens = torch.LongTensor(batch.sent2_length)

    with torch.set_grad_enabled(training):
        example = {'batch_size': batch_size,
                   'context': context.to(device) if device else context,
                   'context_lens': context_lens.to(device) if device else context_lens,
                   'targets': targets.to(device) if device else targets,
                   'sen_adj': sen_adj.to(device) if device else sen_adj,# jia
                   'dep_adj': dep_adj.to(device) if device else dep_adj}  # jia 正则化之后的  x先不加正则

        if batch.has_sent2:
            example['context2'] = context2.to(device) if device else context2
            example['context2_lens'] = context2_lens.to(device) if device else context2_lens
        return example

def prepare_datasets(config):  # 用load-data 加载数据
    data = {}
    if config['data_type'] == 'text':
        train_set, dev_set, test_set = text_data_utils.load_data(config)
        print('# of training examples: {}'.format(len(train_set)))
        print('# of dev examples: {}'.format(len(dev_set)))
        print('# of testing examples: {}'.format(len(test_set)))
        data = {'train': train_set, 'dev': dev_set, 'test': test_set}
    else:
        raise ValueError('Unknown data_type: {}'.format(config['data_type']))
    return data


class DataStream(object):
    def __init__(self, all_instances, word_vocab, config=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1):
        self.config = config
        if batch_size == -1: batch_size = config['batch_size']
        # sort instances based on length
        if isSort:  # TRUE
            all_instances = sorted(all_instances, key=lambda instance: [len(x) for x in instance[:-1]])
        else:
            random.shuffle(all_instances)
            random.shuffle(all_instances)
        self.num_instances = len(all_instances)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = all_instances[batch_start: batch_end]
            cur_batch = InstanceBatch(cur_instances, config, word_vocab)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[i]

class InstanceBatch(object):
    def __init__(self, instances, config, word_vocab):
        self.instances = instances
        self.batch_size = len(instances)

        if len(instances[0]) == 2:
            self.has_sent2 = False
        elif len(instances[0]) == 3:
            self.has_sent2 = True
        else:
            raise RuntimeError('{} elements per example, should be 2 or 3'.format(len(instances[0])))

        self.sent1_word = [] # [batch_size, sent1_len]
        self.sent1_length = [] # [batch_size]
        self.labels = [] # [batch_size]

        if self.has_sent2:
            self.sent2_word = [] # [batch_size, sent2_len]
            self.sent2_length = [] # [batch_size]

        length = 0
        senticNet = load_sentic_word()
        self.s_adj = []
        self.d_adj = []

        for instance in self.instances:
            if (length < len(instance[0])):
                length = len(instance[0])


        for instance in self.instances:

            sent1_cut = instance[0][: config.get('max_seq_len', None)]
            self.sent1_word.append([word_vocab.getIndex(word) for word in sent1_cut])
            self.sent1_length.append(len(sent1_cut))
            if self.has_sent2:
                sent2_cut = instance[1][: config.get('max_seq_len', None)]
                self.sent2_word.append([word_vocab.getIndex(word) for word in sent2_cut])
                self.sent2_length.append(len(sent2_cut))
            self.labels.append(instance[-1])

            s = sentic_adj_matrix(instance[0], senticNet, length)
            d = dependency_adj_matrix(instance[0], length)
            self.s_adj.append(s)
            self.d_adj.append(d)

        self.sent1_word = padding_utils.pad_2d_vals_no_size(self.sent1_word)
        self.sent1_length = np.array(self.sent1_length, dtype=np.int32)
        if self.has_sent2:
            self.sent2_word = padding_utils.pad_2d_vals_no_size(self.sent2_word)
            self.sent2_length = np.array(self.sent2_length, dtype=np.int32)



def sentic_adj_matrix(word_list, senticNet, length):

    seq_len = len(word_list)
    matrix = np.zeros((length, length)).astype('float32')

    for i in range(seq_len):
        for j in range(i, seq_len):
            word_i = word_list[i]
            word_j = word_list[j]
            if word_i not in senticNet or word_j not in senticNet or word_i == word_j:
                continue
            sentic = abs(float(senticNet[word_i] - senticNet[word_j]))
            matrix[i][j] = sentic
            matrix[j][i] = sentic
    matrix = torch.FloatTensor(matrix)

    return matrix

def dependency_adj_matrix(word_list, length):

    str = ' '.join(word_list)
    word_list_new = nlp(str)
    matrix = np.zeros((length, length)).astype('float32')
    num = 0

    if (len(word_list_new) == len(word_list)):

        for token in word_list_new:
            matrix[token.i][token.i] = 1
            for child in token.children:
                matrix[token.i][child.i] = 1
                matrix[child.i][token.i] = 1
    else:
        num = num+1
    matrix = torch.FloatTensor(matrix)
    return matrix


def load_sentic_word():
    """
    load senticNet
    """
    path = '../senticNet/sentiwordnet.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = float(sentic)
    fp.close()
    return senticNet