import os
import re
import codecs
import string
import spacy
from collections import defaultdict
import numpy as np
import nltk
import string
from nltk.tokenize import wordpunct_tokenize



nlp = spacy.load('en_core_web_sm')


tokenize = lambda s: wordpunct_tokenize(re.sub('[%s]' % re.escape(string.punctuation), ' ', s))  # 又去了标点

def load_data(config):
    data_split = [float(x) for x in config['data_split_ratio'].replace(' ', '').split(',')]
    if config['dataset_name'] == 'riloff':
        file_path = os.path.join(config['data_dir'], 'data.txt')
        train_set, dev_set, test_set = load_ril_data(file_path, data_split, config.get('data_seed', 1234))
    elif config['dataset_name'] == 'ghosh':
        file_path = os.path.join(config['data_dir'], 'processed_data.txt')
        train_set, dev_set, test_set = load_pta_data(file_path, data_split, config.get('data_seed', 1234))
    elif config['dataset_name'] == 'reddic':
        file_path = os.path.join(config['data_dir'], 'processed_data.txt')
        train_set, dev_set, test_set = load_pta_data(file_path, data_split, config.get('data_seed', 1234))
    elif config['dataset_name'] == 'pta':
        file_path = os.path.join(config['data_dir'], 'processed_data.txt')
        train_set, dev_set, test_set = load_pta_data(file_path, data_split, config.get('data_seed', 1234))
    else:
        raise ValueError('Unknown dataset_name: {}'.format(config['dataset_name']))

    return train_set, dev_set, test_set

def load_ril_data(file_path, data_split, seed):
    '''Loads the Movie Review Data (https://www.cs.cornell.edu/people/pabo/movie-review-data/).'''

    all_instances = []
    all_seq_len = []
    with open(file_path, 'r') as fp:
        for line in fp:
            subj, lab = line.split(' ==sep== ')
            word_list = tokenize(subj.lower())

            lab = int(lab)
            all_instances.append([word_list, lab])
            all_seq_len.append(len(word_list))
    print('[ Max seq length: {} ]'.format(np.max(all_seq_len)))
    print('[ Min seq length: {} ]'.format(np.min(all_seq_len)))
    print('[ Mean seq length: {} ]'.format(int(np.mean(all_seq_len))))

    # Random data split
    train_ratio, dev_ratio, test_ratio = data_split
    assert train_ratio + dev_ratio + test_ratio == 1
    n_train = int(len(all_instances) * train_ratio)
    n_dev = int(len(all_instances) * dev_ratio)
    n_test = len(all_instances) - n_train - n_dev

    random = np.random.RandomState(seed)
    random.shuffle(all_instances)

    train_instances = all_instances[:n_train]
    dev_instances = all_instances[n_train: n_train + n_dev]
    test_instances = all_instances[-n_test:]

    return train_instances, dev_instances, test_instances




def load_pta_data(file_path, data_split, seed):

    all_instances = []
    all_seq_len = []
    with open(file_path, 'r') as fp:
        for line in fp:
            idx, subj, lab = line.split(' ==sep== ')
            word_list = tokenize(subj.lower())

            lab = int(lab)
            if len(word_list) != 0:
                all_instances.append([word_list, lab])
                all_seq_len.append(len(word_list))


    print('[ Max seq length: {} ]'.format(np.max(all_seq_len)))
    print('[ Min seq length: {} ]'.format(np.min(all_seq_len)))
    print('[ Mean seq length: {} ]'.format(int(np.mean(all_seq_len))))

    # Random data split
    train_ratio, dev_ratio, test_ratio = data_split
    assert train_ratio + dev_ratio + test_ratio == 1
    n_train = int(len(all_instances) * train_ratio)
    n_dev = int(len(all_instances) * dev_ratio)
    n_test = len(all_instances) - n_train - n_dev

    random = np.random.RandomState(seed)
    random.shuffle(all_instances)

    train_instances = all_instances[:n_train]
    dev_instances = all_instances[n_train: n_train + n_dev]
    test_instances = all_instances[-n_test:]

    return train_instances, dev_instances, test_instances



