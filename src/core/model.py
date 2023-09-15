import os
import random
import numpy as np
from sklearn.metrics import f1_score


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from .models.text_graph import TextGraphClf
from .utils.text_data.vocab_utils import VocabModel
from .utils import constants as Constants
from .utils.radam import RAdam


class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """
    def __init__(self, config, train_set=None):
        self.config = config
        self.net_module = TextGraphClf
        print('[ Running {} model ]'.format(self.config['model_name']))

        if config['data_type'] == 'text':
            saved_vocab_file = os.path.join(config['data_dir'], '{}_seed{}.vocab'.format(config['dataset_name'], config.get('data_seed', 1234)))
            self.vocab_model = VocabModel.build(saved_vocab_file, train_set, self.config)  # 变成embedii


        if config['task_type'] == 'classification':
            # self.criterion = F.nll_loss #gai
            self.criterion = nn.CrossEntropyLoss()
            self.score_func = accuracy
            self.metric_name = 'acc'
            self.score_func1 = f1
            self.metric_name1 = 'f1'
        else:
            self.criterion = F.nll_loss
            self.score_func = None
            self.metric_name = None



        if self.config['pretrained']:
            self.init_saved_network(self.config['pretrained'])
        else:
            # Building network.
            self._init_new_network()

        num_params = 0
        for name, p in self.network.named_parameters():
            print('{}: {}'.format(name, str(p.size())))
            num_params += p.numel()

        print('#Parameters = {}\n'.format(num_params))
        self._init_optimizer()


    def init_saved_network(self, saved_dir):
        _ARGUMENTS = ['word_embed_dim', 'hidden_size', 'f_qem', 'f_pos', 'f_ner',
                      'word_dropout', 'rnn_dropout',
                      'ctx_graph_hops', 'ctx_graph_topk',
                      'score_unk_threshold', 'score_yes_threshold',
                      'score_no_threshold']

        # Load all saved fields.
        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        self.state_dict = saved_params['state_dict']
        # for k in _ARGUMENTS:
        #     if saved_params['config'][k] != self.config[k]:
        #         print('Overwrite {}: {} -> {}'.format(k, self.config[k], saved_params['config'][k]))
        #         self.config[k] = saved_params['config'][k]

        if self.config['data_type'] == 'text':
            w_embedding = self._init_embedding(len(self.vocab_model.word_vocab), self.config['word_embed_dim'])
            self.network = self.net_module(self.config, w_embedding, self.vocab_model.word_vocab)
        else:
            self.network = self.net_module(self.config)

        # Merge the arguments
        if self.state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in self.state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

    def _init_new_network(self):
        if self.config['data_type'] == 'text':
            w_embedding = self._init_embedding(len(self.vocab_model.word_vocab), self.config['word_embed_dim'],
                                               pretrained_vecs=self.vocab_model.word_vocab.embeddings)
            self.network = self.net_module(self.config, w_embedding, self.vocab_model.word_vocab)
        else:
            self.network = self.net_module(self.config)

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters, lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'radam':
            self.optimizer = RAdam(parameters, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config['lr_reduce_factor'], \
                    patience=self.config['lr_patience'], verbose=True)

    def _init_embedding(self, vocab_size, embed_size, pretrained_vecs=None):
        """Initializes the embeddings
        """
        return nn.Embedding(vocab_size, embed_size, padding_idx=0,
                            _weight=torch.from_numpy(pretrained_vecs).float()
                            if pretrained_vecs is not None else None)

    def save(self, dirname):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'config': self.config,
            'dir': dirname,
        }
        try:
            torch.save(params, os.path.join(dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')


    def clip_grad(self):
        # Clip gradients
        if self.config['grad_clipping']:
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])

def f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    f1 = f1_score(true, preds, average='weighted')
    return f1

def accuracy(labels, output, test=False):
    preds = output.max(1)[1]
    preds = preds.type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum().item()
    return correct / len(labels)


