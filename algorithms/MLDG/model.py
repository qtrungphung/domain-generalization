import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD

import mlp
from data_loader import BatchImageGenerator
from utils import fix_seed, write_log



class ModelBaseline:
    def __init__(self, flags):

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        # fix the random seed or not
        # fix_seed()

        self.setup_path(flags)

        self.network = mlp.MLPNet(num_classes=flags.num_classes)
        self.network.to(device)

        print(self.network)
        print('flags:', flags)

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        write_log(flags, flags_log)

        self.load_state_dict(flags.state_dict)
        self.configure(flags)

    def setup_path(self, flags):
        """Stitch data to model
        output:
            self.train_paths
            self.val_paths
            self.unseen_data_path
        """
        root_folder = flags.data_root
        train_data = ['art_painting_train_features.hdf5',
                      'cartoon_train_features.hdf5',
                      'photo_train_features.hdf5',
                      'sketch_train_features.hdf5']

        val_data = ['art_painting_val_features.hdf5',
                    'cartoon_val_features.hdf5',
                    'photo_val_features.hdf5',
                    'sketch_val_features.hdf5']

        test_data = ['art_painting_features.hdf5',
                     'cartoon_features.hdf5',
                     'photo_features.hdf5',
                     'sketch_features.hdf5']

        self.train_paths = []
        for data in train_data:
            path = os.path.join(root_folder, data)
            self.train_paths.append(path)

        self.val_paths = []
        for data in val_data:
            path = os.path.join(root_folder, data)
            self.val_paths.append(path)

    def load_state_dict(self, state_dict=''):
        """ torch load state dict """
        pass

    def configure(self, flags):
        for name, para in self.network.named_parameters():
            print(name, para.size())

        self.optimizer = SGD(params=self.network.parameters(),
                             lr=flags.lr,
                             weight_decay=flags.weight_decay,
                             momentum=flags.momentum)

        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer,
                                             step_size=flags.step_size, gamma=0.1)
        self.loss_fn = nn.CrossEntropyLoss()
