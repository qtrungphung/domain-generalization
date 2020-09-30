import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler, SGD

import mlp
from data_loader import to_data_loader
from utils import fix_seed, write_log



class ModelBaseline:
    """ Base model for Meta Learning Domain Generalization """

    def __init__(self, flags):

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # fix the random seed or not
        # fix_seed()

        self.to_data_loader(flags)

        self.network = mlp.MLPNet(num_classes=flags.num_classes)
        self.network.to(self.device)

        print('NETWORK ARCHITECTURE\n', self.network)
        print('FLAGS:', flags)

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        write_log(flags, flags_log)

        self.configure(flags)

    def to_data_loader(self, flags):
        """Stitch data to model
        output:
            self.train_paths
            self.val_paths
            self.unseen_data_path
            self.datasets
            self.dataloaders
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

        self.unseen_index = flags.unseen_index
        self.unseen_data_path = os.path.join(
            root_folder, test_data[self.unseen_index])

        self.train_paths.remove(self.train_paths[self.unseen_index])
        self.val_paths.remove(self.val_paths[self.unseen_index])

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)

        flags_log = os.path.join(flags.logs, 'path_log.txt')
        write_log(str(self.train_paths), flags_log)
        write_log(str(self.val_paths), flags_log)
        write_log(str(self.unseen_data_path), flags_log)

        # to data loader
        self.train_datasets = []
        self.train_dataloaders = []
        for train_path in self.train_paths:
            dataset, dataloader = to_data_loader(
                flags=flags, stage='train', file_path=train_path)
            self.train_datasets.append(dataset)
            self.train_dataloaders.append(dataloader)

        self.val_datasets = []
        self.val_dataloaders = []
        for val_path in self.val_paths:
            dataset, dataloader = to_data_loader(
                flags=flags, stage='val', file_path=val_path)
            self.val_datasets.append(dataset)
            self.val_dataloaders.append(dataloader)

    def load(self, PATH):
        """ torch load state dict
            Input:
                PATH: path to state dict, this is a dict
                state_dict should be saved under 'state'
        """

        checkpoint = torch.load(PATH)

        self.network.load_state_dict(checkpoint['state'])
        ite = checkpoint['ite']

        return

    def save(self, PATH, ite):
        """ Save model """

        torch.save({'ite': ite,
                    'state': self.network.state_dict()}, PATH)
        return

    def configure(self, flags):
        """ Setting up optimizer, scheduler, loss function"""

        print('NETWORK SIZE')
        for name, para in self.network.named_parameters():
            print(name, para.size())

        self.optimizer = SGD(params=self.network.parameters(),
                             lr=flags.lr,
                             weight_decay=flags.weight_decay,
                             momentum=flags.momentum)

        self.scheduler = lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=flags.step_size, gamma=0.1)

        self.loss_fn = F.cross_entropy 

        return

    def train(self, flags):
        """ base line training:
        get a batch from each domain, compute loss, add from all domains
        backprop from total loss
        """

        self.network.train()
        self.best_accuracy_val = -1

        for ite in range(flags.inner_loops):

            total_loss = 0.0
            iter_loader = iter(self.train_dataloaders[1])

            for loader in self.train_dataloaders:
                for samples, labels in loader:
                    samples = samples.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.network(samples)
                    loss = self.loss_fn(outputs, labels)
                    total_loss += loss

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            total_loss.backward()

            # optimize the parameters
            self.optimizer.step()

            print('ite:', ite, 'loss:', total_loss.item(), 'lr:',
                  self.scheduler.get_lr()[0])

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(str(total_loss.item()), flags_log)

            del total_loss, outputs

            if ite % flags.test_every == 0 and ite is not 0 or flags.debug:
                self.test_workflow(flags, ite)
                self.scheduler.step()

        return

    def test_workflow(self, flags, ite):
        """ Test model on validation set
            validation set is provided by dataloader attached to model:
            self.val_dataloaders
        """

        accuracies = []
        for count, loader in enumerate(self.val_dataloaders):
            accuracy_val = self._test(
                dataloader=loader, flags=flags, ite=ite,
                log_prefix='val_index_{}'.format(count), log_dir=flags.logs)
            accuracies.append(accuracy_val)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write('ite:{}, best val accuracy:{}\n'.format(
                ite, self.best_accuracy_val))
            f.close()

            if not os.path.exists(flags.model_path):
                os.mkdir(flags.model_path)

            state_dict_path = os.path.join(flags.model_path, 'best_model.pt')
            self.save(state_dict_path, ite)

        return

    def heldout_test(self, flags, state_dict_file):
        """ Test a saved model on unseen domain
        input: state_dict_file : name of the trained model's state dict """

        # load the best model in the validation data
        state_dict_path = os.path.join(flags.model_path, state_dict_file)
        self.load(state_dict_path)

        # test dataset and data loader
        dataset, dataloader = to_data_loader(
            flags=flags, stage='test', file_path=self.unseen_data_path)

        # call test method
        accuracy = self._test(dataloader=dataloader, flags=flags, ite=0,
                              log_prefix='held', log_dir=flags.logs)
        return accuracy

    def _test(self, dataloader, flags, ite, log_prefix, log_dir='logs/'):
        """ Test model on input dataloader
            test_dataloader: dataloader of the test dataset
            flags: flags
            ite: iteration, to write to log
            log_prefix
            log_dir

            return: accuracy
        """

        assert dataloader is not None

        self.network.eval()

        corrects = 0
        totals = 0
        for samples, labels in dataloader:
            samples = samples.to(self.device)
            labels = labels.to(self.device)
            outputs = self.network(samples)
            _, preds = torch.max(outputs, 1)
            corrects += (preds == labels).sum().item()
            totals += labels.size(0)
        accuracy = corrects / totals
        print("total: {}, corrects: {}, accuracy {}".format(
            totals, corrects, accuracy))

        log_path = os.path.join(log_dir, '{}.txt'.format(log_prefix))
        write_log(str('ite:{}, accuracy:{}'.format(ite, accuracy)),
                  log_path=log_path)

        # switch on the network train mode after test
        self.network.train()

        return accuracy



class ModelMLDG(ModelBaseline):
    """ meta learning domain generalization
    """

    def __init__(self, flags,):
        ModelBaseline.__init__(self, flags)

    def train(self, flags):
        """ MLDG training """

        self.network.train()
        self.best_accuracy_val = -1

        # make iter, use next to get samples
        self.train_iter_loaders = []
        self.val_iter_loaders = []
        for loader in self.train_dataloaders:
            self.train_iter_loaders.append(iter(loader))
        for loader in self.val_dataloaders:
            self.val_iter_loaders.append(iter(loader))

        for ite in range(flags.inner_loops):
            for idx in range(len(self.train_dataloaders)):
                if (ite % len(self.train_iter_loaders[idx])) == 0:
                    self.train_iter_loaders[idx] = iter(self.train_dataloaders[idx])
                if (ite % len(self.val_iter_loaders[idx])) == 0:
                    self.val_iter_loaders[idx] = iter(self.val_dataloaders[idx])

            # meta_query is the held out domain
            # calc F loss on not meta_query (refer to paper)
            # calc G on meta_query
            meta_query_idx = np.random.choice(
                np.arange(0, len(self.train_datasets)), size=1)[0]

            # Calc meta train loss F
            meta_train_loss = 0.0

            for idx in range(len(self.train_dataloaders)):

                if idx == meta_query_idx:
                    continue
                train_loader = self.train_iter_loaders[idx]
                samples, labels = next(train_loader)
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                outputs = self.network(samples)
                loss = self.loss_fn(outputs, labels)
                meta_train_loss += loss

            # Calc meta query loss G
            idx = meta_query_idx
            train_loader = self.val_iter_loaders[idx]
            samples, labels = next(train_loader)
            samples = samples.to(self.device)
            labels = labels.to(self.device)
            outputs = self.network(x=samples,
                                   meta_loss=meta_train_loss,
                                   meta_step_size=flags.meta_step_size,
                                   stop_gradient=flags.stop_gradient)
            meta_query_loss = self.loss_fn(outputs, labels)

            total_loss = (meta_train_loss
                          + meta_query_loss * flags.meta_val_beta)

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            total_loss.backward()

            # optimize the parameters
            self.optimizer.step()

            print('ite:', ite, 'loss:', total_loss.item(), 'lr:',
                  self.scheduler.get_lr()[0])

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(str(total_loss.item()), flags_log)

            del total_loss, meta_query_loss, outputs

            if ite % flags.test_every == 0 and ite is not 0 or flags.debug:
                self.test_workflow(flags, ite)
                self.scheduler.step()
        return

