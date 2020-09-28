import h5py
import numpy as np
import torch

from utils import shuffle_data



class Dataset(torch.utils.data.Dataset):
    """ torch Dataset object for DataLoader to load.
        Input:
            flags : configuration flags (from argsparser)
            stage : train val test
            file_path :
    """
    def __init__(self, flags, stage, file_path):

        if stage not in ['train', 'val', 'test']:
            assert ValueError('invalid stage!')

        self.file_path = file_path
        self.stage = stage

        # open file
        f = h5py.File(file_path, "r")
        self.images = torch.from_numpy(np.array(f['images']))
        self.labels = torch.from_numpy(np.array(f['labels']))
        f.close()

    def __getitem__(self, idx):
        """ get item method to fetch a sample"""
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)


def to_data_loader(flags, stage, file_path):
    """ Generate torch Dataset and DataLoader
    Input:
            flags : configuration flags (from argsparser)
            stage : train val test
            file_path :
    Ourput: (dataset, dataloader)
    """

    dataset = Dataset(flags=flags, stage=stage, file_path=file_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0)
    return dataset, dataloader


class BatchImageGenerator:
    """Load data from paths, yield batches"""
    def __init__(self, flags, stage, file_path, b_unfold_label):

        if stage not in ['train', 'val', 'test']:
            assert ValueError('invalid stage!')

        self.configuration(flags, stage, file_path)
        self.load_data(b_unfold_label)


    def configuration(self, flags, stage, file_path):
        """ Set up config """
        self.batch_size = flags.batch_size
        self.current_index = -1
    #    self.file_path = file_path
        self.stage = stage
        self.shuffled = False


    def load_data(self, b_unfold_label):
        file_path = self.file_path
        # f = h5py.File(file_path, "r")
        # self.images = np.array(f['images'])
        # self.labels = np.array(f['labels'])
        # f.close()

        # shift the labels to start from 0
        self.labels -= np.min(self.labels)

        # if b_unfold_label:
        #     self.labels = unfold_label(labels=self.labels, classes=len(np.unique(self.labels)))
        # assert len(self.images) == len(self.labels)

        self.file_num_train = len(self.labels)
        print('data num loaded:', self.file_num_train)

        if self.stage is 'train':
            self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)
        print('data size', torch.from_numpy(self.images).size())


    def get_images_labels_batch(self):

        images = []
        labels = []
        for index in range(self.batch_size):
            self.current_index += 1

            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.current_index %= self.file_num_train

                self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)

            images.append(self.images[self.current_index])
            labels.append(self.labels[self.current_index])

        images = np.stack(images)
        labels = np.stack(labels)

        return images, labels
