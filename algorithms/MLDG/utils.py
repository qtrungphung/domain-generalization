import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder



# def onehot_label(labels, classes):
#     """turn labels into one hot vectors"""
#     assert len(np.unique(labels)) == classes

#     encoder = LabelEncoder(dtype=np.int8)
#     encoder.fit(labels)
#     return encoder.transform(labels)


# def unfold_label(labels, classes):
#     new_labels = []

#     assert len(np.unique(labels)) == classes
#     # minimum value of labels
#     mini = np.min(labels)

#     for index in range(len(labels)):
#         dump = np.full(shape=[classes], fill_value=0).astype(np.int8)
#         _class = int(labels[index]) - mini
#         dump[_class] = 1
#         new_labels.append(dump)

#     return np.array(new_labels)


def shuffle_data(samples, labels):
    """Input: samples(list-like), labels(list-like)
       Output: shuffled_samples(list-like), shuffled_labels(list-like)
    """
    num = len(labels)
    shuffle_index = np.random.permutation(np.arange(num))
    shuffled_samples = samples[shuffle_index]
    shuffled_labels = labels[shuffle_index]
    return shuffled_samples, shuffled_labels


def shuffle_list(li):
    np.random.shuffle(li)
    return li


def shuffle_list_with_ind(li):
    shuffle_index = np.random.permutation(np.arange(len(li)))
    li = li[shuffle_index]
    return li, shuffle_index


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def fix_seed():
    # torch.manual_seed(1108)
    # np.random.seed(1108)
    pass


def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()


def compute_accuracy(predictions, labels):
    accuracy = accuracy_score(y_true=np.argmax(labels, axis=-1), y_pred=np.argmax(predictions, axis=-1))
    return accuracy
