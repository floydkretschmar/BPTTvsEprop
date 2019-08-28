import numpy as np
import torch

from util import to_device
from config import SR_NUM_CLASSES as NUM_CLASSES


def generate_data(num_observations, sequence_length, time_delta=None):
    '''
    Generates the sequences of form (num_observations, sequence_length) where the entire sequence is 
    filled with 0s, except for one singular signal at a random position inside the sequence.
    The label of a sequence is the true class of the signal - 1, because the NLLLoss used to train 
    the network expects labels in the range between 0 and num_classes - 1 instead of 1 and num_classes.
    '''
    size = ((num_observations, sequence_length, 1))
    data_stream = np.random.randint(1, NUM_CLASSES + 1, size)
    store_signal = np.zeros(size)
    recall_signal = np.zeros(size)
    labels = np.zeros(size)
    data = []
    time_delta = time_delta if time_delta else 1

    for i, row in enumerate(data_stream):
        # select time step where store signal is sent
        store = np.random.choice(list(range(sequence_length - time_delta)))
        store_signal[i][store] = 1

        # select time step where recall signal is sent
        recall = np.random.choice(list(range(store + time_delta, sequence_length)))
        recall_signal[i][recall] = 1
        labels[i][recall] = row[store]

        data.append(np.hstack((row, store_signal[i], recall_signal[i])))

    data = np.array(data)
    return to_device(torch.from_numpy(data).float()), to_device(torch.from_numpy(labels).long())