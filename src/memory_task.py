import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import memory_config as conf
from lstm import MemoryLSTM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_data(num_obervations, num_classes, sequence_length, time_delta=None):
    '''
    Generates the sequences of form (num_observations, sequence_lenght) where the entire sequence is 
    filled with 0s, except for one singular signal at a random position inside the sequence.
    The label of a sequence is the true class of the signal - 1, because the NLLLoss used to train 
    the network expects labels in the range between 0 and num_classes - 1 instead of 1 and num_classes.
    '''

    size = (num_obervations, sequence_length)
    data = np.zeros(size)    
    labels = np.zeros((num_obervations, 1))
    for i, row in enumerate(data):
        signal = np.random.randint(1, num_classes, 1)[0]
        last_possible_signal = sequence_length if not time_delta else sequence_length - time_delta
        
        column = np.random.randint(0, last_possible_signal, 1)[0]
        row[column] = signal
        labels[i] = signal - 1

    return torch.from_numpy(data).float().to(DEVICE), torch.from_numpy(labels).long().to(DEVICE)


def train(model, loss_function, optimizer, train_X, train_Y):

    training_results = []

    print('######################### TRAINING #########################')
    for epoch in range(conf.NUM_EPOCHS): 
        permutation = torch.randperm(train_X.size()[0])

        total_loss = 0
        batch_num = 0
        for i in range(0, train_X.size()[0], conf.BATCH_SIZE):
            # reset gradient
            optimizer.zero_grad()

            # Get mini-batch of sequences
            indices = permutation[i:i + conf.BATCH_SIZE]
            batch_x, batch_y = train_X[indices], train_Y[indices]

            prediction = model(batch_x)
            
            # Compute the loss, gradients, and update the parameters
            loss = loss_function(prediction, batch_y.squeeze())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_num += 1

        training_results.append('Epoch {} \t => Loss: {}'.format(epoch, total_loss / batch_num))
        print(training_results[-1])

        if epoch % 50 == 0:
            model.save(conf.SAVE_PATH, epoch)

    with open('{}/results.txt'.format(conf.SAVE_PATH), 'w') as file:
        for result in training_results:
            file.write("{}\n".format(result))


def test(model, loss_function, test_X, test_Y):
    print('######################### TESTING #########################')
    with torch.no_grad():
        prediction = model(test_X)
        loss = loss_function(prediction, test_Y.squeeze())
        print('Loss: {}'.format(loss))
        print('GT', test_Y)
        print('Predictions (Post-training)', np.argmax(prediction))


def main(size_train_data, size_test_data):
    # generate data sets
    train_X, train_Y = generate_data(size_train_data, conf.NUM_CLASSES, conf.SEQ_LENGTH)
    test_X, test_Y = generate_data(size_train_data, conf.NUM_CLASSES, conf.SEQ_LENGTH, time_delta=conf.SEQ_MIN_DELTA)

    # the default LSTM implementation with bptt
    model = MemoryLSTM(conf.INPUT_SIZE, conf.HIDEN_SIZE, conf.NUM_CLASSES).to(DEVICE)

    # Use negative log-likelihood and ADAM
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.NLLLoss()

    # See what the predictions are before training: output = neg_softmax of classes 1 to num_classes
    with torch.no_grad():
        predicition = model(test_X[0].view(1, -1))
        print('GT', test_Y[0])
        print('Predictions (Pre-training)', predicition)

    train(model, loss_function, optimizer, train_X, train_Y)
    test(model, loss_function, test_X, test_Y)


if __name__ == '__main__':
    main(conf.TRAIN_SIZE, conf.TEST_SIZE)