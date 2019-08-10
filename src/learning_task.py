import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from util import to_device


class LearningTask:
    def __init__(
            self,
            num_epochs, 
            size_batch,
            path,
            loss_function):
        self.size_batch = size_batch
        self.num_epochs = num_epochs
        self.path = path

        self.loss_function = loss_function

    def get_batched_data(self, data, labels):
        permutation = torch.randperm(data.size()[0])
        for i in range(0, data.size()[0], self.size_batch):
            indices = permutation[i:i + self.size_batch]
            yield data[indices], labels[indices]

    def train(self, model, size_train, sequence_length, learning_rate):
        # Use negative log-likelihood and ADAM
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_X, train_Y = self.generate_data(size_train, sequence_length)
        training_results = []

        print('######################### TRAINING #########################')
        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            total_loss = 0
            batch_num = 0

            for batch_x, batch_y in self.get_batched_data(train_X, train_Y):
                # reset gradient
                optimizer.zero_grad()
                prediction = model(batch_x)
                if len(prediction.size()) > 2:
                    prediction = prediction.view(-1, prediction.size(2))
                    targets = batch_y.squeeze().view(-1)
                else:
                    targets = batch_y.squeeze()
                
                # Compute the loss, gradients, and update the parameters
                loss = self.loss_function(prediction, batch_y.squeeze())
                loss.backward()

                optimizer.step()
                
                total_loss += loss.item()
                batch_num += 1

            training_results.append('Epoch {} \t => Loss: {} [Batch-Time = {}s]'.format(epoch, total_loss / batch_num, round(time.time() - start_time, 2)))
            print(training_results[-1])

            if epoch % 50 == 0:
                model.save(self.path, epoch)

        with open('{}/results.txt'.format(self.path), 'w') as file:
            for result in training_results:
                file.write("{}\n".format(result))

    def test(self, model, test_data):
        test_X, test_Y = test_data
        print('######################### TESTING #########################')
        with torch.no_grad():
            prediction = model(test_X)
            if len(prediction.size()) > 2:
                prediction = prediction.view(-1, prediction.size(2))
                targets = test_Y.squeeze().view(-1)
            else:
                targets = test_Y.squeeze()

            example_recall = torch.argmax(test_Y[0])

            loss = self.loss_function(prediction, targets)
            print('Loss: {}'.format(loss))
            print('Targets: \n {}'.format(targets[example_recall]))
            print('Predictions: \n {}'.format(torch.argmax(prediction[example_recall])))


class MemoryTask(LearningTask):
    def __init__(
            self,
            num_epochs, 
            size_batch,
            path,
            num_classes):
        super(MemoryTask, self).__init__(
            num_epochs, 
            size_batch,
            path,
            nn.CrossEntropyLoss())
        self.num_classes = num_classes

    def generate_data(self, num_observations, sequence_length, time_delta=None):
        '''
        Generates the sequences of form (num_observations, sequence_length) where the entire sequence is 
        filled with 0s, except for one singular signal at a random position inside the sequence.
        The label of a sequence is the true class of the signal - 1, because the NLLLoss used to train 
        the network expects labels in the range between 0 and num_classes - 1 instead of 1 and num_classes.
        '''
        size = ((num_observations, sequence_length, 1))
        data = np.zeros(size)    
        labels = np.zeros((num_observations, 1))
        for i, row in enumerate(data):
            signal = np.random.randint(1, self.num_classes, 1)[0]
            last_possible_signal = sequence_length if not time_delta else sequence_length - time_delta
            column = np.random.randint(0, last_possible_signal, 1)[0]
            row[column] = signal
            labels[i] = signal - 1

        return to_device(torch.from_numpy(data).float()), to_device(torch.from_numpy(labels).long())

    def test_all_deltas(self, model, size_test_data, sequence_length):
        for delta in range(1, sequence_length):
            test_data = self.generate_data(size_test_data, sequence_length, delta)
            print("Delta {}:".format(delta))
            self.test(model, test_data)


class StoreRecallTask(LearningTask):
    def __init__(
            self,
            num_epochs, 
            size_batch,
            path,
            num_classes):
        super(StoreRecallTask, self).__init__(
            num_epochs, 
            size_batch,
            path,
            nn.CrossEntropyLoss())
        self.num_classes = num_classes

    def generate_data(self, num_observations, sequence_length, time_delta=None):
        '''
        Generates the sequences of form (num_observations, sequence_length) where the entire sequence is 
        filled with 0s, except for one singular signal at a random position inside the sequence.
        The label of a sequence is the true class of the signal - 1, because the NLLLoss used to train 
        the network expects labels in the range between 0 and num_classes - 1 instead of 1 and num_classes.
        '''
        size = ((num_observations, sequence_length, 1))
        data_stream = np.random.randint(1, self.num_classes + 1, size)
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

    def test_all_deltas(self, model, size_test_data, sequence_length):
        for delta in range(1, sequence_length):
            test_data = self.generate_data(size_test_data, sequence_length, delta)
            print("Delta {}:".format(delta))
            self.test(model, test_data)