import torch
import numpy as np


class LearningTask:
    def __init__(
            self,
            size_training_data, 
            size_test_data, 
            sequence_length, 
            num_epochs, 
            size_batch,
            device,
            path):
        self.device = device
        self.size_training_data = size_training_data
        self.size_test_data = size_test_data
        self.sequence_length = sequence_length
        self.size_batch = size_batch
        self.num_epochs = num_epochs
        self.path = path

    def get_batched_data(self, data, labels):
        permutation = torch.randperm(data.size()[0])
        for i in range(0, data.size()[0], self.size_batch):
            indices = permutation[i:i + self.size_batch]
            yield data[indices], labels[indices]

    def train(self, model, loss_function, optimizer):
        # Compute the loss, gradients, and update the parameters
        train_X, train_Y = self.generate_train_data()
        training_results = []

        print('######################### TRAINING #########################')
        for epoch in range(1, self.num_epochs + 1):
            total_loss = 0
            batch_num = 0

            for batch_x, batch_y in self.get_batched_data(train_X, train_Y):
                # reset gradient
                optimizer.zero_grad()
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
                model.save(self.path, epoch)

        with open('{}/results.txt'.format(self.path), 'w') as file:
            for result in training_results:
                file.write("{}\n".format(result))

    def test(self, model, loss_function):
        print('######################### TESTING #########################')
        with torch.no_grad():
            index = 1
            for test_X, test_Y in self.generate_test_data():
                prediction = model(test_X)
                loss = loss_function(prediction, test_Y.squeeze())
                print('Loss for test set {}: {}'.format(index, loss))
                index += 1

    def generate_test_data():
        pass

    def generate_train_data():
        pass


class MemoryTask(LearningTask):
    def __init__(
            self,
            size_training_data, 
            size_test_data, 
            sequence_length, 
            num_epochs, 
            size_batch,
            device,
            path,
            num_classes):
        super(MemoryTask, self).__init__(
            size_training_data, 
            size_test_data, 
            sequence_length, 
            num_epochs, 
            size_batch,
            device,
            path)
        self.num_classes = num_classes

    def generate_data(self, num_observations, time_delta=None):
        '''
        Generates the sequences of form (num_observations, sequence_length) where the entire sequence is 
        filled with 0s, except for one singular signal at a random position inside the sequence.
        The label of a sequence is the true class of the signal - 1, because the NLLLoss used to train 
        the network expects labels in the range between 0 and num_classes - 1 instead of 1 and num_classes.
        '''
        size = (num_observations, self.sequence_length)
        data = np.zeros(size)    
        labels = np.zeros((num_observations, 1))
        for i, row in enumerate(data):
            signal = np.random.randint(1, self.num_classes, 1)[0]
            last_possible_signal = self.sequence_length if not time_delta else self.sequence_length - time_delta
            column = np.random.randint(0, last_possible_signal, 1)[0]
            row[column] = signal
            labels[i] = signal - 1

        return torch.from_numpy(data).float().to(self.device), torch.from_numpy(labels).long().to(self.device)

    def generate_train_data(self):
        return self.generate_data(self.size_training_data)

    def generate_test_data(self):
        for delta in range(1, self.sequence_length):
            yield self.generate_data(self.size_test_data, delta)
