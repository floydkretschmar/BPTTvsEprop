import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, forward_hook=None, backward_hook=None, descriptor='LSTM_BPTT'):
        self.output_size = output_size
        self.input_size = input_size
        self.descriptor = descriptor
        super(MemoryLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        input = input.view(len(input), -1, self.input_size)
        lstm_out, _ = self.lstm(input)
        dense_out = self.dense(lstm_out[:, -1, :])
        predictions = F.log_softmax(dense_out, dim=1)
        return predictions

    def save(self, path, iteration):
        torch.save(self.state_dict(), '{}{}{}.pth'.format(path, self.descriptor, iteration, '.pth'))

    def load(self, path):
        self.load_state_dict(torch.load('{}{}{}.pth'.format(path, self.descriptor, '.pth')))
        self.eval()
