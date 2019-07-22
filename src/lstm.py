import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, forward_hook=None, backward_hook=None, model_name='LSTM_BPTT'):
        super(MemoryLSTM, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.model_name = model_name

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        input = input.view(len(input), -1, self.input_size)
        lstm_out, _ = self.lstm(input)
        dense_out = self.dense(lstm_out[:, -1, :])
        predictions = F.log_softmax(dense_out, dim=1)
        return predictions

    def save(self, path, epoch):
        torch.save(self.state_dict(), '{}{}_{}.pth'.format(path, self.model_name, epoch, '.pth'))

    def load(self, path):
        self.load_state_dict(torch.load('{}/load/{}{}'.format(path, self.model_name, '.pth')))
        self.eval()
