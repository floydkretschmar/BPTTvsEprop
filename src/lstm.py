import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        # Create for weight matrices, one for each gate + recurrent connections
        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

    def recurrence(self, input, hidden):
        # get last hidden output and cell state
        hx, cx = hidden 

        # calculate net activations of all gates at once
        gates = self.input_weights(input) + self.hidden_weights(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # calculate true activations of all gates
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate) 
        outgate = torch.sigmoid(outgate)

        # calculate new cell state and hidden output
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy) 

        return hy, cy

    def forward(self, input):
        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        hidden = torch.zeros(input.size(1), self.hidden_size), torch.zeros(input.size(1), self.hidden_size)
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


class MemoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, backward_hook=None, model_name='LSTM_BPTT'):
        super(MemoryLSTM, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.model_name = model_name

        # self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm = CustomLSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)

        # if backward_hook:
        # self.lstm.register_backward_hook(backward_hook)
        # self.dense.register_backward_hook(backward_hook)

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

