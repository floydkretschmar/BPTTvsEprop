import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F

import math

from util import to_device


class Eprop1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, eligibility_trace, bias=None):
        # store new eligibility trace for t+1
        ctx.save_for_backward(input, weight, bias, eligibility_trace)

        # calculate the actual weighted sum
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, eligibility_trace = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # TODO: calculate gradient according to eligibility_trace and 
        # the learning signal (=grad_output in case of eprop1)

        return grad_input, grad_weight, grad_bias


class Linear(jit.ScriptModule):
    """ 
    Default linear layer implementation adopted so that it can be extended to implement e-prop
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    @jit.script_method
    def forward(self, input):
        # Implement on eprop-function with forward and backward to replace F.linear
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class LSTMCell(jit.ScriptModule):
    """ 
    Custom LSTM Cell implementation using jit adopted from:
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Create for weight matrices, one for each gate + recurrent connections
        self.input_weights = Linear(input_size, 4 * hidden_size, bias=bias)
        self.hidden_weights = Linear(hidden_size, 4 * hidden_size, bias=bias)

    @jit.script_method
    def forward(self, input, hx, cx):
        # net activations...
        gates = self.input_weights(input) + self.hidden_weights(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # ... and gate activations
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, hy, cy


class LSTM(jit.ScriptModule):
    """ 
    Custom LSTM implementation using jit adopted from:
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM, self).__init__()
        self.cell = LSTMCell(input_size, hidden_size, bias)

    @jit.script_method
    def forward(self, input, initial_h, initial_c):
        # input (seq_len x batch_size x input_size)
        # initial_hidden (batch x hidden_size)
        # initial_state (batch x hidden_size)
        inputs = input.unbind(0)
        hx = initial_h
        cx = initial_c

        outputs = []
        for i in range(len(inputs)):
            out, hx, cx = self.cell(inputs[i], hx, cx)
            outputs += [out]

        return torch.stack(outputs), hx, cx


class MemoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias=True, batch_first=False, model_name='LSTM_BPTT'):
        super(MemoryLSTM, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.model_name = model_name
        self.batch_first = batch_first

        # LSTM layer
        # self.lstm = LSTM(input_size, hidden_size, bias)
        self.lstm = nn.LSTM(input_size, hidden_size)

        # LSTM to output mapping
        self.dense = nn.Linear(hidden_size, output_size, bias)

    def forward(self, input):
        # prepare input and initial state
        if self.batch_first:
            input = input.permute(1, 0, 2)

        initial_h = to_device(torch.zeros(input.size(1), self.hidden_size))
        initial_c = to_device(torch.zeros(input.size(1), self.hidden_size))

        # lstm and dense pass for prediction
        # lstm_out, _, _ = self.lstm(input, initial_h, initial_c)
        lstm_out, _ = self.lstm(input)

        # mapping to outputs
        dense_out = self.dense(lstm_out[-1, :, :])
        predictions = F.log_softmax(dense_out, dim=1)
        return predictions

    def save(self, path, epoch):
        torch.save(self.state_dict(), '{}{}_{}.pth'.format(path, self.model_name, epoch, '.pth'))

    def load(self, path):
        self.load_state_dict(torch.load('{}/load/{}{}'.format(path, self.model_name, '.pth')))
        self.eval()