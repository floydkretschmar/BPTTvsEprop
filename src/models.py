import torch
import torch.nn as nn
import torch.jit as jit

import math

import lstm
from util import to_device
from autograd import EProp1, EProp3, SyntheticGradient


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def forward(self):
        pass

    def get_name(self):
        return "BaseNetwork"


class BaseLSTM(BaseNetwork):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 lstm,
                 bias=True, 
                 batch_first=True, 
                 single_output=True):
        super(BaseLSTM, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_first = batch_first
        self.single_output = single_output

        self.lstm = lstm

        # LSTM to output mapping
        self.dense = nn.Linear(hidden_size, output_size, bias)

    def forward(self, input):
        # prepare input and initial state
        if self.batch_first:
            input = input.permute(1, 0, 2)
        
        return self.forward_core(input)

    def forward_dense(self, lstm_out):
        # mapping to outputs
        if self.single_output:
            lstm_out = lstm_out[-1, :, :]       
        else:
            lstm_out = lstm_out.permute(1, 0, 2)
            
        dense_out = self.dense(lstm_out)
        return dense_out 

    def forward_core(self, input):
        initial_h = to_device(torch.zeros(input.size(1), self.hidden_size))
        initial_c = to_device(torch.zeros(input.size(1), self.hidden_size))

        # lstm and dense pass for prediction
        lstm_out = self.lstm(input, initial_h, initial_c)[0]
        dense_out = self.forward_dense(lstm_out)
        return dense_out

    def get_name(self):
        return "BaseLSTM"

    def reset(self):
        pass

class BPTT_LSTM(BaseLSTM):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 bias=True, 
                 batch_first=True, 
                 single_output=True):
        super(BPTT_LSTM, self).__init__(
            input_size, 
            hidden_size, 
            output_size, 
            lstm.BPTT_LSTM(input_size, hidden_size), 
            bias=bias, 
            batch_first=batch_first, 
            single_output=single_output)
            
    def get_name(self):        
        return "LSTM_BPTT"


class EPROP1_LSTM(BaseLSTM):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 bias=True, 
                 batch_first=True, 
                 single_output=True):
        super(EPROP1_LSTM, self).__init__(
            input_size, 
            hidden_size, 
            output_size, 
            lstm.EPropLSTM(input_size, hidden_size, eprop_func=EProp1.apply), 
            bias=bias, 
            batch_first=batch_first, 
            single_output=single_output)

    def get_name(self):        
        return "LSTM_EPROP1"


class EPROP3_LSTM(BaseLSTM):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 bias=True, 
                 batch_first=True, 
                 single_output=True):
        super(EPROP3_LSTM, self).__init__(
            input_size, 
            hidden_size, 
            output_size, 
            lstm.EPropLSTM(input_size, hidden_size, eprop_func=EProp3.apply), 
            bias=bias, 
            batch_first=batch_first, 
            single_output=single_output)

        # sythetic gradient that tries to emulate dE/d(s_j^{tm+1})
        self.synthetic_gradient_net = nn.Sequential(nn.Linear(output_size, 512, bias=bias),
                                nn.ReLU(),
                                nn.Linear(512, 256, bias=bias),
                                nn.ReLU(),
                                nn.Linear(256, 128, bias=bias),
                                nn.ReLU(),
                                nn.Linear(128, hidden_size, bias=bias))

        self.initial_c = None
        self.eligibility_vectors = None

    def forward_core(self, input):
        batch_size = input.size(1)
        initial_h = to_device(torch.zeros(batch_size, self.hidden_size))

        # initialize eligibility vectors only on first run
        if type(self.eligibility_vectors) == type(None):
            self.initial_c = to_device(torch.zeros(batch_size, self.hidden_size)).requires_grad_()
            self.eligibility_vectors = [to_device(torch.zeros(batch_size, 3 * self.hidden_size, self.input_size, requires_grad=False)), 
                to_device(torch.zeros(batch_size, 3 * self.hidden_size, self.hidden_size, requires_grad=False)),
                to_device(torch.zeros(batch_size, 3 * self.hidden_size, 1, requires_grad=False))]

        initial_c = self.initial_c
        # lstm and dense pass for prediction
        lstm_out, final_c, self.eligibility_vectors = self.lstm(input, initial_h.detach(), initial_c, self.eligibility_vectors)
        lstm_out = self.forward_dense(lstm_out)

        # take the last output of the network to let the synth grad network predict the synthetic gradient
        synth_grad = self.synthetic_gradient_net(lstm_out[:,-1,:].detach())

        # set the gradient of the last internal state equal to the synthetic gradient
        self.initial_c, lstm_out, _ = SyntheticGradient.apply(final_c, lstm_out, synth_grad.detach())

        # detach initial state from the compute graph of [t_{m-1}+1, ..., t_{m}] but 
        # build a new compute graph ...
        self.initial_c = self.initial_c.detach().requires_grad_()
        # ... and make sure to also detach eligibility vectors
        self.eligibility_vectors = [self.eligibility_vectors[0].detach(), self.eligibility_vectors[1].detach(),self.eligibility_vectors[2].detach()]

        # return both the output as well as the synthetic gradient 
        return lstm_out, initial_c, synth_grad

    def get_name(self):        
        return "LSTM_EPROP3"

    def reset(self):
        self.initial_c = None
        self.eligibility_vectors = None
