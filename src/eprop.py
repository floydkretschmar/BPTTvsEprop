import torch
import torch.jit as jit

from lstm_jit import LSTMCell, LSTM


class EProp1(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, 
            ev_w_ih_x,
            ev_w_hh_x,
            ev_b_x,
            forgetgate_x,
            input, 
            hx, 
            cx,
            weight_ih, 
            weight_hh, 
            bias_ih=None, 
            bias_hh=None):
        gates = (torch.mm(input, weight_ih.t()) + bias_ih + torch.mm(hx, weight_hh.t()) + bias_hh)
        ingate, forgetgate_y, cellgate, outgate = gates.chunk(4, 1)

        # ... and gate activations
        ingate = torch.sigmoid(ingate)
        forgetgate_y = torch.sigmoid(forgetgate_y)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        gates = [ingate, forgetgate_y, cellgate]

        cy = (forgetgate_y * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        # TODO: calculate new eligibility vector and trace
        # There exist distinct eligibility traces and vectors for the followiug parts of the LSTM cell:
        # - input to hidden connections, hidden to hidden connections, bias 
        # all for each: 
        # - ... inputgate, forgetgate and cellgate
        # => overall 3 * 3 = 9 eligibility traces
        input_size = input.size(1)

        # the new eligibility vectors ...
        ev_w_ih_y = torch.Tensor(ev_w_ih_x.size())
        ev_w_hh_y = torch.Tensor(ev_w_hh_x.size())
        ev_b_y = torch.Tensor(ev_b_x.size())

        # ... and eligibility traces
        et_w_ih_y = torch.Tensor(ev_w_ih_x.size(0), )
        et_w_hh_y = torch.Tensor(ev_w_hh_x.size(0))
        et_b_y = torch.Tensor(ev_b_x.size(0))

        ev_w_ih_i, ev_w_ih_f, ev_w_ih_c, ev_w_ih_o = ev_w_ih_x.chunk(4, 1)
        ev_w_hh_i, ev_w_hh_f, ev_w_hh_c, ev_w_hh_o = ev_w_hh_x.chunk(4, 1)
        ev_b_i, ev_b_f, ev_b_c, ev_b_o = ev_b_x.chunk(4, 1)

        w_ih_i, w_ih_f, w_ih_c, w_ih_o = weight_ih.chunk(4, 1)
        w_hh_i, w_hh_f, w_hh_c, w_hh_o = weight_hh.chunk(4, 1)

        ones = torch.ones(ingate.size())

        # ingate
        base = ingate * (ones - ingate) * cellgate
        ds_dw_hh_i = base * hx
        ds_dw_ih_i = base * input
        ds_dbias_i = base
        
        ev_w_ih_y[:, :input_size] = forgetgate_x * ev_w_ih_i + ds_dw_hh_i
        ev_w_hh_y[:, :input_size] = forgetgate_x * ev_w_hh_i + ds_dw_ih_i
        ev_b_y[:, :input_size] = forgetgate_x * ev_b_i + ds_dbias_i
        
        # forgetgate
        base = forgetgate_y * (ones - forgetgate_y) * cellgate
        ds_dw_hh_f = base * hx
        ds_dw_ih_f = base * input
        ds_dbias_f = base
        
        ev_w_ih_y[:, input_size:(2 * input_size)] = forgetgate_x * ev_w_ih_f + ds_dw_hh_f
        ev_w_hh_y[:, input_size:(2 * input_size)] = forgetgate_x * ev_w_hh_f + ds_dw_ih_f
        ev_b_y[:, input_size:(2 * input_size)] = forgetgate_x * ev_b_f + ds_dbias_f
        
        # cellgate
        base = ingate * (ones - cellgate**2)
        ds_dw_hh_c = base * hx
        ds_dw_ih_c = base * input
        ds_dbias_c = base
        
        ev_w_ih_y[:, (2 * input_size):(3 * input_size)] = forgetgate_x * ev_w_ih_c + ds_dw_hh_c
        ev_w_hh_y[:, (2 * input_size):(3 * input_size)] = forgetgate_x * ev_w_hh_c + ds_dw_ih_c
        ev_b_y[:, (2 * input_size):(3 * input_size)] = forgetgate_x * ev_b_c + ds_dbias_c

        # calculate eligibility traces by multiplying the eligibility vectors with the outgate
        for i in range(0, 3 * input_size, input_size):
            et_w_ih_y[:, i:(i + input_size)] = ev_w_ih_y[:, i:(i + input_size)] * outgate
            et_w_hh_y[:, i:(i + input_size)] = ev_w_hh_y[:, i:(i + input_size)] * outgate
            et_b_y[:, i:(i + input_size)] = ev_b_y[:, i:(i + input_size)] * outgate
        
        # The gradient of the output gate is only dependent on the observable state
        # => just use normal gradient calculation of dE/dh * dh/dweight 
        # => calculate second part of that equation now for input to hidden, hidden to hidden 
        #    and bias connections and multiply in the backward pass
        base = outgate * (ones - outgate) * cy
        et_w_ih_y[:, (3 * input_size):(4 * input_size)] = base * hx
        et_w_hh_y[:, (3 * input_size):(4 * input_size)] = base * input
        et_b_y[:, (3 * input_size):(4 * input_size)] = base

        ctx.save_for_backward(et_w_ih_y, et_w_hh_y, et_b_y, input_size)

        return hy, cy, ev_w_ih_y, ev_w_ih_y, ev_w_ih_y, forgetgate_y

    @staticmethod
    # grad_ev_ih and grad_ev_hh should always be None
    def backward(ctx, grad_hy, grad_cy, grad_ev_ih, grad_ev_hh):
        et_w_ih_y, et_w_hh_y, et_b_y, input_size = ctx.saved_tensors
        grad_weight_ih = grad_weight_hh = grad_bias = None

        # create tensors for the weight gradient
        grad_weight_ih = torch.Tensor(et_w_ih_y.size())
        grad_weight_hh = torch.Tensor(et_w_hh_y.size())
        grad_bias = torch.Tensor(et_b_y.size())
        
        # ingate, forgetgate and cellgate
        for i in range(0, 4 * input_size, input_size):
            grad_weight_ih[:, i:i + input_size] = et_w_ih_y[:, i:i + input_size] * grad_hy[:, i:i + input_size]
            grad_weight_hh[:, i:i + input_size] = et_w_hh_y[:, i:i + input_size] * grad_hy[:, i:i + input_size]
            grad_bias[:, i:i + input_size] = et_b_y[:, i:i + input_size] * grad_hy[:, i:i + input_size]

        # TODO: calculate for bias

        # grad_input, grad_ev_ih, grad_ev_hh, grad_hx, grad_cx, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh
        return None, None, None, None, None, grad_weight_ih, grad_weight_hh, grad_bias, grad_bias


class EpropCell(LSTMCell):
    """ 
    Custom LSTM Cell implementation using jit adopted from:
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(EpropCell, self).__init__(input_size, hidden_size, bias)

        # Initialize eligibility traces and forgetgate to zero
        self.ev_w_ih_x = torch.zeros(hidden_size, 3 * input_size)
        self.ev_w_hh_x = torch.zeros(hidden_size, 3 * input_size)
        self.ev_b_x = torch.zeros(hidden_size, 3 * input_size)
        self.forgetgate = torch.zeros(hidden_size, input_size)

    @jit.script_method
    def forward(self, input, hx, cx):
        hy, cy, self.ev_w_ih_x, self.ev_w_hh_x, self.ev_b_x, self.forgetgate = EProp1.apply(
            self.ev_w_ih_x,
            self.ev_w_hh_x,
            self.ev_b_x,
            self.forgetgate,
            input, 
            hx, 
            cx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh)

        return hy, hy, cy


class EpropLSTM(LSTM):
    """ 
    Custom LSTM implementation using jit adopted from:
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM, self).__init__()
        self.cell = LSTMCell(input_size, hidden_size, bias)