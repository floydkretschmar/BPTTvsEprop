import torch
import torch.jit as jit

from lstm_jit import LSTMCell


class EProp1(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, 
            ev_ih_x,
            ev_hh_x,
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
        input_size = input.size(1)
        ev_ih_y = torch.Tensor(ev_ih_x.size())
        ev_hh_y = torch.Tensor(ev_hh_x.size())

        et_ih_y = torch.Tensor(weight_ih.size())
        et_hh_y = torch.Tensor(weight_ih.size())

        ev_ih_i, ev_ih_f, ev_ih_c, ev_ih_o = ev_ih_x.chunk(4, 1)
        ev_hh_i, ev_hh_f, ev_hh_c, ev_hh_o = ev_hh_x.chunk(4, 1)

        ctx.save_for_backward(et_ih_y, et_hh_y, input_size)

        return hy, cy, ev_ih_y, ev_hh_y, forgetgate_y

    @staticmethod
    # grad_ev_ih and grad_ev_hh should always be None
    def backward(ctx, grad_hy, grad_cy, grad_ev_ih, grad_ev_hh):
        et_ih_y, et_hh_y, input_size = ctx.saved_tensors
        grad_weight_ih = grad_weight_hh = grad_bias_ih = grad_bias_hh = None

        # create tensors for the weight gradient
        grad_weight_ih = torch.Tensor(et_ih_y.size())
        grad_weight_hh = torch.Tensor(et_hh_y.size())
        
        # ingate, forgetgate and cellgate
        for i in range(0, 4 * input_size, input_size):
            grad_weight_ih[:, i:i + input_size] = et_ih_y[:, i:i + input_size] * grad_hy[:, i:i + input_size]
            grad_weight_hh[:, i:i + input_size] = et_hh_y[:, i:i + input_size] * grad_hy[:, i:i + input_size]

        # TODO: calculate for bias

        # grad_input, grad_ev_ih, grad_ev_hh, grad_hx, grad_cx, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh
        return None, None, None, None, None, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh


class EpropCell(LSTMCell):
    """ 
    Custom LSTM Cell implementation using jit adopted from:
    https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(EpropCell, self).__init__(input_size, hidden_size, bias)

        # Initialize eligibility trace to zero
        self.eligibility_vector_ih = torch.zeros(hidden_size, input_size)
        self.eligibility_vector_hh = torch.zeros(hidden_size, hidden_size)

    @jit.script_method
    def forward(self, input, hx, cx):
        hy, cy, self.eligibility_vector_ih, self.eligibility_vector_hh = EProp1.apply(
            self.eligibility_vector_ih,
            self.eligibility_vector_hh,
            input, 
            hx, 
            cx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh)

        return hy, hy, cy