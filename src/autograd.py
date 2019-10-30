import torch
from util import to_device


class SyntheticGradient(torch.autograd.Function):
    """
    This is a passthrough autograd function that is used to set the error dE/d(s^{tm+1}) = SG(z^{tm}, \phi)
    In this implementation dE/d(s^{tm+1}) is equal to grad_last_h and should be of dimension (batch_size, hidden_size)
    with all 0. SG(z^{tm}, \phi) is equal to synth_grad and has the same dimension as grad_last_h.
    """

    @staticmethod
    def forward(
            ctx, 
            last_c,
            output,
            synth_grad):
        ctx.intermediate_results = synth_grad
        return last_c, output, synth_grad

    @staticmethod
    # grad_ev_ih and grad_ev_hh should always be None
    def backward(ctx, grad_last_c, grad_output, grad_synth_grad):
        synth_grad = ctx.intermediate_results
        #print(synth_grad)
        return synth_grad, grad_output, None
        

class ForgetGate(torch.autograd.Function):
    """
    This is a passthrough autograd function that is used to let the eprop autograd function
    know about future forget gate values by setting grad_forgetgate_x = forgetgate_y, since
    the gradient is not used otherwise.
    """

    @staticmethod
    def forward(
            ctx, 
            forgetgate_x,
            forgetgate_y):
        ctx.intermediate_results = forgetgate_y
        return forgetgate_x, forgetgate_y

    @staticmethod
    # grad_ev_ih and grad_ev_hh should always be None
    def backward(ctx, grad_forgetgate_y, grad_forgetgate_x):
        forgetgate_y = ctx.intermediate_results
        #print(forgetgate_y)
        return forgetgate_y, forgetgate_y


class EPropBase(torch.autograd.Function):
    """
    This is the base class for the autograd function that implements the Eprop algorithm. This
    base class defines the forward pass including the default LSTM forward pass as well as the
    calculation of the eligibility traces and vectors.
    """

    @staticmethod
    def forward(
            ctx, 
            ev_w_ih_x,
            ev_w_hh_x,
            ev_b_x,
            forgetgate_x,
            input_data, 
            hx, 
            cx,
            weight_ih, 
            weight_hh, 
            bias_ih=None, 
            bias_hh=None):
        # calculate gates ...
        gates = (torch.mm(input_data, weight_ih.t()) + bias_ih + torch.mm(hx, weight_hh.t()) + bias_hh)
        ingate, forgetgate_y, cellgate, outgate = gates.chunk(4, 1)

        # ... and gate activations
        ingate = torch.sigmoid(ingate)
        forgetgate_y = torch.sigmoid(forgetgate_y)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate_y * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        # TODO: calculate new eligibility vector and trace
        # There exist distinct eligibility traces and vectors for the followiug parts of the LSTM cell:
        # - input to hidden connections, hidden to hidden connections, bias 
        # all for each: 
        # - ... inputgate, forgetgate and cellgate
        # => overall 3 * 3 = 9 eligibility traces        
        hidden_size = hy.size(1)
        batch_size = input_data.size(0)
        input_size = input_data.size(1)

        ingate = ingate.unsqueeze(2)
        forgetgate_y = forgetgate_y.unsqueeze(2)
        cellgate = cellgate.unsqueeze(2)
        outgate = outgate.unsqueeze(2)
        input_data = input_data.unsqueeze(1)
        hx = hx.unsqueeze(2)
        cx = cx.unsqueeze(2)

        forgetgate_x = forgetgate_x.repeat(1, 3, 1)        
        ones = to_device(torch.ones(ingate.size()))

        # the new eligibility vectors ...
        ev_w_ih_y = ev_w_ih_x * forgetgate_x
        ev_w_hh_y = ev_w_hh_x * forgetgate_x
        ev_b_y = ev_b_x * forgetgate_x

        # ingate
        base = ingate * (ones - ingate) * cellgate
        ev_w_hh_y[:, :hidden_size, :] += base * hx
        ev_w_ih_y[:, :hidden_size, :] += base * input_data
        ev_b_y[:, :hidden_size, :] += base
        
        # forgetgate
        #base = forgetgate_y * (ones - forgetgate_y) * cellgate
        base = forgetgate_y * (ones - forgetgate_y) * cx
        ev_w_hh_y[:, hidden_size:(2 * hidden_size), :] += base * hx
        ev_w_ih_y[:, hidden_size:(2 * hidden_size), :] += base * input_data
        ev_b_y[:, hidden_size:(2 * hidden_size), :] += base

        # cellgate
        base = ingate * (ones - cellgate**2)
        ev_w_hh_y[:, (2 * hidden_size):(3 * hidden_size), :] += base * hx
        ev_w_ih_y[:, (2 * hidden_size):(3 * hidden_size), :] += base * input_data
        ev_b_y[:, (2 * hidden_size):(3 * hidden_size), :] += base

         # ... and eligibility traces
        et_w_ih_y = to_device(torch.zeros(batch_size, 4 * hidden_size, input_size, requires_grad=False))
        et_w_hh_y = to_device(torch.zeros(batch_size, 4 * hidden_size, hidden_size, requires_grad=False))
        et_b_y = to_device(torch.zeros(batch_size, 4 * hidden_size, 1, requires_grad=False))

        # calculate eligibility traces by multiplying the eligibility vectors with the outgate
        tmp_outgate = outgate.repeat(1, 3, 1)
        et_w_ih_y[:, :3 * hidden_size, :] = ev_w_ih_y * tmp_outgate
        et_w_hh_y[:, :3 * hidden_size, :] = ev_w_hh_y * tmp_outgate
        et_b_y[:, :3 * hidden_size, :] = ev_b_y * tmp_outgate
        
        # The gradient of the output gate is only dependent on the observable state
        # => just use normal gradient calculation of dE/dh * dh/dweight 
        # => calculate second part of that equation now for input to hidden, hidden to hidden 
        #    and bias connections and multiply in the backward pass
        base = outgate * (ones - outgate) * cy.unsqueeze(2)
        et_w_hh_y[:, (3 * hidden_size):(4 * hidden_size)] = base * hx
        et_w_ih_y[:, (3 * hidden_size):(4 * hidden_size)] = base * input_data
        et_b_y[:, (3 * hidden_size):(4 * hidden_size)] = base

        ctx.save_for_backward(et_w_ih_y, et_w_hh_y, et_b_y, weight_hh, cx.clone().squeeze(), cy.clone().squeeze(), outgate.squeeze(), ingate.squeeze(), cellgate.squeeze(), forgetgate_y.squeeze())

        return hy, cy, ev_w_ih_y, ev_w_hh_y, ev_b_y, forgetgate_y


class EProp1(EPropBase):
    """
    This is the autograd function that implements EProp1: This means we approximate the error gradient
    dE/dz which is composed of the current error partialE/partialz and the backpropagated future error
    by only using the current error.
    """

    @staticmethod
    # grad_ev_ih and grad_ev_hh should always be None
    def backward(ctx, grad_hy, grad_cy, grad_ev_w_ih, grad_ev_w_hh, grad_ev_b, grad_forgetgate_y):
        et_w_ih_y, et_w_hh_y, et_b_y, _, _, _, _, _, _, _ = ctx.saved_variables

        # Approximate dE/dh by substituting only with local error (\partial E)/(\partial h)
        tmp_grad_hy = grad_hy.unsqueeze(2).repeat(1, 4, 1)

        grad_weight_ih = et_w_ih_y * tmp_grad_hy
        grad_weight_hh = et_w_hh_y * tmp_grad_hy
        grad_bias = et_b_y * tmp_grad_hy

        # grad_ev_ih, grad_ev_hh, grad_ev_b, grad_forgetgate_x, grad_input, grad_hx, grad_cx, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh
        return None, None, None, None, None, None, None, grad_weight_ih, grad_weight_hh, grad_bias.squeeze(), grad_bias.squeeze()


class EProp3(EPropBase):
    """
    This is the autograd function that implements EProp3: This means we no longer look at the entire time series,
    but at truncated parts of length deltaT. Within these truncated parts we actually calculate the error gradient
    dE/dz by adding the local error partialE/partialz to the backpropagated future error. Since we are truncating the
    time series 
    """

    @staticmethod
    # grad_ev_ih and grad_ev_hh should always be None
    def backward(ctx, grad_hy, grad_cy, grad_ev_w_ih, grad_ev_w_hh, grad_ev_b, forgetgate_y):
        # grad_hy = dE/dh^t = partial(E)/partial(h^{t}) + sum_i dE/dc^{t+1} * partial(c^{t+1})/partial(h^t) (the backpropagated error w.r.t. the output)
        # grad_cy = dE/dc^{t+1} (the backpropagated error w.r.t. the next cell state)

        et_w_ih_y, et_w_hh_y, et_b_y, weight_hh, cx, cy, outgate, ingate, cellgate, forgetgate_x,  = ctx.saved_variables
        #print(grad_hy[0])

        # first calcualate the gradient of the current cell state dE/ds^{t} = putput_part + hidden_part where
        # output_part = dE/dh^t * partial(h^t)/partial(c^t) = grad_hy * (outgate * sig_deriv_cy)
        # hidden_part = dE/dc^{t+1} * partial(c^{t+1})/partial(c^t) = grad_cy * (forgetgate_y)
        ones = torch.ones_like(cy)
        sig_deriv_cy = torch.sigmoid(cy) * (ones - torch.sigmoid(cy))
        output_part = outgate * sig_deriv_cy * grad_hy
        hidden_part = grad_cy * forgetgate_y.squeeze()
        grad_cx = output_part + hidden_part

        # next calculate the gradient of the gates ...
        tanh_deriv_cellgate = (ones - torch.sigmoid(cellgate)**2)
        sig_deriv_ingate = torch.sigmoid(ingate) * (ones - torch.sigmoid(ingate))
        sig_deriv_forgetgate = torch.sigmoid(forgetgate_x) * (ones - torch.sigmoid(forgetgate_x))
        sig_deriv_outgate = torch.sigmoid(outgate) * (ones - torch.sigmoid(outgate))

        grad_cellgate = grad_cx * tanh_deriv_cellgate * ingate
        grad_ingate = grad_cx * sig_deriv_ingate * cellgate
        grad_forgetgate_x = grad_cx * sig_deriv_forgetgate * cx
        grad_outgate = grad_hy * sig_deriv_outgate * torch.tanh(cy)

        # ... to finally calculate sum_i dE/dc^{t} * partial(c^{t})/partial(h^{t-1}) = grad_hx which will be added to the local error
        # of the next backpropagation step
        grad_gates = torch.cat([grad_ingate, grad_forgetgate_x, grad_cellgate, grad_outgate], dim=1) 
        grad_hx = torch.matmul(weight_hh.t(), grad_gates.t()).t()

        # Calculate the gradient of the weights by multiplying the learning signal dE/dh^{t} = grad_hy with the
        # eligibility traces calculated in the forward pass
        temp_grad_hy = grad_hy.unsqueeze(2).repeat(1, 4, 1)

        grad_weight_ih = et_w_ih_y * temp_grad_hy
        grad_weight_hh = et_w_hh_y * temp_grad_hy
        grad_bias = et_b_y * temp_grad_hy

        # grad_ev_ih, grad_ev_hh, grad_ev_b, grad_forgetgate_x, grad_input, grad_hx, grad_cx, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh
        return None, None, None, None, None, grad_hx, grad_cx, grad_weight_ih, grad_weight_hh, grad_bias.squeeze(), grad_bias.squeeze()