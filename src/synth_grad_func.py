import torch

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