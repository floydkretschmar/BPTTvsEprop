import torch

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