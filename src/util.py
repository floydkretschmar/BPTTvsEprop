import torch
import torch.jit as jit

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_device(torch_object, mixed_prec=False):
    if torch.cuda.is_available():
        return torch_object.cuda() if not mixed_prec else torch_object.cuda().half()
    else:
        return torch_object

    '''# dirty hack because .to(device) is not available for ScriptModules right now
    if isinstance(torch_object, jit.ScriptModule):
        if DEVICE.type == 'cuda':
            return torch_object.cuda()
        else:
            return torch_object

    return torch_object.to(DEVICE)

def prepare_parameter_lists(model):
    model_params = [p for p in model.parameters() if p.requires_grad]
    master_params = [p.detach().clone().float() for p in model_params]

    for p in master_params:
        p.requires_grad = True

    return model_params, master_params

def copy_master_parameters_to_model(model_parameters, master_parameters):
    for model, master in zip(model_parameters, master_parameters):
        model.data.copy_(master.data)

def copy_model_gradients_to_master(model_parameters, master_parameters):
    for model, master in zip(model_parameters, master_parameters):
        if model.grad is not None:
            if master.grad is None:
                master.grad = torch.autograd.Variable(master.data.new(*master.data.size()))
            master.grad.data.copy_(model.grad.data)'''