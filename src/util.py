import torch
import torch.jit as jit

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_device(torch_object):
    # dirty hack because .to(device) is not available for ScriptModules right now
    if isinstance(torch_object, jit.ScriptModule):
        if DEVICE.type == 'cuda':
            return torch_object.cuda()
        else:
            return torch_object

    return torch_object.to(DEVICE)