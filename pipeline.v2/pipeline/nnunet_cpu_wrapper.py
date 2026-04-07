"""
Wrapper to run nnUNetv2_predict on CPU by patching torch after import.
"""
import sys
import torch

# Patch AFTER torch is loaded - redirect all CUDA calls to CPU
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

# Patch Tensor.to() to redirect CUDA to CPU
_original_to = torch.Tensor.to
def cpu_only_to(self, *args, **kwargs):
    if len(args) > 0:
        device = args[0]
        if isinstance(device, str) and 'cuda' in device:
            args = ('cpu',) + args[1:]
        elif isinstance(device, torch.device) and device.type == 'cuda':
            args = (torch.device('cpu'),) + args[1:]
    if 'device' in kwargs and isinstance(kwargs['device'], str) and 'cuda' in kwargs['device']:
        kwargs['device'] = 'cpu'
    return _original_to(self, *args, **kwargs)

torch.Tensor.to = cpu_only_to

# Patch nn.Module.to() similarly
_original_module_to = torch.nn.Module.to
def cpu_only_module_to(self, *args, **kwargs):
    if len(args) > 0:
        device = args[0]
        if isinstance(device, str) and 'cuda' in device:
            args = ('cpu',) + args[1:]
        elif isinstance(device, torch.device) and device.type == 'cuda':
            args = (torch.device('cpu'),) + args[1:]
    if 'device' in kwargs and isinstance(kwargs['device'], str) and 'cuda' in kwargs['device']:
        kwargs['device'] = 'cpu'
    return _original_module_to(self, *args, **kwargs)

torch.nn.Module.to = cpu_only_module_to

# Patch pin_memory to no-op on CPU
_original_pin_memory = torch.Tensor.pin_memory
def cpu_safe_pin_memory(self):
    if self.device.type == 'cuda':
        return _original_pin_memory(self)
    return self

torch.Tensor.pin_memory = cpu_safe_pin_memory

# Now import and run nnUNet
from nnunetv2.inference.predict_from_raw_data import predict_entry_point

if __name__ == "__main__":
    sys.exit(predict_entry_point())
