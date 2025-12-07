torch.cuda.synchronize 
================================================================================

torch.cuda. synchronize ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py#L1075) 
:   Wait for all kernels in all streams on a CUDA device to complete. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – device for which to synchronize.
It uses the current device, given by [`current_device()`](torch.cuda.current_device.html#torch.cuda.current_device "torch.cuda.current_device")  ,
if [`device`](torch.cuda.device.html#torch.cuda.device "torch.cuda.device")  is `None`  (default).

