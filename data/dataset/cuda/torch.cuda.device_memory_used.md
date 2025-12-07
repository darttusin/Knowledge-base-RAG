torch.cuda.device_memory_used 
================================================================================================

torch.cuda. device_memory_used ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py#L1308) 
:   Return used global (device) memory in bytes as given by *nvidia-smi* or *amd-smi* . 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – selected device. Returns
statistic for the current device, given by [`current_device()`](torch.cuda.current_device.html#torch.cuda.current_device "torch.cuda.current_device")  ,
if [`device`](torch.cuda.device.html#torch.cuda.device "torch.cuda.device")  is `None`  (default).

Return type
:   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")

