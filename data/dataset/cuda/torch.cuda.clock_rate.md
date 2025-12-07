torch.cuda.clock_rate 
===============================================================================

torch.cuda. clock_rate ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py#L1408) 
:   Return the clock speed of the GPU SM in MHz (megahertz) over the past sample period as given by *nvidia-smi* . 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – selected device. Returns
statistic for the current device, given by [`current_device()`](torch.cuda.current_device.html#torch.cuda.current_device "torch.cuda.current_device")  ,
if [`device`](torch.cuda.device.html#torch.cuda.device "torch.cuda.device")  is `None`  (default).

Return type
:   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")

Warning: Each sample period may be between 1 second and 1/6 second,
depending on the product being queried.

