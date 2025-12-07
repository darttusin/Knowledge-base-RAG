torch.xpu.memory.memory_reserved 
=====================================================================================================

torch.xpu.memory. memory_reserved ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/memory.py#L153) 
:   Return the current GPU memory managed by the caching allocator in bytes for a given device. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – selected device. Returns
statistic for the current device, given by [`current_device()`](torch.xpu.current_device.html#torch.xpu.current_device "torch.xpu.current_device")  ,
if `device`  is `None`  (default).

Return type
:   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")

