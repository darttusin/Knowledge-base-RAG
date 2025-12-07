torch.xpu.memory.memory_allocated 
=======================================================================================================

torch.xpu.memory. memory_allocated ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/memory.py#L120) 
:   Return the current GPU memory occupied by tensors in bytes for a given device. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – selected device. Returns
statistic for the current device, given by [`current_device()`](torch.xpu.current_device.html#torch.xpu.current_device "torch.xpu.current_device")  ,
if `device`  is `None`  (default).

Return type
:   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")

Note 

This is likely less than the amount shown in *xpu-smi* since some
unused memory can be held by the caching allocator and some context
needs to be created on GPU.

