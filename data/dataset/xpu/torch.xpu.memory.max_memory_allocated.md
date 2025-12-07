torch.xpu.memory.max_memory_allocated 
================================================================================================================

torch.xpu.memory. max_memory_allocated ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/memory.py#L136) 
:   Return the maximum GPU memory occupied by tensors in bytes for a given device. 

By default, this returns the peak allocated memory since the beginning of
this program. `reset_peak_memory_stats()`  can be used to
reset the starting point in tracking this metric. For example, these two
functions can measure the peak allocated memory usage of each iteration in a
training loop. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – selected device. Returns
statistic for the current device, given by [`current_device()`](torch.xpu.current_device.html#torch.xpu.current_device "torch.xpu.current_device")  ,
if `device`  is `None`  (default).

Return type
:   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")

