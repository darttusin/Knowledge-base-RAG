torch.xpu.memory.mem_get_info 
================================================================================================

torch.xpu.memory. mem_get_info ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/memory.py#L181) 
:   Return the global free and total GPU memory for a given device. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – selected device. Returns
statistic for the current device, given by [`current_device()`](torch.xpu.current_device.html#torch.xpu.current_device "torch.xpu.current_device")  ,
if `device`  is `None`  (default).

Returns
:   the memory available on the device in units of bytes.
int: the total memory on the device in units of bytes

Return type
:   [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")

