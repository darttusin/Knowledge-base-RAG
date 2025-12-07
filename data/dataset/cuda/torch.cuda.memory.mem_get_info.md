torch.cuda.memory.mem_get_info 
==================================================================================================

torch.cuda.memory. mem_get_info ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/memory.py#L822) 
:   Return the global free and total GPU memory for a given device using cudaMemGetInfo. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – selected device. Returns
statistic for the current device, given by [`current_device()`](torch.cuda.current_device.html#torch.cuda.current_device "torch.cuda.current_device")  ,
if `device`  is `None`  (default) or if the device index is not specified.

Return type
:   [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  , [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ]

Note 

See [Memory management](../notes/cuda.html#cuda-memory-management)  for more
details about GPU memory management.

