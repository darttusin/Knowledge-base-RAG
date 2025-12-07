torch.mtia.memory_stats 
===================================================================================

torch.mtia. memory_stats ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/mtia/memory.py#L13) 
:   Return a dictionary of MTIA memory allocator statistics for a given device. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *, or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – statistics for the current device, given by current_device(),
if device is None (default).

Return type
:   [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)")  [ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.13)")  ]

