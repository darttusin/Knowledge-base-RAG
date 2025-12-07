torch.mtia.get_device_capability 
======================================================================================================

torch.mtia. get_device_capability ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/mtia/__init__.py#L207) 
:   Return capability of a given device as a tuple of (major version, minor version). 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – statistics for the current device, given by current_device(),
if device is None (default).

Return type
:   [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.13)")  [ [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  , [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ]

