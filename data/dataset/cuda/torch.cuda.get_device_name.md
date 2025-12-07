torch.cuda.get_device_name 
==========================================================================================

torch.cuda. get_device_name ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py#L572) 
:   Get the name of a device. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – device for which to return the
name. This function is a no-op if this argument is a negative
integer. It uses the current device, given by [`current_device()`](torch.cuda.current_device.html#torch.cuda.current_device "torch.cuda.current_device")  ,
if [`device`](torch.cuda.device.html#torch.cuda.device "torch.cuda.device")  is `None`  (default).

Returns
:   the name of the device

Return type
:   [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")

