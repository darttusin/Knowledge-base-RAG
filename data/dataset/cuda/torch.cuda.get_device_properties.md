torch.cuda.get_device_properties 
======================================================================================================

torch.cuda. get_device_properties ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py#L604) 
:   Get the properties of a device. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – device for which to return the
properties of the device. It uses the current device, given by [`current_device()`](torch.cuda.current_device.html#torch.cuda.current_device "torch.cuda.current_device")  , if [`device`](torch.cuda.device.html#torch.cuda.device "torch.cuda.device")  is `None`  (default).

Returns
:   the properties of the device

Return type
:   _CudaDeviceProperties

