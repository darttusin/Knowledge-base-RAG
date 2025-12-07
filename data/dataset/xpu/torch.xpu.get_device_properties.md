torch.xpu.get_device_properties 
====================================================================================================

torch.xpu. get_device_properties ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/__init__.py#L250) 
:   Get the properties of a device. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – device for which to return the
properties of the device.

Returns
:   the properties of the device

Return type
:   _XpuDeviceProperties

