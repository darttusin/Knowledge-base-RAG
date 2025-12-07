torch.xpu.get_device_capability 
====================================================================================================

torch.xpu. get_device_capability ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/__init__.py#L223) 
:   Get the xpu capability of a device. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* *optional*  ) – device for which to
return the device capability. This function is a no-op if this
argument is a negative integer. It uses the current device, given by [`current_device()`](torch.xpu.current_device.html#torch.xpu.current_device "torch.xpu.current_device")  , if [`device`](torch.xpu.device.html#torch.xpu.device "torch.xpu.device")  is `None`  (default).

Returns
:   the xpu capability dictionary of the device

Return type
:   Dict[ [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , Any]

