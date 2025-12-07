torch.mtia.current_stream 
=======================================================================================

torch.mtia. current_stream ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/mtia/__init__.py#L151) 
:   Return the currently selected [`Stream`](torch.mtia.Stream.html#torch.mtia.Stream "torch.mtia.Stream")  for a given device. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – selected device. Returns
the currently selected [`Stream`](torch.mtia.Stream.html#torch.mtia.Stream "torch.mtia.Stream")  for the current device, given
by [`current_device()`](torch.mtia.current_device.html#torch.mtia.current_device "torch.mtia.current_device")  , if [`device`](torch.mtia.device.html#torch.mtia.device "torch.mtia.device")  is `None`  (default).

Return type
:   [*Stream*](torch.Stream.html#torch.Stream "torch.Stream")

