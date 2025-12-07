torch.xpu.current_stream 
=====================================================================================

torch.xpu. current_stream ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/__init__.py#L368) 
:   Return the currently selected [`Stream`](torch.xpu.Stream.html#torch.xpu.Stream "torch.xpu.Stream")  for a given device. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – selected device. Returns
the currently selected [`Stream`](torch.xpu.Stream.html#torch.xpu.Stream "torch.xpu.Stream")  for the current device, given
by [`current_device()`](torch.xpu.current_device.html#torch.xpu.current_device "torch.xpu.current_device")  , if [`device`](torch.xpu.device.html#torch.xpu.device "torch.xpu.device")  is `None`  (default).

Return type
:   [*Stream*](torch.xpu.Stream.html#torch.xpu.Stream "torch.xpu.streams.Stream")

