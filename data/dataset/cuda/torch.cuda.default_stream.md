torch.cuda.default_stream 
=======================================================================================

torch.cuda. default_stream ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py#L1119) 
:   Return the default [`Stream`](torch.cuda.Stream.html#torch.cuda.Stream "torch.cuda.Stream")  for a given device. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – selected device. Returns
the default [`Stream`](torch.cuda.Stream.html#torch.cuda.Stream "torch.cuda.Stream")  for the current device, given by [`current_device()`](torch.cuda.current_device.html#torch.cuda.current_device "torch.cuda.current_device")  , if [`device`](torch.cuda.device.html#torch.cuda.device "torch.cuda.device")  is `None`  (default).

Return type
:   [*Stream*](torch.cuda.Stream.html#torch.cuda.Stream "torch.cuda.streams.Stream")

