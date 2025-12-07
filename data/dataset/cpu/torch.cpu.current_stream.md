torch.cpu.current_stream 
=====================================================================================

torch.cpu. current_stream ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cpu/__init__.py#L124) 
:   Returns the currently selected [`Stream`](torch.cpu.Stream.html#torch.cpu.Stream "torch.cpu.Stream")  for a given device. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – Ignored.

Return type
:   [*Stream*](torch.cpu.Stream.html#torch.cpu.Stream "torch.cpu.Stream")

N.B. This function only exists to facilitate device-agnostic code

