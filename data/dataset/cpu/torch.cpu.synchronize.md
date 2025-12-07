torch.cpu.synchronize 
==============================================================================

torch.cpu. synchronize ( *device = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cpu/__init__.py#L78) 
:   Waits for all kernels in all streams on the CPU device to complete. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – ignored, there’s only one CPU device.

N.B. This function only exists to facilitate device-agnostic code.

