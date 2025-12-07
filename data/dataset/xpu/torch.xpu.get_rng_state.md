torch.xpu.get_rng_state 
====================================================================================

torch.xpu. get_rng_state ( *device = 'xpu'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/xpu/random.py#L11) 
:   Return the random number generator state of the specified GPU as a ByteTensor. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – The device to return the RNG state of.
Default: `'xpu'`  (i.e., `torch.device('xpu')`  , the current XPU device).

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

Warning 

This function eagerly initializes XPU.

