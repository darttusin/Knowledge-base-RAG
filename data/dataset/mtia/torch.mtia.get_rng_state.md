torch.mtia.get_rng_state 
======================================================================================

torch.mtia. get_rng_state ( *device = 'mtia'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/mtia/__init__.py#L349) 
:   Returns the random number generator state as a ByteTensor. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – The device to return the RNG state of.
Default: `'mtia'`  (i.e., `torch.device('mtia')`  , the current mtia device).

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

