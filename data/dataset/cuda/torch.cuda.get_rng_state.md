torch.cuda.get_rng_state 
======================================================================================

torch.cuda. get_rng_state ( *device = 'cuda'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/random.py#L24) 
:   Return the random number generator state of the specified GPU as a ByteTensor. 

Parameters
: **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – The device to return the RNG state of.
Default: `'cuda'`  (i.e., `torch.device('cuda')`  , the current CUDA device).

Return type
:   [*Tensor*](../tensors.html#torch.Tensor "torch.Tensor")

Warning 

This function eagerly initializes CUDA.

