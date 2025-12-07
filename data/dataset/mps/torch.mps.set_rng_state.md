torch.mps.set_rng_state 
====================================================================================

torch.mps. set_rng_state ( *new_state*  , *device = 'mps'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/mps/__init__.py#L47) 
:   Sets the random number generator state. 

Parameters
:   * **new_state** ( *torch.ByteTensor*  ) – The desired state
* **device** ( [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* *optional*  ) – The device to set the RNG state.
Default: `'mps'`  (i.e., `torch.device('mps')`  , the current MPS device).

