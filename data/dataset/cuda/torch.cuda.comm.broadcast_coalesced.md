torch.cuda.comm.broadcast_coalesced 
===========================================================================================================

torch.cuda.comm. broadcast_coalesced ( *tensors*  , *devices*  , *buffer_size = 10485760* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/parallel/comm.py#L49) 
:   Broadcast a sequence of tensors to the specified GPUs. 

Small tensors are first coalesced into a buffer to reduce the number of synchronizations. 

Parameters
:   * **tensors** ( *sequence*  ) – tensors to broadcast. Must be on the same device,
either CPU or GPU.
* **devices** ( *Iterable* *[* [*torch.device*](../tensor_attributes.html#torch.device "torch.device") *,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *or* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]*  ) – an iterable of GPU
devices, among which to broadcast.
* **buffer_size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – maximum size of the buffer used for coalescing

Returns
:   A tuple containing copies of `tensor`  , placed on `devices`  .

