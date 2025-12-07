torch.Tensor.new_empty 
=================================================================================

Tensor. new_empty ( *size*  , *** , *dtype = None*  , *device = None*  , *requires_grad = False*  , *layout = torch.strided*  , *pin_memory = False* ) → [Tensor](../tensors.html#torch.Tensor "torch.Tensor") 
:   Returns a Tensor of size [`size`](torch.Tensor.size.html#torch.Tensor.size "torch.Tensor.size")  filled with uninitialized data.
By default, the returned Tensor has the same [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  and [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  as this tensor. 

Parameters
: **size** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *...*  ) – a list, tuple, or [`torch.Size`](../size.html#torch.Size "torch.Size")  of integers defining the
shape of the output tensor.

Keyword Arguments
:   * **dtype** ( [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  , optional) – the desired type of returned tensor.
Default: if None, same [`torch.dtype`](../tensor_attributes.html#torch.dtype "torch.dtype")  as this tensor.
* **device** ( [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  , optional) – the desired device of returned tensor.
Default: if None, same [`torch.device`](../tensor_attributes.html#torch.device "torch.device")  as this tensor.
* **requires_grad** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If autograd should record operations on the
returned tensor. Default: `False`  .
* **layout** ( [`torch.layout`](../tensor_attributes.html#torch.layout "torch.layout")  , optional) – the desired layout of returned Tensor.
Default: `torch.strided`  .
* **pin_memory** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *optional*  ) – If set, returned tensor would be allocated in
the pinned memory. Works only for CPU tensors. Default: `False`  .

Example: 

```
>>> tensor = torch.ones(())
>>> tensor.new_empty((2, 3))
tensor([[ 5.8182e-18,  4.5765e-41, -1.0545e+30],
        [ 3.0949e-41,  4.4842e-44,  0.0000e+00]])

```

